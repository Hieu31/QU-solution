"""Compare candidate confidence estimators on pinned ReparoS top-10 output.

All labels are strict matches to diagnostic gold, not human correctness labels.
The final test split is chosen by canonical expected text and never used to
select the method. Development methods are compared with grouped 5-fold CV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from train_reparos_candidate_confidence import (
    DEFAULT_INPUT, FEATURE_NAMES, features, fit_logistic, normalized, sigmoid,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/reparos_base_v3_confidence_14088/confidence_comparison"
BASE = {name: index for index, name in enumerate(FEATURE_NAMES)}
EXTRA_NAMES = (
    "log1p_margin", "log1p_gap", "gap_to_next", "gap_from_previous",
    "margin_x_rank1", "margin_x_rank2", "change_x_rank1", "score_x_rank1",
    "accent_fold_change", "score_squared",
)
ALL_NAMES = FEATURE_NAMES + EXTRA_NAMES
IDX = {name: index for index, name in enumerate(ALL_NAMES)}
RANKS = tuple(f"rank_{rank}" for rank in range(1, 11))
CORE = ("sequence_score", "gap_from_top_score", *RANKS)
FULL = FEATURE_NAMES
VARIANTS = {
    "rank_prior": (),
    "score_only": ("sequence_score",),
    "rank_only": RANKS,
    "score_rank": ("sequence_score", *RANKS),
    "score_rank_gap": CORE,
    "score_rank_margin": ("sequence_score", "top1_top2_margin", *RANKS),
    "score_rank_gap_margin": ("sequence_score", "gap_from_top_score", "top1_top2_margin", *RANKS),
    "full_without_margin": tuple(name for name in FULL if name != "top1_top2_margin"),
    "full_with_margin": FULL,
    "full_plus_local_gap": (*FULL, "gap_to_next", "gap_from_previous"),
    "full_plus_nonlinear": (*FULL, "log1p_margin", "log1p_gap", "score_squared"),
    "full_plus_accent_fold": (*FULL, "accent_fold_change"),
    "full_plus_interactions": (*FULL, "margin_x_rank1", "margin_x_rank2", "change_x_rank1", "score_x_rank1"),
    "full_all_extras": ALL_NAMES,
}


def group_hash(seed: int, expected: str) -> float:
    key = f"{seed}\0{normalized(expected)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64


def accent_fold(value: str) -> str:
    text = unicodedata.normalize("NFD", normalized(value).lower())
    return "".join(char for char in text if unicodedata.category(char) != "Mn").replace("đ", "d")


def load_data(path: Path, seed: int, folds: int, test_fraction: float) -> dict:
    queries, rows, labels, suites, final_test, cv_fold = [], [], [], [], [], []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            query = json.loads(line)
            hypotheses, scores, gold_labels = (
                query["hypotheses"], query["sequence_scores"], query["strict_exact_labels"]
            )
            if len(hypotheses) != len(scores) or len(scores) != 10 or len(gold_labels) != 10:
                raise ValueError(f"unaligned top 10: {query['suite']}/{query['query_id']}")
            queries.append(query)
            test = group_hash(seed, query["expected"]) < test_fraction
            fold = min(int(group_hash(seed + 1, query["expected"]) * folds), folds - 1)
            top_score, margin = float(scores[0]), float(scores[0]) - float(scores[1])
            input_fold = accent_fold(query["input"])
            for index, (hypothesis, score, label) in enumerate(zip(hypotheses, scores, gold_labels)):
                rank, score = index + 1, float(score)
                base = features(query["input"], hypothesis, score, top_score, margin, rank)
                gap = top_score - score
                change = base[BASE["character_change_ratio"]]
                accent_change = 1 - SequenceMatcher(
                    None, input_fold, accent_fold(hypothesis), autojunk=False
                ).ratio()
                extras = [
                    np.log1p(max(margin, 0)), np.log1p(max(gap, 0)),
                    score - float(scores[index + 1]) if index < 9 else 0.0,
                    float(scores[index - 1]) - score if index > 0 else 0.0,
                    margin * (rank == 1), margin * (rank == 2),
                    change * (rank == 1), score * (rank == 1),
                    accent_change, score * score,
                ]
                rows.append(base + extras)
                labels.append(int(label))
                suites.append(query["suite"])
                final_test.append(test)
                cv_fold.append(fold)
    return {
        "queries": queries,
        "x": np.asarray(rows, dtype=np.float64),
        "y": np.asarray(labels, dtype=np.float64),
        "suite": np.asarray(suites, dtype=object),
        "test": np.asarray(final_test, dtype=bool),
        "fold": np.asarray(cv_fold, dtype=np.int8),
        "rank": np.tile(np.arange(1, 11), len(queries)),
    }


def fit_variant(name: str, x: np.ndarray, y: np.ndarray, ranks: np.ndarray, mask: np.ndarray) -> dict:
    if name == "rank_prior":
        priors = []
        for rank in range(1, 11):
            selected = mask & (ranks == rank)
            priors.append(float((y[selected].sum() + 1) / (selected.sum() + 2)))
        return {"kind": "rank_prior", "priors": priors}
    columns = [IDX[feature] for feature in VARIANTS[name]]
    train_x = x[mask][:, columns]
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack((np.ones(len(train_x)), (train_x - mean) / scale))
    parameters, iterations = fit_logistic(design, y[mask], 1e-3, 40)
    return {
        "kind": "logistic", "features": VARIANTS[name], "mean": mean.tolist(),
        "scale": scale.tolist(), "parameters": parameters.tolist(), "iterations": iterations,
    }


def predict_variant(model: dict, x: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    if model["kind"] == "rank_prior":
        return np.asarray(model["priors"])[ranks - 1]
    columns = [IDX[feature] for feature in model["features"]]
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    parameters = np.asarray(model["parameters"])
    design = np.column_stack((np.ones(len(x)), (x[:, columns] - mean) / scale))
    return sigmoid(design @ parameters)


def metrics(p: np.ndarray, y: np.ndarray, ranks: np.ndarray) -> dict:
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    result = {"candidates": len(y), "positives": int(y.sum()), "mean_confidence": float(p.mean())}
    for name, mask in (("all", np.ones(len(y), dtype=bool)), ("top1", ranks == 1), ("rank2to10", ranks > 1)):
        pred, truth = clipped[mask], y[mask]
        bins = calibration_bins(pred, truth)
        ece = sum(
            item["count"] / len(pred) * abs(item["mean_confidence"] - item["observed_exact_rate"])
            for item in bins if item["count"]
        )
        result[name] = {
            "brier": float(np.mean((pred - truth) ** 2)),
            "log_loss": float(np.mean(-truth * np.log(pred) - (1 - truth) * np.log1p(-pred))),
            "observed_rate": float(truth.mean()),
            "mean_confidence": float(pred.mean()),
            "ece10": float(ece),
            "confidence_ge_0_8_count": int((pred >= 0.8).sum()),
            "confidence_ge_0_8_observed_rate": float(truth[pred >= 0.8].mean()) if (pred >= 0.8).any() else None,
        }
    return result


def calibration_bins(p: np.ndarray, y: np.ndarray) -> list[dict]:
    bins = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        mask = (p >= lower) & (p < upper if index < 9 else p <= upper)
        bins.append({
            "range": [lower, upper], "count": int(mask.sum()),
            "mean_confidence": float(p[mask].mean()) if mask.any() else None,
            "observed_exact_rate": float(y[mask].mean()) if mask.any() else None,
        })
    return bins


def paired_brier_gain(reference: np.ndarray, selected: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    # Candidate rows are contiguous blocks of 10 for each query.
    ref_error = np.mean((reference - labels).reshape(-1, 10) ** 2, axis=1)
    selected_error = np.mean((selected - labels).reshape(-1, 10) ** 2, axis=1)
    gain = ref_error - selected_error
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(gain), size=(1000, len(gain)))
    sampled = gain[draws].mean(axis=1)
    return {
        "mean_brier_gain": float(gain.mean()),
        "bootstrap_95pct_interval": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "resampling_unit": "query; 1000 paired bootstrap resamples",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7193)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 0.5 or args.folds < 3:
        parser.error("test fraction must be between 0 and 0.5; folds must be at least 3")
    started = time.perf_counter()
    data = load_data(args.input.resolve(), args.seed, args.folds, args.test_fraction)
    x, y, ranks, test, fold = (data[key] for key in ("x", "y", "rank", "test", "fold"))
    dev = ~test
    report = {
        "schema_version": 1,
        "task": "strict gold-exact candidate confidence, not production correctness",
        "queries": len(data["queries"]),
        "query_splits": {
            "development": int(dev.sum() // 10), "final_test": int(test.sum() // 10),
        },
        "split": "by normalized expected text; five grouped CV folds within development",
        "seed": args.seed,
        "variants": {},
    }
    oof_metrics = {}
    for name in VARIANTS:
        out_of_fold = np.full(len(y), np.nan)
        for current_fold in range(args.folds):
            train_mask = dev & (fold != current_fold)
            validation_mask = dev & (fold == current_fold)
            if not validation_mask.any():
                raise ValueError(f"empty fold {current_fold}")
            model = fit_variant(name, x, y, ranks, train_mask)
            out_of_fold[validation_mask] = predict_variant(
                model, x[validation_mask], ranks[validation_mask]
            )
        cv = metrics(out_of_fold[dev], y[dev], ranks[dev])
        oof_metrics[name] = cv
        print(f"CV {name}: Brier={cv['all']['brier']:.5f}, top1={cv['top1']['brier']:.5f}", flush=True)
    selected = min(VARIANTS, key=lambda name: oof_metrics[name]["all"]["brier"])
    report["selected_by_cv_all_candidate_brier"] = selected
    test_predictions = {}
    for name in VARIANTS:
        model = fit_variant(name, x, y, ranks, dev)
        final_predictions = predict_variant(model, x[test], ranks[test])
        test_predictions[name] = final_predictions
        by_suite = {}
        for suite in sorted(set(data["suite"])):
            suite_mask = data["suite"][test] == suite
            by_suite[suite] = metrics(
                final_predictions[suite_mask], y[test][suite_mask], ranks[test][suite_mask]
            )
        report["variants"][name] = {
            "cv": oof_metrics[name],
            "final_test": metrics(final_predictions, y[test], ranks[test]),
            "final_test_by_suite": by_suite,
        }
        if name == selected:
            report["selected_model"] = model
    report["selected_test_calibration_bins"] = calibration_bins(
        test_predictions[selected], y[test]
    )
    report["paired_test_brier_gain_vs_selected"] = {}
    for reference in ("full_with_margin", "full_plus_accent_fold"):
        if reference != selected:
            report["paired_test_brier_gain_vs_selected"][reference] = paired_brier_gain(
                test_predictions[reference], test_predictions[selected], y[test], args.seed
            )
    report["leave_one_suite_out"] = {}
    expected_groups = np.repeat(
        np.asarray([normalized(query["expected"]) for query in data["queries"]], dtype=object), 10
    )
    for target_suite in sorted(set(data["suite"])):
        target = data["suite"] == target_suite
        target_groups = set(expected_groups[target])
        train = (data["suite"] != target_suite) & ~np.isin(expected_groups, list(target_groups))
        report["leave_one_suite_out"][target_suite] = {
            "train_queries": int(train.sum() // 10), "test_queries": int(target.sum() // 10),
        }
        for name in ("full_with_margin", "full_plus_accent_fold", selected):
            model = fit_variant(name, x, y, ranks, train)
            prediction = predict_variant(model, x[target], ranks[target])
            report["leave_one_suite_out"][target_suite][name] = metrics(
                prediction, y[target], ranks[target]
            )
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_score_path = output / "top10_selected_confidence.jsonl"
    selected_scores = predict_variant(report["selected_model"], x, ranks)
    with selected_score_path.open("w", encoding="utf-8") as stream:
        for index, query in enumerate(data["queries"]):
            start = index * 10
            stream.write(json.dumps({
                "suite": query["suite"], "query_id": query["query_id"],
                "input": query["input"], "expected": query["expected"],
                "split": "final_test" if test[start] else "development_in_sample",
                "hypotheses": query["hypotheses"],
                "sequence_scores": query["sequence_scores"],
                "strict_exact_labels": query["strict_exact_labels"],
                "confidence_gold_exact": selected_scores[start : start + 10].tolist(),
            }, ensure_ascii=False) + "\n")
    report["selected_scores"] = str(selected_score_path)
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# ReparoS candidate confidence comparison", "",
        f"Queries: {report['queries']}; development: {report['query_splits']['development']}; "
        f"final test: {report['query_splits']['final_test']}.", "",
        "| Method | CV Brier | Test Brier | Test top1 Brier | Test log loss | Test ECE10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(VARIANTS, key=lambda item: oof_metrics[item]["all"]["brier"]):
        item = report["variants"][name]
        lines.append(
            f"| {name} | {item['cv']['all']['brier']:.5f} | "
            f"{item['final_test']['all']['brier']:.5f} | "
            f"{item['final_test']['top1']['brier']:.5f} | "
            f"{item['final_test']['all']['log_loss']:.5f} | "
            f"{item['final_test']['all']['ece10']:.5f} |"
        )
    lines.extend(["", f"Selected on CV: **{selected}**.", "",
                  "Scores measure strict gold agreement on diagnostic data, not semantic correctness on production traffic."])
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Selected by CV: {selected}; report: {output / 'comparison.md'}")


if __name__ == "__main__":
    main()
