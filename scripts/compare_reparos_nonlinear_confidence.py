"""Compare nonlinear gold-exact confidence models with the pinned logistic baseline.

Run with the project venv after installing scikit-learn:
    .venv/Scripts/python.exe scripts/compare_reparos_nonlinear_confidence.py

The expected-text grouped split and five CV folds match
compare_reparos_confidence_methods.py. The diagnostic suites are not an
independent production sample. The previous test report has been inspected,
so this is an exploratory comparison, not a fresh blind test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from compare_reparos_confidence_methods import (
    ALL_NAMES, DEFAULT_INPUT, load_data, metrics,
)
from train_reparos_candidate_confidence import normalized


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/reparos_base_v3_confidence_14088/nonlinear_confidence"
BASELINE_REPORT = ROOT / "artifacts/reparos_base_v3_confidence_14088/confidence_comparison/comparison.json"


def build_model(name: str):
    if name == "hgb_shallow":
        return HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, min_samples_leaf=80,
            l2_regularization=1.0, learning_rate=0.07,
            early_stopping=False, random_state=7193,
        )
    if name == "hgb_deep":
        return HistGradientBoostingClassifier(
            max_iter=160, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=2.0, learning_rate=0.05,
            early_stopping=False, random_state=7193,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=120, max_features=0.8, min_samples_leaf=30,
            n_jobs=-1, random_state=7193,
        )
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=("hgb_shallow", "hgb_deep", "random_forest"),
                        default=("hgb_shallow", "hgb_deep", "random_forest"))
    args = parser.parse_args()
    started = time.perf_counter()
    data = load_data(args.input.resolve(), seed=7193, folds=5, test_fraction=0.2)
    x, y, rank, test, fold = (data[key] for key in ("x", "y", "rank", "test", "fold"))
    dev = ~test
    report = {
        "task": "strict gold-exact candidate confidence",
        "split": "same grouped expected-text split and five folds as logistic comparison",
        "caveat": "Exploratory: the existing final-test report was already inspected; no fresh blind test.",
        "queries": len(data["queries"]),
        "development_queries": int(dev.sum() // 10),
        "test_queries": int(test.sum() // 10),
        "features": ALL_NAMES,
        "models": {},
    }
    baseline_report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    baseline = baseline_report["variants"]["full_all_extras"]
    report["logistic_baseline"] = {
        "cv": baseline["cv"], "final_test": baseline["final_test"],
        "final_test_by_suite": baseline["final_test_by_suite"],
    }
    for name in args.models:
        oof = np.full(len(y), np.nan)
        for current_fold in range(5):
            train = dev & (fold != current_fold)
            valid = dev & (fold == current_fold)
            model = build_model(name)
            model.fit(x[train], y[train].astype(np.int8))
            oof[valid] = model.predict_proba(x[valid])[:, 1]
            print(f"{name}: fold {current_fold + 1}/5", flush=True)
        cv = metrics(oof[dev], y[dev], rank[dev])
        model = build_model(name)
        model.fit(x[dev], y[dev].astype(np.int8))
        prediction = model.predict_proba(x[test])[:, 1]
        by_suite = {}
        for suite in sorted(set(data["suite"])):
            mask = data["suite"][test] == suite
            by_suite[suite] = metrics(prediction[mask], y[test][mask], rank[test][mask])
        report["models"][name] = {
            "parameters": model.get_params(),
            "cv": cv,
            "final_test": metrics(prediction, y[test], rank[test]),
            "final_test_by_suite": by_suite,
        }
        print(f"{name}: CV Brier {cv['all']['brier']:.5f}, test Brier "
              f"{report['models'][name]['final_test']['all']['brier']:.5f}, "
              f"top1 {report['models'][name]['final_test']['top1']['brier']:.5f}", flush=True)
    expected_groups = np.repeat(
        np.asarray([normalized(query["expected"]) for query in data["queries"]], dtype=object), 10
    )
    report["leave_one_suite_out"] = {}
    for suite in sorted(set(data["suite"])):
        target = data["suite"] == suite
        target_groups = set(expected_groups[target])
        train = (data["suite"] != suite) & ~np.isin(expected_groups, list(target_groups))
        item = {
            "train_queries": int(train.sum() // 10),
            "test_queries": int(target.sum() // 10),
            "logistic_full_all_extras": baseline_report["leave_one_suite_out"][suite]["full_all_extras"],
        }
        for name in args.models:
            model = build_model(name)
            model.fit(x[train], y[train].astype(np.int8))
            prediction = model.predict_proba(x[target])[:, 1]
            item[name] = metrics(prediction, y[target], rank[target])
        report["leave_one_suite_out"][suite] = item
        print(f"Transfer to {suite} complete", flush=True)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Nonlinear confidence comparison", "",
        f"Queries: {report['queries']}; development: {report['development_queries']}; "
        f"test: {report['test_queries']}.", "",
        "| Model | CV Brier all | Test Brier all | Test Brier top1 | Test ECE top1 | Test top1 mean p | Test top1 exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in (("logistic_full_all_extras", report["logistic_baseline"]), *report["models"].items()):
        cv, final = result["cv"], result["final_test"]
        lines.append(
            f"| {name} | {cv['all']['brier']:.5f} | {final['all']['brier']:.5f} | "
            f"{final['top1']['brier']:.5f} | {final['top1']['ece10']:.5f} | "
            f"{final['top1']['mean_confidence']:.5f} | {final['top1']['observed_rate']:.5f} |"
        )
    lines.extend(["", "Same strict-gold labels and expected-text grouped split. The test was previously inspected; treat this comparison as exploratory.", ""])
    lines.extend(["## Leave-one-suite-out transfer", "",
                  "| Held-out suite | Model | All Brier | Top1 Brier | Top1 ECE |",
                  "|---|---|---:|---:|---:|"])
    for suite, item in report["leave_one_suite_out"].items():
        for name in ("logistic_full_all_extras", *args.models):
            result = item[name]
            lines.append(f"| {suite} | {name} | {result['all']['brier']:.5f} | "
                         f"{result['top1']['brier']:.5f} | {result['top1']['ece10']:.5f} |")
    lines.append("")
    (args.output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {args.output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
