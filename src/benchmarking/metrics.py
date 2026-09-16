from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Sequence

from benchmarking.io import normalize_text


def edit_distance(left: Sequence, right: Sequence) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_item in enumerate(right, 1):
        current = [row]
        for column, left_item in enumerate(left, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def token_edit_slots(source: str, target: str) -> set[tuple[str, int]]:
    """Return deterministic source-token and insertion-boundary edit positions."""
    left, right = normalize_text(source).split(), normalize_text(target).split()
    rows, cols = len(left), len(right)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1): dp[i][0] = i
    for j in range(cols + 1): dp[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (left[i - 1] != right[j - 1]),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    edits: set[tuple[str, int]] = set()
    i, j = rows, cols
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (left[i - 1] != right[j - 1]):
            if left[i - 1] != right[j - 1]: edits.add(("token", i - 1))
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            edits.add(("token", i - 1)); i -= 1
        else:
            edits.add(("boundary", i)); j -= 1
    return edits


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {
        "precision": precision, "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
    }


def _rank(hypotheses: list[str], target: str, k: int) -> int | None:
    unique = []
    for item in hypotheses:
        item = normalize_text(item)
        if item not in unique: unique.append(item)
    try:
        rank = unique[:k].index(target) + 1
    except ValueError:
        return None
    return rank


def score_rows(joined: list[tuple[dict, dict]], include_breakdown: bool = True) -> dict:
    if not joined:
        raise ValueError("cannot score an empty benchmark")
    counts = Counter()
    char_before = char_after = char_den = 0
    token_before = token_after = token_den = 0
    macro_cer_before, macro_cer_after, macro_ter_before, macro_ter_after = [], [], [], []
    token_tp = token_fp = token_fn = 0
    structural = 0
    reciprocal_ranks = []
    regressions = []
    per_type: dict[str, list[tuple[dict, dict]]] = {}
    for gold, prediction in joined:
        source = normalize_text(str(gold["input"]))
        target = normalize_text(str(gold["expected"]))
        output = normalize_text(str(prediction["output"]))
        noisy, changed, exact = source != target, output != source, output == target
        counts["queries"] += 1
        counts["noisy"] += noisy; counts["clean"] += not noisy
        counts["tp"] += noisy and changed; counts["fp"] += (not noisy) and changed
        counts["fn"] += noisy and not changed; counts["tn"] += (not noisy) and not changed
        counts["correct_autocorrect"] += noisy and changed and exact
        counts["changed"] += changed; counts["exact"] += exact
        counts["exact_noisy"] += noisy and exact
        before, after = edit_distance(source, target), edit_distance(output, target)
        relation = "improved" if after < before else "regressed" if after > before else "unchanged"
        counts[relation] += 1
        if relation == "regressed":
            regressions.append({"query_id": gold["query_id"], "input": source, "expected": target, "output": output})
        char_before += before; char_after += after; char_den += max(len(target), 1)
        source_tokens, target_tokens, output_tokens = source.split(), target.split(), output.split()
        tb, ta = edit_distance(source_tokens, target_tokens), edit_distance(output_tokens, target_tokens)
        token_before += tb; token_after += ta; token_den += max(len(target_tokens), 1)
        macro_cer_before.append(before / max(len(target), 1)); macro_cer_after.append(after / max(len(target), 1))
        macro_ter_before.append(tb / max(len(target_tokens), 1)); macro_ter_after.append(ta / max(len(target_tokens), 1))
        gold_slots, predicted_slots = token_edit_slots(source, target), token_edit_slots(source, output)
        token_tp += len(gold_slots & predicted_slots); token_fp += len(predicted_slots - gold_slots); token_fn += len(gold_slots - predicted_slots)
        structural += int(any(kind == "boundary" for kind, _ in gold_slots | predicted_slots))
        hypotheses = [normalize_text(str(item)) for item in prediction.get("hypotheses", [output])]
        if not hypotheses: hypotheses = [output]
        for k in (1, 3, 5, 10): counts[f"recall@{k}"] += _rank(hypotheses, target, k) is not None
        rank = _rank(hypotheses, target, 10)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        per_type.setdefault(str(gold.get("error_type", "unknown")), []).append((gold, prediction))

    detection = _prf(counts["tp"], counts["fp"], counts["fn"])
    token_detection = _prf(token_tp, token_fp, token_fn)
    autocorrect_precision = _ratio(counts["correct_autocorrect"], counts["changed"])
    autocorrect_recall = _ratio(counts["correct_autocorrect"], counts["noisy"])
    result = {
        "queries": counts["queries"], "clean_queries": counts["clean"], "noisy_queries": counts["noisy"],
        "safety": {
            "clean_preservation_rate": _ratio(counts["tn"], counts["clean"]),
            "false_correction_rate": _ratio(counts["fp"], counts["clean"]),
            "improvement_rate": _ratio(counts["improved"], counts["queries"]),
            "unchanged_rate": _ratio(counts["unchanged"], counts["queries"]),
            "regression_rate": _ratio(counts["regressed"], counts["queries"]),
        },
        "query_detection": {**detection, "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "tn": counts["tn"]},
        "token_detection": {**token_detection, "tp": token_tp, "fp": token_fp, "fn": token_fn,
                            "alignment_coverage": _ratio(counts["queries"] - structural, counts["queries"]),
                            "structural_queries": structural},
        "correction": {
            "autocorrect_precision": autocorrect_precision,
            "autocorrect_recall": autocorrect_recall,
            "autocorrect_f1": _ratio(2 * autocorrect_precision * autocorrect_recall, autocorrect_precision + autocorrect_recall),
            "exact_noisy_accuracy": _ratio(counts["exact_noisy"], counts["noisy"]),
            "exact_overall_accuracy": _ratio(counts["exact"], counts["queries"]),
        },
        "ranking": {**{f"recall@{k}": _ratio(counts[f"recall@{k}"], counts["queries"]) for k in (1, 3, 5, 10)}, "mrr@10": mean(reciprocal_ranks)},
        "distance": {
            "cer_before_micro": _ratio(char_before, char_den), "cer_after_micro": _ratio(char_after, char_den),
            "cer_net_reduction": _ratio(char_before - char_after, char_before),
            "ter_before_micro": _ratio(token_before, token_den), "ter_after_micro": _ratio(token_after, token_den),
            "ter_net_reduction": _ratio(token_before - token_after, token_before),
            "cer_before_macro": mean(macro_cer_before), "cer_after_macro": mean(macro_cer_after),
            "ter_before_macro": mean(macro_ter_before), "ter_after_macro": mean(macro_ter_after),
        },
        "regressions": regressions,
    }
    if include_breakdown:
        result["by_error_type"] = {key: score_rows(value, False) for key, value in sorted(per_type.items())}
        metric_keys = ("exact_noisy_accuracy", "autocorrect_precision", "autocorrect_recall", "autocorrect_f1")
        noisy_groups = [value for key, value in result["by_error_type"].items() if key != "clean"]
        result["macro_error_type"] = {key: mean(group["correction"][key] for group in noisy_groups) if noisy_groups else 0.0 for key in metric_keys}
    return result
