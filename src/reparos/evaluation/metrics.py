from __future__ import annotations

from collections.abc import Sequence


def evaluate_queries(
    observed: Sequence[str], intended: Sequence[str], hypotheses: Sequence[Sequence[str]],
) -> dict[str, float | int]:
    if not observed or len(observed) != len(intended) or len(observed) != len(hypotheses):
        raise ValueError('observed, intended and hypotheses must have equal non-zero lengths')
    exact = topk = clean = clean_preserved = false_corrections = 0
    for source, target, candidates in zip(observed, intended, hypotheses, strict=True):
        if not candidates:
            raise ValueError('every query requires at least one hypothesis')
        exact += int(candidates[0] == target)
        topk += int(target in candidates)
        if source == target:
            clean += 1
            clean_preserved += int(candidates[0] == source)
            false_corrections += int(candidates[0] != source)
    count = len(observed)
    return {
        'queries': count, 'exact_queries': exact,
        'query_exact_accuracy': exact / count,
        'topk_oracle_accuracy': topk / count,
        'clean_queries': clean, 'clean_preserved': clean_preserved,
        'clean_preservation': clean_preserved / clean if clean else 0.0,
        'clean_false_corrections': false_corrections,
        'clean_false_correction_rate': false_corrections / clean if clean else 0.0,
    }
