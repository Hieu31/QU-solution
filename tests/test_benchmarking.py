from __future__ import annotations

import json

import pytest

from benchmarking.io import strict_join
from benchmarking.metrics import edit_distance, score_rows, token_edit_slots
from benchmarking.performance import latency_summary, percentile
from benchmarking.runners import parity_rows
from reparos.architecture import DecodingConfig


def row(query_id: str, source: str, target: str, output: str, error_type: str = "keyboard_edit"):
    gold = {"query_id": query_id, "input": source, "expected": target, "error_type": error_type}
    prediction = {**gold, "output": output, "hypotheses": [output, target]}
    return gold, prediction


def test_edit_distance_and_structural_slots() -> None:
    assert edit_distance("abc", "adc") == 1
    assert token_edit_slots("ha noi", "ha noi") == set()
    assert any(kind == "boundary" for kind, _ in token_edit_slots("hanoi", "ha noi"))
    assert ("token", 1) in token_edit_slots("ha nọi", "ha nội")


def test_metric_contract_distinguishes_detection_from_correction() -> None:
    joined = [
        row("1", "ha noi", "hà nội", "hồ chí minh"),
        row("2", "hà nội", "hà nội", "hà nội", "clean"),
        row("3", "da nang", "đà nẵng", "đà nẵng"),
    ]
    metrics = score_rows(joined)
    assert metrics["query_detection"]["tp"] == 2
    assert metrics["correction"]["exact_noisy_accuracy"] == 0.5
    assert metrics["safety"]["clean_preservation_rate"] == 1.0
    assert metrics["ranking"]["recall@1"] == pytest.approx(2 / 3)
    assert metrics["ranking"]["recall@3"] == 1.0


def test_clean_false_correction_complements_preservation() -> None:
    metrics = score_rows([row("1", "hà nội", "hà nội", "hạ nội", "clean")])
    safety = metrics["safety"]
    assert safety["clean_preservation_rate"] + safety["false_correction_rate"] == 1.0
    assert safety["regression_rate"] == 1.0


def test_strict_join_rejects_missing_and_modified_gold() -> None:
    gold, prediction = row("1", "a", "b", "b")
    with pytest.raises(ValueError, match="missing"):
        strict_join([gold], [])
    prediction["expected"] = "c"
    with pytest.raises(ValueError, match="changed"):
        strict_join([gold], [prediction])


def test_parity_reports_order_only_and_top1() -> None:
    left = [{"query_id": "1", "hypotheses": ["a", "b"]}, {"query_id": "2", "hypotheses": ["x"]}]
    right = [{"query_id": "1", "hypotheses": ["a", "b"]}, {"query_id": "2", "hypotheses": ["y"]}]
    report = parity_rows(left, right)
    assert report["top1_parity"] == 0.5
    assert report["ordered_topk_parity"] == 0.5
    assert report["mismatches"][0]["category"] == "top1_mismatch"


def test_latency_percentiles_are_interpolated_and_labeled() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    summary = latency_summary([1.0, 2.0, 3.0], official=False, note="smoke")
    assert summary["p95_ms"] == pytest.approx(2.9)
    assert summary["official"] is False


def test_ctranslate2_penalty_is_separate_from_opennmt_alpha() -> None:
    decoding = DecodingConfig()
    assert decoding.length_penalty == 1.0
    assert decoding.ctranslate2_length_penalty == 0.0
