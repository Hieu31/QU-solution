from __future__ import annotations

import json
from pathlib import Path

from reparos.curriculum_training import build_stage_config, evaluate_pilot_gate


def test_stage_config_reuses_vocab_and_advances_checkpoint_step(tmp_path: Path) -> None:
    for name in ("vocab.src", "vocab.tgt", "tokenizer.model"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    checkpoint = tmp_path / "reparos_base_step_25000.pt"
    checkpoint.write_text("x", encoding="utf-8")
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "src_vocab": str(tmp_path / "vocab.src"), "tgt_vocab": str(tmp_path / "vocab.tgt"),
        "src_subword_model": str(tmp_path / "tokenizer.model"),
        "tgt_subword_model": str(tmp_path / "tokenizer.model"), "data": {},
    }), encoding="utf-8")
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in ("train.src", "train.tgt", "validation.src", "validation.tgt"):
        (stage / name).write_text("x\n", encoding="utf-8")
    path = build_stage_config(base, stage, tmp_path / "run", checkpoint, 3000)
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["train_steps"] == 28000
    assert result["train_from"] == str(checkpoint)
    assert result["src_vocab"] == str(tmp_path / "vocab.src")


def test_gate_rejects_one_failed_metric() -> None:
    reports = {
        "diagnostic": {"safety": {"clean_preservation_rate": 0.95}},
        "composition": {"by_error_type": {
            "missing_diacritics_only": {"correction": {"exact_noisy_accuracy": 0.7}},
            "boundary_one_only": {"correction": {"exact_noisy_accuracy": 0.7}},
        }},
        "user": {"by_error_type": {
            "telex_malformed": {"correction": {"exact_overall_accuracy": 0.8}},
            "vni_and_input_method": {"correction": {"exact_overall_accuracy": 1.0}},
        }},
    }
    result = evaluate_pilot_gate("stage1-primitives", reports)
    assert not result["passed"]
    assert not result["checks"]["boundary_only"]["passed"]
