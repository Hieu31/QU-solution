from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping


CHECKPOINT_RE = re.compile(r"_step_(\d+)\.pt$")

PILOT_GATES: dict[str, dict[str, tuple[str, str, float]]] = {
    "stage1-primitives": {
        "clean_preservation": ("diagnostic", "safety.clean_preservation_rate", 0.90),
        "telex": ("user", "by_error_type.telex_malformed.correction.exact_overall_accuracy", 0.75),
        "vni": ("user", "by_error_type.vni_and_input_method.correction.exact_overall_accuracy", 0.95),
        "missing_only": ("composition", "by_error_type.missing_diacritics_only.correction.exact_noisy_accuracy", 0.60),
        "boundary_only": ("composition", "by_error_type.boundary_one_only.correction.exact_noisy_accuracy", 0.80),
    },
    "stage2-composition": {
        "clean_preservation": ("diagnostic", "safety.clean_preservation_rate", 0.90),
        "boundary_plus_diacritics": ("composition", "by_error_type.missing_diacritics_boundary_one.correction.exact_noisy_accuracy", 0.50),
        "all_joined": ("composition", "by_error_type.missing_diacritics_boundary_all.correction.exact_noisy_accuracy", 0.15),
        "user_boundary": ("user", "by_error_type.diacritics_boundary.correction.exact_overall_accuracy", 0.50),
    },
    "stage3-lexical": {
        "clean_preservation": ("diagnostic", "safety.clean_preservation_rate", 0.90),
        "location_acronym": ("user", "by_error_type.location_acronym.correction.exact_overall_accuracy", 0.70),
        "abbreviation": ("user", "by_error_type.abbreviation.correction.exact_overall_accuracy", 0.50),
        "abbreviation_boundary": ("user", "by_error_type.abbreviation_boundary.correction.exact_overall_accuracy", 0.40),
        "lexical_typo": ("user", "by_error_type.lexical_typo.correction.exact_overall_accuracy", 0.60),
    },
}


def checkpoint_step(path: str | Path) -> int:
    match = CHECKPOINT_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"not an OpenNMT step checkpoint: {path}")
    return int(match.group(1))


def latest_checkpoint(folder: str | Path, prefix: str = "reparos") -> Path:
    paths = list(Path(folder).glob(f"{prefix}*_step_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint in {folder}")
    return max(paths, key=checkpoint_step)


def build_stage_config(
    base_config: str | Path,
    stage_data: str | Path,
    output: str | Path,
    train_from: str | Path,
    additional_steps: int,
    *,
    valid_steps: int = 1000,
    save_checkpoint_steps: int = 1000,
    keep_checkpoint: int = 3,
) -> Path:
    if additional_steps < 1 or valid_steps < 1 or save_checkpoint_steps < 1:
        raise ValueError("training and checkpoint steps must be positive")
    base_path, data_root, output_root, checkpoint = (
        Path(base_config), Path(stage_data), Path(output), Path(train_from)
    )
    required = [
        data_root / "train.src", data_root / "train.tgt",
        data_root / "validation.src", data_root / "validation.tgt",
        checkpoint,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing curriculum training inputs: {missing}")
    config = json.loads(base_path.read_text(encoding="utf-8"))
    # Reuse the exact tokenizer vocabulary and architecture from Base. Building
    # a new vocabulary would make checkpoint embeddings incompatible.
    for key in ("src_vocab", "tgt_vocab", "src_subword_model", "tgt_subword_model"):
        value = Path(str(config[key]))
        if not value.is_file():
            raise FileNotFoundError(f"Base config {key} does not exist: {value}")
    config["data"] = {
        "corpus_1": {
            "path_src": str(data_root / "train.src"),
            "path_tgt": str(data_root / "train.tgt"),
            "transforms": ["sentencepiece"], "weight": 1,
        },
        "valid": {
            "path_src": str(data_root / "validation.src"),
            "path_tgt": str(data_root / "validation.tgt"),
            "transforms": ["sentencepiece"],
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    config["save_model"] = str(output_root / "reparos")
    config["train_from"] = str(checkpoint)
    config["train_steps"] = checkpoint_step(checkpoint) + additional_steps
    config["valid_steps"] = valid_steps
    config["save_checkpoint_steps"] = save_checkpoint_steps
    config["keep_checkpoint"] = keep_checkpoint
    config["overwrite"] = False
    path = output_root / "opennmt-stage.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _metric(payload: Mapping[str, object], path: str) -> float:
    value: object = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"metric path missing: {path}")
        value = value[part]
    return float(value)


def evaluate_pilot_gate(stage: str, reports: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    if stage not in PILOT_GATES:
        raise ValueError(f"unknown stage: {stage}")
    checks = {}
    for name, (report_name, metric_path, minimum) in PILOT_GATES[stage].items():
        value = _metric(reports[report_name], metric_path)
        checks[name] = {"value": value, "minimum": minimum, "passed": value >= minimum}
    return {"stage": stage, "passed": all(item["passed"] for item in checks.values()), "checks": checks}

