from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Mapping

from reparos.curriculum_training import (
    build_stage_config,
    checkpoint_step,
    evaluate_pilot_gate,
    latest_checkpoint,
)


STAGES = ("stage1-primitives", "stage2-composition", "stage3-lexical")


def prepare_base_template(
    original_config: str | Path,
    base_artifact_dir: str | Path,
    tokenizer_model: str | Path,
    output: str | Path,
) -> Path:
    config = json.loads(Path(original_config).read_text(encoding="utf-8"))
    artifact = Path(base_artifact_dir)
    paths = {
        "src_vocab": artifact / "vocab.src",
        "tgt_vocab": artifact / "vocab.tgt",
        "src_subword_model": Path(tokenizer_model),
        "tgt_subword_model": Path(tokenizer_model),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Base vocabulary/tokenizer: {missing}")
    config.update({key: str(value) for key, value in paths.items()})
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def train_sequence(
    *,
    profile_root: str | Path,
    output_root: str | Path,
    base_config: str | Path,
    initial_checkpoint: str | Path,
    additional_steps: Mapping[str, int],
    python_executable: str | Path,
    valid_steps: int,
    checkpoint_steps: int,
) -> tuple[Path, dict[str, Path]]:
    current = Path(initial_checkpoint)
    outputs: dict[str, Path] = {}
    for stage in STAGES:
        stage_output = Path(output_root) / stage
        if list(stage_output.glob("reparos_step_*.pt")):
            raise FileExistsError(f"stage output already has checkpoints: {stage_output}")
        config = build_stage_config(
            base_config, Path(profile_root) / stage, stage_output, current,
            additional_steps[stage], valid_steps=valid_steps,
            save_checkpoint_steps=checkpoint_steps,
        )
        subprocess.run([
            str(python_executable), "-m", "onmt.bin.train", "-config", str(config)
        ], check=True)
        current = latest_checkpoint(stage_output, "reparos")
        outputs[stage] = current
        print(stage, "checkpoint:", current)
    return current, outputs


def evaluate_checkpoint(
    *, checkpoint: str | Path, tokenizer: str | Path,
    decoding_config: str | Path, evaluation_root: str | Path,
    output_root: str | Path, reparos_cli: str | Path,
    benchmark_cli: str | Path, device: str = "cpu",
    compute_type: str = "float32",
) -> dict[str, Mapping[str, object]]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    ct2 = output / "ctranslate2"
    if ct2.exists():
        shutil.rmtree(ct2)
    subprocess.run([
        str(reparos_cli), "export-ctranslate2", "--model", str(checkpoint),
        "--tokenizer", str(tokenizer), "--output", str(ct2),
        "--compute-type", "float32", "--trust-checkpoint",
    ], check=True)
    golds = {
        "user": Path(evaluation_root) / "user-centric-v2.jsonl",
        "composition": Path(evaluation_root) / "composition-4k.jsonl",
        "diagnostic": Path(evaluation_root) / "diagnostic-10k.jsonl",
    }
    reports: dict[str, Mapping[str, object]] = {}
    for name, gold in golds.items():
        prediction, metrics = output / f"{name}.predictions.jsonl", output / f"{name}.metrics.json"
        subprocess.run([
            str(benchmark_cli), "run-ctranslate2", "--gold", str(gold),
            "--model", str(ct2), "--decoding-config", str(decoding_config),
            "--device", device, "--compute-type", compute_type, "--output", str(prediction),
        ], check=True)
        subprocess.run([
            str(benchmark_cli), "score", "--gold", str(gold),
            "--prediction", f"reparos={prediction}", "--output", str(metrics),
        ], check=True)
        reports[name] = json.loads(metrics.read_text(encoding="utf-8"))["systems"]["reparos"]
    return reports


def run_pilot_gates(
    checkpoints: Mapping[str, Path], *, tokenizer: str | Path,
    decoding_config: str | Path, evaluation_root: str | Path,
    output_root: str | Path, reparos_cli: str | Path,
    benchmark_cli: str | Path, device: str = "cpu",
    compute_type: str = "float32",
) -> dict[str, object]:
    results = {}
    for stage in STAGES:
        reports = evaluate_checkpoint(
            checkpoint=checkpoints[stage], tokenizer=tokenizer,
            decoding_config=decoding_config, evaluation_root=evaluation_root,
            output_root=Path(output_root) / stage, reparos_cli=reparos_cli,
            benchmark_cli=benchmark_cli, device=device, compute_type=compute_type,
        )
        results[stage] = evaluate_pilot_gate(stage, reports)
    payload = {"passed": all(item["passed"] for item in results.values()), "stages": results}
    path = Path(output_root) / "pilot-gates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def train_and_select_pilot_sequence(
    *, profile_root: str | Path, output_root: str | Path,
    base_config: str | Path, initial_checkpoint: str | Path,
    additional_steps: Mapping[str, int], python_executable: str | Path,
    tokenizer: str | Path, decoding_config: str | Path,
    evaluation_root: str | Path, reparos_cli: str | Path,
    benchmark_cli: str | Path, valid_steps: int, checkpoint_steps: int,
    device: str = "cpu", compute_type: str = "float32",
) -> tuple[Path, dict[str, Path], dict[str, object]]:
    """Train each pilot stage from the newest passing retained checkpoint."""
    root = Path(output_root)
    current = Path(initial_checkpoint)
    selected: dict[str, Path] = {}
    stage_results: dict[str, object] = {}
    for stage in STAGES:
        stage_output = root / "pilot" / stage
        if list(stage_output.glob("reparos_step_*.pt")):
            raise FileExistsError(f"stage output already has checkpoints: {stage_output}")
        config = build_stage_config(
            base_config, Path(profile_root) / stage, stage_output, current,
            additional_steps[stage], valid_steps=valid_steps,
            save_checkpoint_steps=checkpoint_steps,
        )
        subprocess.run([str(python_executable), "-m", "onmt.bin.train", "-config", str(config)], check=True)
        candidates = sorted(stage_output.glob("reparos_step_*.pt"), key=checkpoint_step)
        if not candidates:
            raise FileNotFoundError(f"no checkpoint in {stage_output}")
        candidate_results = []
        passing: list[Path] = []
        for checkpoint in candidates:
            step = checkpoint_step(checkpoint)
            reports = evaluate_checkpoint(
                checkpoint=checkpoint, tokenizer=tokenizer,
                decoding_config=decoding_config, evaluation_root=evaluation_root,
                output_root=root / "pilot-evaluation" / stage / f"step-{step}",
                reparos_cli=reparos_cli, benchmark_cli=benchmark_cli,
                device=device, compute_type=compute_type,
            )
            gate = evaluate_pilot_gate(stage, reports)
            candidate_results.append({"checkpoint": str(checkpoint), "step": step, **gate})
            if gate["passed"]:
                passing.append(checkpoint)
        chosen = max(passing, key=checkpoint_step) if passing else None
        stage_results[stage] = {
            "stage": stage, "passed": chosen is not None,
            "selected_checkpoint": str(chosen) if chosen else None,
            "candidates": candidate_results,
        }
        payload = {"passed": len(selected) + (chosen is not None) == len(STAGES), "stages": stage_results}
        gates_path = root / "pilot-evaluation" / "pilot-gates.json"
        gates_path.parent.mkdir(parents=True, exist_ok=True)
        gates_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if chosen is None:
            raise RuntimeError(f"{stage} failed: no retained checkpoint passed its pilot gate; inspect {gates_path}")
        selected[stage] = chosen
        current = chosen
        print(stage, "selected checkpoint:", chosen)
    return current, selected, {"passed": True, "stages": stage_results}
