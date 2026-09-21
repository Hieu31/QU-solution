from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from reparos.dependencies import require


def checkpoint_step(path: str | Path) -> int:
    """Extract step number from OpenNMT checkpoint filename."""
    name = Path(path).name
    import re
    match = re.search(r"_step_(\d+)\.pt$", name)
    if not match:
        raise ValueError(f"Not an OpenNMT step checkpoint: {path}")
    return int(match.group(1))


def prepare_finetune_corpora(
    dataset_root: str | Path,
    run_root: str | Path,
    arm: str = "B",
) -> dict[str, Any]:
    """Prepares line-by-line .src and .tgt files for OpenNMT from interleaved-continual-v1.

    Args:
        dataset_root: Path to data/reparos/interleaved-continual-v1
        run_root: Directory to write training corpus text files
        arm: "B" (Balanced Interleaved: 32% Plasticity, 46% Retention, 22% Protection)
             or "A" (Ablation: 100% Plasticity)

    Returns:
        OpenNMT 'data' dictionary configuration
    """
    data_dir = Path(dataset_root)

    # Auto-normalize if pointed to train/ or eval/ child folder
    if data_dir.name in ("train", "eval") and (data_dir.parent / "train").is_dir() and (data_dir.parent / "eval").is_dir():
        data_dir = data_dir.parent
    elif not ((data_dir / "train").is_dir() and (data_dir / "eval").is_dir()):
        children = [p for p in data_dir.glob("*") if p.is_dir() and (p / "train").is_dir() and (p / "eval").is_dir()]
        if children:
            data_dir = children[0]

    out_dir = Path(run_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    eval_dir = data_dir / "eval"

    if not train_dir.is_dir() or not eval_dir.is_dir():
        raise FileNotFoundError(f"Missing train or eval directory in {data_dir} (passed: {dataset_root})")

    def read_jsonl_pairs(folder: Path) -> list[tuple[str, str]]:
        pairs = []
        for file in sorted(folder.glob("*.jsonl")):
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    pairs.append((row["input"].strip(), row["expected"].strip()))
        return pairs

    def write_pairs(src_path: Path, tgt_path: Path, pairs: list[tuple[str, str]]) -> None:
        with src_path.open("w", encoding="utf-8") as fs, tgt_path.open("w", encoding="utf-8") as ft:
            for s, t in pairs:
                fs.write(f"{s}\n")
                ft.write(f"{t}\n")

    # 1. Plasticity pairs (Group A: 32,000)
    plasticity_pairs = read_jsonl_pairs(train_dir / "plasticity")
    write_pairs(out_dir / "train.plasticity.src", out_dir / "train.plasticity.tgt", plasticity_pairs)

    # 2. Retention pairs (Group B: 46,000)
    retention_pairs = read_jsonl_pairs(train_dir / "retention")
    write_pairs(out_dir / "train.retention.src", out_dir / "train.retention.tgt", retention_pairs)

    # 3. Protection pairs (Group C: 22,000)
    protection_pairs = read_jsonl_pairs(train_dir / "protection_seen")
    write_pairs(out_dir / "train.protection.src", out_dir / "train.protection.tgt", protection_pairs)

    # 4. Combined Validation pairs (from all eval splits: 2,800 pairs)
    eval_pairs = []
    for split_name in ("plasticity", "retention", "protection_seen", "protection_heldout"):
        eval_pairs.extend(read_jsonl_pairs(eval_dir / split_name))
    write_pairs(out_dir / "validation.src", out_dir / "validation.tgt", eval_pairs)

    arm = arm.upper()
    if arm == "B":
        data_config = {
            "plasticity": {
                "path_src": str(out_dir / "train.plasticity.src"),
                "path_tgt": str(out_dir / "train.plasticity.tgt"),
                "transforms": ["sentencepiece"],
                "weight": 32,
            },
            "retention": {
                "path_src": str(out_dir / "train.retention.src"),
                "path_tgt": str(out_dir / "train.retention.tgt"),
                "transforms": ["sentencepiece"],
                "weight": 46,
            },
            "protection": {
                "path_src": str(out_dir / "train.protection.src"),
                "path_tgt": str(out_dir / "train.protection.tgt"),
                "transforms": ["sentencepiece"],
                "weight": 22,
            },
            "valid": {
                "path_src": str(out_dir / "validation.src"),
                "path_tgt": str(out_dir / "validation.tgt"),
                "transforms": ["sentencepiece"],
            },
        }
    elif arm == "A":
        data_config = {
            "plasticity": {
                "path_src": str(out_dir / "train.plasticity.src"),
                "path_tgt": str(out_dir / "train.plasticity.tgt"),
                "transforms": ["sentencepiece"],
                "weight": 100,
            },
            "valid": {
                "path_src": str(out_dir / "validation.src"),
                "path_tgt": str(out_dir / "validation.tgt"),
                "transforms": ["sentencepiece"],
            },
        }
    else:
        raise ValueError(f"Unknown experiment arm: {arm}. Must be 'A' or 'B'.")

    return data_config


def build_finetune_config(
    base_config_path: str | Path,
    base_checkpoint_path: str | Path,
    run_root: str | Path,
    data_config: dict[str, Any],
    tokenizer_model: str | Path,
    vocab_src: str | Path,
    vocab_tgt: str | Path,
    *,
    additional_steps: int = 3000,
    save_checkpoint_steps: int = 250,
    valid_steps: int = 250,
    keep_checkpoint: int = 15,
    batch_size: int = 4096,
    bucket_size: int = 16384,
    num_workers: int = 2,
) -> Path:
    """Builds OpenNMT configuration for continual fine-tuning from Base V2 checkpoint."""
    base_cfg = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    out_dir = Path(run_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = Path(base_checkpoint_path)
    base_step = checkpoint_step(base_ckpt)

    for src_path, dst_name in ((vocab_src, "vocab.src"), (vocab_tgt, "vocab.tgt")):
        dst_path = out_dir / dst_name
        if not dst_path.exists() or dst_path.resolve() != Path(src_path).resolve():
            shutil.copy2(src_path, dst_path)

    base_cfg.update({
        "save_data": str(out_dir / "vocab"),
        "src_vocab": str(out_dir / "vocab.src"),
        "tgt_vocab": str(out_dir / "vocab.tgt"),
        "src_subword_model": str(tokenizer_model),
        "tgt_subword_model": str(tokenizer_model),
        "save_model": str(out_dir / "reparos_continual"),
        "train_from": str(base_ckpt),
        "train_steps": base_step + additional_steps,
        "valid_steps": valid_steps,
        "save_checkpoint_steps": save_checkpoint_steps,
        "keep_checkpoint": keep_checkpoint,
        "data": data_config,
        "batch_size": batch_size,
        "bucket_size": bucket_size,
        "num_workers": num_workers,
        "accum_count": [1],
        "overwrite": True,
    })

    out_cfg_path = out_dir / "opennmt-continual.json"
    out_cfg_path.write_text(json.dumps(base_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_cfg_path


def evaluate_continual_splits(
    predictor: Any,
    eval_root: str | Path,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluates a CTranslate2 predictor across the 4 continual evaluation splits."""
    eval_dir = Path(eval_root)
    if (eval_dir / "eval").is_dir() and not (eval_dir / "plasticity").is_dir():
        eval_dir = eval_dir / "eval"
    results: dict[str, Any] = {}

    def run_split(folder: Path) -> dict[str, Any]:
        sub_results: dict[str, Any] = {}
        all_items: list[dict] = []
        for file in sorted(folder.glob("*.jsonl")):
            items = []
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            all_items.extend(items)

            # Evaluate this individual file
            sub_results[file.stem] = evaluate_item_list(predictor, items, batch_size=batch_size)

        sub_results["_overall"] = evaluate_item_list(predictor, all_items, batch_size=batch_size)
        return sub_results

    # 1. Plasticity
    results["plasticity"] = run_split(eval_dir / "plasticity")
    # 2. Retention
    results["retention"] = run_split(eval_dir / "retention")
    # 3. Protection Seen
    results["protection_seen"] = run_split(eval_dir / "protection_seen")
    # 4. Protection Held-Out
    results["protection_heldout"] = run_split(eval_dir / "protection_heldout")

    return results


def evaluate_item_list(predictor: Any, items: list[dict], batch_size: int = 64) -> dict[str, Any]:
    """Batch prediction and metrics calculation on a list of item records."""
    if not items:
        return {"count": 0, "exact_accuracy": 0.0, "noop_rate": 0.0, "fcr": 0.0}

    inputs = [item["input"] for item in items]
    expected = [item["expected"] for item in items]

    preds: list[str] = []
    for i in range(0, len(inputs), batch_size):
        chunk = inputs[i : i + batch_size]
        # Tokenize chunk with sentencepiece
        pieces_batch = [predictor.tokenizer.processor.encode(text, out_type=str) for text in chunk]
        res_batch = predictor.translator.translate_batch(pieces_batch, beam_size=1)
        for r in res_batch:
            preds.append(predictor.tokenizer.processor.decode(r.hypotheses[0]))

    exact = 0
    noop = 0
    fcr = 0
    count = len(items)

    for inp, exp, pred in zip(inputs, expected, preds, strict=True):
        if pred == exp:
            exact += 1
        if pred == inp:
            noop += 1
        if inp == exp and pred != inp:
            fcr += 1

    return {
        "count": count,
        "exact_accuracy": exact / count,
        "noop_rate": noop / count,
        "fcr": fcr / count,
    }


def evaluate_3tier_gating(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Evaluates candidate checkpoint against Base V2 baseline using 3-Tier Gating."""
    # Gate 0: Protection
    fcr_heldout = candidate_metrics["protection_heldout"]["_overall"]["fcr"]
    fcr_seen = candidate_metrics["protection_seen"]["_overall"]["fcr"]
    gate0_passed = bool((fcr_heldout <= 0.020) and (fcr_seen <= 0.010))

    # Gate 1: Retention
    base_ret_acc = baseline_metrics["retention"]["_overall"]["exact_accuracy"]
    cand_ret_acc = candidate_metrics["retention"]["_overall"]["exact_accuracy"]
    ret_deg = base_ret_acc - cand_ret_acc
    gate1_passed = bool(gate0_passed and (ret_deg <= 0.015))

    # Gate 2: Plasticity
    base_plas_acc = baseline_metrics["plasticity"]["_overall"]["exact_accuracy"]
    cand_plas_acc = candidate_metrics["plasticity"]["_overall"]["exact_accuracy"]
    plas_gain = cand_plas_acc - base_plas_acc
    gate2_passed = bool(gate0_passed and gate1_passed and (plas_gain >= 0.250))

    # Outcome Classification
    if gate2_passed:
        outcome = "Case A: Continual Fine-Tuning Success"
        status = "PASSED"
    elif not gate0_passed:
        outcome = "Case B: Protection Failure (Semantic / Acronym Over-generalization)"
        status = "FAILED_GATE_0"
    elif not gate1_passed:
        outcome = "Case C: Selective Forgetting (Retention Drop > 1.5%)"
        status = "FAILED_GATE_1"
    else:
        outcome = "Case D: Plasticity Under-Learning (Gain < 25.0%)"
        status = "FAILED_GATE_2"

    return {
        "status": status,
        "outcome": outcome,
        "gate_0_protection": {
            "passed": gate0_passed,
            "fcr_heldout": fcr_heldout,
            "fcr_heldout_threshold": 0.020,
            "fcr_seen": fcr_seen,
            "fcr_seen_threshold": 0.010,
        },
        "gate_1_retention": {
            "passed": gate1_passed,
            "baseline_accuracy": base_ret_acc,
            "candidate_accuracy": cand_ret_acc,
            "degradation": ret_deg,
            "max_allowed_degradation": 0.015,
        },
        "gate_2_plasticity": {
            "passed": gate2_passed,
            "baseline_accuracy": base_plas_acc,
            "candidate_accuracy": cand_plas_acc,
            "gain": plas_gain,
            "min_required_gain": 0.250,
            "noop_rate": candidate_metrics["plasticity"]["_overall"]["noop_rate"],
        },
    }
