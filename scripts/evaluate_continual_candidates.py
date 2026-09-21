from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from reparos.continual_finetune import (
    checkpoint_step,
    evaluate_3tier_gating,
    evaluate_continual_splits,
)
from reparos.serving.ctranslate2 import CTranslate2Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate continual candidate checkpoints and run 3-Tier Gating")
    parser.add_argument("--run-root", required=True, help="Directory containing fine-tuning checkpoints")
    parser.add_argument("--continual-data-root", required=True, help="Dataset root containing eval/ folder")
    parser.add_argument("--base-checkpoint", required=True, help="Path to Base V2 step 10k checkpoint")
    parser.add_argument("--tokenizer-model", required=True, help="Path to SentencePiece tokenizer.model")
    parser.add_argument("--reparos-cli", required=True, help="Path to reparos CLI binary in venv")
    parser.add_argument("--device", default="cuda", help="Inference device (cuda or cpu)")
    parser.add_argument("--compute-type", default="float16", help="Compute type (float16 or float32)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for CTranslate2 inference")
    parser.add_argument("--output", required=True, help="Output JSON path for gating results")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    eval_dir = Path(args.continual_data_root) / "eval"
    base_ckpt = Path(args.base_checkpoint)
    tokenizer_path = Path(args.tokenizer_model)
    reparos_cli = Path(args.reparos_cli)
    output_path = Path(args.output)

    # 1. Evaluate Base V2 baseline
    base_ct2_dir = run_root / "ct2_base_v2"
    if not base_ct2_dir.exists():
        print("Exporting Base V2 to CTranslate2...")
        subprocess.run([
            str(reparos_cli), "export-ctranslate2",
            "--model", str(base_ckpt),
            "--tokenizer", str(tokenizer_path),
            "--output", str(base_ct2_dir),
            "--compute-type", args.compute_type,
            "--trust-checkpoint",
        ], check=True)

    print("Evaluating Base V2 baseline across continual splits...")
    base_predictor = CTranslate2Predictor(base_ct2_dir, device=args.device, compute_type=args.compute_type)
    baseline_metrics = evaluate_continual_splits(base_predictor, eval_dir, batch_size=args.batch_size)
    del base_predictor

    print("=== Base V2 Baseline Metrics ===")
    print(f"  Plasticity Acc : {baseline_metrics['plasticity']['_overall']['exact_accuracy']*100:.2f}%")
    print(f"  Retention Acc  : {baseline_metrics['retention']['_overall']['exact_accuracy']*100:.2f}%")
    print(f"  FCR Seen       : {baseline_metrics['protection_seen']['_overall']['fcr']*100:.2f}%")
    print(f"  FCR Heldout    : {baseline_metrics['protection_heldout']['_overall']['fcr']*100:.2f}%")

    # 2. Locate and evaluate all candidate checkpoints
    saved_checkpoints = sorted(run_root.glob("reparos_continual_step_*.pt"), key=checkpoint_step)
    if not saved_checkpoints:
        raise FileNotFoundError(f"No reparos_continual_step_*.pt checkpoints found in {run_root}")

    print(f"\nEvaluating {len(saved_checkpoints)} continual candidate checkpoints...")
    candidate_results = []
    for ckpt in saved_checkpoints:
        step = checkpoint_step(ckpt)
        ct2_dir = run_root / f"ct2_step_{step}"
        if ct2_dir.exists():
            shutil.rmtree(ct2_dir)

        subprocess.run([
            str(reparos_cli), "export-ctranslate2",
            "--model", str(ckpt),
            "--tokenizer", str(tokenizer_path),
            "--output", str(ct2_dir),
            "--compute-type", args.compute_type,
            "--trust-checkpoint",
        ], check=True)

        predictor = CTranslate2Predictor(ct2_dir, device=args.device, compute_type=args.compute_type)
        split_metrics = evaluate_continual_splits(predictor, eval_dir, batch_size=args.batch_size)
        del predictor

        gating = evaluate_3tier_gating(baseline_metrics, split_metrics)
        record = {
            "step": step,
            "checkpoint": str(ckpt),
            "ct2_dir": str(ct2_dir),
            "status": gating["status"],
            "outcome": gating["outcome"],
            "plasticity_acc": split_metrics["plasticity"]["_overall"]["exact_accuracy"],
            "plasticity_gain": gating["gate_2_plasticity"]["gain"],
            "plasticity_noop": split_metrics["plasticity"]["_overall"]["noop_rate"],
            "retention_acc": split_metrics["retention"]["_overall"]["exact_accuracy"],
            "retention_deg": gating["gate_1_retention"]["degradation"],
            "fcr_seen": split_metrics["protection_seen"]["_overall"]["fcr"],
            "fcr_heldout": split_metrics["protection_heldout"]["_overall"]["fcr"],
            "gating_details": gating,
            "split_metrics": split_metrics,
        }
        candidate_results.append(record)
        print(f"  Step {step:5d} | {gating['status']:14s} | Plas={record['plasticity_acc']*100:5.2f}% ({record['plasticity_gain']*100:+5.2f}%) | Ret={record['retention_acc']*100:5.2f}% ({record['retention_deg']*100:+5.2f}%) | FCR Seen={record['fcr_seen']*100:4.1f}% | FCR Held={record['fcr_heldout']*100:4.1f}%")

    # 3. Pareto selection
    passing_candidates = [c for c in candidate_results if c["status"] == "PASSED"]
    if passing_candidates:
        winning_candidate = min(
            passing_candidates,
            key=lambda c: (c["fcr_seen"] + c["fcr_heldout"], -c["plasticity_gain"]),
        )
    else:
        print("\nWARNING: No checkpoint passed all 3 tiers. Selecting candidate with minimum Gate violations.")
        winning_candidate = min(
            candidate_results,
            key=lambda c: (c["fcr_heldout"], c["retention_deg"], -c["plasticity_gain"]),
        )

    print("\n" + "=" * 60)
    print("WINNING CHECKPOINT:", Path(winning_candidate["checkpoint"]).name)
    print("Step             :", winning_candidate["step"])
    print("Outcome          :", winning_candidate["outcome"])
    print(f"Plasticity Acc   : {winning_candidate['plasticity_acc']*100:.2f}% (gain: {winning_candidate['plasticity_gain']*100:+.2f}%)")
    print(f"Retention Acc    : {winning_candidate['retention_acc']*100:.2f}% (drop: {winning_candidate['retention_deg']*100:+.2f}%)")
    print(f"FCR Seen         : {winning_candidate['fcr_seen']*100:.2f}%")
    print(f"FCR Held-out     : {winning_candidate['fcr_heldout']*100:.2f}%")
    print("=" * 60 + "\n")

    report = {
        "baseline_metrics": baseline_metrics,
        "winning_candidate": winning_candidate,
        "candidate_results": candidate_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved complete evaluation report to {output_path}")


if __name__ == "__main__":
    main()
