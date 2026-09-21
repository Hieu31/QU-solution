#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def run_cmd(cmd: list[str], desc: str):
    print(f"\n>>> [{desc}] Running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f">>> [{desc}] Completed in {elapsed:.1f}s (code {res.returncode})", flush=True)


def train_and_eval_run(
    run_name: str,
    ablation_dir: Path,
    tokenizer_path: Path,
    eval_dir: Path,
    device: str = "cuda",
    batch_size: int = 32768,
    train_steps: int = 20000,
) -> Path:
    run_dir = ablation_dir / run_name
    config_path = run_dir / "opennmt_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found at {config_path}. Run prepare_base_v3_ablation_configs.py first.")

    # 1. Build Vocab
    vocab_src = run_dir / "vocab.src"
    vocab_tgt = run_dir / "vocab.tgt"
    if not vocab_src.is_file() or not vocab_tgt.is_file() or vocab_src.stat().st_size == 0:
        run_cmd(
            [sys.executable, "-m", "onmt.bin.build_vocab", "-config", str(config_path), "-n_sample", "-1"],
            f"Build Vocab for {run_name}",
        )

    # 2. Train OpenNMT
    checkpoints_dir = Path("checkpoints") / f"base_v3_ablation_{run_name}"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if checkpoint exists
    target_ckpt = checkpoints_dir / f"reparos_base_v3_{run_name}_step_{train_steps}.pt"
    existing_ckpts = sorted(checkpoints_dir.glob(f"reparos_base_v3_{run_name}_step_*.pt"))

    if not existing_ckpts:
        run_cmd(
            [sys.executable, "-m", "onmt.bin.train", "-config", str(config_path)],
            f"Train {run_name} for {train_steps} steps",
        )
        existing_ckpts = sorted(checkpoints_dir.glob(f"reparos_base_v3_{run_name}_step_*.pt"))

    if not existing_ckpts:
        raise RuntimeError(f"No checkpoint produced for {run_name} in {checkpoints_dir}")

    best_ckpt = existing_ckpts[-1]
    print(f"\nFinal checkpoint for {run_name}: {best_ckpt}")
    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description="End-to-End Orchestrator for Base V3 Ablation Experiments.")
    parser.add_argument("--runs", nargs="+", default=["dataset_run_a", "dataset_run_b", "dataset_run_c"],
                        choices=["dataset_run_a", "dataset_run_b", "dataset_run_c"])
    parser.add_argument("--ablation-dir", type=Path, default=Path("data/base_v3_ablation"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/base_v3_eval"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer_v3/tokenizer.model"))
    parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--train-steps", type=int, default=20000)
    args = parser.parse_args()

    print("=" * 80)
    print("      REPAROS BASE V3 ABLATION ORCHESTRATOR (20,000 STEPS FROM STEP 0)      ")
    print("=" * 80)
    print(f"Runs to execute: {args.runs}")
    print(f"Device: {args.device} | Batch Size: {args.batch_size} tokens | Steps: {args.train_steps:,}")

    completed_ckpts = {}
    for run in args.runs:
        print(f"\n{'#' * 80}\n# Starting {run.upper()}\n{'#' * 80}")
        ckpt = train_and_eval_run(
            run_name=run,
            ablation_dir=args.ablation_dir,
            tokenizer_path=args.tokenizer,
            eval_dir=args.eval_dir,
            device=args.device,
            batch_size=args.batch_size,
            train_steps=args.train_steps,
        )
        completed_ckpts[run] = ckpt

    # Unified Evaluation
    print("\n" + "=" * 80)
    print("Running Unified Frozen Evaluation across all completed runs...")
    print("=" * 80)

    eval_cmd = [
        sys.executable, "scripts/evaluate_base_v3_ablation.py",
        "--eval-dir", str(args.eval_dir),
        "--tokenizer", str(args.tokenizer),
        "--device", args.device,
        "--checkpoints", *[str(c) for c in completed_ckpts.values()],
        "--names", *list(completed_ckpts.keys()),
        "--output-json", "data/ablation_comparison_report.json"
    ]
    run_cmd(eval_cmd, "Unified Frozen Evaluation")
    print("\nAll Ablation Experiments Completed Successfully!")


if __name__ == "__main__":
    main()
