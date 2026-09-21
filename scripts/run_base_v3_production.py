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


def create_production_config(
    data_dir: Path,
    tokenizer_path: Path,
    checkpoints_dir: Path,
    batch_size: int = 65536,
    bucket_size: int = 131072,
    num_workers: int = 8,
    gpu_rank: int | None = 0,
    train_steps: int = 50000,
    valid_steps: int = 1000,
    save_checkpoint_steps: int = 2000,
    report_every: int = 2000,
) -> Path:
    train_src = data_dir / "train.src"
    train_tgt = data_dir / "train.tgt"
    valid_src = data_dir / "valid.src"
    valid_tgt = data_dir / "valid.tgt"

    for p in [train_src, train_tgt, valid_src, valid_tgt]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required data file: {p}")

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / "opennmt_config.json"

    config = {
        "save_data": str(data_dir / "vocab").replace("\\", "/"),
        "src_vocab": str(data_dir / "vocab.src").replace("\\", "/"),
        "tgt_vocab": str(data_dir / "vocab.tgt").replace("\\", "/"),
        "src_vocab_size": 12000,
        "tgt_vocab_size": 12000,
        "overwrite": True,
        "data": {
            "corpus_1": {
                "path_src": str(train_src).replace("\\", "/"),
                "path_tgt": str(train_tgt).replace("\\", "/"),
                "transforms": ["sentencepiece"],
                "weight": 1,
            },
            "valid": {
                "path_src": str(valid_src).replace("\\", "/"),
                "path_tgt": str(valid_tgt).replace("\\", "/"),
                "transforms": ["sentencepiece"],
            },
        },
        "src_subword_model": str(tokenizer_path).replace("\\", "/"),
        "tgt_subword_model": str(tokenizer_path).replace("\\", "/"),
        "save_model": str(checkpoints_dir / "reparos_base_v3_production").replace("\\", "/"),
        "encoder_type": "transformer",
        "decoder_type": "transformer",
        "enc_layers": 1,
        "dec_layers": 1,
        "heads": 8,
        "hidden_size": 128,
        "word_vec_size": 128,
        "transformer_ff": 512,
        "position_encoding": True,
        "self_attn_type": "scaled-dot",
        "dropout": [0.1],
        "attention_dropout": [0.1],
        "model_dtype": "fp16",
        "param_init": 0.0,
        "param_init_glorot": True,
        "optim": "adam",
        "learning_rate": 1.0,
        "adam_beta1": 0.8,
        "adam_beta2": 0.998,
        "adam_eps": 1e-8,
        "decay_method": "noam",
        "warmup_steps": 4000,
        "max_grad_norm": 1.0,
        "batch_type": "tokens",
        "batch_size": batch_size,
        "bucket_size": bucket_size,
        "num_workers": num_workers,
        "normalization": "tokens",
        "report_every": report_every,
        "train_steps": train_steps,
        "valid_steps": valid_steps,
        "save_checkpoint_steps": save_checkpoint_steps,
        "keep_checkpoint": 5,
        "seed": 2026,
        "world_size": 1,
        "gpu_ranks": [gpu_rank] if gpu_rank is not None else [],
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Generated production OpenNMT config at {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="End-to-End Orchestrator for Base V3 Production Training (4M pairs).")
    parser.add_argument("--data-dir", type=Path, default=Path("data/base_v3_production"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer_v3/tokenizer.model"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/base_v3_eval"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints/base_v3_production"))
    parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--bucket-size", type=int, default=131072)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu-rank", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=50000)
    parser.add_argument("--valid-steps", type=int, default=1000)
    parser.add_argument("--save-checkpoint-steps", type=int, default=2000)
    parser.add_argument("--report-every", type=int, default=2000)
    parser.add_argument("--build-vocab-only", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("      REPAROS BASE V3 PRODUCTION PRE-TRAINING (RUN C WINNING RECIPE, 4M PAIRS)      ")
    print("=" * 80)
    print(f"Data Dir: {args.data_dir}")
    print(f"Device: {args.device} | Batch Size: {args.batch_size} tokens | Steps: {args.train_steps:,}")

    config_path = create_production_config(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        checkpoints_dir=args.checkpoints_dir,
        batch_size=args.batch_size,
        bucket_size=args.bucket_size,
        num_workers=args.num_workers,
        gpu_rank=args.gpu_rank if args.device == "cuda" else None,
        train_steps=args.train_steps,
        valid_steps=args.valid_steps,
        save_checkpoint_steps=args.save_checkpoint_steps,
        report_every=args.report_every,
    )

    # 1. Build Vocab
    vocab_src = args.data_dir / "vocab.src"
    vocab_tgt = args.data_dir / "vocab.tgt"
    if not vocab_src.is_file() or not vocab_tgt.is_file() or vocab_src.stat().st_size == 0:
        run_cmd(
            [sys.executable, "-m", "onmt.bin.build_vocab", "-config", str(config_path), "-n_sample", "-1"],
            "Build Vocab for Base V3 Production",
        )
    else:
        print(f"Vocab files already exist at {vocab_src} and {vocab_tgt}")

    if args.build_vocab_only:
        print("Build vocab completed (--build-vocab-only specified). Exiting.")
        return

    # 2. Train OpenNMT
    target_ckpt = args.checkpoints_dir / f"reparos_base_v3_production_step_{args.train_steps}.pt"
    existing_ckpts = sorted(args.checkpoints_dir.glob("reparos_base_v3_production_step_*.pt"))

    if not target_ckpt.exists():
        print(f"\nStarting OpenNMT Pre-training to {args.train_steps:,} steps...")
        run_cmd(
            [sys.executable, "-m", "onmt.bin.train", "-config", str(config_path)],
            f"Train Base V3 Production ({args.train_steps} steps)",
        )
        existing_ckpts = sorted(args.checkpoints_dir.glob("reparos_base_v3_production_step_*.pt"))

    if not existing_ckpts:
        raise RuntimeError(f"No checkpoint found in {args.checkpoints_dir}")

    best_ckpt = existing_ckpts[-1]
    print(f"\nBase V3 Production Training Completed! Checkpoint: {best_ckpt}")

    # 3. Run Frozen Evaluation
    print("\nRunning Frozen Unified Evaluation Suite...")
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_base_v3_ablation.py",
        "--checkpoints-dir", str(args.checkpoints_dir.parent),
        "--eval-dir", str(args.eval_dir),
        "--tokenizer", str(args.tokenizer),
        "--device", args.device,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")
    subprocess.run(eval_cmd, check=True, env=env)


if __name__ == "__main__":
    main()
