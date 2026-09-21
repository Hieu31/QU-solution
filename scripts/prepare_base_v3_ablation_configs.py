#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def create_run_config(
    run_name: str,
    ablation_dir: Path,
    tokenizer_path: Path,
    batch_size: int = 32768,
    bucket_size: int = 65536,
    num_workers: int = 4,
    gpu_rank: int | None = 0,
    train_steps: int = 20000,
    valid_steps: int = 1000,
    save_checkpoint_steps: int = 2000,
) -> Path:
    run_dir = ablation_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_src = run_dir / "train.src"
    train_tgt = run_dir / "train.tgt"
    valid_src = run_dir / "valid.src"
    valid_tgt = run_dir / "valid.tgt"

    for p in [train_src, train_tgt, valid_src, valid_tgt]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required data file: {p}")

    checkpoints_dir = Path("checkpoints") / f"base_v3_ablation_{run_name}"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "save_data": str(run_dir / "vocab").replace("\\", "/"),
        "src_vocab": str(run_dir / "vocab.src").replace("\\", "/"),
        "tgt_vocab": str(run_dir / "vocab.tgt").replace("\\", "/"),
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
        "save_model": str(checkpoints_dir / f"reparos_base_v3_{run_name}").replace("\\", "/"),
        "encoder_type": "transformer",
        "decoder_type": "transformer",
        "enc_layers": 1,
        "dec_layers": 1,
        "heads": 8,
        "hidden_size": 128,
        "word_vec_size": 128,
        "transformer_ff": 512,
        "position_encoding": True,
        # Scaled-dot ensures compatibility with CTranslate2
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
        "train_steps": train_steps,
        "valid_steps": valid_steps,
        "save_checkpoint_steps": save_checkpoint_steps,
        "keep_checkpoint": 3,
        "seed": 2026,
        "world_size": 1,
        "gpu_ranks": [gpu_rank] if gpu_rank is not None else [],
    }

    config_path = run_dir / "opennmt_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Generated {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Generate OpenNMT configs for Base V3 Ablation Runs.")
    parser.add_argument("--ablation-dir", type=Path, default=Path("data/base_v3_ablation"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer_v3/tokenizer.model"))
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--bucket-size", type=int, default=65536)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-rank", type=int, default=0)
    args = parser.parse_args()

    for run in ["dataset_run_a", "dataset_run_b", "dataset_run_c"]:
        create_run_config(
            run_name=run,
            ablation_dir=args.ablation_dir,
            tokenizer_path=args.tokenizer,
            batch_size=args.batch_size,
            bucket_size=args.bucket_size,
            num_workers=args.num_workers,
            gpu_rank=args.gpu_rank,
        )

    print("\nAll 3 Base V3 ablation configs successfully generated!")


if __name__ == "__main__":
    main()
