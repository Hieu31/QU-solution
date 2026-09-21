#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def main():
    notebook_path = Path("notebook/train_reparos_base_v3_ablation_kaggle.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    cells = []

    # Cell 1: Intro Markdown
    cells.append(md("""# ReparoS Base V3 — 3-Way Empirical Lane 4 Ablation Study (20,000 Steps)

This notebook runs the official **Base V3 Pre-training from Scratch Ablation Suite** across 3 controlled variants to determine the optimal real-query adaptation mixture:

- **Run A (No Lane 4):** 45% Lane 1 + 30% Lane 2 + 25% Lane 3 + 0% Lane 4
- **Run B (Lane 4 DAE-Only):** 40% Lane 1 + 25% Lane 2 + 25% Lane 3 + 10% Lane 4 DAE
- **Run C (Lane 4 DAE + Identity):** 40% Lane 1 + 25% Lane 2 + 25% Lane 3 + 7% DAE + 3% Identity

### Key Architectural & Data Upgrades in Base V3:
1. **Tokenizer V3 (12,000 Vocab, Byte-Fallback, Custom Symbols):** Solves the Base V2 `<unk>` brand truncation bottleneck (`Biti's`, `Pizza 4P's`, `McDonald's`, `J&T Express` are 100% tokenizable with 0.0000% UNK).
2. **Confidence-Gated Lane 4 (3-Tier Filter):** Resolves the label contradiction hazards of raw `zero_click.csv`.
3. **Strict Disjoint Brand Split:** 50 Seen vs 50 Held-out Brands with mathematical 0-leakage verification.
4. **Unified Frozen Evaluation Suite (~3,100 pairs):** Frozen across Plasticity, Retention, Protection (Seen & Held-out), and User-Centric benchmarks.
"""))

    # Cell 2: Hyperparameters & Configuration
    cells.append(code("""import os
import sys
from pathlib import Path

# --- Training Configurations ---
TRAIN_STEPS = 20_000
VALID_STEPS = 1_000
SAVE_CHECKPOINT_STEPS = 2_000
KEEP_CHECKPOINTS = 3
BATCH_SIZE_TOKENS = 32_768  # 32k for T4 / 65k for RTX 6000
BUCKET_SIZE = 65_536
NUM_WORKERS = 4
MODEL_DTYPE = "fp16"
SEED = 2026

RUNS = ["dataset_run_a", "dataset_run_b", "dataset_run_c"]
print(f"Configured {len(RUNS)} ablation runs with {TRAIN_STEPS:,} steps each.")
"""))

    # Cell 3: GPU & Environment Setup
    cells.append(code("""!nvidia-smi

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
"""))

    # Cell 4: Locate Repository Root & Data
    cells.append(code("""# Locate workspace root
import os
import sys
from pathlib import Path

# When running in Kaggle environment:
if Path("/kaggle/working/QU-solution").is_dir():
    REPO_ROOT = Path("/kaggle/working/QU-solution")
elif Path("../input/qu-solution").is_dir():
    REPO_ROOT = Path("../input/qu-solution")
else:
    REPO_ROOT = Path(os.getcwd())

os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))
print(f"Current Working Directory: {os.getcwd()}")

TOKENIZER_MODEL = REPO_ROOT / "data/tokenizer_v3/tokenizer.model"
EVAL_DIR = REPO_ROOT / "data/base_v3_eval"
ABLATION_DIR = REPO_ROOT / "data/base_v3_ablation"

assert TOKENIZER_MODEL.is_file(), f"Tokenizer not found at {TOKENIZER_MODEL}"
assert EVAL_DIR.is_dir(), f"Eval directory not found at {EVAL_DIR}"
assert ABLATION_DIR.is_dir(), f"Ablation directory not found at {ABLATION_DIR}"
print("All inputs located and verified!")
"""))

    # Cell 5: Verify Tokenizer V3 Zero-UNK Guarantee
    cells.append(code("""import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_MODEL))
print(f"Tokenizer Vocab Size: {sp.vocab_size():,}")

test_queries = [
    "Pizza 4P's Hai Bà Trưng",
    "Biti's Hunter Street x VietMax",
    "J&T Express Cầu Giấy",
    "McDonald's Bưu điện Trung tâm",
    "Co.opmart Nguyễn Kiệm Gò Vấp",
    "L'Oréal Paris kem chống nắng",
    "Quán ăn Quận 1 Đường 3/2",
]

print("\\nTesting Tokenizer V3 on special-symbol brand queries:")
all_clean = True
for q in test_queries:
    pieces = sp.encode_as_pieces(q.lower())
    ids = sp.encode_as_ids(q.lower())
    has_unk = sp.unk_id() in ids
    status = "❌ UNK DETECTED" if has_unk else "✅ CLEAN (0 UNK)"
    print(f"  [{status}] {q:<35} -> {pieces}")
    if has_unk:
        all_clean = False

assert all_clean, "Tokenizer V3 failed special-symbol verification!"
print("\\n100% PASS: Tokenizer V3 has mathematical 0 UNK on all target brands!")
"""))

    # Cell 6: Generate OpenNMT Configs & Build Vocabularies
    cells.append(code("""import subprocess

# Generate configs
!python scripts/prepare_base_v3_ablation_configs.py --batch-size $BATCH_SIZE_TOKENS --bucket-size $BUCKET_SIZE --num-workers $NUM_WORKERS

# Build OpenNMT Vocabularies (-n_sample -1)
for run_name in RUNS:
    config_file = ABLATION_DIR / run_name / "opennmt_config.json"
    vocab_src = ABLATION_DIR / run_name / "vocab.src"
    if not vocab_src.is_file() or vocab_src.stat().st_size == 0:
        print(f"\\nBuilding OpenNMT full vocabulary for {run_name}...")
        subprocess.run([
            sys.executable, "-m", "onmt.bin.build_vocab",
            "-config", str(config_file),
            "-n_sample", "-1"
        ], check=True)
    print(f"Vocab ready for {run_name}: {vocab_src}")
"""))

    # Cell 7: Execute Ablation Pre-training from Scratch
    cells.append(code("""import time

checkpoints = {}

for run_name in RUNS:
    print(f"\\n{'='*70}\\n>>> STARTING PRE-TRAINING FOR {run_name.upper()} ({TRAIN_STEPS:,} STEPS)\\n{'='*70}")
    config_file = ABLATION_DIR / run_name / "opennmt_config.json"
    ckpt_dir = Path("checkpoints") / f"base_v3_ablation_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    subprocess.run([
        sys.executable, "-m", "onmt.bin.train",
        "-config", str(config_file)
    ], check=True)
    
    elapsed = (time.time() - t0) / 60
    print(f"Pre-training {run_name} finished in {elapsed:.1f} minutes.")
    
    # Locate final checkpoint
    ckpts = sorted(ckpt_dir.glob(f"reparos_base_v3_{run_name}_step_*.pt"))
    assert len(ckpts) > 0, f"No checkpoint saved for {run_name}!"
    checkpoints[run_name] = str(ckpts[-1])
    print(f"Final checkpoint: {checkpoints[run_name]}")
"""))

    # Cell 8: Run Unified Frozen Evaluation & Comparison
    cells.append(code("""import subprocess

eval_cmd = [
    sys.executable, "scripts/evaluate_base_v3_ablation.py",
    "--eval-dir", str(EVAL_DIR),
    "--tokenizer", str(TOKENIZER_MODEL),
    "--device", "cuda" if torch.cuda.is_available() else "cpu",
    "--checkpoints", *list(checkpoints.values()),
    "--names", *list(checkpoints.keys()),
    "--output-json", "data/ablation_evaluation_report.json"
]

print("Running Unified Frozen Evaluation across Run A, Run B, Run C...")
subprocess.run(eval_cmd, check=True)
"""))

    # Cell 9: Display Final Markdown Summary & Best Run Selection
    cells.append(code("""import json

with open("data/ablation_evaluation_report.json", "r", encoding="utf-8") as f:
    results = json.load(f)

print("\\n" + "="*80)
print("                   FINAL EMPIRICAL DECISION MATRIX                           ")
print("="*80)

# Display comparative summary
for run, data in results.items():
    p = data["metrics"]["plasticity"]["exact_match"]
    r = data["metrics"]["retention"]["exact_match"]
    ps = data["metrics"]["protection_seen"]["exact_match"]
    ph = data["metrics"]["protection_heldout"]["exact_match"]
    uc = data["metrics"]["user_centric"]["exact_match"]
    br = data["metrics"]["protection_heldout"].get("brand_retention", 100.0)
    avg_score = (p + r + ps + ph + uc) / 5.0
    print(f"{run:<16} | Plasticity: {p:.1f}% | Retention: {r:.1f}% | Seen: {ps:.1f}% | Heldout: {ph:.1f}% (Brand: {br:.1f}%) | User: {uc:.1f}% | Overall: {avg_score:.2f}%")

print("="*80)
print("\\nCheck qualitative diagnostics in the stdout log above to verify zero error preservation on zero-click queries.")
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 4
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Generated self-contained Kaggle Notebook at {notebook_path}")


if __name__ == "__main__":
    main()
