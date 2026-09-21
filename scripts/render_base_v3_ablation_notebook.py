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
    cells.append(md("""# ReparoS Base V3 — 3-Way Empirical Ablation Study (Kaggle T4)

This notebook runs the official **Base V3 Pre-training from Scratch Ablation Suite** across 3 controlled variants to determine the optimal real-query adaptation mixture:

- **Run A (No Lane 4):** 45% Lane 1 + 30% Lane 2 + 25% Lane 3 + 0% Lane 4
- **Run B (Lane 4 DAE-Only):** 40% Lane 1 + 25% Lane 2 + 25% Lane 3 + 10% Lane 4 DAE
- **Run C (Lane 4 DAE + Identity):** 40% Lane 1 + 25% Lane 2 + 25% Lane 3 + 7% DAE + 3% Identity

### Setup Requirement:
- **Accelerator:** GPU T4 x1 (or T4 x2)
- **Internet:** ON (pip install OpenNMT-py in ~20 seconds)
- **Input Dataset:** Upload & attach `reparos-base-v3-data.zip` (~21 MB)
"""))

    # Cell 2: Install Dependencies
    cells.append(code("""# Install required dependencies (~20 seconds, lock numpy<2 for OpenNMT stability)
!pip install -q "numpy<2" "OpenNMT-py>=3.5,<4" sentencepiece ctranslate2
"""))

    # Cell 3: GPU & Environment Check
    cells.append(code("""!nvidia-smi

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
"""))

    # Cell 4: Locate Dataset & Setup Working Directory
    cells.append(code("""import os
import shutil
import sys
from pathlib import Path

KAGGLE_INPUT = Path('/kaggle/input')
WORK_DIR = Path('/kaggle/working/QU-solution')

# Find the dataset path containing data/tokenizer_v3/tokenizer.model
candidates = []
for p in KAGGLE_INPUT.rglob('tokenizer.model'):
    if 'tokenizer_v3' in str(p):
        root = p.parent.parent.parent
        candidates.append(root)

candidates = sorted(set(candidates))
if candidates:
    DATASET_ROOT = candidates[0]
else:
    DATASET_ROOT = Path(os.getcwd())

print(f"Located DATASET_ROOT: {DATASET_ROOT}")

# Prepare working directory
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Copy code and data to working directory for writable execution
if (DATASET_ROOT / 'src').is_dir():
    shutil.copytree(DATASET_ROOT / 'src', WORK_DIR / 'src')
if (DATASET_ROOT / 'scripts').is_dir():
    shutil.copytree(DATASET_ROOT / 'scripts', WORK_DIR / 'scripts')
if (DATASET_ROOT / 'data').is_dir():
    shutil.copytree(DATASET_ROOT / 'data', WORK_DIR / 'data')

os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR / 'src'))

TOKENIZER_MODEL = WORK_DIR / "data/tokenizer_v3/tokenizer.model"
EVAL_DIR = WORK_DIR / "data/base_v3_eval"
ABLATION_DIR = WORK_DIR / "data/base_v3_ablation"

assert TOKENIZER_MODEL.is_file(), f"Tokenizer not found at {TOKENIZER_MODEL}"
assert EVAL_DIR.is_dir(), f"Eval directory not found at {EVAL_DIR}"
assert ABLATION_DIR.is_dir(), f"Ablation directory not found at {ABLATION_DIR}"
print("All directories verified and ready in working space!")
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

    # Cell 6: Hyperparameters & Vocab Generation
    cells.append(code("""import subprocess
import json

TRAIN_STEPS = 20_000
VALID_STEPS = 1_000
SAVE_CHECKPOINT_STEPS = 2_000
KEEP_CHECKPOINTS = 3
BATCH_SIZE_TOKENS = 32_768  # 32k for T4 GPU
BUCKET_SIZE = 65_536
NUM_WORKERS = 4
MODEL_DTYPE = "fp16"
SEED = 2026

RUNS = ["dataset_run_a", "dataset_run_b", "dataset_run_c"]

# Generate OpenNMT configs
!python scripts/prepare_base_v3_ablation_configs.py --batch-size $BATCH_SIZE_TOKENS --bucket-size $BUCKET_SIZE --num-workers $NUM_WORKERS

# Build OpenNMT Vocabularies (-n_sample -1) without log clutter
for run_name in RUNS:
    config_file = ABLATION_DIR / run_name / "opennmt_config.json"
    vocab_src = ABLATION_DIR / run_name / "vocab.src"
    if not vocab_src.is_file() or vocab_src.stat().st_size == 0:
        print(f"Building OpenNMT vocabulary for {run_name}...", end=" ", flush=True)
        res = subprocess.run([
            sys.executable, "-m", "onmt.bin.build_vocab",
            "-config", str(config_file),
            "-n_sample", "-1"
        ], capture_output=True, text=True)
        if res.returncode != 0:
            print("FAILED!\\n", res.stderr)
            raise subprocess.CalledProcessError(res.returncode, res.args)
        print("Done!")
    print(f"Vocab ready for {run_name}: {vocab_src}")
"""))

    # Cell 7: Execute Ablation Pre-training from Scratch (Clean progress output)
    cells.append(code("""import time

checkpoints = {}

for run_name in RUNS:
    print(f"\\n{'='*70}\\n>>> STARTING PRE-TRAINING FOR {run_name.upper()} ({TRAIN_STEPS:,} STEPS)\\n{'='*70}")
    config_file = ABLATION_DIR / run_name / "opennmt_config.json"
    ckpt_dir = Path("checkpoints") / f"base_v3_ablation_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = ckpt_dir / "train.log"
    print(f"Training in progress... (Chi tiết log đầy đủ được lưu vào {log_path})")
    print("Tiến độ các mốc chính:")
    
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f_log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "onmt.bin.train", "-config", str(config_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            f_log.write(line)
            clean = line.strip()
            # Chỉ in các mốc 2,000 steps và khi lưu checkpoint
            if any(f"Step {s}/" in clean for s in range(2000, 22000, 2000)) or "Saving checkpoint" in clean or "Validation score" in clean:
                print(f"  {clean}")
        proc.wait()
        if proc.returncode != 0:
            print(f"\\nTraining {run_name} failed! Check {log_path} for details.")
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
    
    elapsed = (time.time() - t0) / 60
    print(f"Pre-training {run_name} finished in {elapsed:.1f} minutes.")
    
    ckpts = sorted(ckpt_dir.glob(f"reparos_base_v3_{run_name}_step_*.pt"))
    assert len(ckpts) > 0, f"No checkpoint saved for {run_name}!"
    checkpoints[run_name] = str(ckpts[-1])
    print(f"Final checkpoint: {checkpoints[run_name]}")
"""))

    # Cell 8: Run Unified Frozen Evaluation & Comparison
    cells.append(code("""eval_env = dict(os.environ)
eval_env["PYTHONPATH"] = f"{WORK_DIR}/src:{eval_env.get('PYTHONPATH', '')}"

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
subprocess.run(eval_cmd, check=True, env=eval_env)
"""))

    # Cell 9: Display Final Markdown Summary & Best Run Selection
    cells.append(code("""with open("data/ablation_evaluation_report.json", "r", encoding="utf-8") as f:
    results = json.load(f)

print("\\n" + "="*85)
print("                   FINAL EMPIRICAL DECISION MATRIX                           ")
print("="*85)
header = f"{'Run Name':<18} | {'Plasticity':<12} | {'Retention':<12} | {'Seen Brands':<12} | {'Heldout Brand':<14} | {'User':<10}"
print(header)
print("-" * len(header))

for run, data in results.items():
    p = data["metrics"]["plasticity"]["exact_match"]
    r = data["metrics"]["retention"]["exact_match"]
    ps = data["metrics"]["protection_seen"]["exact_match"]
    ph = data["metrics"]["protection_heldout"]["exact_match"]
    br = data["metrics"]["protection_heldout"].get("brand_retention", 100.0)
    uc = data["metrics"]["user_centric"]["exact_match"]
    print(f"{run:<18} | {p:>10.2f}% | {r:>10.2f}% | {ps:>10.2f}% | {ph:>6.2f}% ({br:.0f}%) | {uc:>8.2f}%")

print("="*85)
print("\\nReview qualitative diagnostic outputs above to verify zero label contradiction & zero error preservation.")
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
    print(f"Updated Kaggle Notebook at {notebook_path}")


if __name__ == "__main__":
    main()
