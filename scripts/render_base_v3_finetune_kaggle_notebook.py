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
    notebook_path = Path("notebook/train_reparos_base_v3_1_finetune_kaggle.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    cells = []

    # Cell 1: Header
    cells.append(md("""# ReparoS Base V3.1 — Production Fine-Tuning (1,000,000 Pairs)
### Targeted Optimization: Address Symbols + Nested Acronyms + Replay Anchor

This notebook fine-tunes **ReparoS Base V3** (Step 50,000 checkpoint) into **ReparoS V3.1 Production** using the 1,000,000-pair dataset.
- **Warm-start:** Checkpoint `reparos_base_v3_production_step_50000.pt`
- **Training schedule:** Step 50,000 $\\to$ 56,000 (6,000 steps, ~15-20 min on T4 GPU).
- **Safety certified:** 60% Replay Anchor (600k pairs) prevents catastrophic forgetting of VNI/Telex/Brands.
- **Automated Benchmark:** Automatically exports to CTranslate2 INT8 and evaluates across all 3 legacy benchmarks (`diagnostic-10k`, `compositional-4k`, `user-centric-v2`).
"""))

    # Cell 2: Hyperparameters
    cells.append(code("""START_STEP = 50_000
TRAIN_STEPS = 56_000
SAVE_CHECKPOINT_STEPS = 1_000
VALID_STEPS = 500
LEARNING_RATE = 0.0001
BATCH_SIZE_TOKENS = 32_768
MODEL_DTYPE = "fp16"
SEED = 2026
"""))

    # Cell 3: Environment Setup & Dataset Detection
    cells.append(code("""import json, os, shutil, subprocess, sys, time
from pathlib import Path

KAGGLE_INPUT = Path('/kaggle/input')
REPO_ROOT = Path('/kaggle/working/QU-solution')

print("Locating dataset under /kaggle/input...")
extracted_candidate = None
zip_candidate = None

for p in KAGGLE_INPUT.rglob('data/base_v3_finetune'):
    if p.is_dir():
        extracted_candidate = p.parent.parent
        break

if not extracted_candidate:
    zips = sorted(list(KAGGLE_INPUT.rglob('*.zip')))
    if zips:
        zip_candidate = zips[0]

if REPO_ROOT.exists():
    shutil.rmtree(REPO_ROOT)
REPO_ROOT.mkdir(parents=True, exist_ok=True)

if extracted_candidate:
    print(f"Found already extracted dataset at: {extracted_candidate}")
    for folder in ['src', 'scripts', 'data', 'benchmark', 'checkpoints']:
        src_dir = extracted_candidate / folder
        dst_dir = REPO_ROOT / folder
        if src_dir.exists():
            print(f"  Copying {folder}/ to {dst_dir}...")
            shutil.copytree(src_dir, dst_dir)
elif zip_candidate:
    print(f"Found zip bundle at: {zip_candidate}")
    print("Unpacking zip bundle into /kaggle/working/QU-solution...")
    subprocess.run(['unzip', '-q', str(zip_candidate), '-d', str(REPO_ROOT)], check=True)
else:
    raise FileNotFoundError("Could not find dataset in /kaggle/input! Please ensure your Kaggle dataset is attached.")

print("Project workspace successfully initialized at /kaggle/working/QU-solution!")
"""))

    # Cell 4: Install Dependencies
    cells.append(code("""print("Installing OpenNMT-py and CTranslate2...")
subprocess.run(['pip', 'install', '-q', 'OpenNMT-py==3.5.1', 'ctranslate2==4.8.2', 'sentencepiece==0.2.0'], check=True)
print("Dependencies installed successfully!")
"""))

    # Cell 5: Fine-Tuning Execution
    cells.append(code("""import time
os.chdir(str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / 'data/base_v3_finetune/opennmt_finetune_config.json'
LOG_FILE = REPO_ROOT / 'train_finetune.log'

# Ensure checkpoints destination directory exists
(REPO_ROOT / 'checkpoints/base_v3_1_finetune').mkdir(parents=True, exist_ok=True)

# Ensure vocabs exist; if not, build them automatically
vocab_src = REPO_ROOT / 'data/base_v3_finetune/vocab.src'
vocab_tgt = REPO_ROOT / 'data/base_v3_finetune/vocab.tgt'
if not vocab_src.is_file() or not vocab_tgt.is_file() or vocab_src.stat().st_size == 0:
    print("Building OpenNMT vocab with Tokenizer V3...")
    subprocess.run([
        'onmt_build_vocab', '-config', str(CONFIG_PATH), '-n_sample', '100000'
    ], check=True)
    print("Vocab built successfully!")

cmd = ['onmt_train', '-config', str(CONFIG_PATH)]
print(f"Starting Fine-Tuning (Step {START_STEP} -> {TRAIN_STEPS})...")
print(f"Command: {' '.join(cmd)}")
t0 = time.time()

with open(LOG_FILE, 'w', encoding='utf-8') as log_f:
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

# Monitor progress
last_line = ""
while proc.poll() is None:
    time.sleep(10)
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            lines = [l.strip() for l in f if l.strip()]
            for l in lines[-10:]:
                if any(k in l for k in ['Step', 'Validation', 'Saving', 'loss:']):
                    if l != last_line:
                        print(f"[{time.strftime('%H:%M:%S')}] {l}")
                        last_line = l

if proc.returncode != 0:
    print(f"ERROR: Fine-tuning failed with return code {proc.returncode}!")
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            print("\nLast 30 lines of train_finetune.log:")
            print("\n".join(f.readlines()[-30:]))
    raise RuntimeError("Fine-tuning failed.")

print(f"\nFine-Tuning finished successfully in {(time.time() - t0)/60:.1f} minutes!")
"""))

    # Cell 6: Export to CTranslate2 INT8 (Fixing NumPy 2.x ABI incompatibility via Subprocess)
    cells.append(code("""print("Ensuring numpy<2 is installed for CTranslate2 INT8 quantization compatibility...")
subprocess.run(['pip', 'install', '-q', 'numpy<2'], check=True)

CKPT_DIR = REPO_ROOT / 'checkpoints/base_v3_1_finetune'
ckpts = sorted(CKPT_DIR.glob('reparos_base_v3_1_finetune_step_*.pt'))
if not ckpts:
    raise FileNotFoundError("No checkpoint found in checkpoints/base_v3_1_finetune!")

final_ckpt = ckpts[-1]
print(f"Selected Checkpoint for Export: {final_ckpt.name}")

EXPORT_DIR = REPO_ROOT / 'artifacts/checkpoints/base_v3_1_finetune/ctranslate2_export'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Run conversion in a fresh subprocess to ensure clean NumPy 1.x runtime
export_script = REPO_ROOT / 'scripts/run_export_ct2.py'
export_script.parent.mkdir(parents=True, exist_ok=True)
export_script.write_text(f'''import argparse, sys
from pathlib import Path
import torch
import ctranslate2

try:
    torch.serialization.add_safe_globals([argparse.Namespace])
except Exception:
    pass

ckpt = "{final_ckpt.as_posix()}"
out = "{EXPORT_DIR.as_posix()}"
print(f"Converting {{ckpt}} to INT8 at {{out}}...")
converter = ctranslate2.converters.OpenNMTPyConverter(ckpt, unsafe_deserialization=True)
converter.convert(out, quantization="int8", force=True)
print("Conversion successful!")
''', encoding='utf-8')

print("Executing CTranslate2 export in fresh process...")
subprocess.run(['python', str(export_script)], check=True)

# Copy tokenizer.model
tok_src = REPO_ROOT / 'data/tokenizer_v3/tokenizer.model'
shutil.copy2(tok_src, EXPORT_DIR / 'tokenizer.model')

print("\\nExport complete! Files in export directory:")
for f in EXPORT_DIR.glob("*"):
    print(f"  {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
"""))

    # Cell 7: Benchmark Evaluation (Base V3.1 FT vs Base V1, V2, V3)
    cells.append(code("""print("Running Comprehensive Evaluation: Base V3.1 FT vs Base V1, Base V2, Base V3...")
subprocess.run(['python', 'scripts/evaluate_finetuned_v3_1.py'], check=True)
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"\nKaggle Fine-Tuning Notebook rendered successfully at {notebook_path}!")


if __name__ == "__main__":
    main()
