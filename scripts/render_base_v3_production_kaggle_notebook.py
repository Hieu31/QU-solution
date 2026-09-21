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
    notebook_path = Path("notebook/train_reparos_base_v3_production_kaggle.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    cells = []

    # Cell 1: Intro Markdown
    cells.append(md("""# ReparoS Base V3 — Production Pre-Training (4,000,000 Pairs)
### Winning Recipe: Run C (40% L1 + 25% L2 + 25% L3 + 7% DAE + 3% Identity)

This notebook executes official pre-training from scratch for **ReparoS Base V3** using the production dataset of **4,000,000 unique pairs**.

### Key Features:
- **Clean Milestone Logging:** Verbose OpenNMT step logs are safely redirected to `train.log`. The notebook cell displays milestone progress every 2,000 steps and checkpoint saves, keeping the output clean.
- **Auto-Crash Diagnostics:** If an error occurs, the notebook automatically surfaces the last 25 lines of `train.log` right inside the cell output.
- **Zero-Leakage Certified:** 100% leak-free, zero held-out brands, zero eval overlaps, zero label contradictions.
- **Automated Frozen Evaluation:** Benchmarks the final checkpoint against all 3,150 test queries across all 6 frozen test suites upon completion.
"""))

    # Cell 2: Hyperparameters
    cells.append(code("""TRAIN_STEPS = 50_000
VALID_STEPS = 2_000
SAVE_CHECKPOINT_STEPS = 5_000
KEEP_CHECKPOINTS = 5
BATCH_SIZE_TOKENS = 32_768  # 32,768 tokens/batch (FP16 on Kaggle T4 GPU)
BUCKET_SIZE = 65_536
NUM_WORKERS = 4
MODEL_DTYPE = "fp16"
SEED = 2026
REPORT_EVERY = 2_000
"""))

    # Cell 3: Environment Setup, Dataset Detection & Dependency Management
    cells.append(code("""import json, os, shutil, subprocess, sys, time
from pathlib import Path

KAGGLE_INPUT = Path('/kaggle/input')
REPO_ROOT = Path('/kaggle/working/QU-solution')

# 1. Locate dataset bundle (supports both Kaggle auto-extracted folders and .zip files)
print("Locating dataset under /kaggle/input...")
extracted_candidate = None
zip_candidate = None

for p in KAGGLE_INPUT.rglob('data/base_v3_production'):
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
    for folder in ['src', 'scripts', 'data']:
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

# 2. Setup Python environment and path
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / 'src'))

# 3. Ensure NumPy < 2 for OpenNMT / PyTorch C-extension compatibility
try:
    import numpy as np
    if int(np.__version__.split('.')[0]) >= 2:
        print(f"NumPy {np.__version__} detected. Downgrading to numpy<2 for OpenNMT compatibility...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'numpy<2', '--quiet'], check=True)
except Exception:
    pass

# 4. Install OpenNMT-py and sentencepiece if not present
try:
    import onmt
    import sentencepiece
    print("OpenNMT-py and sentencepiece are ready.")
except ImportError:
    print("Installing OpenNMT-py==3.5.1, sentencepiece, and numpy<2...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'numpy<2', 'OpenNMT-py==3.5.1', 'sentencepiece', '--quiet'], check=True)
"""))

    # Cell 4: Verify Production Dataset Integrity
    cells.append(code("""DATA_DIR = REPO_ROOT / 'data/base_v3_production'
TOKENIZER_MODEL = REPO_ROOT / 'data/tokenizer_v3/tokenizer.model'
EVAL_DIR = REPO_ROOT / 'data/base_v3_eval'

assert (DATA_DIR / 'train.src').exists(), f"train.src not found in {DATA_DIR}!"
assert (DATA_DIR / 'train.tgt').exists(), f"train.tgt not found in {DATA_DIR}!"
assert TOKENIZER_MODEL.exists(), f"tokenizer.model not found in {TOKENIZER_MODEL}!"

# Verify line counts
src_lines = sum(1 for _ in open(DATA_DIR / 'train.src', 'r', encoding='utf-8'))
tgt_lines = sum(1 for _ in open(DATA_DIR / 'train.tgt', 'r', encoding='utf-8'))
print(f"Verified Production Dataset: {src_lines:,} src lines | {tgt_lines:,} tgt lines")
assert src_lines == tgt_lines == 4_000_000, f"Expected exactly 4,000,000 pairs, found {src_lines:,}"
"""))

    # Cell 5: Generate OpenNMT Config & Build Vocab
    cells.append(code("""CKPT_DIR = Path('/kaggle/working/checkpoints/base_v3_production')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

config_path = DATA_DIR / 'opennmt_config.json'
config = {
    "save_data": str(DATA_DIR / "vocab"),
    "src_vocab": str(DATA_DIR / "vocab.src"),
    "tgt_vocab": str(DATA_DIR / "vocab.tgt"),
    "src_vocab_size": 12000,
    "tgt_vocab_size": 12000,
    "overwrite": True,
    "data": {
        "corpus_1": {
            "path_src": str(DATA_DIR / "train.src"),
            "path_tgt": str(DATA_DIR / "train.tgt"),
            "transforms": ["sentencepiece"],
            "weight": 1
        },
        "valid": {
            "path_src": str(DATA_DIR / "valid.src"),
            "path_tgt": str(DATA_DIR / "valid.tgt"),
            "transforms": ["sentencepiece"]
        }
    },
    "src_subword_model": str(TOKENIZER_MODEL),
    "tgt_subword_model": str(TOKENIZER_MODEL),
    "save_model": str(CKPT_DIR / "reparos_base_v3_production"),
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
    "model_dtype": MODEL_DTYPE,
    "param_init": 0.0,
    "param_init_glorot": True,
    "optim": "adam",
    "learning_rate": 1.0,
    "adam_beta1": 0.8,
    "adam_beta2": 0.998,
    "adam_eps": 1e-08,
    "decay_method": "noam",
    "warmup_steps": 4000,
    "max_grad_norm": 1.0,
    "batch_type": "tokens",
    "batch_size": BATCH_SIZE_TOKENS,
    "bucket_size": BUCKET_SIZE,
    "num_workers": NUM_WORKERS,
    "normalization": "tokens",
    "report_every": REPORT_EVERY,
    "train_steps": TRAIN_STEPS,
    "valid_steps": VALID_STEPS,
    "save_checkpoint_steps": SAVE_CHECKPOINT_STEPS,
    "keep_checkpoint": KEEP_CHECKPOINTS,
    "seed": SEED,
    "world_size": 1,
    "gpu_ranks": [0]
}

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)

# Build Vocab
vocab_src = DATA_DIR / 'vocab.src'
vocab_tgt = DATA_DIR / 'vocab.tgt'
if not vocab_src.is_file() or not vocab_tgt.is_file() or vocab_src.stat().st_size == 0:
    print("Building OpenNMT vocab with Tokenizer V3...")
    subprocess.run([
        sys.executable, '-m', 'onmt.bin.build_vocab',
        '-config', str(config_path), '-n_sample', '-1'
    ], check=True)
print("Vocab built successfully! Ready for training.")
"""))

    # Cell 6: Train OpenNMT with Clean Output Logging & Crash Diagnostics
    cells.append(code("""log_file = CKPT_DIR / 'train.log'
print(f"Starting Pre-training to {TRAIN_STEPS:,} steps...")
print(f"Detailed logs redirected to {log_file} (clean milestone display below):\\n")

t0 = time.time()
with open(log_file, 'w', encoding='utf-8') as log_f:
    proc = subprocess.Popen(
        [sys.executable, '-m', 'onmt.bin.train', '-config', str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in proc.stdout:
        log_f.write(line)
        # Only print clean milestone step summaries and checkpoint saves
        if 'Step ' in line and ('acc:' in line or 'loss:' in line or 'Saving checkpoint' in line):
            print(f"[{time.strftime('%H:%M:%S')}] {line.strip()}", flush=True)
    proc.wait()

    if proc.returncode != 0:
        print("\\n" + "!" * 75)
        print(f"TRAINING ENCOUNTERED AN ERROR (EXIT CODE {proc.returncode})!")
        print("Last 25 lines of train.log:")
        print("!" * 75)
        with open(log_file, 'r', encoding='utf-8') as f_read:
            last_lines = f_read.readlines()[-25:]
            print(''.join(last_lines))
        print("!" * 75)
        raise subprocess.CalledProcessError(proc.returncode, 'onmt.bin.train')

elapsed_hours = (time.time() - t0) / 3600
print(f"\\nPre-training completed successfully in {elapsed_hours:.2f} hours!")

ckpts = sorted(CKPT_DIR.glob('reparos_base_v3_production_step_*.pt'))
assert ckpts, "No checkpoint found!"
best_ckpt = ckpts[-1]
print(f"Final Checkpoint: {best_ckpt}")
"""))

    # Cell 7: Unified Frozen Evaluation Across 3,150 Test Queries
    cells.append(code("""eval_script = REPO_ROOT / 'scripts/evaluate_base_v3_ablation.py'
eval_cmd = [
    sys.executable, str(eval_script),
    '--eval-dir', str(EVAL_DIR),
    '--tokenizer', str(TOKENIZER_MODEL),
    '--device', 'cuda',
    '--checkpoints', str(best_ckpt),
    '--names', 'base_v3_production',
    '--output-json', '/kaggle/working/production_evaluation_report.json'
]

env = os.environ.copy()
env['PYTHONPATH'] = str(REPO_ROOT / 'src')

print("Executing Unified Frozen Evaluation across all 3,150 test queries...")
subprocess.run(eval_cmd, check=True, env=env)

with open('/kaggle/working/production_evaluation_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

print("\\n" + "="*70)
print("       REPAROS BASE V3 PRODUCTION EVALUATION RESULTS       ")
print("="*70)
for k, v in report['base_v3_production'].items():
    print(f"  {k:30s}: {v:6.2f}%")
print("="*70)
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated clean production Kaggle notebook at {notebook_path}")


if __name__ == "__main__":
    main()
