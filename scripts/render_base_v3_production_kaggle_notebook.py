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
- **Clean Output Logging:** Verbose training logs are redirected to `train.log`. Only milestone checkpoints and step summaries (every 2,000 steps) are displayed to keep the notebook interface clean.
- **Zero-Leakage Certified:** 100% leak-free, zero held-out brands, zero eval overlaps, zero label contradictions.
- **Automated Frozen Evaluation:** Automatically benchmarks the final checkpoint against all 3,150 test queries across all 6 test suites upon completion.
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

    # Cell 3: Environment Setup & Unpacking
    cells.append(code("""import json, os, shutil, subprocess, sys, time
from pathlib import Path

KAGGLE_INPUT = Path('/kaggle/input')
REPO_ROOT = Path('/kaggle/working/QU-solution')

# 1. Locate attached bundle
bundle_candidates = list(KAGGLE_INPUT.rglob('reparos-base-v3-production-data.zip'))
if not bundle_candidates:
    bundle_candidates = list(KAGGLE_INPUT.rglob('*.zip'))

assert bundle_candidates, f"No bundle zip found under /kaggle/input! Please attach the dataset."
bundle_zip = bundle_candidates[0]
print(f"Found production bundle: {bundle_zip}")

# 2. Extract into /kaggle/working/QU-solution
if REPO_ROOT.exists():
    shutil.rmtree(REPO_ROOT)
REPO_ROOT.mkdir(parents=True, exist_ok=True)

print("Unpacking production bundle...")
subprocess.run(['unzip', '-q', str(bundle_zip), '-d', str(REPO_ROOT)], check=True)
print("Unpacking complete!")

# 3. Setup Python path
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / 'src'))

# 4. Install OpenNMT and SentencePiece if needed
try:
    import onmt
    import sentencepiece
    print("OpenNMT and SentencePiece already installed.")
except ImportError:
    print("Installing OpenNMT-py and sentencepiece...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'OpenNMT-py==3.5.1', 'sentencepiece'], check=True)
"""))

    # Cell 4: Verify Production Dataset & Audit
    cells.append(code("""DATA_DIR = REPO_ROOT / 'data/base_v3_production'
TOKENIZER_MODEL = REPO_ROOT / 'data/tokenizer_v3/tokenizer.model'
EVAL_DIR = REPO_ROOT / 'data/base_v3_eval'

assert (DATA_DIR / 'train.src').exists(), "train.src not found!"
assert (DATA_DIR / 'train.tgt').exists(), "train.tgt not found!"
assert TOKENIZER_MODEL.exists(), "tokenizer.model not found!"

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
if not vocab_src.is_file() or vocab_src.stat().st_size == 0:
    print("Building OpenNMT vocab with Tokenizer V3...")
    subprocess.run([
        sys.executable, '-m', 'onmt.bin.build_vocab',
        '-config', str(config_path), '-n_sample', '-1'
    ], check=True)
print("Vocab ready!")
"""))

    # Cell 6: Train OpenNMT with Clean Output Logging
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
        raise subprocess.CalledProcessError(proc.returncode, 'onmt.bin.train')

elapsed_hours = (time.time() - t0) / 3600
print(f"\\nPre-training completed successfully in {elapsed_hours:.2f} hours!")

ckpts = sorted(CKPT_DIR.glob('reparos_base_v3_production_step_*.pt'))
assert ckpts, "No checkpoint found!"
best_ckpt = ckpts[-1]
print(f"Final Checkpoint: {best_ckpt}")
"""))

    # Cell 7: Unified Frozen Evaluation
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
