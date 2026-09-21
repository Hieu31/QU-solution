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
    notebook_path = Path("notebook/train_reparos_base_v3_ablation_kaggle_offline.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    cells = []

    # Cell 1: Intro Markdown
    cells.append(md("""# ReparoS Base V3 — 3-Way Ablation Study (Kaggle Offline 100%)

This notebook runs the official **ReparoS Base V3 3-Way Empirical Ablation Study** completely **offline** using the attached Kaggle Dataset bundle (`kaggle-bundle`).

### 3 Controlled Ablation Runs (20,000 steps each):
- **Run A (No Lane 4):** 45% Lane 1 + 30% Lane 2 + 25% Lane 3 + 0% Lane 4
- **Run B (Lane 4 DAE-Only):** 40% Lane 1 + 25% Lane 2 + 25% Lane 3 + 10% Lane 4 DAE
- **Run C (Lane 4 DAE + Identity):** 40% Lane 1 + 25% Lane 2 + 25% Lane 3 + 7% DAE + 3% Identity

No internet access, external downloads, or virtualenv creation required.
"""))

    # Cell 2: Hyperparameters
    cells.append(code("""TRAIN_STEPS = 20_000
VALID_STEPS = 1_000
SAVE_CHECKPOINT_STEPS = 2_000
KEEP_CHECKPOINTS = 3
BATCH_SIZE_TOKENS = 32_768  # 32,768 tokens/batch (FP16 on T4)
BUCKET_SIZE = 65_536
NUM_WORKERS = 4
MODEL_DTYPE = "fp16"
SEED = 2026

RUNS = ["dataset_run_a", "dataset_run_b", "dataset_run_c"]
"""))

    # Cell 3: Offline Runtime Setup & Wheels Installation
    cells.append(code("""import json, os, shutil, subprocess, sys
from pathlib import Path

KAGGLE_INPUT = Path('/kaggle/input')
REPO_ROOT = Path('/kaggle/working/QU-solution')

# Locate kaggle-bundle attached to this notebook
bundle_candidates = []
for p in KAGGLE_INPUT.rglob('wheels'):
    root = p.parent
    if (root / 'wheels').is_dir() and (root / 'data').is_dir():
        bundle_candidates.append(root)
bundle_candidates = sorted(set(bundle_candidates))
assert len(bundle_candidates) >= 1, f"Expected at least one kaggle-bundle under /kaggle/input, found: {bundle_candidates}"
BUNDLE_ROOT = bundle_candidates[0]
print("Found BUNDLE_ROOT:", BUNDLE_ROOT)

# Copy source code out of read-only /kaggle/input
if REPO_ROOT.exists():
    shutil.rmtree(REPO_ROOT)
if (BUNDLE_ROOT / 'project').is_dir():
    shutil.copytree(BUNDLE_ROOT / 'project', REPO_ROOT)
else:
    REPO_ROOT.mkdir(parents=True, exist_ok=True)

# Install bundled Linux wheels into /kaggle/working/site-packages
SITE_PACKAGES = Path('/kaggle/working/site-packages')
if SITE_PACKAGES.exists():
    shutil.rmtree(SITE_PACKAGES)
SITE_PACKAGES.mkdir(parents=True)

wheels = sorted((BUNDLE_ROOT / 'wheels').glob('*.whl'))
assert wheels, f"No wheels found in {BUNDLE_ROOT / 'wheels'}"
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    '--no-index', '--no-deps', '--target', str(SITE_PACKAGES),
    *map(str, wheels),
], check=True)

os.chdir(REPO_ROOT)
sys.path.insert(0, str(SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / 'src'))

import torch, onmt, sentencepiece, ctranslate2
assert torch.cuda.is_available(), "GPU accelerator is required!"
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("GPU Device:", torch.cuda.get_device_name(0))
print("OpenNMT-py:", onmt.__version__, "| CTranslate2:", ctranslate2.__version__)
"""))

    # Cell 4: Verify Tokenizer V3 & Data Paths
    cells.append(code("""TOKENIZER_MODEL = BUNDLE_ROOT / 'data/tokenizer_v3/tokenizer.model'
EVAL_DIR = BUNDLE_ROOT / 'data/base_v3_eval'
ABLATION_DIR = BUNDLE_ROOT / 'data/base_v3_ablation'

assert TOKENIZER_MODEL.is_file(), f"Missing {TOKENIZER_MODEL}"
assert EVAL_DIR.is_dir(), f"Missing {EVAL_DIR}"
assert ABLATION_DIR.is_dir(), f"Missing {ABLATION_DIR}"

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_MODEL))
print(f"Tokenizer V3 loaded successfully: {sp.vocab_size():,} vocab.")

# Smoke test special brands
test_queries = ["Pizza 4P's Hai Bà Trưng", "Biti's Hunter", "J&T Express", "McDonald's", "Co.opmart"]
for q in test_queries:
    ids = sp.encode_as_ids(q.lower())
    assert sp.unk_id() not in ids, f"UNK detected on {q}!"
print("Zero-UNK verification: 100% PASS on special-character brands!")
"""))

    # Cell 5: Train All 3 Ablation Runs (20,000 steps each)
    cells.append(code("""import time

# Copy ablation data to working directory so OpenNMT can build vocab and checkpoints locally
LOCAL_ABLATION_DIR = Path('/kaggle/working/data/base_v3_ablation')
if LOCAL_ABLATION_DIR.exists():
    shutil.rmtree(LOCAL_ABLATION_DIR)
shutil.copytree(ABLATION_DIR, LOCAL_ABLATION_DIR)

checkpoints = {}

for run_name in RUNS:
    print(f"\\n{'='*70}\\n>>> PRE-TRAINING {run_name.upper()} ({TRAIN_STEPS:,} STEPS)\\n{'='*70}")
    run_dir = LOCAL_ABLATION_DIR / run_name
    config_file = run_dir / 'opennmt_config.json'
    
    # Update config paths to local working directories
    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    cfg['src_subword_model'] = str(TOKENIZER_MODEL)
    cfg['tgt_subword_model'] = str(TOKENIZER_MODEL)
    cfg['data']['corpus_1']['path_src'] = str(run_dir / 'train.src')
    cfg['data']['corpus_1']['path_tgt'] = str(run_dir / 'train.tgt')
    cfg['data']['valid']['path_src'] = str(run_dir / 'valid.src')
    cfg['data']['valid']['path_tgt'] = str(run_dir / 'valid.tgt')
    cfg['src_vocab'] = str(run_dir / 'vocab.src')
    cfg['tgt_vocab'] = str(run_dir / 'vocab.tgt')
    cfg['save_data'] = str(run_dir / 'vocab')
    
    ckpt_dir = Path(f'/kaggle/working/checkpoints/base_v3_ablation_{run_name}')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg['save_model'] = str(ckpt_dir / f'reparos_base_v3_{run_name}')
    cfg['train_steps'] = TRAIN_STEPS
    cfg['valid_steps'] = VALID_STEPS
    cfg['save_checkpoint_steps'] = SAVE_CHECKPOINT_STEPS
    cfg['batch_size'] = BATCH_SIZE_TOKENS
    cfg['bucket_size'] = BUCKET_SIZE
    cfg['num_workers'] = NUM_WORKERS
    cfg['model_dtype'] = MODEL_DTYPE
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    # 1. Build Vocab
    vocab_src = run_dir / 'vocab.src'
    if not vocab_src.is_file() or vocab_src.stat().st_size == 0:
        print(f"Building OpenNMT vocab for {run_name}...")
        subprocess.run([
            sys.executable, '-m', 'onmt.bin.build_vocab',
            '-config', str(config_file), '-n_sample', '-1'
        ], check=True)

    # 2. Train OpenNMT
    t0 = time.time()
    subprocess.run([
        sys.executable, '-m', 'onmt.bin.train',
        '-config', str(config_file)
    ], check=True)
    elapsed = (time.time() - t0) / 60
    print(f"Run {run_name} completed in {elapsed:.1f} minutes.")

    ckpts = sorted(ckpt_dir.glob(f'reparos_base_v3_{run_name}_step_*.pt'))
    assert ckpts, f"No checkpoint produced for {run_name}!"
    checkpoints[run_name] = str(ckpts[-1])
    print(f"Final checkpoint: {checkpoints[run_name]}")
"""))

    # Cell 6: Run Unified Evaluation on Frozen Evaluation Suite
    cells.append(code("""eval_script = REPO_ROOT / 'scripts/evaluate_base_v3_ablation.py'

eval_cmd = [
    sys.executable, str(eval_script),
    '--eval-dir', str(EVAL_DIR),
    '--tokenizer', str(TOKENIZER_MODEL),
    '--device', 'cuda',
    '--checkpoints', *list(checkpoints.values()),
    '--names', *list(checkpoints.keys()),
    '--output-json', '/kaggle/working/ablation_evaluation_report.json'
]

print("Executing Unified Frozen Evaluation across all 3 runs...")
subprocess.run(eval_cmd, check=True)
"""))

    # Cell 7: Display Comparative Matrix
    cells.append(code("""with open('/kaggle/working/ablation_evaluation_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

print("\\n" + "="*85)
print("                   FINAL REPAROS BASE V3 DECISION MATRIX                     ")
print("="*85)
header = f"{'Run Name':<18} | {'Plasticity':<12} | {'Retention':<12} | {'Seen Brands':<12} | {'Heldout Brand':<14} | {'User':<10}"
print(header)
print("-" * len(header))

for name, res in report.items():
    p = res['metrics']['plasticity']['exact_match']
    r = res['metrics']['retention']['exact_match']
    ps = res['metrics']['protection_seen']['exact_match']
    ph = res['metrics']['protection_heldout']['exact_match']
    br = res['metrics']['protection_heldout'].get('brand_retention', 100.0)
    uc = res['metrics']['user_centric']['exact_match']
    print(f"{name:<18} | {p:>10.2f}% | {r:>10.2f}% | {ps:>10.2f}% | {ph:>6.2f}% ({br:.0f}%) | {uc:>8.2f}%")
print("="*85)
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": False,
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
    print(f"Generated offline Kaggle notebook at {notebook_path}")


if __name__ == "__main__":
    main()
