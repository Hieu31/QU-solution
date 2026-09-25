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
    cells.append(md("""# ReparoS Base V3 (2E / 2D Arm E) — Production Pre-Training (4,000,000 Pairs)
### Hardware Profile: NVIDIA RTX 6000 PRO (96GB VRAM) / High-Throughput Workstation
### Winning Recipe: Run C (40% L1 + 25% L2 + 25% L3 + 7% DAE + 3% Identity)

This notebook executes official pre-training from scratch for **ReparoS Base V3** using the production dataset of **4,000,000 unique pairs** with the winning **2E / 2D Transformer architecture (Arm E Sweet Spot, 7.1M Params)**.

### Key Features:
- **RTX 6000 PRO High-Throughput:** Configured with `65,536 tokens/batch`, `131,072 bucket size`, and 8 workers to maximize throughput on 96GB VRAM.
- **Arm E Architecture:** 2 Encoder / 2 Decoder layers, hidden size 128, FFN 2048, 8 attention heads (solves word omission and repetition in long queries).
- **Clean Milestone Logging:** Verbose OpenNMT step logs are safely redirected to `train.log`. Milestone progress is displayed every 2,000 steps.
- **Auto-Crash Diagnostics:** If an error occurs, the notebook automatically surfaces the last 25 lines of `train.log` right inside the cell output.
- **Zero-Leakage Certified:** 100% leak-free, zero held-out brands, zero eval overlaps, zero label contradictions.
- **Automated Frozen Evaluation:** Benchmarks the final checkpoint against all 3,100 test queries across 5 frozen test suites upon completion.
"""))

    # Cell 2: Hyperparameters
    cells.append(code("""# Model Architecture (2E / 2D Arm E Sweet Spot)
ENC_LAYERS = 2
DEC_LAYERS = 2

# Training Schedule
TRAIN_STEPS = 50_000
VALID_STEPS = 2_000
SAVE_CHECKPOINT_STEPS = 5_000
KEEP_CHECKPOINTS = 5
SEED = 2026
REPORT_EVERY = 2_000
MODEL_DTYPE = "fp16"

# Hardware Profile: NVIDIA RTX 6000 PRO (96GB VRAM) Maximum Throughput
BATCH_SIZE_TOKENS = 65_536   # 65,536 tokens/batch (Saturates 96GB VRAM for fastest convergence)
BUCKET_SIZE = 131_072        # 131,072 tokens bucket size for minimal padding overhead
NUM_WORKERS = 8              # 8 CPU workers for multi-threaded dataloader

# Adaptive Safety Check for GPU Memory
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Hardware Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 24:
            print(f"Notice: Running on smaller GPU ({vram_gb:.1f}GB < 24GB). Auto-scaling batch size to 32,768.")
            BATCH_SIZE_TOKENS = 32_768
            BUCKET_SIZE = 65_536
            NUM_WORKERS = 4
        else:
            print(f"RTX 6000 PRO / High-VRAM GPU verified! Active profile: batch={BATCH_SIZE_TOKENS:,} tokens, workers={NUM_WORKERS}.")
except Exception as e:
    print(f"Defaulting to RTX 6000 PRO configuration: {e}")
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

# 3. Offline Wheels & Dependency Management (100% Offline Compatible)
SITE_PACKAGES = Path('/kaggle/working/site-packages')
if not SITE_PACKAGES.exists():
    SITE_PACKAGES.mkdir(parents=True)
sys.path.insert(0, str(SITE_PACKAGES))

# Check if dependencies are already installed
need_install = False
try:
    import onmt, sentencepiece, ctranslate2, torch
    print("Dependencies (OpenNMT-py, sentencepiece, ctranslate2, torch) already loaded successfully.")
except ImportError:
    need_install = True

if need_install:
    print("Dependencies not preloaded. Scanning ALL datasets under /kaggle/input for offline wheels...")
    all_wheels = sorted(list(KAGGLE_INPUT.rglob('*.whl')))
    
    # Filter out torch/nvidia wheels to preserve Kaggle's native GPU PyTorch
    target_wheels = [w for w in all_wheels if not w.name.lower().startswith(('torch-', 'torch==', 'nvidia_'))]
    
    has_onmt = any('opennmt' in w.name.lower() or 'onmt' in w.name.lower() for w in target_wheels)
    
    if target_wheels:
        print(f"Found {len(target_wheels)} offline wheels across attached datasets.")
        # Install directly into Python environment and also into SITE_PACKAGES for maximum compatibility
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--quiet',
            '--no-index', '--no-deps',
            *[str(w) for w in target_wheels]
        ], check=False)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--quiet',
            '--no-index', '--no-deps', '--target', str(SITE_PACKAGES),
            *[str(w) for w in target_wheels]
        ], check=False)
        
    if not has_onmt:
        print("\n" + "!" * 80)
        print("CẢNH BÁO THIẾU BÁNH XE (WHEEL) OpenNMT-py:")
        print("Dataset 'buildwheel1' bạn đang attach chỉ chứa các thư viện của HuggingFace (transformers, v.v.).")
        print("Để huấn luyện Base V3 (OpenNMT), bạn cần ATTACH THÊM dataset chứa OpenNMT-py:")
        print("  -> Hãy bấm 'Add Input' trên Kaggle và thêm dataset: 'vanhieu1125/reparos-haha'")
        print("     (Dataset này có sẵn wheels/OpenNMT_py-3.5.1-py3-none-any.whl)")
        print("!" * 80 + "\n")
        # If internet happens to be ON, attempt automatic online install as safety fallback
        try:
            print("Đang thử cài đặt OpenNMT-py online (nếu Notebook có bật Internet)...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--quiet',
                'numpy<2', 'OpenNMT-py==3.5.1', 'sentencepiece', 'ctranslate2'
            ], check=True)
            print("Cài đặt OpenNMT-py thành công qua internet fallback!")
        except Exception:
            pass

# Set global PYTHONPATH for all child processes and subprocesses
python_path = os.pathsep.join([str(SITE_PACKAGES), str(REPO_ROOT / 'src')])
os.environ['PYTHONPATH'] = python_path + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, str(SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / 'src'))

# Fix PyTorch 2.6+ safe globals unpickling for OpenNMT checkpoints
try:
    import argparse, torch
    torch.serialization.add_safe_globals([argparse.Namespace])
except Exception:
    pass

import onmt, sentencepiece, ctranslate2, torch
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("GPU Accelerator:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("OpenNMT-py:", onmt.__version__, "| CTranslate2:", ctranslate2.__version__)
print("Offline environment verified successfully!")
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
    "enc_layers": 2,
    "dec_layers": 2,
    "heads": 8,
    "hidden_size": 128,
    "word_vec_size": 128,
    "transformer_ff": 2048,
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
    ], check=True, env=os.environ.copy())
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
        bufsize=1,
        env=os.environ.copy()
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

    # Cell 7: Unified Frozen Evaluation Across 3,100 Test Queries (In-Process)
    cells.append(code("""import argparse, json, os, shutil, sys, time
from pathlib import Path
import torch

# 0. Clean evaluate_base_v3_ablation.py on disk (fix Jupyter OutStream & syntax)
eval_file = REPO_ROOT / 'scripts/evaluate_base_v3_ablation.py'
if eval_file.exists():
    txt = eval_file.read_text(encoding='utf-8')
    txt = txt.replace('sys.stdout.reconfigure', '# sys.stdout.reconfigure')
    if 'from __future__ import annotations' in txt:
        idx = txt.find('from __future__ import annotations')
        txt = '#!/usr/bin/env python\n' + txt[idx:]
    eval_file.write_text(txt, encoding='utf-8')

# 1. PyTorch 2.6 safe unpickling allowlist
try:
    torch.serialization.add_safe_globals([argparse.Namespace])
except Exception:
    pass
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# 2. Reset ctranslate2 OpenNMTPyConverter cleanly if previously wrapped in kernel
import ctranslate2
import importlib
try:
    import ctranslate2.converters.opennmt_py
    importlib.reload(ctranslate2.converters.opennmt_py)
    ctranslate2.converters.OpenNMTPyConverter = ctranslate2.converters.opennmt_py.OpenNMTPyConverter
except Exception:
    pass

# 3. Import evaluation logic directly in Python kernel
if str(REPO_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))

from evaluate_base_v3_ablation import evaluate_run, print_comparison_report

# 4. Locate best checkpoint safely
if 'best_ckpt' not in globals() or not Path(str(best_ckpt)).exists():
    ckpts = sorted(Path('/kaggle/working/checkpoints/base_v3_production').glob('reparos_base_v3_production_step_*.pt'))
    assert ckpts, "No checkpoint found in /kaggle/working/checkpoints/base_v3_production!"
    best_ckpt = ckpts[-1]

print(f"Executing Unified Frozen Evaluation on {best_ckpt.name} across all 3,100 test queries...")
eval_res = evaluate_run(
    checkpoint_or_ct2=best_ckpt,
    tokenizer_path=TOKENIZER_MODEL,
    eval_dir=EVAL_DIR,
    device="cuda" if torch.cuda.is_available() else "cpu",
    trust_checkpoint=True,
)

all_results = {"base_v3_production": eval_res}
print_comparison_report(all_results)

out_report_path = Path('/kaggle/working/production_evaluation_report.json')
with open(out_report_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\\nReport successfully saved to {out_report_path}!")
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
