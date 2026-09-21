from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md("""# ReparoS Base V2-32 — train and evaluate on Kaggle

Train a fresh Base model on the 7.06M-pair Base V2-32 dataset. The notebook then evaluates both the original Base and Base V2 on exactly the same User-centric 88, Composition 4K and Diagnostic 10K benchmarks. No curriculum checkpoint is used for training.
"""),
    md("""## 1. Configuration

Attach three Kaggle datasets before running: Base V2 data, the original Base artifact, and the curriculum/evaluation dataset. Paths are discovered from their required files, so the Kaggle dataset slug may change.
"""),
    code("""from pathlib import Path

REPO_URL = 'https://github.com/Hieu31/QU-solution.git'
REPO_ROOT = Path('/kaggle/working/QU-solution')
RUN_ROOT = Path('/kaggle/working/reparos/base-v2-32-run')
KAGGLE_INPUT = Path('/kaggle/input')

TRAIN_STEPS = 75_000
VALID_STEPS = 2_500
CHECKPOINT_STEPS = 5_000
BATCH_SIZE_TOKENS = 16_384
BUCKET_SIZE = 32_768
NUM_WORKERS = 2
RESUME_IF_AVAILABLE = True
SEED = 2026
"""),
    md("## 2. Runtime setup"),
    code("""import json, os, shutil, subprocess, sys

if not REPO_ROOT.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_ROOT)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO_ROOT), 'pull', '--ff-only'], check=True)
os.chdir(REPO_ROOT)
SRC_ROOT = str(REPO_ROOT / 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run(['uv', 'sync', '--python', '3.11', '--extra', 'reparos-opennmt'], check=True)

VENV_PY = REPO_ROOT / '.venv/bin/python'
REPAROS = REPO_ROOT / '.venv/bin/reparos'
BENCHMARK = REPO_ROOT / '.venv/bin/benchmark-three'
assert all(path.is_file() for path in (VENV_PY, REPAROS, BENCHMARK))
subprocess.run(['nvidia-smi'], check=True)
subprocess.run(['git', 'rev-parse', 'HEAD'], check=True)
"""),
    md("## 3. Locate and validate attached artifacts"),
    code("""def unique_match(pattern, label, predicate=lambda path: True):
    matches = [path for path in KAGGLE_INPUT.rglob(pattern) if predicate(path)]
    assert len(matches) == 1, f'{label}: expected exactly one match, found {matches}'
    return matches[0]

# Base V2 root is the directory containing train.noisy.src.
BASE_V2_ROOT = unique_match(
    'train.noisy.src', 'Base V2 train data',
    lambda p: (p.parent / 'train.clean.src').is_file() and (p.parent / 'validation.noisy.src').is_file(),
).parent

# Original Base artifact supplies the frozen tokenizer/vocabulary, decoding config,
# and the Base checkpoint used only as an evaluation baseline.
ORIGINAL_CONFIG = unique_match(
    'opennmt-base.json', 'original Base config',
    lambda p: list(p.parent.glob('reparos_base_step_*.pt')) and (p.parent / 'vocab.src').is_file(),
)
BASE_ARTIFACT_DIR = ORIGINAL_CONFIG.parent
TOKENIZER_MODEL = unique_match('tokenizer.model', 'tokenizer')
DECODING_CONFIG = BASE_ARTIFACT_DIR / 'decoding-config.json'

USER_GOLD = unique_match('user-centric-v2.jsonl', 'User-centric 88')
EVALUATION_ROOT = USER_GOLD.parent
for name in ('composition-4k.jsonl', 'diagnostic-10k.jsonl'):
    assert (EVALUATION_ROOT / name).is_file(), EVALUATION_ROOT / name

MANIFEST = BASE_V2_ROOT.parents[1] / 'base-v2-manifest.json'
assert MANIFEST.is_file(), MANIFEST
manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
assert manifest['leakage'] == {'train_validation': 0, 'train_test': 0, 'validation_test': 0}

required = []
for split in ('train', 'validation', 'test'):
    for lane in ('noisy', 'clean'):
        for suffix in ('src', 'tgt', 'meta.jsonl'):
            required.append(BASE_V2_ROOT / f'{split}.{lane}.{suffix}')
assert all(path.is_file() for path in required), [str(p) for p in required if not p.is_file()]

print('Base V2:', BASE_V2_ROOT)
print('Manifest:', MANIFEST)
print('Original Base:', BASE_ARTIFACT_DIR)
print('Tokenizer:', TOKENIZER_MODEL)
print('Evaluation:', EVALUATION_ROOT)
"""),
    md("""## 4. Build a fresh Base V2 OpenNMT configuration

The architecture, tokenizer and vocabulary are copied from Base. Model weights are **not** loaded. OpenNMT samples noisy and clean as separate corpora with weights 75/25. Validation contains both lanes.
"""),
    code("""RUN_ROOT.mkdir(parents=True, exist_ok=True)

def concatenate(destination, sources):
    with destination.open('wb') as target:
        for source in sources:
            with source.open('rb') as stream:
                shutil.copyfileobj(stream, target, length=16 * 1024 * 1024)

VALID_SRC = RUN_ROOT / 'validation.src'
VALID_TGT = RUN_ROOT / 'validation.tgt'
concatenate(VALID_SRC, [BASE_V2_ROOT / 'validation.noisy.src', BASE_V2_ROOT / 'validation.clean.src'])
concatenate(VALID_TGT, [BASE_V2_ROOT / 'validation.noisy.tgt', BASE_V2_ROOT / 'validation.clean.tgt'])

for name in ('vocab.src', 'vocab.tgt'):
    shutil.copy2(BASE_ARTIFACT_DIR / name, RUN_ROOT / name)

config = json.loads(ORIGINAL_CONFIG.read_text(encoding='utf-8'))
config.update({
    'save_data': str(RUN_ROOT / 'vocab'),
    'src_vocab': str(RUN_ROOT / 'vocab.src'),
    'tgt_vocab': str(RUN_ROOT / 'vocab.tgt'),
    'src_subword_model': str(TOKENIZER_MODEL),
    'tgt_subword_model': str(TOKENIZER_MODEL),
    'save_model': str(RUN_ROOT / 'reparos_base_v2'),
    'train_steps': TRAIN_STEPS,
    'valid_steps': VALID_STEPS,
    'save_checkpoint_steps': CHECKPOINT_STEPS,
    'batch_size': BATCH_SIZE_TOKENS,
    'bucket_size': BUCKET_SIZE,
    'num_workers': NUM_WORKERS,
    'keep_checkpoint': 5,
    'seed': SEED,
    'transformer_ff': 2048,
    'enc_layers': 2,
    'dec_layers': 1,
    'data': {
        'noisy': {
            'path_src': str(BASE_V2_ROOT / 'train.noisy.src'),
            'path_tgt': str(BASE_V2_ROOT / 'train.noisy.tgt'),
            'transforms': ['sentencepiece'],
            'weight': 75,
        },
        'clean': {
            'path_src': str(BASE_V2_ROOT / 'train.clean.src'),
            'path_tgt': str(BASE_V2_ROOT / 'train.clean.tgt'),
            'transforms': ['sentencepiece'],
            'weight': 25,
        },
        'valid': {
            'path_src': str(VALID_SRC),
            'path_tgt': str(VALID_TGT),
            'transforms': ['sentencepiece'],
        },
    },
})
config.pop('train_from', None)

def checkpoint_step(path):
    return int(path.stem.rsplit('_step_', 1)[1])

existing = sorted(RUN_ROOT.glob('reparos_base_v2_step_*.pt'), key=checkpoint_step)
if existing and RESUME_IF_AVAILABLE:
    config['train_from'] = str(existing[-1])
    assert checkpoint_step(existing[-1]) < TRAIN_STEPS, (existing[-1], TRAIN_STEPS)
    print('Resuming from:', existing[-1])
elif existing:
    raise FileExistsError(f'Existing checkpoints found; clear {RUN_ROOT} or enable resume: {existing}')
else:
    print('Fresh training: no train_from checkpoint')

CONFIG_PATH = RUN_ROOT / 'opennmt-base-v2.json'
CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')
print(CONFIG_PATH.read_text(encoding='utf-8'))
"""),
    md("## 5. Train Base V2-32"),
    code("""subprocess.run([
    str(VENV_PY), '-m', 'onmt.bin.train', '-config', str(CONFIG_PATH)
], check=True)

checkpoints = sorted(RUN_ROOT.glob('reparos_base_v2_step_*.pt'), key=checkpoint_step)
assert checkpoints, RUN_ROOT
BASE_V2_CHECKPOINT = checkpoints[-1]
assert checkpoint_step(BASE_V2_CHECKPOINT) == TRAIN_STEPS, BASE_V2_CHECKPOINT
print('Base V2 checkpoint:', BASE_V2_CHECKPOINT)
"""),
    md("""## 6. Evaluate original Base and Base V2 in the same session

Both checkpoints use the same tokenizer, decoding configuration, CTranslate2 conversion and three frozen benchmarks. This cell may be rerun without retraining.
"""),
    code("""from reparos.curriculum_notebook import evaluate_checkpoint
from reparos.curriculum_training import latest_checkpoint

ORIGINAL_BASE_CHECKPOINT = latest_checkpoint(BASE_ARTIFACT_DIR, 'reparos_base')

reports = {}
for system, checkpoint in (
    ('base', ORIGINAL_BASE_CHECKPOINT),
    ('base_v2_32', BASE_V2_CHECKPOINT),
):
    reports[system] = evaluate_checkpoint(
        checkpoint=checkpoint,
        tokenizer=TOKENIZER_MODEL,
        decoding_config=DECODING_CONFIG,
        evaluation_root=EVALUATION_ROOT,
        output_root=RUN_ROOT / 'evaluation' / system,
        reparos_cli=REPAROS,
        benchmark_cli=BENCHMARK,
        device='cuda',
        compute_type='float16',
    )

REPORT_PATH = RUN_ROOT / 'base-v2-comparison.json'
REPORT_PATH.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')
print(json.dumps(reports, ensure_ascii=False, indent=2))
"""),
    md("## 7. Show metric deltas"),
    code("""def flatten_numbers(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f'{prefix}.{key}' if prefix else key
            result.update(flatten_numbers(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result

rows = []
for benchmark in ('user', 'composition', 'diagnostic'):
    old = flatten_numbers(reports['base'][benchmark])
    new = flatten_numbers(reports['base_v2_32'][benchmark])
    for metric in sorted(old.keys() & new.keys()):
        rows.append({
            'benchmark': benchmark,
            'metric': metric,
            'base': old[metric],
            'base_v2_32': new[metric],
            'delta': new[metric] - old[metric],
        })

try:
    import pandas as pd
    display(pd.DataFrame(rows))
except ImportError:
    print(json.dumps(rows, ensure_ascii=False, indent=2))
"""),
    md("## 8. Package checkpoint, CT2 model, predictions and reports"),
    code("""EXPORT_ROOT = RUN_ROOT / 'final-artifact'
if EXPORT_ROOT.exists():
    shutil.rmtree(EXPORT_ROOT)
EXPORT_ROOT.mkdir(parents=True)

shutil.copy2(BASE_V2_CHECKPOINT, EXPORT_ROOT / BASE_V2_CHECKPOINT.name)
shutil.copy2(TOKENIZER_MODEL, EXPORT_ROOT / 'tokenizer.model')
shutil.copy2(DECODING_CONFIG, EXPORT_ROOT / 'decoding-config.json')
shutil.copy2(CONFIG_PATH, EXPORT_ROOT / 'opennmt-base-v2.json')
shutil.copy2(MANIFEST, EXPORT_ROOT / 'base-v2-manifest.json')
shutil.copy2(REPORT_PATH, EXPORT_ROOT / REPORT_PATH.name)
shutil.copytree(RUN_ROOT / 'evaluation/base_v2_32/ctranslate2', EXPORT_ROOT / 'ctranslate2')
shutil.copytree(RUN_ROOT / 'evaluation', EXPORT_ROOT / 'evaluation')

zip_path = Path(shutil.make_archive('/kaggle/working/reparos-base-v2-32-final', 'zip', EXPORT_ROOT))
print('DOWNLOAD:', zip_path, f'({zip_path.stat().st_size / 1024**2:.1f} MiB)')
print('Comparison report:', REPORT_PATH)
"""),
]


def main() -> None:
    payload = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = Path("notebook/train_reparos_base_v2_kaggle.ipynb")
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(destination, len(CELLS), "cells")


if __name__ == "__main__":
    main()
