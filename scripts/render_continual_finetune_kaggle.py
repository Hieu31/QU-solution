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
    md("""# ReparoS Interleaved Continual Fine-Tuning — Kaggle Pipeline

This notebook executes **Interleaved Continual Fine-Tuning** on **ReparoS Base V2-32** (step 10,000 checkpoint).

### Core Objectives & Gating
1. **Plasticity (Group A: 32% weight)**: Expand vocabulary and pattern coverage on **Address Abbreviations** (`d.` $\\rightarrow$ `đường`, `q.` $\\rightarrow$ `quận`, `p.` $\\rightarrow$ `phường`), **Contextual Acronyms** (`tp hcm`, `hn`, `tt`), and **Compositions** without changing base architecture.
2. **Capability Retention (Group B: 46% weight)**: Preserve Base V2's strong typing noise recovery (VNI, Telex, diacritics, keyboard typos). Gate 1 allows $\\le 1.5\\%$ degradation.
3. **Hard No-Change / Protection (Group C: 22% weight)**: Strictly preserve brand entities and clean queries on both seen and **held-out** (strictly disjoint) distributions. Gate 0 limits $FCR_{\\text{heldout}} \\le 2.0\\%$ and $FCR_{\\text{seen}} \\le 1.0\\%$.
4. **Hierarchical 3-Tier Gate & Pareto Selection**: Evaluates checkpoints every 250 steps to select the optimal model.
"""),
    md("""## 1. Hyperparameter & Experiment Configuration

Set `EXPERIMENT_ARM = 'B'` for the recommended Balanced Interleaved training, or `'A'` for Plasticity-only ablation.
"""),
    code("""from pathlib import Path

REPO_URL = 'https://github.com/Hieu31/QU-solution.git'
REPO_ROOT = Path('/kaggle/working/QU-solution')
RUN_ROOT = Path('/kaggle/working/reparos/continual-finetune-run')
KAGGLE_INPUT = Path('/kaggle/input')

# Experiment Arm: 'B' = Balanced Interleaved (Recommended), 'A' = Plasticity Only (Ablation)
EXPERIMENT_ARM = 'B'

# Continual Fine-Tuning Schedule (from step 10,000 to 13,000)
ADDITIONAL_STEPS = 3000
SAVE_STEPS = 250
VALID_STEPS = 250
KEEP_CHECKPOINTS = 15
BATCH_SIZE_TOKENS = 32768
NUM_WORKERS = 4
SEED = 2026
"""),
    md("""## 2. Environment Setup

Clones/updates the repository, installs `uv`, and synchronizes Python 3.11 with OpenNMT and CTranslate2 dependencies.
"""),
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
assert all(path.is_file() for path in (VENV_PY, REPAROS, BENCHMARK)), 'Virtual environment binaries missing'

subprocess.run(['nvidia-smi'], check=True)
subprocess.run(['git', 'rev-parse', 'HEAD'], check=True)
"""),
    md("""## 3. Locate and Validate Attached Artifacts

Locates the Base V2 checkpoint artifacts and the new `interleaved-continual-v1` dataset from `/kaggle/input`.
Supports either:
- A single all-in-one dataset (`reparos-continual-all-in-one`)
- OR separate Base V2 and Continual dataset attachments.
"""),
    code("""def search_file(pattern, label, predicate=lambda p: True):
    matches = [p for p in KAGGLE_INPUT.rglob(pattern) if predicate(p)]
    if not matches:
        raise FileNotFoundError(f'{label}: no matching file found for pattern \"{pattern}\" under {KAGGLE_INPUT}')
    return matches[0]

# 1. Locate Base V2 starting checkpoint (step 10,000)
BASE_CHECKPOINT = search_file('reparos_base_v2_step_10000.pt', 'Base V2 Checkpoint')
BASE_ARTIFACT_DIR = BASE_CHECKPOINT.parent

# 2. Locate Base configuration and tokenizer/vocabularies
BASE_CONFIG = search_file('opennmt-base-v2.json', 'Base V2 Config')
TOKENIZER_MODEL = search_file('tokenizer.model', 'SentencePiece Tokenizer')
VOCAB_SRC = search_file('vocab.src', 'Source Vocabulary')
VOCAB_TGT = search_file('vocab.tgt', 'Target Vocabulary')
DECODING_CONFIG = search_file('decoding-config.json', 'Decoding Config')

# 3. Locate Interleaved Continual Dataset Root (contains train/ and eval/)
CONTINUAL_DATA_ROOT = search_file(
    'address_abbreviation.jsonl',
    'Continual Dataset (plasticity)',
    lambda p: (p.parents[1] / 'eval').is_dir() or (p.parents[2] / 'eval').is_dir()
).parents[1]
if CONTINUAL_DATA_ROOT.name == 'plasticity':
    CONTINUAL_DATA_ROOT = CONTINUAL_DATA_ROOT.parent.parent

# 4. Standard evaluation benchmarks (from git repo or kaggle input)
BENCHMARK_ROOT = REPO_ROOT / 'benchmark'
USER_GOLD = BENCHMARK_ROOT / 'reparos-user-centric-v2/gold.jsonl'
COMPOSITION_GOLD = BENCHMARK_ROOT / 'reparos-compositional-4k/gold.jsonl'
DIAGNOSTIC_GOLD = BENCHMARK_ROOT / 'reparos-diagnostic-10k/gold.jsonl'
for p in (USER_GOLD, COMPOSITION_GOLD, DIAGNOSTIC_GOLD):
    assert p.is_file(), f'Missing benchmark gold file: {p}'

print('--- Resolved Paths ---')
print('Base Checkpoint   :', BASE_CHECKPOINT)
print('Base Config       :', BASE_CONFIG)
print('Tokenizer Model   :', TOKENIZER_MODEL)
print('Continual Dataset :', CONTINUAL_DATA_ROOT)
print('Benchmarks Gold   :', BENCHMARK_ROOT)
"""),
    md("""## 4. Prepare Training Corpora and OpenNMT Configuration

Extracts the `.jsonl` records into line-by-line `.src` and `.tgt` files weighted according to the experiment arm.
"""),
    code("""from reparos.continual_finetune import prepare_finetune_corpora, build_finetune_config

RUN_ROOT.mkdir(parents=True, exist_ok=True)

# 1. Prepare training and validation text files
print(f'Preparing corpora for Arm {EXPERIMENT_ARM}...')
data_config = prepare_finetune_corpora(CONTINUAL_DATA_ROOT, RUN_ROOT, arm=EXPERIMENT_ARM)

# 2. Build OpenNMT configuration
CONFIG_PATH = build_finetune_config(
    base_config_path=BASE_CONFIG,
    base_checkpoint_path=BASE_CHECKPOINT,
    run_root=RUN_ROOT,
    data_config=data_config,
    tokenizer_model=TOKENIZER_MODEL,
    vocab_src=VOCAB_SRC,
    vocab_tgt=VOCAB_TGT,
    additional_steps=ADDITIONAL_STEPS,
    save_checkpoint_steps=SAVE_STEPS,
    valid_steps=VALID_STEPS,
    keep_checkpoint=KEEP_CHECKPOINTS,
)

print('Generated OpenNMT Continual Config at:', CONFIG_PATH)
config_json = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
print(json.dumps({
    'train_from': config_json.get('train_from'),
    'train_steps': config_json.get('train_steps'),
    'valid_steps': config_json.get('valid_steps'),
    'save_checkpoint_steps': config_json.get('save_checkpoint_steps'),
    'data_corpora': list(config_json.get('data', {}).keys()),
}, indent=2))
"""),
    md("""## 5. Execute Continual Fine-Tuning

Trains for 3,000 steps starting from step 10,000 to step 13,000, saving checkpoints every 250 steps.
"""),
    code("""subprocess.run([
    str(VENV_PY), '-m', 'onmt.bin.train', '-config', str(CONFIG_PATH)
], check=True)

from reparos.continual_finetune import checkpoint_step
saved_checkpoints = sorted(RUN_ROOT.glob('reparos_continual_step_*.pt'), key=checkpoint_step)
print(f'Training finished! Generated {len(saved_checkpoints)} checkpoints:')
for ckpt in saved_checkpoints:
    print(' ', ckpt.name)
assert saved_checkpoints, 'No checkpoints found after training'
"""),
    md("""## 6. Baseline & 3-Tier Hierarchical Gate Evaluation

1. Evaluates Base V2 step 10,000 on `interleaved-continual-v1/eval/` to establish the exact baseline metrics.
2. Converts each fine-tuned candidate checkpoint to CTranslate2 float16.
3. Evaluates all candidates across the 4 evaluation splits:
   - **Gate 0 (Protection)**: $FCR_{\\text{heldout}} \\le 2.0\\%$, $FCR_{\\text{seen}} \\le 1.0\\%$.
   - **Gate 1 (Retention)**: Retention drop $\\le 1.5\\%$.
   - **Gate 2 (Plasticity)**: Plasticity gain $\\ge +25.0\\%$.
4. Classifies outcomes (Case A, B, C, D) and selects the **Pareto Optimal Checkpoint**.
"""),
    code("""from reparos.continual_finetune import (
    evaluate_continual_splits,
    evaluate_3tier_gating,
    checkpoint_step,
)
from reparos.serving.ctranslate2 import CTranslate2Predictor

EVAL_DIR = CONTINUAL_DATA_ROOT / 'eval'

# Step A: Evaluate Base V2 baseline
BASE_CT2_DIR = RUN_ROOT / 'ct2_base_v2'
if not BASE_CT2_DIR.exists():
    subprocess.run([
        str(REPAROS), 'export-ctranslate2',
        '--model', str(BASE_CHECKPOINT),
        '--tokenizer', str(TOKENIZER_MODEL),
        '--output', str(BASE_CT2_DIR),
        '--compute-type', 'float16',
        '--trust-checkpoint',
    ], check=True)

print('Evaluating Base V2 baseline...')
base_predictor = CTranslate2Predictor(BASE_CT2_DIR, device='cuda', compute_type='float16')
baseline_metrics = evaluate_continual_splits(base_predictor, EVAL_DIR, batch_size=128)
del base_predictor

print('=== Base V2 Baseline Metrics ===')
print(f'Plasticity Acc : {baseline_metrics[\"plasticity\"][\"_overall\"][\"exact_accuracy\"]*100:.2f}%')
print(f'Retention Acc  : {baseline_metrics[\"retention\"][\"_overall\"][\"exact_accuracy\"]*100:.2f}%')
print(f'FCR Seen       : {baseline_metrics[\"protection_seen\"][\"_overall\"][\"fcr\"]*100:.2f}%')
print(f'FCR Heldout    : {baseline_metrics[\"protection_heldout\"][\"_overall\"][\"fcr\"]*100:.2f}%')

# Step B: Evaluate all candidate checkpoints
candidate_results = []
for ckpt in saved_checkpoints:
    step = checkpoint_step(ckpt)
    ct2_dir = RUN_ROOT / f'ct2_step_{step}'
    if ct2_dir.exists():
        shutil.rmtree(ct2_dir)

    subprocess.run([
        str(REPAROS), 'export-ctranslate2',
        '--model', str(ckpt),
        '--tokenizer', str(TOKENIZER_MODEL),
        '--output', str(ct2_dir),
        '--compute-type', 'float16',
        '--trust-checkpoint',
    ], check=True)

    predictor = CTranslate2Predictor(ct2_dir, device='cuda', compute_type='float16')
    split_metrics = evaluate_continual_splits(predictor, EVAL_DIR, batch_size=128)
    del predictor

    gating = evaluate_3tier_gating(baseline_metrics, split_metrics)
    record = {
        'step': step,
        'checkpoint': str(ckpt),
        'ct2_dir': str(ct2_dir),
        'status': gating['status'],
        'outcome': gating['outcome'],
        'plasticity_acc': split_metrics['plasticity']['_overall']['exact_accuracy'],
        'plasticity_gain': gating['gate_2_plasticity']['gain'],
        'plasticity_noop': split_metrics['plasticity']['_overall']['noop_rate'],
        'retention_acc': split_metrics['retention']['_overall']['exact_accuracy'],
        'retention_deg': gating['gate_1_retention']['degradation'],
        'fcr_seen': split_metrics['protection_seen']['_overall']['fcr'],
        'fcr_heldout': split_metrics['protection_heldout']['_overall']['fcr'],
        'gating_details': gating,
        'split_metrics': split_metrics,
    }
    candidate_results.append(record)

# Step C: Select Pareto Optimal Checkpoint
passing_candidates = [c for c in candidate_results if c['status'] == 'PASSED']
if passing_candidates:
    # Pareto criterion: lowest total FCR, then highest plasticity gain
    winning_candidate = min(
        passing_candidates,
        key=lambda c: (c['fcr_seen'] + c['fcr_heldout'], -c['plasticity_gain'])
    )
else:
    print('WARNING: No checkpoint passed all 3 tiers. Selecting candidate with minimum Gate violations.')
    winning_candidate = min(
        candidate_results,
        key=lambda c: (c['fcr_heldout'], c['retention_deg'], -c['plasticity_gain'])
    )

print('=' * 60)
print('SELECTED WINNING CHECKPOINT: Step', winning_candidate['step'])
print('Outcome Status :', winning_candidate['outcome'])
print(f'Plasticity Acc : {winning_candidate[\"plasticity_acc\"]*100:.2f}% (gain: {winning_candidate[\"plasticity_gain\"]*100:+.2f}%)')
print(f'Retention Acc  : {winning_candidate[\"retention_acc\"]*100:.2f}% (drop: {winning_candidate[\"retention_deg\"]*100:+.2f}%)')
print(f'FCR Seen       : {winning_candidate[\"fcr_seen\"]*100:.2f}%')
print(f'FCR Held-out   : {winning_candidate[\"fcr_heldout\"]*100:.2f}%')
print('=' * 60)

# Display tabular comparison
import pandas as pd
display_df = pd.DataFrame([{
    'Step': c['step'],
    'Status': c['status'],
    'Plasticity %': f\"{c['plasticity_acc']*100:.1f}% (+{c['plasticity_gain']*100:.1f}%)\",
    'Retention %': f\"{c['retention_acc']*100:.1f}% (-{c['retention_deg']*100:.1f}%)\",
    'FCR Seen %': f\"{c['fcr_seen']*100:.1f}%\",
    'FCR Held-out %': f\"{c['fcr_heldout']*100:.1f}%\",
} for c in candidate_results])
display(display_df)
"""),
    md("""## 7. Standard Frozen Diagnostic Benchmarks Comparison

Evaluates the winning Continual Fine-Tuned model against Base V2 on the three frozen benchmarks:
- `reparos-diagnostic-10k`
- `reparos-compositional-4k`
- `reparos-user-centric-v2`
"""),
    code("""from reparos.curriculum_notebook import evaluate_checkpoint

WINNING_CHECKPOINT = Path(winning_candidate['checkpoint'])

bench_reports = {}
for system_name, ckpt in (('base_v2', BASE_CHECKPOINT), ('continual_v1', WINNING_CHECKPOINT)):
    bench_reports[system_name] = evaluate_checkpoint(
        checkpoint=ckpt,
        tokenizer=TOKENIZER_MODEL,
        decoding_config=DECODING_CONFIG,
        evaluation_root=BENCHMARK_ROOT,
        output_root=RUN_ROOT / 'benchmarks' / system_name,
        reparos_cli=REPAROS,
        benchmark_cli=BENCHMARK,
        device='cuda',
        compute_type='float16',
    )

print(json.dumps(bench_reports, indent=2, ensure_ascii=False))

# Show Diagnostic & Compositional Comparison
for bname in ('user', 'composition', 'diagnostic'):
    b_base = bench_reports['base_v2'][bname].get('overall', {})
    b_cont = bench_reports['continual_v1'][bname].get('overall', {})
    print(f'--- Benchmark: {bname} ---')
    for metric_name in ('query_exact_accuracy', 'clean_preservation_rate', 'clean_false_correction_rate'):
        v1 = b_base.get(metric_name)
        v2 = b_cont.get(metric_name)
        if v1 is not None and v2 is not None:
            print(f'  {metric_name:28s}: Base V2={v1*100:5.2f}% | Continual={v2*100:5.2f}% | Delta={(v2-v1)*100:+5.2f}%')
"""),
    md("""## 8. Export Final Continual Fine-Tuned Artifacts

Packages the winning model, tokenizer, decoding configuration, CTranslate2 runtime model, and evaluation reports into a single zip archive for download.
"""),
    code("""EXPORT_DIR = RUN_ROOT / 'final_artifact'
if EXPORT_DIR.exists():
    shutil.rmtree(EXPORT_DIR)
EXPORT_DIR.mkdir(parents=True)

# Copy winning checkpoint and assets
shutil.copy2(WINNING_CHECKPOINT, EXPORT_DIR / WINNING_CHECKPOINT.name)
shutil.copy2(TOKENIZER_MODEL, EXPORT_DIR / 'tokenizer.model')
shutil.copy2(VOCAB_SRC, EXPORT_DIR / 'vocab.src')
shutil.copy2(VOCAB_TGT, EXPORT_DIR / 'vocab.tgt')
shutil.copy2(DECODING_CONFIG, EXPORT_DIR / 'decoding-config.json')
shutil.copy2(CONFIG_PATH, EXPORT_DIR / 'opennmt-continual.json')

# Copy CTranslate2 converted model
WINNING_CT2 = Path(winning_candidate['ct2_dir'])
shutil.copytree(WINNING_CT2, EXPORT_DIR / 'ctranslate2')

# Save gating and benchmark report
REPORT = {
    'selected_checkpoint': winning_candidate['checkpoint'],
    'selected_step': winning_candidate['step'],
    'outcome': winning_candidate['outcome'],
    'status': winning_candidate['status'],
    'baseline_metrics': baseline_metrics,
    'candidate_evaluations': candidate_results,
    'benchmark_reports': bench_reports,
}
(EXPORT_DIR / 'continual-finetune-report.json').write_text(
    json.dumps(REPORT, indent=2, ensure_ascii=False) + '\\n', encoding='utf-8'
)

zip_file = Path(shutil.make_archive('/kaggle/working/reparos-continual-v1-final', 'zip', EXPORT_DIR))
print('=' * 60)
print('ARTIFACT READY FOR DOWNLOAD:')
print('File Path :', zip_file)
print(f'File Size : {zip_file.stat().st_size / (1024 * 1024):.2f} MB')
print('=' * 60)
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
    destination = Path("notebook/train_reparos_continual_finetune_kaggle.ipynb")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(destination, len(CELLS), "cells successfully rendered!")


if __name__ == "__main__":
    main()
