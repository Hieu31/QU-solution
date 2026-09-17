from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


COMMON_SETUP = r'''import os, sys, subprocess, json
from pathlib import Path

if not REPO_ROOT.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_ROOT)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO_ROOT), 'pull', '--ff-only'], check=True)
os.chdir(REPO_ROOT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run(['uv', 'sync', '--python', '3.11', '--extra', 'reparos-opennmt'], check=True)

VENV_PY = REPO_ROOT / '.venv/bin/python'
REPAROS = REPO_ROOT / '.venv/bin/reparos'
BENCHMARK = REPO_ROOT / '.venv/bin/benchmark-three'
assert all(path.is_file() for path in (VENV_PY, REPAROS, BENCHMARK))
subprocess.run(['nvidia-smi'], check=True)
subprocess.run(['git', 'rev-parse', 'HEAD'], check=True)
'''


VALIDATE = r'''from reparos.curriculum_notebook import prepare_base_template
from reparos.curriculum_training import latest_checkpoint

for profile, expected in [('pilot', {'stage1-primitives': 120000, 'stage2-composition': 160000, 'stage3-lexical': 120000}),
                          ('full', {'stage1-primitives': 1000000, 'stage2-composition': 1200000, 'stage3-lexical': 600000})]:
    for stage, expected_rows in expected.items():
        folder = CURRICULUM_ROOT / profile / stage
        paths = [folder / name for name in ('train.src', 'train.tgt', 'validation.src', 'validation.tgt')]
        assert all(path.is_file() for path in paths), paths
        train_counts = [sum(1 for _ in path.open(encoding='utf-8')) for path in paths[:2]]
        valid_counts = [sum(1 for _ in path.open(encoding='utf-8')) for path in paths[2:]]
        assert train_counts == [expected_rows, expected_rows], (profile, stage, train_counts)
        assert valid_counts[0] == valid_counts[1] and valid_counts[0] > 0

EVALUATION_ROOT = CURRICULUM_ROOT / 'evaluation'
for name in ('user-centric-v2.jsonl', 'composition-4k.jsonl', 'diagnostic-10k.jsonl'):
    assert (EVALUATION_ROOT / name).is_file(), EVALUATION_ROOT / name

BASE_CHECKPOINT = latest_checkpoint(BASE_ARTIFACT_DIR, 'reparos_base')
ORIGINAL_BASE_CONFIG = BASE_ARTIFACT_DIR / 'opennmt-base.json'
DECODING_CONFIG = BASE_ARTIFACT_DIR / 'decoding-config.json'
assert ORIGINAL_BASE_CONFIG.is_file() and DECODING_CONFIG.is_file()
BASE_TEMPLATE = prepare_base_template(
    ORIGINAL_BASE_CONFIG, BASE_ARTIFACT_DIR, TOKENIZER_MODEL,
    RUN_ROOT / 'base-template.json',
)
print('Base checkpoint:', BASE_CHECKPOINT)
print('Curriculum:', CURRICULUM_ROOT)
print('Evaluation:', EVALUATION_ROOT)
'''


PILOT_TRAIN = r'''from reparos.curriculum_notebook import train_sequence

assert RUN_PILOT, 'Set RUN_PILOT=True to run the three pilot stages.'
PILOT_ROOT = RUN_ROOT / 'pilot'
PILOT_FINAL, PILOT_CHECKPOINTS = train_sequence(
    profile_root=CURRICULUM_ROOT / 'pilot', output_root=PILOT_ROOT,
    base_config=BASE_TEMPLATE, initial_checkpoint=BASE_CHECKPOINT,
    additional_steps=PILOT_STEPS, python_executable=VENV_PY,
    valid_steps=500, checkpoint_steps=500,
)
print('Pilot final checkpoint:', PILOT_FINAL)
'''


PILOT_GATE = r'''from reparos.curriculum_notebook import run_pilot_gates

PILOT_GATES = run_pilot_gates(
    PILOT_CHECKPOINTS, tokenizer=TOKENIZER_MODEL,
    decoding_config=DECODING_CONFIG, evaluation_root=EVALUATION_ROOT,
    output_root=RUN_ROOT / 'pilot-evaluation', reparos_cli=REPAROS,
    benchmark_cli=BENCHMARK,
)
print(json.dumps(PILOT_GATES, ensure_ascii=False, indent=2))
ALL_PILOT_GATES_PASSED = bool(PILOT_GATES['passed'])
assert ALL_PILOT_GATES_PASSED, 'Pilot failed. Inspect pilot-gates.json; full training remains locked.'
print('ALL THREE PILOT STAGES PASSED. Full training may now be explicitly unlocked.')
'''


FULL_TRAIN = r'''assert ALL_PILOT_GATES_PASSED, 'Full training is locked: pilot gates did not pass.'
assert ALLOW_FULL_TRAIN, 'Review pilot-gates.json, then explicitly set ALLOW_FULL_TRAIN=True in the config cell.'

# Full is a clean branch from Base, not a continuation of the pilot checkpoint.
FULL_ROOT = RUN_ROOT / 'full'
FULL_FINAL, FULL_CHECKPOINTS = train_sequence(
    profile_root=CURRICULUM_ROOT / 'full', output_root=FULL_ROOT,
    base_config=BASE_TEMPLATE, initial_checkpoint=BASE_CHECKPOINT,
    additional_steps=FULL_STEPS, python_executable=VENV_PY,
    valid_steps=1000, checkpoint_steps=1000,
)
print('Full final checkpoint:', FULL_FINAL)
'''


FINAL_EVAL = r'''from reparos.curriculum_notebook import evaluate_checkpoint

FINAL_REPORTS = evaluate_checkpoint(
    checkpoint=FULL_FINAL, tokenizer=TOKENIZER_MODEL,
    decoding_config=DECODING_CONFIG, evaluation_root=EVALUATION_ROOT,
    output_root=RUN_ROOT / 'full-evaluation', reparos_cli=REPAROS,
    benchmark_cli=BENCHMARK,
)
(RUN_ROOT / 'full-evaluation-summary.json').write_text(
    json.dumps(FINAL_REPORTS, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
)
print('Saved final reports:', RUN_ROOT / 'full-evaluation-summary.json')
'''


def notebook(platform: str) -> dict:
    if platform == "colab":
        config = r'''from google.colab import drive
drive.mount('/content/drive')

REPO_URL = 'https://github.com/Hieu31/QU-solution.git'
REPO_ROOT = Path('/content/QU-solution') if 'Path' in globals() else __import__('pathlib').Path('/content/QU-solution')
DRIVE_ROOT = __import__('pathlib').Path('/content/drive/MyDrive/reparos')
CURRICULUM_ZIP = DRIVE_ROOT / 'reparos-curriculum-v2.zip'
LOCAL_CURRICULUM_PARENT = __import__('pathlib').Path('/content/reparos-curriculum')
BASE_ARTIFACT_DIR = DRIVE_ROOT / 'opennmt-production-kaggle-t4-v1'
TOKENIZER_MODEL = DRIVE_ROOT / 'tokenizer-production-v1/tokenizer.model'
RUN_ROOT = DRIVE_ROOT / 'curriculum-v2-run'

RUN_PILOT = True
ALLOW_FULL_TRAIN = False  # Change only after the three pilot gates pass.
PILOT_STEPS = {'stage1-primitives': 3000, 'stage2-composition': 3000, 'stage3-lexical': 3000}
FULL_STEPS = {'stage1-primitives': 12000, 'stage2-composition': 15000, 'stage3-lexical': 8000}
'''
        data = r'''import shutil
assert CURRICULUM_ZIP.is_file(), f'Missing: {CURRICULUM_ZIP}'
if LOCAL_CURRICULUM_PARENT.exists():
    shutil.rmtree(LOCAL_CURRICULUM_PARENT)
LOCAL_CURRICULUM_PARENT.mkdir(parents=True)
subprocess.run(['unzip', '-q', str(CURRICULUM_ZIP), '-d', str(LOCAL_CURRICULUM_PARENT)], check=True)
candidates = [p.parent.parent for p in LOCAL_CURRICULUM_PARENT.rglob('pilot/stage1-primitives/train.src')]
assert len(candidates) == 1, candidates
CURRICULUM_ROOT = candidates[0]
print('Curriculum root:', CURRICULUM_ROOT)
'''
        title = "# ReparoS curriculum v2 — Colab\n\nTrain three pilot stages, run frozen gates, and only then unlock a separate full run."
    else:
        config = r'''from pathlib import Path

REPO_URL = 'https://github.com/Hieu31/QU-solution.git'
REPO_ROOT = Path('/kaggle/working/QU-solution')
RUN_ROOT = Path('/kaggle/working/reparos/curriculum-v2-run')

curriculum_candidates = [p.parent.parent for p in Path('/kaggle/input').rglob('pilot/stage1-primitives/train.src')]
assert curriculum_candidates, 'Attach the curriculum-v2 dataset to this notebook.'
CURRICULUM_ROOT = curriculum_candidates[0]
base_configs = list(Path('/kaggle/input').rglob('opennmt-base.json'))
assert base_configs, 'Attach the trusted Base OpenNMT artifact dataset.'
BASE_ARTIFACT_DIR = base_configs[0].parent
tokenizers = list(Path('/kaggle/input').rglob('tokenizer.model'))
assert tokenizers, 'Attach tokenizer-production-v1.'
TOKENIZER_MODEL = tokenizers[0]

RUN_PILOT = True
ALLOW_FULL_TRAIN = False  # Change only after the three pilot gates pass.
PILOT_STEPS = {'stage1-primitives': 3000, 'stage2-composition': 3000, 'stage3-lexical': 3000}
FULL_STEPS = {'stage1-primitives': 12000, 'stage2-composition': 15000, 'stage3-lexical': 8000}
'''
        data = "print('Curriculum root:', CURRICULUM_ROOT)\nprint('Base artifacts:', BASE_ARTIFACT_DIR)\nprint('Tokenizer:', TOKENIZER_MODEL)\n"
        title = "# ReparoS curriculum v2 — Kaggle\n\nTrain three pilot stages, run frozen gates, and only then unlock a separate full run."
    cells = [
        md(title),
        md("## 1. Configuration\n\n`ALLOW_FULL_TRAIN` must remain `False` until all pilot gates pass."),
        code("import os, sys, subprocess, json\nfrom pathlib import Path\n\n" + config),
        md("## 2. Runtime setup"), code(COMMON_SETUP),
        md("## 3. Locate curriculum and trusted Base artifacts"), code(data),
        md("## 4. Validate all data and lock Base vocabulary/checkpoint"), code(VALIDATE),
        md("## 5. Train Pilot Stage 1 → Stage 2 → Stage 3\n\nEach stage resumes from the preceding checkpoint. No full data is touched here."), code(PILOT_TRAIN),
        md("## 6. Evaluate every pilot checkpoint and enforce gates\n\nThis runs the frozen 88-case user set, 4K composition set, and 10K stratified set for each stage."), code(PILOT_GATE),
        md("## 7. FULL TRAIN — manually locked\n\nOnly after the prior cell prints that all gates passed, set `ALLOW_FULL_TRAIN=True` in the configuration cell and run this cell. Full starts again from Base, not Pilot."), code(FULL_TRAIN),
        md("## 8. Final full evaluation and CTranslate2 export"), code(FINAL_EVAL),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    targets = {
        Path("notebook/train_reparos.ipynb"): notebook("colab"),
        Path("notebook/train_reparos_kaggle.ipynb"): notebook("kaggle"),
    }
    for path, payload in targets.items():
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path, len(payload["cells"]), "cells")


if __name__ == "__main__":
    main()
