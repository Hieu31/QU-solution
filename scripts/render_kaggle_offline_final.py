from __future__ import annotations

import ast
import json
from pathlib import Path


SOURCE = Path("notebook/train_reparos_base_v2_kaggle.ipynb")
DESTINATION = Path("notebook/train_reparos_base_v2_kaggle_offline.ipynb")


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


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    notebook["cells"][0] = md("""# ReparoS Base V2-32 — Kaggle offline

One-notebook competition workflow. The attached Kaggle Dataset already contains `project/`, `wheels/`, `data/`, `artifacts/` and `evaluation/`. No network access, archive extraction, virtualenv or Torch replacement is used.
""")

    config = "".join(notebook["cells"][2]["source"])
    config = "\n".join(
        line for line in config.splitlines()
        if not line.startswith("REPO_URL =")
    ) + "\n"
    notebook["cells"][2] = code(config)
    notebook["cells"][3] = md("## 2. Offline runtime setup")
    notebook["cells"][4] = code(r'''import json, os, shutil, subprocess, sys

# Kaggle expands Dataset contents under /kaggle/input. Find the single root
# containing all five required bundle directories.
bundle_candidates = []
for project_dir in KAGGLE_INPUT.rglob('project'):
    root = project_dir.parent
    if all((root / name).is_dir() for name in ('project', 'wheels', 'data', 'artifacts', 'evaluation')):
        bundle_candidates.append(root)
bundle_candidates = sorted(set(bundle_candidates))
assert len(bundle_candidates) == 1, f'Expected one expanded kaggle-bundle, found: {bundle_candidates}'
BUNDLE_ROOT = bundle_candidates[0]

# Copy source out of read-only /kaggle/input.
if REPO_ROOT.exists():
    shutil.rmtree(REPO_ROOT)
shutil.copytree(BUNDLE_ROOT / 'project', REPO_ROOT)

# Install only bundled Linux CPython 3.12 wheels. Torch is deliberately absent:
# keep Kaggle's CUDA-enabled torch 2.10.0+cu128 for Blackwell sm_120.
SITE_PACKAGES = Path('/kaggle/working/site-packages')
if SITE_PACKAGES.exists():
    shutil.rmtree(SITE_PACKAGES)
SITE_PACKAGES.mkdir(parents=True)
wheels = sorted((BUNDLE_ROOT / 'wheels').glob('*.whl'))
assert wheels, BUNDLE_ROOT / 'wheels'
assert not any(path.name.startswith(('torch-', 'nvidia_')) for path in wheels)
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    '--no-index', '--no-deps', '--target', str(SITE_PACKAGES),
    *map(str, wheels),
], check=True)

os.chdir(REPO_ROOT)
SRC_ROOT = REPO_ROOT / 'src'
python_path = os.pathsep.join((str(SRC_ROOT), str(SITE_PACKAGES)))
os.environ['PYTHONPATH'] = python_path + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path[:0] = [str(SRC_ROOT), str(SITE_PACKAGES)]

# Helpers expect executable CLI paths. Small local wrappers invoke the repo
# modules with the Kaggle Python interpreter and inherited PYTHONPATH.
BIN_ROOT = Path('/kaggle/working/qu-solution-bin')
BIN_ROOT.mkdir(parents=True, exist_ok=True)
VENV_PY = Path(sys.executable)
REPAROS = BIN_ROOT / 'reparos'
BENCHMARK = BIN_ROOT / 'benchmark-three'
REPAROS.write_text(f'#!{sys.executable}\nfrom reparos.cli import main\nmain()\n', encoding='utf-8')
BENCHMARK.write_text(f'#!{sys.executable}\nfrom benchmarking.cli import main\nmain()\n', encoding='utf-8')
REPAROS.chmod(0o755)
BENCHMARK.chmod(0o755)

# All subsequent artifact discovery must search this bundle, not every attached
# Kaggle Dataset.
KAGGLE_INPUT = BUNDLE_ROOT

import torch, onmt, pyonmttok, sentencepiece, ctranslate2
assert torch.cuda.is_available(), 'GPU accelerator is not enabled'
print('Bundle:', BUNDLE_ROOT)
print('Python:', sys.version)
print('Torch:', torch.__version__, torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('OpenNMT:', getattr(onmt, '__version__', 'unknown'))
print('CTranslate2:', ctranslate2.__version__)
''')

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{DESTINATION}:cell-{index}")
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DESTINATION, len(notebook["cells"]), "cells")


if __name__ == "__main__":
    main()
