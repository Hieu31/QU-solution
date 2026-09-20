from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path("notebook/train_reparos_base_v2_kaggle_offline.ipynb")
OLD = '''# Kaggle expands Dataset contents under /kaggle/input. Find the single root
# containing all five required bundle directories.
bundle_candidates = []
for project_dir in KAGGLE_INPUT.rglob('project'):
    root = project_dir.parent
    if all((root / name).is_dir() for name in ('project', 'wheels', 'data', 'artifacts', 'evaluation')):
        bundle_candidates.append(root)
bundle_candidates = sorted(set(bundle_candidates))
assert len(bundle_candidates) == 1, f'Expected one expanded kaggle-bundle, found: {bundle_candidates}'
BUNDLE_ROOT = bundle_candidates[0]
'''
NEW = '''# Fixed path of the attached private Kaggle Dataset.
BUNDLE_ROOT = Path('/kaggle/input/datasets/vanhieu1125/reparos-haha/kaggle-bundle')
required_bundle_dirs = ('project', 'wheels', 'data', 'artifacts', 'evaluation')
assert all((BUNDLE_ROOT / name).is_dir() for name in required_bundle_dirs), BUNDLE_ROOT
'''


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][4]["source"])
    if OLD not in source:
        raise RuntimeError("expected auto-discovery block was not found")
    notebook["cells"][4]["source"] = source.replace(OLD, NEW).splitlines(keepends=True)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK}:cell-{index}")
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
