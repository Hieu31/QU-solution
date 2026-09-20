from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path("notebook/train_reparos_base_v2_kaggle_offline.ipynb")
OLD = "RUN_ROOT = Path('/kaggle/working/reparos/base-v2-32-run')"
NEW = "RUN_ROOT = Path('/kaggle/working/reparos/base-v2-32-tokenizer-v2-run')"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][2]["source"])
    assert OLD in source
    notebook["cells"][2]["source"] = source.replace(OLD, NEW).splitlines(keepends=True)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK}:cell-{index}")
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
