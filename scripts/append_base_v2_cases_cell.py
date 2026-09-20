from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebook/train_reparos_base_v2_kaggle.ipynb")


def cell(kind: str, source: str) -> dict:
    payload = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        payload.update({"execution_count": None, "outputs": []})
    return payload


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    marker = "## 8. Inspect improvements and regressions"
    if any(marker in "".join(item["source"]) for item in notebook["cells"]):
        return

    markdown = cell("markdown", marker)
    source = r'''def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

case_rows = []
for benchmark in ('user', 'composition', 'diagnostic'):
    base_rows = {
        row['query_id']: row
        for row in load_jsonl(RUN_ROOT / 'evaluation' / 'base' / f'{benchmark}.predictions.jsonl')
    }
    v2_rows = load_jsonl(RUN_ROOT / 'evaluation' / 'base_v2_32' / f'{benchmark}.predictions.jsonl')
    for new in v2_rows:
        old = base_rows[new['query_id']]
        old_ok = old['output'] == old['expected']
        new_ok = new['output'] == new['expected']
        if old_ok != new_ok:
            case_rows.append({
                'status': 'IMPROVED' if new_ok else 'REGRESSED',
                'benchmark': benchmark,
                'error_type': new['error_type'],
                'input': new['input'],
                'expected': new['expected'],
                'base': old['output'],
                'base_v2_32': new['output'],
            })

case_rows.sort(key=lambda row: (row['status'] != 'IMPROVED', row['benchmark'], row['error_type']))
CASE_PATH = RUN_ROOT / 'base-v2-cases.json'
CASE_PATH.write_text(json.dumps(case_rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
try:
    import pandas as pd
    frame = pd.DataFrame(case_rows)
    print(frame['status'].value_counts())
    display(frame.groupby(['status', 'benchmark']).head(20))
except ImportError:
    print(json.dumps(case_rows[:120], ensure_ascii=False, indent=2))
'''
    code_cell = cell("code", source)

    package_index = next(
        index for index, item in enumerate(notebook["cells"])
        if "Package checkpoint" in "".join(item["source"])
    )
    notebook["cells"][package_index:package_index] = [markdown, code_cell]
    notebook["cells"][package_index + 2]["source"] = [
        line.replace("## 8. Package", "## 9. Package")
        for line in notebook["cells"][package_index + 2]["source"]
    ]
    package_code = notebook["cells"][package_index + 3]["source"]
    report_copy = "shutil.copy2(REPORT_PATH, EXPORT_ROOT / REPORT_PATH.name)\n"
    case_copy = "shutil.copy2(CASE_PATH, EXPORT_ROOT / CASE_PATH.name)\n"
    insertion = package_code.index(report_copy) + 1
    package_code.insert(insertion, case_copy)

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
