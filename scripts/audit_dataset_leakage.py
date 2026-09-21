from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from reparos.continual_dataset import (
    HELDOUT_PROTECTED_ENTITIES,
    SEEN_PROTECTED_ENTITIES,
)


def load_lines(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audit_leakage(dataset_dir: Path) -> dict:
    train_dir = dataset_dir / "train"
    eval_dir = dataset_dir / "eval"

    print("=" * 60)
    print(f"AUDITING DATASET LEAKAGE: {dataset_dir}")
    print("=" * 60)

    # 1. Collect all train strings
    train_files = list(train_dir.rglob("*.jsonl"))
    print(f"Found {len(train_files)} train jsonl files:")
    for tf in train_files:
        print(f"  - {tf.relative_to(dataset_dir)}")

    train_inputs: set[str] = set()
    train_expecteds: set[str] = set()
    train_full_text: list[str] = []
    total_train_rows = 0

    for tf in train_files:
        rows = load_lines(tf)
        total_train_rows += len(rows)
        for r in rows:
            inp = r["input"].strip().casefold()
            exp = r["expected"].strip().casefold()
            train_inputs.add(inp)
            train_expecteds.add(exp)
            train_full_text.append(inp)
            train_full_text.append(exp)

    print(f"Total train rows: {total_train_rows:,}")
    print(f"Total unique train inputs: {len(train_inputs):,}")
    print(f"Total unique train targets: {len(train_expecteds):,}")

    # 2. Check eval/protection_heldout against train
    heldout_files = list((eval_dir / "protection_heldout").rglob("*.jsonl"))
    print(f"\nAuditing {len(heldout_files)} heldout evaluation files:")
    for hf in heldout_files:
        print(f"  - {hf.relative_to(dataset_dir)}")

    exact_query_leaks = []
    substring_entity_leaks = []
    total_heldout_rows = 0

    for hf in heldout_files:
        rows = load_lines(hf)
        total_heldout_rows += len(rows)
        for r in rows:
            inp = r["input"].strip().casefold()
            exp = r["expected"].strip().casefold()
            # Exact match check
            if inp in train_inputs:
                exact_query_leaks.append({"file": str(hf.name), "type": "input_exact_match", "text": r["input"]})
            if exp in train_expecteds:
                exact_query_leaks.append({"file": str(hf.name), "type": "target_exact_match", "text": r["expected"]})

    # 3. Word-Boundary Regex scan: Verify NO held-out entity appears in ANY train text as a whole entity/word
    print("\nScanning train corpus for forbidden held-out entity mentions...")
    forbidden_entities = [e.strip() for e in HELDOUT_PROTECTED_ENTITIES]
    
    for ent in forbidden_entities:
        pattern = re.compile(r"\b" + re.escape(ent) + r"\b", re.IGNORECASE)
        matches = [line for line in train_full_text if pattern.search(line)]
        if matches:
            substring_entity_leaks.append({
                "forbidden_entity": ent,
                "matches_count": len(matches),
                "examples": matches[:3],
            })

    # 4. Summary & Report
    passed = (len(exact_query_leaks) == 0) and (len(substring_entity_leaks) == 0)
    report = {
        "dataset_dir": str(dataset_dir),
        "audit_passed": passed,
        "total_train_rows": total_train_rows,
        "total_heldout_rows": total_heldout_rows,
        "exact_query_leaks_count": len(exact_query_leaks),
        "exact_query_leaks": exact_query_leaks[:10],
        "substring_entity_leaks_count": len(substring_entity_leaks),
        "substring_entity_leaks": substring_entity_leaks,
        "forbidden_entities_checked": len(forbidden_entities),
    }

    report_path = dataset_dir / "leak_audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    if passed:
        print(">>> AUDIT RESULT: [100% PASS - ZERO LEAKAGE CONFIRMED] <<<")
        print(f"Zero exact query leaks and zero held-out entity mentions in {total_train_rows:,} train rows.")
    else:
        print(">>> AUDIT RESULT: [FAIL - LEAKAGE DETECTED] <<<")
        print(f"Exact leaks: {len(exact_query_leaks)}, Substring leaks: {len(substring_entity_leaks)}")
    print(f"Report written to: {report_path}")
    print("=" * 60)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset for leakage between train and eval/protection_heldout")
    parser.add_argument("--data", default="data/reparos/interleaved-continual-v1", help="Path to dataset root")
    args = parser.parse_args()

    report = audit_leakage(Path(args.data))
    if not report["audit_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
