from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


VALID_ACTIONS = {"keep", "correct", "incomplete", "out_of_domain", "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows, counts = [], Counter()
    with Path(args.annotations).open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            action = row["action"].strip()
            if action not in VALID_ACTIONS:
                raise ValueError(f"invalid action at line {line}: {action!r}")
            status = row["review_status"].strip()
            if status not in {"reviewed", "adjudicated"}:
                raise ValueError(f"unreviewed row at line {line}: {status!r}")
            expected = row["expected"].strip()
            if action == "keep":
                expected = row["input"].strip()
            if action == "correct" and not expected:
                raise ValueError(f"correct row lacks expected value at line {line}")
            counts[action] += 1
            if action not in {"keep", "correct"}:
                continue
            rows.append({
                "query_id": row["query_id"],
                "input": row["input"].strip(),
                "expected": expected,
                "error_type": "clean" if action == "keep" else (row["error_types"].strip() or "production_other"),
                "split": "zero-click-production-5k",
                "stratum": row["stratum"],
                "frequency": int(row["frequency"]),
                "review_status": status,
            })

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "profile": "zero-click-production-benchmark/gold-v1",
        "annotated_actions": dict(counts),
        "scorable_rows": len(rows),
        "gold_sha256": digest,
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
