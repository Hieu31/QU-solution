from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    counts: Counter[str] = Counter(); gold = []
    with Path(args.annotations).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            action = row["action"].strip(); counts[action] += 1
            if row["review_status"].strip() != "auto_labeled":
                raise ValueError("weak-gold exporter only accepts auto_labeled rows")
            if action not in {"keep", "correct"}:
                continue
            gold.append({"query_id": row["query_id"], "input": row["input"].strip(),
                "expected": row["expected"].strip(), "error_type": "clean" if action == "keep" else (row["error_types"].strip() or "production_other"),
                "split": "zero-click-production-5k", "stratum": row["stratum"],
                "frequency": int(row["frequency"]), "review_status": "auto_labeled"})
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in gold: stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"schema_version": 1, "profile": "zero-click-production-benchmark/weak-gold-v1",
        "label_provenance": "weak-auto", "annotated_actions": dict(counts),
        "scorable_rows": len(gold), "gold_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
