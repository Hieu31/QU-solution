from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

PARTIAL_SUFFIXES = (" b", " be", " ben", " bệnh v", " bệnh vi", " bệnh viê", " trường t", " trường th", " trường thc", " trường mầm", " sân b", " sân ba", " bến x", " trung t")

def looks_incomplete(text: str) -> bool:
    value = " " + re.sub(r"\s+", " ", text.strip().casefold())
    return any(value.endswith(suffix) for suffix in PARTIAL_SUFFIXES)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, destination = Path(args.input), Path(args.output)
    counts: Counter[str] = Counter()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream); rows = list(reader); fieldnames = list(reader.fieldnames or [])
    for row in rows:
        text, stratum = row["input"].strip(), row["stratum"]
        if stratum == "exact_osm_keep":
            action, expected, reason = "keep", text, "exact_normalized_osm_match"
        elif stratum == "number_address" and not looks_incomplete(text):
            action, expected, reason = "keep", text, "structured_number_address_preservation"
        elif looks_incomplete(text):
            action, expected, reason = "incomplete", "", "query_looks_prefix_truncated"
        else:
            action, expected, reason = "unknown", "", "insufficient_evidence_for_gold_label"
        row["action"], row["expected"], row["review_status"] = action, expected, "auto_labeled"
        row["annotator_confidence"] = "high" if action == "keep" else "low"
        row["notes"] = "; ".join(x for x in (row.get("notes", "").strip(), f"auto_rule={reason}") if x)
        counts[action] += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "actions": dict(counts), "output": str(destination)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
