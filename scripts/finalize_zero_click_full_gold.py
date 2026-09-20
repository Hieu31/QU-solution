from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.input).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    counts: Counter[str] = Counter()
    for row in rows:
        previous = row["action"].strip()
        if previous not in {"keep", "correct"}:
            row["action"] = "keep"
            row["expected"] = row["input"].strip()
            row["error_types"] = "clean_protected"
            row["annotator_confidence"] = "medium"
            row["notes"] = (
                f"reviewer=codex; previous_action={previous}; "
                "scoring_policy=preserve_when_correction_is_not_uniquely_supported"
            )
        elif previous == "keep":
            row["expected"] = row["input"].strip()
            row["error_types"] = row["error_types"].strip() or "clean_protected"
        row["review_status"] = "adjudicated"
        counts[row["action"]] += 1

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)} keep={counts['keep']} correct={counts['correct']}")


if __name__ == "__main__":
    main()
