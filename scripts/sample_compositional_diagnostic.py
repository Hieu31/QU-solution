from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path


LETTERS_AND_SPACES = re.compile(r"^[^\W\d_]+(?: [^\W\d_]+){1,5}$", re.UNICODE)


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def remove_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    plain = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", plain).replace("đ", "d").replace("Đ", "D")


def stable_id(kind: str, target: str) -> str:
    digest = hashlib.sha256(f"compositional-v1\0{kind}\0{target}".encode()).hexdigest()[:20]
    return f"composition:{kind}:{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired diacritic/boundary diagnostic data")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = Path(args.data)
    targets = (root / "test.tgt").read_text(encoding="utf-8").splitlines()
    metadata = [json.loads(line) for line in (root / "test.meta.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(targets) != len(metadata):
        raise ValueError("test target/meta row counts differ")

    candidates: dict[str, tuple[str, dict]] = {}
    for target, meta in zip(targets, metadata):
        target = normalize(target)
        if meta.get("error_type") != "clean" or not LETTERS_AND_SPACES.fullmatch(target):
            continue
        if remove_diacritics(target) == target:
            continue
        candidates.setdefault(target.casefold(), (target, meta))
    values = list(candidates.values())
    random.Random(args.seed).shuffle(values)
    if len(values) < args.queries:
        raise ValueError(f"only {len(values)} eligible clean queries; requested {args.queries}")

    rows: list[dict] = []
    for target, meta in values[: args.queries]:
        words = target.split()
        join_index = random.Random(f"{args.seed}:{target}").randrange(len(words) - 1)
        joined_words = words[:join_index] + [words[join_index] + words[join_index + 1]] + words[join_index + 2 :]
        variants = {
            "missing_diacritics_only": remove_diacritics(target),
            "boundary_one_only": " ".join(joined_words),
            "missing_diacritics_boundary_one": remove_diacritics(" ".join(joined_words)),
            "missing_diacritics_boundary_all": remove_diacritics(target).replace(" ", ""),
        }
        for kind, source in variants.items():
            item = dict(meta)
            item.update({
                "query_id": stable_id(kind, target),
                "input": source,
                "expected": target,
                "error_type": kind,
                "split": "test",
                "diagnostic_family": "diacritics_x_word_boundary",
            })
            rows.append(item)
    random.Random(args.seed).shuffle(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "purpose": "paired compositional diagnostic; never train on this file",
        "source": str(root.resolve()),
        "seed": args.seed,
        "base_queries": args.queries,
        "rows": len(rows),
        "buckets": {name: args.queries for name in variants},
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
