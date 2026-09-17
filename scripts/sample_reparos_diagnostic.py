from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


BUCKETS = (
    "clean",
    "combined",
    "address_abbreviation",
    "address_symbol",
    "keyboard_edit",
    "missing_diacritics_full",
    "missing_diacritics_partial",
    "telex_leak",
    "vni_leak",
    "word_boundary",
    "wrong_diacritic",
)


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def bucket(error_type: str) -> str:
    return "combined" if error_type.startswith("combined") else error_type


def stable_id(split: str, index: int, source: str, target: str) -> str:
    digest = hashlib.sha256(f"{split}\0{index}\0{source}\0{target}".encode()).hexdigest()[:16]
    return f"{split}:{index:09d}:{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed 10K stratified ReparoS diagnostic set")
    parser.add_argument("--data", required=True, help="parallel dataset directory containing test.src/tgt/meta.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = Path(args.data)
    sources = (root / "test.src").read_text(encoding="utf-8").splitlines()
    targets = (root / "test.tgt").read_text(encoding="utf-8").splitlines()
    metadata = [json.loads(line) for line in (root / "test.meta.jsonl").read_text(encoding="utf-8").splitlines()]
    if not (len(sources) == len(targets) == len(metadata)):
        raise ValueError("test src/tgt/meta row counts differ")

    limits = {name: 800 for name in BUCKETS}
    limits["clean"] = limits["combined"] = 1400
    rngs = {name: random.Random(f"{args.seed}:{name}") for name in BUCKETS}
    reservoirs: dict[str, list[tuple[int, str, str, dict]]] = defaultdict(list)
    seen: Counter[str] = Counter()
    available: Counter[str] = Counter()
    for index, (source, target, meta) in enumerate(zip(sources, targets, metadata)):
        name = bucket(str(meta.get("error_type", "unknown")))
        if name not in limits:
            continue
        available[name] += 1
        seen[name] += 1
        row = (index, normalize(source), normalize(target), meta)
        current = reservoirs[name]
        if len(current) < limits[name]:
            current.append(row)
        else:
            replacement = rngs[name].randrange(seen[name])
            if replacement < limits[name]:
                current[replacement] = row

    missing = {name: limits[name] - len(reservoirs[name]) for name in BUCKETS if len(reservoirs[name]) < limits[name]}
    if missing:
        raise ValueError(f"insufficient rows for requested buckets: {missing}; available={dict(available)}")

    rows: list[dict] = []
    for name in BUCKETS:
        for index, source, target, meta in reservoirs[name]:
            item = dict(meta)
            item.update({
                "query_id": stable_id("test", index, source, target),
                "input": source,
                "expected": target,
                "error_type": name,
                "original_error_type": meta.get("error_type", "unknown"),
                "split": "test",
                "row_index": index,
            })
            rows.append(item)
    random.Random(args.seed).shuffle(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "purpose": "ReparoS 10K stratified diagnostic; not a training dataset",
        "source": str(root.resolve()),
        "seed": args.seed,
        "rows": len(rows),
        "requested_distribution": limits,
        "actual_distribution": dict(sorted(Counter(row["error_type"] for row in rows).items())),
        "available_distribution": dict(sorted(available.items())),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
