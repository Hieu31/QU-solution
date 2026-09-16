from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Iterable


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _stable_id(split: str, index: int, source: str, target: str) -> str:
    digest = hashlib.sha256(f"{split}\0{index}\0{source}\0{target}".encode()).hexdigest()[:16]
    return f"{split}:{index:09d}:{digest}"


def load_parallel_dataset(root: str | Path, split: str = "test", limit: int = 0) -> list[dict]:
    base = Path(root)
    sources = (base / f"{split}.src").read_text(encoding="utf-8").splitlines()
    targets = (base / f"{split}.tgt").read_text(encoding="utf-8").splitlines()
    meta_path = base / f"{split}.meta.jsonl"
    metadata = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()]
    if not (len(sources) == len(targets) == len(metadata)):
        raise ValueError("src, tgt and metadata row counts differ")
    size = len(sources) if limit <= 0 else min(limit, len(sources))
    rows = []
    for index in range(size):
        source, target = normalize_text(sources[index]), normalize_text(targets[index])
        item = dict(metadata[index])
        item.update({
            "query_id": _stable_id(split, index, source, target),
            "input": source, "expected": target,
            "error_type": item.get("error_type", "unknown"),
            "split": split, "row_index": index,
        })
        rows.append(item)
    return rows


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    seen: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = str(row.get("query_id", ""))
        if not query_id:
            raise ValueError(f"missing query_id at {path}:{number}")
        if query_id in seen:
            raise ValueError(f"duplicate query_id {query_id!r} in {path}")
        seen.add(query_id)
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return destination


def strict_join(gold: list[dict], predictions: list[dict]) -> list[tuple[dict, dict]]:
    expected = {row["query_id"]: row for row in gold}
    actual = {row["query_id"]: row for row in predictions}
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    if missing or extra:
        raise ValueError(f"prediction join mismatch: missing={missing[:5]}, extra={extra[:5]}")
    joined = []
    for gold_row in gold:
        prediction = actual[gold_row["query_id"]]
        for field in ("input", "expected", "error_type"):
            if field in prediction and normalize_text(str(prediction[field])) != normalize_text(str(gold_row[field])):
                raise ValueError(f"gold field {field!r} changed for {gold_row['query_id']}")
        joined.append((gold_row, prediction))
    return joined
