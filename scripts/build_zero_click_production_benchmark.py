from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz, process


QUOTAS = {
    "head_frequent": 1000,
    "random_tail": 1000,
    "exact_osm_keep": 750,
    "plain_osm_candidate": 750,
    "short_ambiguous": 500,
    "number_address": 500,
    "fuzzy_osm_candidate": 500,
}
PHONE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){8,10}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def plain(text: str) -> str:
    value = unicodedata.normalize("NFD", normalize(text))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", value).replace("đ", "d")


def safe_query(text: str) -> bool:
    return bool(text and len(text) <= 100 and not PHONE.search(text) and not EMAIL.search(text))


def stable_id(query: str) -> str:
    return "zero-click:" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:20]


def load_zero_click(path: Path) -> tuple[Counter[str], dict[str, str]]:
    counts: Counter[str] = Counter()
    representative: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            raw = unicodedata.normalize("NFC", row["keyword"].strip())
            key = normalize(raw)
            if not safe_query(raw) or not key:
                continue
            counts[key] += 1
            representative.setdefault(key, raw)
    return counts, representative


def load_osm(root: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    exact: dict[str, dict[str, dict]] = defaultdict(dict)
    for split in ("train", "validation", "test"):
        path = root / split / "noisy_pairs.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                target = unicodedata.normalize("NFC", row["correct_query"].strip())
                key = normalize(target)
                if not safe_query(target) or not key:
                    continue
                exact[key].setdefault(target, {
                    "text": target,
                    "entity_id": row.get("entity_id"),
                    "group_id": row.get("group_id"),
                    "source_split": split,
                })
    exact_rows = {key: list(items.values()) for key, items in exact.items()}
    plain_rows: dict[str, list[dict]] = defaultdict(list)
    for key, items in exact_rows.items():
        for item in items:
            plain_rows[plain(key)].append(item)
    return exact_rows, plain_rows


def record(
    key: str, raw: str, frequency: int, stratum: str,
    *, action: str | None = None, expected: str | None = None,
    candidates: list[dict] | None = None, notes: str = "",
) -> dict:
    return {
        "query_id": stable_id(key),
        "input": raw,
        "normalized_input": key,
        "frequency": frequency,
        "stratum": stratum,
        "action": action,
        "expected": expected,
        "error_types": [],
        "intent_type": None,
        "candidate_entity_id": None,
        "annotator_confidence": None,
        "contains_pii": False,
        "review_status": "prefilled_keep" if action == "keep" else "pending",
        "candidates": candidates or [],
        "notes": notes,
    }


def choose(rows: list[dict], quota: int, used: set[str]) -> list[dict]:
    selected = []
    for item in rows:
        key = item["normalized_input"]
        if key in used:
            continue
        selected.append(item)
        used.add(key)
        if len(selected) == quota:
            break
    if len(selected) != quota:
        raise RuntimeError(f"quota not met: wanted {quota}, got {len(selected)}")
    return selected


def build(args: argparse.Namespace) -> None:
    counts, representatives = load_zero_click(Path(args.zero_click))
    exact_osm, plain_osm = load_osm(Path(args.osm))
    used: set[str] = set()
    strata: dict[str, list[dict]] = {}

    exact_rows = []
    for key, frequency in counts.items():
        if key not in exact_osm:
            continue
        raw = representatives[key]
        exact_rows.append(record(
            key, raw, frequency, "exact_osm_keep",
            action="keep", expected=raw,
            candidates=exact_osm[key][:5],
            notes="Exact normalized OSM match; prefilled keep, still requires reviewer confirmation.",
        ))
    exact_rows.sort(key=lambda item: (-item["frequency"], item["normalized_input"]))
    strata["exact_osm_keep"] = choose(exact_rows, QUOTAS["exact_osm_keep"], used)

    plain_rows = []
    for key, frequency in counts.items():
        if key in exact_osm:
            continue
        candidates = plain_osm.get(plain(key), [])
        distinct = {normalize(item["text"]): item for item in candidates}
        if len(distinct) != 1:
            continue
        candidate = next(iter(distinct.values()))
        plain_rows.append(record(
            key, representatives[key], frequency, "plain_osm_candidate",
            candidates=[candidate],
            notes="Unique match after removing diacritics; candidate is not an automatic label.",
        ))
    plain_rows.sort(key=lambda item: (-item["frequency"], item["normalized_input"]))
    strata["plain_osm_candidate"] = choose(plain_rows, QUOTAS["plain_osm_candidate"], used)

    short_rows = []
    for key, frequency in counts.items():
        raw = representatives[key]
        if len(key) <= 4 or (len(key.split()) == 1 and len(key) <= 6):
            short_rows.append(record(key, raw, frequency, "short_ambiguous"))
    short_rows.sort(key=lambda item: (-item["frequency"], item["normalized_input"]))
    strata["short_ambiguous"] = choose(short_rows, QUOTAS["short_ambiguous"], used)

    number_rows = [
        record(key, representatives[key], frequency, "number_address")
        for key, frequency in counts.items() if any(char.isdigit() for char in key)
    ]
    number_rows.sort(key=lambda item: (-item["frequency"], item["normalized_input"]))
    strata["number_address"] = choose(number_rows, QUOTAS["number_address"], used)

    # Candidate pools by first plain character and approximate length keep fuzzy
    # retrieval tractable while allowing errors after the first character.
    pools: dict[tuple[str, int], list[str]] = defaultdict(list)
    target_meta: dict[str, dict] = {}
    for items in exact_osm.values():
        for item in items:
            target = normalize(item["text"])
            target_meta.setdefault(target, item)
            value = plain(target)
            if value:
                pools[(value[:3], len(value) // 5)].append(target)

    fuzzy_rows = []
    # The frequent 30K unmatched queries are sufficient for a 500-row
    # high-confidence stratum and keep runtime bounded.
    fuzzy_source = sorted(counts, key=lambda key: (-counts[key], key))[:5_000]
    for key in fuzzy_source:
        if key in used or key in exact_osm or plain(key) in plain_osm or len(key) < 5:
            continue
        value = plain(key)
        length_bucket = len(value) // 5
        choices = []
        for bucket in range(max(0, length_bucket - 1), length_bucket + 2):
            choices.extend(pools.get((value[:3], bucket), ()))
        if not choices:
            continue
        choices = list(dict.fromkeys(choices))
        matches = process.extract(value, choices, scorer=fuzz.WRatio, limit=2, score_cutoff=80)
        if not matches:
            continue
        top_score = float(matches[0][1])
        second_score = float(matches[1][1]) if len(matches) > 1 else 0.0
        if top_score < 85 or top_score - second_score < 3:
            continue
        candidates = []
        for target, score, _ in matches:
            item = dict(target_meta[target])
            item["score"] = float(score)
            candidates.append(item)
        fuzzy_rows.append(record(
            key, representatives[key], counts[key], "fuzzy_osm_candidate",
            candidates=candidates,
            notes=f"High-confidence lexical candidate; top margin={top_score-second_score:.1f}; human review required.",
        ))
        if len(fuzzy_rows) >= QUOTAS["fuzzy_osm_candidate"] * 3:
            break
    fuzzy_rows.sort(key=lambda item: (-item["frequency"], -item["candidates"][0]["score"], item["normalized_input"]))
    strata["fuzzy_osm_candidate"] = choose(fuzzy_rows, QUOTAS["fuzzy_osm_candidate"], used)

    head_rows = [
        record(key, representatives[key], frequency, "head_frequent")
        for key, frequency in counts.items()
    ]
    head_rows.sort(key=lambda item: (-item["frequency"], item["normalized_input"]))
    strata["head_frequent"] = choose(head_rows, QUOTAS["head_frequent"], used)

    rng = random.Random(args.seed)
    tail_keys = [key for key, frequency in counts.items() if frequency == 1 and key not in used]
    rng.shuffle(tail_keys)
    tail_rows = [record(key, representatives[key], 1, "random_tail") for key in tail_keys]
    strata["random_tail"] = choose(tail_rows, QUOTAS["random_tail"], used)

    rows = [item for name in QUOTAS for item in strata[name]]
    assert len(rows) == sum(QUOTAS.values()) == 5000
    assert len({item["normalized_input"] for item in rows}) == len(rows)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    queue = output / "annotation-queue.jsonl"
    with queue.open("w", encoding="utf-8", newline="\n") as stream:
        for item in rows:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    review_csv = output / "annotation-queue.csv"
    fields = [
        "query_id", "input", "frequency", "stratum", "action", "expected",
        "error_types", "intent_type", "candidate_entity_id",
        "annotator_confidence", "contains_pii", "review_status", "notes", "candidates_json",
    ]
    with review_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            flat = {key: item.get(key) for key in fields}
            flat["error_types"] = "|".join(item["error_types"])
            flat["candidates_json"] = json.dumps(item["candidates"], ensure_ascii=False)
            writer.writerow(flat)
    manifest = {
        "schema_version": 1,
        "profile": "zero-click-production-benchmark/annotation-queue-v1",
        "seed": args.seed,
        "rows": len(rows),
        "unique_normalized_queries": len(used),
        "quotas": QUOTAS,
        "actual": {name: len(items) for name, items in strata.items()},
        "privacy": {
            "coordinates_exported": False,
            "phone_email_filtered": True,
            "maximum_query_characters": 100,
        },
        "label_policy": {
            "prefilled": {"exact_osm_keep": "keep; reviewer confirmation required"},
            "candidate_only": ["plain_osm_candidate", "fuzzy_osm_candidate"],
            "benchmark_ready": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-click", default="data/zero_click.csv")
    parser.add_argument("--osm", default="data/osm/prepared-v4-leakfree")
    parser.add_argument("--output", default="benchmark/zero-click-production-5k")
    parser.add_argument("--seed", type=int, default=2026)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
