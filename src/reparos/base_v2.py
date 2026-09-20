from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from reparos.data import _source_query_group
from reparos.typing_curriculum import abbreviate, encode_ime


SCHEMA_VERSION = "reparos-base-v2/v1"
SPLITS = ("train", "validation", "test")
PROFILES = (8, 16, 32)


@dataclass(frozen=True)
class BaseV2Config:
    seed: int = 2026
    profiles: tuple[int, ...] = PROFILES
    max_groups_per_split: int | None = None
    materialize: bool = False
    sample_per_family: int = 20


@dataclass(frozen=True)
class CleanSeed:
    text: str
    query_group: str
    source_group_id: str
    split: str


@dataclass(frozen=True)
class Variant:
    source: str
    target: str
    query_group: str
    source_group_id: str
    split: str
    rank: int
    family: str
    operations: tuple[str, ...]


def _stable_seed(seed: int, *parts: object) -> int:
    raw = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _plain(text: str) -> str:
    value = unicodedata.normalize("NFD", text)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", value).replace("đ", "d").replace("Đ", "D")


def _partial_plain(text: str, rng: random.Random) -> str:
    words = text.split()
    eligible = [i for i, word in enumerate(words) if _plain(word) != word]
    if not eligible:
        return text
    count = min(len(eligible), 1 if len(eligible) == 1 else rng.randint(1, len(eligible) - 1))
    for index in rng.sample(eligible, count):
        words[index] = _plain(words[index])
    return " ".join(words)


def _wrong_diacritic(text: str, rng: random.Random) -> str:
    tone_marks = "\u0301\u0300\u0309\u0303\u0323"
    chars = list(unicodedata.normalize("NFD", text))
    vowels = [i for i, ch in enumerate(chars) if ch.casefold() in "aeiouy"]
    if not vowels:
        return text
    index = rng.choice(vowels)
    end = index + 1
    while end < len(chars) and unicodedata.category(chars[end]) == "Mn":
        end += 1
    current = next((ch for ch in chars[index + 1 : end] if ch in tone_marks), None)
    replacement = rng.choice([mark for mark in tone_marks if mark != current])
    marks = [ch for ch in chars[index + 1 : end] if ch not in tone_marks]
    chars[index + 1 : end] = marks + [replacement]
    return unicodedata.normalize("NFC", "".join(chars))


def _keyboard(text: str, rng: random.Random, operation: str) -> str:
    positions = [i for i, ch in enumerate(text) if ch.isalpha()]
    if not positions:
        return text
    pos = rng.choice(positions)
    if operation == "drop":
        return text[:pos] + text[pos + 1 :]
    if operation == "duplicate":
        return text[:pos] + text[pos] + text[pos:]
    if operation == "transpose":
        adjacent = [i for i in (pos - 1, pos + 1) if 0 <= i < len(text) and text[i].isalpha()]
        if not adjacent:
            return text
        other = rng.choice(adjacent)
        chars = list(text)
        chars[pos], chars[other] = chars[other], chars[pos]
        return "".join(chars)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if operation == "insert":
        return text[:pos] + rng.choice(alphabet) + text[pos:]
    return text


def _join_boundary(text: str, rng: random.Random, count: int = 1, all_spaces: bool = False) -> str:
    if all_spaces:
        return text.replace(" ", "")
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if not spaces:
        return text
    selected = set(rng.sample(spaces, min(count, len(spaces))))
    return "".join(ch for i, ch in enumerate(text) if i not in selected)


def _wrong_split(text: str, rng: random.Random) -> str:
    words = text.split()
    eligible = [i for i, word in enumerate(words) if len(word) >= 6]
    if not eligible:
        return text
    index = rng.choice(eligible)
    word = words[index]
    cut = rng.randint(2, len(word) - 2)
    words[index : index + 1] = (word[:cut], word[cut:])
    return " ".join(words)


def _address_symbol(text: str, rng: random.Random) -> str:
    match = re.search(r"\b(\d{1,4})[ /-](\d{1,4}[A-Za-z]?)\b", text)
    if match:
        separator = rng.choice(("/", "-", "", " "))
        return text[: match.start()] + match.group(1) + separator + match.group(2) + text[match.end() :]
    match = re.search(r"\b(\d+)([A-Za-z])\b", text)
    if match:
        return text[: match.start()] + match.group(1) + " " + match.group(2) + text[match.end() :]
    return text


def _apply(text: str, operations: Iterable[str], rng: random.Random) -> str:
    value = text
    for operation in operations:
        if operation == "missing_full":
            value = _plain(value)
        elif operation == "missing_partial":
            value = _partial_plain(value, rng)
        elif operation == "wrong_diacritic":
            value = _wrong_diacritic(value, rng)
        elif operation in {"keyboard_drop", "keyboard_duplicate", "keyboard_transpose", "keyboard_insert"}:
            value = _keyboard(value, rng, operation.removeprefix("keyboard_"))
        elif operation == "boundary_one":
            value = _join_boundary(value, rng)
        elif operation == "boundary_multi":
            value = _join_boundary(value, rng, count=2)
        elif operation == "boundary_all":
            value = _join_boundary(value, rng, all_spaces=True)
        elif operation == "wrong_split":
            value = _wrong_split(value, rng)
        elif operation == "telex":
            value = encode_ime(value, "telex", rng)
        elif operation == "telex_malformed":
            value = encode_ime(value, "telex", rng, malformed=True)
        elif operation == "vni":
            value = encode_ime(value, "vni", rng)
        elif operation == "vni_malformed":
            value = encode_ime(value, "vni", rng, malformed=True)
        elif operation == "abbreviation":
            value = abbreviate(value, rng)
        elif operation == "address_symbol":
            value = _address_symbol(value, rng)
        else:
            raise ValueError(f"unknown operation: {operation}")
    return _normalize(value)


# Rank order is stable: V2-8 is the prefix of V2-16, which is the prefix of V2-32.
SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("missing_diacritics_full", ("missing_full",)),
    ("missing_diacritics_partial", ("missing_partial",)),
    ("telex", ("telex",)),
    ("vni", ("vni",)),
    ("keyboard_drop", ("keyboard_drop",)),
    ("word_boundary", ("boundary_one",)),
    ("diacritics_boundary", ("missing_full", "boundary_one")),
    ("keyboard_boundary", ("keyboard_drop", "boundary_one")),
    ("wrong_diacritic", ("wrong_diacritic",)),
    ("telex_malformed", ("telex_malformed",)),
    ("vni_malformed", ("vni_malformed",)),
    ("keyboard_duplicate", ("keyboard_duplicate",)),
    ("keyboard_transpose", ("keyboard_transpose",)),
    ("boundary_multi", ("boundary_multi",)),
    ("boundary_all", ("boundary_all",)),
    ("diacritics_boundary_all", ("missing_full", "boundary_all")),
    ("telex_boundary", ("telex", "boundary_one")),
    ("vni_boundary", ("vni", "boundary_one")),
    ("keyboard_diacritics", ("keyboard_drop", "missing_full")),
    ("wrong_diacritic_boundary", ("wrong_diacritic", "boundary_one")),
    ("address_abbreviation", ("abbreviation",)),
    ("abbreviation_boundary", ("abbreviation", "boundary_one")),
    ("abbreviation_diacritics", ("abbreviation", "missing_partial")),
    ("address_symbol", ("address_symbol",)),
    ("keyboard_insert", ("keyboard_insert",)),
    ("partial_diacritics_boundary", ("missing_partial", "boundary_one")),
    ("telex_malformed_boundary", ("telex_malformed", "boundary_one")),
    ("vni_malformed_boundary", ("vni_malformed", "boundary_one")),
    ("three_operation_plain", ("missing_full", "boundary_one", "keyboard_drop")),
    ("three_operation_telex", ("telex", "boundary_one", "keyboard_drop")),
    ("wrong_split", ("wrong_split",)),
    ("address_symbol_diacritics", ("address_symbol", "missing_partial")),
)


def generate_variants(seed: CleanSeed, global_seed: int = 2026) -> list[Variant]:
    seen: set[str] = set()
    result: list[Variant] = []
    target_key = _source_query_group(seed.text)
    for rank, (family, operations) in enumerate(SLOTS, start=1):
        rng = random.Random(_stable_seed(global_seed, seed.split, seed.query_group, rank, family))
        source = _apply(seed.text, operations, rng)
        source_key = _source_query_group(source)
        if not source_key or source_key == target_key or source_key in seen:
            continue
        seen.add(source_key)
        result.append(Variant(
            source, seed.text, seed.query_group, seed.source_group_id,
            seed.split, rank, family, operations,
        ))
    return result


def _load_clean(source_root: Path, split: str, limit: int | None = None) -> list[CleanSeed]:
    path = source_root / split / "noisy_pairs.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, CleanSeed] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for index, item in enumerate(csv.DictReader(stream)):
            if item.get("noise_source") != "clean" and item.get("error_type") != "clean":
                continue
            text = _normalize(item.get("correct_query", ""))
            query_group = _source_query_group(text)
            if not text or not query_group or query_group in rows:
                continue
            rows[query_group] = CleanSeed(
                text, query_group,
                item.get("group_id") or item.get("entity_id") or f"{split}-{index}",
                split,
            )
            if limit is not None and len(rows) >= limit:
                break
    return list(rows.values())


class _ProfileWriter:
    def __init__(self, root: Path, profile: int, split: str):
        folder = root / f"v2-{profile}" / "base"
        folder.mkdir(parents=True, exist_ok=True)
        self.handles = {}
        for lane in ("noisy", "clean"):
            for suffix in ("src", "tgt", "meta.jsonl"):
                self.handles[(lane, suffix)] = (folder / f"{split}.{lane}.{suffix}").open(
                    "w", encoding="utf-8", newline="\n"
                )

    def write_clean(self, seed: CleanSeed) -> None:
        meta = {
            "split": seed.split, "query_group": seed.query_group,
            "source_group_id": seed.source_group_id, "is_clean": True,
            "family": "clean", "rank": 0, "operations": [],
        }
        self.handles[("clean", "src")].write(seed.text + "\n")
        self.handles[("clean", "tgt")].write(seed.text + "\n")
        self.handles[("clean", "meta.jsonl")].write(json.dumps(meta, ensure_ascii=False) + "\n")

    def write_variant(self, row: Variant) -> None:
        meta = {
            "split": row.split, "query_group": row.query_group,
            "source_group_id": row.source_group_id, "is_clean": False,
            "family": row.family, "rank": row.rank,
            "operations": row.operations,
        }
        self.handles[("noisy", "src")].write(row.source + "\n")
        self.handles[("noisy", "tgt")].write(row.target + "\n")
        self.handles[("noisy", "meta.jsonl")].write(json.dumps(meta, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def prepare_base_v2(source: str | Path, output: str | Path, config: BaseV2Config | None = None) -> dict[str, object]:
    settings = config or BaseV2Config()
    invalid = set(settings.profiles).difference(PROFILES)
    if invalid:
        raise ValueError(f"unsupported profiles: {sorted(invalid)}")
    source_root, output_root = Path(source), Path(output)
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source_root.resolve()),
        "config": asdict(settings),
        "distribution_objective": "capability_balanced",
        "production_distribution_claimed": False,
        "quota_status": "design_hypothesis",
        "lanes": {
            "noisy": "correction pairs; profile prefix selected by rank",
            "clean": "unique identity pairs; weight separately during training",
        },
        "recommended_training_mix": {"noisy": 75, "clean": 25},
        "splits": {},
    }
    split_groups: dict[str, set[str]] = {}
    for split in SPLITS:
        seeds = _load_clean(source_root, split, settings.max_groups_per_split)
        split_groups[split] = {row.query_group for row in seeds}
        profile_counts = {str(profile): Counter() for profile in settings.profiles}
        samples: dict[str, list[dict[str, object]]] = defaultdict(list)
        writers = {
            profile: _ProfileWriter(output_root, profile, split)
            for profile in settings.profiles
        } if settings.materialize else {}
        try:
            for seed in seeds:
                variants = generate_variants(seed, settings.seed)
                for profile in settings.profiles:
                    profile_counts[str(profile)]["clean"] += 1
                    if settings.materialize:
                        writers[profile].write_clean(seed)
                    for row in variants:
                        if row.rank > profile:
                            continue
                        profile_counts[str(profile)]["noisy"] += 1
                        profile_counts[str(profile)][f"family:{row.family}"] += 1
                        if settings.materialize:
                            writers[profile].write_variant(row)
                for row in variants:
                    if len(samples[row.family]) < settings.sample_per_family:
                        samples[row.family].append({
                            "source": row.source, "target": row.target,
                            "rank": row.rank, "operations": row.operations,
                        })
        finally:
            for writer in writers.values():
                writer.close()
        report["splits"][split] = {
            "clean_groups": len(seeds),
            "profiles": {name: dict(sorted(counts.items())) for name, counts in profile_counts.items()},
            "samples": dict(sorted(samples.items())),
        }
    leakage = {
        "train_validation": len(split_groups["train"] & split_groups["validation"]),
        "train_test": len(split_groups["train"] & split_groups["test"]),
        "validation_test": len(split_groups["validation"] & split_groups["test"]),
    }
    report["leakage"] = leakage
    if any(leakage.values()):
        raise ValueError(f"query-group leakage detected: {leakage}")
    output_root.mkdir(parents=True, exist_ok=True)
    name = "base-v2-manifest.json" if settings.materialize else "base-v2-dry-run.json"
    (output_root / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
