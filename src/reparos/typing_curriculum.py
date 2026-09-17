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
from typing import Iterable, Iterator, Mapping, Sequence

from reparos.data import _source_query_group


SCHEMA_VERSION = "reparos-typing-curriculum/v1"
SPLITS = ("train", "validation", "test")

# Deliberately conservative. Ambiguous one-letter forms are expanded only when
# an address-shaped context makes the meaning explicit.
ABBREVIATIONS: tuple[tuple[str, str], ...] = (
    ("thành phố", "tp"),
    ("thành phố hồ chí minh", "tphcm"),
    ("hà nội", "hn"),
    ("bệnh viện", "bv"),
    ("đại học", "đh"),
    ("trường đại học", "trường đh"),
    ("trung học phổ thông", "thpt"),
    ("trung học cơ sở", "thcs"),
    ("ủy ban nhân dân", "ubnd"),
    ("khu công nghiệp", "kcn"),
    ("chung cư", "cc"),
    ("quốc lộ", "ql"),
    ("tỉnh lộ", "tl"),
    ("đường", "đg"),
    ("quận", "q"),
    ("phường", "p"),
    ("huyện", "h"),
    ("xã", "x"),
)

TELEX_SHAPE = re.compile(r"[^\W\d_]+", re.UNICODE)
VNI_SHAPE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class CurriculumConfig:
    seed: int = 2026
    ime_variants_per_query: int = 2
    abbreviation_variants_per_query: int = 2
    mixed_variants_per_query: int = 2
    stage1_weights: tuple[int, int] = (70, 30)  # IME, clean/hard-negative
    stage2_weights: tuple[int, int, int] = (60, 20, 20)  # abbrev, IME, clean
    stage3_weights: tuple[int, int, int, int] = (50, 15, 15, 20)
    stage4_weights: tuple[int, int, int, int, int] = (60, 15, 10, 5, 10)


@dataclass(frozen=True)
class Example:
    source: str
    target: str
    group_id: str
    split: str
    error_type: str
    source_family: str
    clean: bool = False
    source_group_id: str = ""


def _stable_seed(seed: int, *parts: object) -> int:
    raw = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _strip_marks(text: str) -> str:
    value = unicodedata.normalize("NFD", text)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", value).replace("đ", "d").replace("Đ", "D")


_TELEX_BASE = str.maketrans({
    "ă": "aw", "â": "aa", "ê": "ee", "ô": "oo", "ơ": "ow", "ư": "uw", "đ": "dd",
    "Ă": "Aw", "Â": "Aa", "Ê": "Ee", "Ô": "Oo", "Ơ": "Ow", "Ư": "Uw", "Đ": "Dd",
})
_VNI_BASE = str.maketrans({
    "ă": "a8", "â": "a6", "ê": "e6", "ô": "o6", "ơ": "o7", "ư": "u7", "đ": "d9",
    "Ă": "A8", "Â": "A6", "Ê": "E6", "Ô": "O6", "Ơ": "O7", "Ư": "U7", "Đ": "D9",
})
_TONE = {
    "á": "s", "à": "f", "ả": "r", "ã": "x", "ạ": "j",
    "é": "s", "è": "f", "ẻ": "r", "ẽ": "x", "ẹ": "j",
    "í": "s", "ì": "f", "ỉ": "r", "ĩ": "x", "ị": "j",
    "ó": "s", "ò": "f", "ỏ": "r", "õ": "x", "ọ": "j",
    "ú": "s", "ù": "f", "ủ": "r", "ũ": "x", "ụ": "j",
    "ý": "s", "ỳ": "f", "ỷ": "r", "ỹ": "x", "ỵ": "j",
}
_TONE_VNI = {"s": "1", "f": "2", "r": "3", "x": "4", "j": "5"}


def _encode_word(word: str, method: str) -> str:
    normalized = unicodedata.normalize("NFC", word)
    tone = ""
    chars: list[str] = []
    for char in normalized:
        decomposed = unicodedata.normalize("NFD", char)
        base = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        marks = [ch for ch in decomposed if unicodedata.category(ch) == "Mn"]
        tone_mark = next((m for m in marks if m in "\u0301\u0300\u0309\u0303\u0323"), None)
        if tone_mark:
            tone_map = {"\u0301": "s", "\u0300": "f", "\u0309": "r", "\u0303": "x", "\u0323": "j"}
            tone = tone_map[tone_mark]
        shape_marks = [m for m in marks if m not in "\u0301\u0300\u0309\u0303\u0323"]
        shaped = unicodedata.normalize("NFC", base + "".join(shape_marks))
        table = _TELEX_BASE if method == "telex" else _VNI_BASE
        chars.append(shaped.translate(table))
    encoded = "".join(chars)
    # Common Unikey Telex shares one ``w`` across the ươ vowel cluster:
    # ``trường`` is normally typed ``truowngf``, not the verbose
    # character-by-character form ``truwowngf``.
    if method == "telex":
        encoded = encoded.replace("uwow", "uow").replace("UwOw", "Uow")
    suffix = tone if method == "telex" else _TONE_VNI.get(tone, "")
    return encoded + suffix


def encode_ime(text: str, method: str, rng: random.Random, malformed: bool = False) -> str:
    pattern = TELEX_SHAPE if method == "telex" else VNI_SHAPE
    encoded = pattern.sub(lambda match: _encode_word(match.group(0), method), text)
    if malformed and encoded:
        positions = [i for i, char in enumerate(encoded) if char.isalnum()]
        if positions:
            pos = rng.choice(positions)
            operation = rng.choice(("duplicate", "transpose", "drop"))
            if operation == "duplicate":
                encoded = encoded[:pos] + encoded[pos] + encoded[pos:]
            elif operation == "drop" and len(positions) > 1:
                encoded = encoded[:pos] + encoded[pos + 1:]
            elif pos + 1 < len(encoded):
                encoded = encoded[:pos] + encoded[pos + 1] + encoded[pos] + encoded[pos + 2:]
    return encoded


def abbreviate(text: str, rng: random.Random) -> str:
    candidates: list[tuple[int, int, str]] = []
    lowered = text.casefold()
    for expanded, short in ABBREVIATIONS:
        start = 0
        needle = expanded.casefold()
        while True:
            index = lowered.find(needle, start)
            if index < 0:
                break
            left_ok = index == 0 or not lowered[index - 1].isalnum()
            end = index + len(needle)
            right_ok = end == len(lowered) or not lowered[end].isalnum()
            if left_ok and right_ok:
                candidates.append((index, end, short))
            start = index + 1
    if not candidates:
        return text
    index, end, short = rng.choice(candidates)
    punctuation = rng.choice(("", "", "."))
    return text[:index] + short + punctuation + text[end:]


def _keyboard_edit(text: str, rng: random.Random) -> str:
    positions = [i for i, char in enumerate(text) if char.isalpha()]
    if not positions:
        return text
    pos = rng.choice(positions)
    op = rng.choice(("drop", "duplicate", "transpose"))
    if op == "drop":
        return text[:pos] + text[pos + 1:]
    if op == "duplicate":
        return text[:pos] + text[pos] + text[pos:]
    if pos + 1 < len(text) and text[pos + 1].isalpha():
        return text[:pos] + text[pos + 1] + text[pos] + text[pos + 2:]
    return text[:pos] + text[pos] + text[pos:]


def _boundary_edit(text: str, rng: random.Random) -> str:
    spaces = [i for i, char in enumerate(text) if char == " "]
    if spaces:
        pos = rng.choice(spaces)
        return text[:pos] + text[pos + 1:]
    words = text.split()
    candidates = [(i, word) for i, word in enumerate(words) if len(word) >= 6]
    if not candidates:
        return text
    index, word = rng.choice(candidates)
    cut = rng.randrange(2, len(word) - 1)
    words[index] = word[:cut] + " " + word[cut:]
    return " ".join(words)


def _read_source(source_root: Path) -> Iterator[Example]:
    for split in SPLITS:
        path = source_root / split / "noisy_pairs.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                target = row.get("correct_query", "").strip()
                if not target:
                    continue
                source_group_id = (row.get("group_id") or row.get("entity_id") or f"{split}-{index}").strip()
                # prepared-v4-leakfree is split by normalized correct_query,
                # not by the OSM POI/canonical group. Alias and canonical
                # forms may share a source group across different splits.
                query_group = _source_query_group(target)
                if not query_group:
                    continue
                yield Example(
                    target, target, query_group, split, "clean", "source-clean",
                    True, source_group_id,
                )


def _unique_clean(rows: Iterable[Example]) -> dict[str, list[Example]]:
    seen: set[tuple[str, str]] = set()
    result = {split: [] for split in SPLITS}
    group_splits: dict[str, str] = {}
    for row in rows:
        previous = group_splits.setdefault(row.group_id, row.split)
        if previous != row.split:
            raise ValueError(f"group leakage in source: {row.group_id!r} occurs in {previous} and {row.split}")
        key = (row.split, row.target)
        if key not in seen:
            seen.add(key)
            result[row.split].append(row)
    return result


def _stage_pools(clean: Mapping[str, Sequence[Example]], config: CurriculumConfig) -> dict[str, dict[str, list[Example]]]:
    stages: dict[str, dict[str, list[Example]]] = {}
    for split, rows in clean.items():
        pools: dict[str, list[Example]] = defaultdict(list)
        for index, row in enumerate(rows):
            pools["clean"].append(row)
            for variant in range(config.ime_variants_per_query):
                method = "telex" if variant % 2 == 0 else "vni"
                rng = random.Random(_stable_seed(config.seed, split, row.group_id, "ime", variant))
                source = encode_ime(row.target, method, rng, malformed=variant >= 2)
                if source != row.target:
                    pools["ime"].append(Example(source, row.target, row.group_id, split, f"{method}_ime", "ime", source_group_id=row.source_group_id))
            for variant in range(config.abbreviation_variants_per_query):
                rng = random.Random(_stable_seed(config.seed, split, row.group_id, "abbreviation", variant))
                source = abbreviate(row.target, rng)
                if source != row.target:
                    pools["abbreviation"].append(Example(source, row.target, row.group_id, split, "address_abbreviation", "abbreviation", source_group_id=row.source_group_id))
            for variant in range(config.mixed_variants_per_query):
                rng = random.Random(_stable_seed(config.seed, split, row.group_id, "mixed", variant))
                source = row.target
                kinds: list[str] = []
                operations = ["ime", "abbreviation", "keyboard", "boundary"]
                rng.shuffle(operations)
                for operation in operations[:rng.randint(2, 3)]:
                    previous = source
                    if operation == "ime":
                        method = rng.choice(("telex", "vni"))
                        source = encode_ime(source, method, rng, malformed=rng.random() < 0.25)
                    elif operation == "abbreviation":
                        source = abbreviate(source, rng)
                    elif operation == "keyboard":
                        source = _keyboard_edit(source, rng)
                    else:
                        source = _boundary_edit(source, rng)
                    if source != previous:
                        kinds.append(operation)
                if source != row.target:
                    pools["mixed"].append(Example(source, row.target, row.group_id, split, "combined:" + "+".join(kinds), "mixed", source_group_id=row.source_group_id))
        stages[split] = dict(pools)
    return stages


def _sample(pool: Sequence[Example], count: int, seed: int, label: str) -> list[Example]:
    if count <= 0 or not pool:
        return []
    rng = random.Random(_stable_seed(seed, label))
    order = list(pool)
    rng.shuffle(order)
    if count <= len(order):
        return order[:count]
    return [order[index % len(order)] for index in range(count)]


def _mix(pools: Mapping[str, Sequence[Example]], weights: Mapping[str, int], seed: int, label: str) -> list[Example]:
    active = {name: weight for name, weight in weights.items() if weight > 0 and pools.get(name)}
    if not active:
        return []
    anchor = max(len(pools[name]) / weight for name, weight in active.items())
    result: list[Example] = []
    for name, weight in active.items():
        result.extend(_sample(pools[name], max(1, round(anchor * weight)), seed, f"{label}:{name}"))
    random.Random(_stable_seed(seed, label, "shuffle")).shuffle(result)
    return result


def _write_stage(root: Path, stage: str, split: str, rows: Sequence[Example]) -> dict[str, object]:
    folder = root / stage
    folder.mkdir(parents=True, exist_ok=True)
    paths = {suffix: folder / f"{split}.{suffix}" for suffix in ("src", "tgt", "meta.jsonl")}
    with paths["src"].open("w", encoding="utf-8", newline="\n") as src, paths["tgt"].open("w", encoding="utf-8", newline="\n") as tgt, paths["meta.jsonl"].open("w", encoding="utf-8", newline="\n") as meta:
        for index, row in enumerate(rows):
            src.write(row.source + "\n")
            tgt.write(row.target + "\n")
            meta.write(json.dumps({
                "example_id": f"{stage}:{split}:{index}", "group_id": row.group_id,
                "query_group": row.group_id, "source_group_id": row.source_group_id,
                "split": split, "stage": stage, "error_type": row.error_type,
                "source_family": row.source_family, "is_clean": row.clean,
            }, ensure_ascii=False) + "\n")
    return {
        "rows": len(rows),
        "clean_rows": sum(row.clean for row in rows),
        "source_families": dict(sorted(Counter(row.source_family for row in rows).items())),
        "error_types": dict(sorted(Counter(row.error_type for row in rows).items())),
    }


def prepare_typing_curriculum(source_root: str | Path, output_root: str | Path, config: CurriculumConfig | None = None) -> dict[str, object]:
    settings = config or CurriculumConfig()
    source = Path(source_root)
    output = Path(output_root)
    clean = _unique_clean(_read_source(source))
    generated = _stage_pools(clean, settings)
    stage_weights = {
        "stage1-ime": dict(zip(("ime", "clean"), settings.stage1_weights)),
        "stage2-abbreviation": dict(zip(("abbreviation", "ime", "clean"), settings.stage2_weights)),
        "stage3-mixed": dict(zip(("mixed", "ime", "abbreviation", "clean"), settings.stage3_weights)),
        # Until real logs exist, domain means the source POI distribution plus replay.
        "stage4-domain": dict(zip(("domain", "mixed", "ime", "abbreviation", "clean"), settings.stage4_weights)),
    }
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(source.resolve()),
        "config": asdict(settings),
        "stages": {},
        "leakage": {"train_validation": 0, "train_test": 0, "validation_test": 0},
        "notes": [
            "No training was started.",
            "Stage 4 uses canonical source-domain queries until reviewed real logs are supplied.",
            "Test files are diagnostic only and must never be replayed into a later stage.",
        ],
    }
    for stage, weights in stage_weights.items():
        stage_summary: dict[str, object] = {"weights": weights, "splits": {}}
        for split in SPLITS:
            pools = dict(generated[split])
            pools["domain"] = clean[split]
            rows = _mix(pools, weights, settings.seed, f"{stage}:{split}")
            stage_summary["splits"][split] = _write_stage(output, stage, split, rows)
        manifest["stages"][stage] = stage_summary
    output.mkdir(parents=True, exist_ok=True)
    with (output / "curriculum-manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest
