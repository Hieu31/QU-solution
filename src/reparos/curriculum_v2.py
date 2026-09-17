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


SCHEMA_VERSION = "reparos-capability-curriculum/v2"
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
GENERIC_WORDS = {
    "duong", "pho", "phuong", "quan", "huyen", "xa", "thi", "tran",
    "thanh", "tinh", "thon", "to", "khu", "ap", "so", "hem", "ngo",
}
CURATED_ACRONYMS = {
    "ltk": "lý thường kiệt",
    "nct": "nguyễn chí thanh",
    "hbt": "hai bà trưng",
    "dbp": "điện biên phủ",
    "pvh": "phạm văn hai",
    "ntmk": "nguyễn thị minh khai",
    "nvl": "nguyễn văn linh",
}
CURATED_QUERY_EXPANSIONS = {
    "thpt clhp": "trung học phổ thông chuyên lê hồng phong",
    "thpt tdn": "trung học phổ thông trần đại nghĩa",
    "thcs nbk": "trung học cơ sở nguyễn bỉnh khiêm",
    "dhbk": "đại học bách khoa",
    "dh bk": "đại học bách khoa",
    "dhqg": "đại học quốc gia",
    "dh qg": "đại học quốc gia",
    "bv bm": "bệnh viện bạch mai",
    "bvbm": "bệnh viện bạch mai",
    "ubnd q1": "ủy ban nhân dân quận 1",
    "ubnd pbn q1": "ủy ban nhân dân phường bến nghé quận 1",
    "kcn tb": "khu công nghiệp tân bình",
}
CURATED_UNIVERSITY_ALIASES = {
    "hust": "đại học bách khoa hà nội",
    "hcmute": "đại học sư phạm kỹ thuật thành phố hồ chí minh",
    "ptit": "học viện công nghệ bưu chính viễn thông",
    "neu": "đại học kinh tế quốc dân",
    "ftu": "đại học ngoại thương",
    "vnu": "đại học quốc gia hà nội",
    "vnuhcm": "đại học quốc gia thành phố hồ chí minh",
    "uet": "đại học công nghệ đại học quốc gia hà nội",
    "uit": "đại học công nghệ thông tin đại học quốc gia thành phố hồ chí minh",
    "hcmus": "đại học khoa học tự nhiên đại học quốc gia thành phố hồ chí minh",
    "hcmut": "đại học bách khoa đại học quốc gia thành phố hồ chí minh",
    "ueh": "đại học kinh tế thành phố hồ chí minh",
    "uel": "đại học kinh tế luật đại học quốc gia thành phố hồ chí minh",
    "tdtu": "đại học tôn đức thắng",
    "iuh": "đại học công nghiệp thành phố hồ chí minh",
    "ufm": "đại học tài chính marketing",
    "hutech": "đại học công nghệ thành phố hồ chí minh",
    "huflit": "đại học ngoại ngữ tin học thành phố hồ chí minh",
    "fptu": "đại học fpt",
    "vinuni": "đại học vinuni",
    "phenikaa": "đại học phenikaa",
}
RESERVED_ABBREVIATIONS = {
    "q", "p", "tp", "bv", "ubnd", "thpt", "thcs", "dh", "kcn", "cc",
    "ql", "tl", "h", "x", "d", "dg", "dt", "tx", "tt", "ct", "cty",
    "tnhh", "hdnd", "cd", "mn", "kdt", "tttm", "bql", "hn", "hcm",
    "sg", "dn", "hp", "bd", "vt", "nt", "gv", "bt", "tb", "ad",
}
ACRONYM_ANCHORS = {
    "duong", "pho", "hem", "ngo", "truong", "benh", "vien", "dai",
    "hoc", "thpt", "thcs", "ubnd", "xa", "phuong", "quan", "huyen",
}


@dataclass(frozen=True)
class Profile:
    name: str
    train_rows: dict[str, int]
    validation_rows: int


PROFILES = {
    "pilot": Profile(
        "pilot",
        {"stage1-primitives": 120_000, "stage2-composition": 160_000, "stage3-lexical": 120_000},
        10_000,
    ),
    "full": Profile(
        "full",
        {"stage1-primitives": 1_000_000, "stage2-composition": 1_200_000, "stage3-lexical": 600_000},
        40_000,
    ),
}

STAGE_WEIGHTS: dict[str, dict[str, int]] = {
    "stage1-primitives": {
        "clean": 30, "missing_diacritics": 30, "boundary": 15,
        "keyboard": 10, "telex": 10, "vni": 5,
    },
    "stage2-composition": {
        "clean": 20, "primitive_replay": 20, "two_operation": 40,
        "three_operation": 20,
    },
    "stage3-lexical": {
        "clean": 25, "acronym": 5, "contextual_abbreviation": 20,
        "lexical_typo": 10, "mixed_replay": 25, "primitive_replay": 10,
        "address_symbol": 5,
    },
}


@dataclass(frozen=True)
class CleanQuery:
    text: str
    query_group: str
    source_group_id: str


@dataclass(frozen=True)
class Row:
    source: str
    target: str
    query_group: str
    source_group_id: str
    family: str
    error_type: str
    operations: tuple[str, ...]


def _seed(seed: int, *parts: object) -> int:
    value = "\0".join((str(seed), *(str(x) for x in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _plain(text: str) -> str:
    value = unicodedata.normalize("NFD", text)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", value).replace("đ", "d").replace("Đ", "D")


def _partial_plain(text: str, rng: random.Random) -> str:
    words = text.split()
    candidates = [i for i, word in enumerate(words) if _plain(word) != word]
    if not candidates:
        return text
    count = max(1, min(len(candidates), rng.randint(1, 2)))
    for index in rng.sample(candidates, count):
        words[index] = _plain(words[index])
    return " ".join(words)


def _join_boundary(text: str, rng: random.Random, all_spaces: bool = False) -> str:
    if all_spaces:
        return text.replace(" ", "")
    spaces = [i for i, char in enumerate(text) if char == " "]
    if not spaces:
        return text
    position = rng.choice(spaces)
    return text[:position] + text[position + 1 :]


def _keyboard(text: str, rng: random.Random) -> str:
    positions = [i for i, char in enumerate(text) if char.isalpha()]
    if not positions:
        return text
    position = rng.choice(positions)
    operation = rng.choice(("drop", "duplicate", "transpose"))
    if operation == "drop":
        return text[:position] + text[position + 1 :]
    if operation == "duplicate":
        return text[:position] + text[position] + text[position:]
    if position + 1 < len(text) and text[position + 1].isalpha():
        return text[:position] + text[position + 1] + text[position] + text[position + 2 :]
    return text


def _address_symbol(text: str, rng: random.Random) -> str:
    match = re.search(r"\b(\d{1,3})[/+-](\d{1,4})\b", text)
    if match:
        return text[: match.start()] + f"{match.group(1)} {match.group(2)}" + text[match.end() :]
    match = re.search(r"\b(\d+)([A-Za-z])\b", text)
    if match:
        return text[: match.start()] + f"{match.group(1)} {match.group(2)}" + text[match.end() :]
    return text


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _acronym(words: Iterable[str]) -> str:
    return "".join(word[0] for word in words if word)


def _acronym_candidates(text: str) -> list[tuple[str, str]]:
    """Return (source, acronym) replacements, preferring name phrases.

    Example: ``đường lý thường kiệt quận 1`` -> ``đường ltk quận 1``.
    Bare ``ltk`` is admitted later only when it maps to one target.
    """
    tokens = text.split()
    results: list[tuple[str, str]] = []
    for length in range(4, 1, -1):
        for start in range(0, len(tokens) - length + 1):
            phrase = tokens[start : start + length]
            if not all(WORD_RE.fullmatch(token) for token in phrase):
                continue
            plain = [_plain(token).casefold() for token in phrase]
            if sum(word not in GENERIC_WORDS for word in plain) < 2:
                continue
            short = _acronym(plain)
            # Two-letter initials collide far too often with Vietnamese
            # administrative/chat abbreviations. They require explicit
            # curation and are never mined automatically.
            if not 3 <= len(short) <= 5:
                continue
            # Automatic acronym mining is permitted only inside an anchored
            # place phrase. Generic sliding n-grams produced collisions such
            # as bv=bùi viện and tp=tân phước.
            context = [_plain(token).casefold() for token in tokens[max(0, start - 5):start]]
            if not any(token in ACRONYM_ANCHORS for token in context):
                continue
            if short in RESERVED_ABBREVIATIONS or short in CURATED_ACRONYMS:
                continue
            source = " ".join(tokens[:start] + [short] + tokens[start + length :])
            if source != text:
                results.append((source, short))
    return results


def _load_clean(source: Path, split: str) -> list[CleanQuery]:
    path = source / split / "noisy_pairs.csv"
    seen: set[str] = set()
    rows: list[CleanQuery] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for index, item in enumerate(csv.DictReader(stream)):
            if item.get("noise_source") != "clean" and item.get("error_type") != "clean":
                continue
            text = " ".join(item["correct_query"].split())
            group = _source_query_group(text)
            if not text or not group or group in seen:
                continue
            seen.add(group)
            rows.append(CleanQuery(text, group, item.get("group_id") or item.get("entity_id") or f"{split}-{index}"))
    return rows


def _row(clean: CleanQuery, source: str, family: str, operations: tuple[str, ...]) -> Row | None:
    source = " ".join(source.split())
    if not source or source == clean.text and family != "clean":
        return None
    return Row(source, clean.text, clean.query_group, clean.source_group_id, family, family, operations)


def _primitive(clean: CleanQuery, family: str, rng: random.Random) -> Row | None:
    if family == "clean":
        return _row(clean, clean.text, family, ())
    if family == "missing_diacritics":
        source = _plain(clean.text) if rng.random() < 0.65 else _partial_plain(clean.text, rng)
    elif family == "boundary":
        source = _join_boundary(clean.text, rng)
    elif family == "keyboard":
        source = _keyboard(clean.text, rng)
    elif family == "telex":
        source = encode_ime(clean.text, "telex", rng, malformed=rng.random() < 0.25)
    elif family == "vni":
        source = encode_ime(clean.text, "vni", rng, malformed=rng.random() < 0.10)
    elif family == "address_symbol":
        source = _address_symbol(clean.text, rng)
    else:
        raise ValueError(f"unknown primitive family: {family}")
    return _row(clean, source, family, (family,))


def _composition(clean: CleanQuery, count: int, rng: random.Random, family: str) -> Row | None:
    operations = ["missing_diacritics", "boundary", "keyboard", "telex"]
    # Telex and plain-diacritic removal are alternatives, not sequentially
    # applied to an already encoded string.
    if count >= 2:
        first = rng.choice(("missing_diacritics", "telex"))
        rest = rng.sample([op for op in operations if op not in {first, "telex" if first == "missing_diacritics" else "missing_diacritics"}], count - 1)
        chosen = [first, *rest]
    else:
        chosen = [rng.choice(operations)]
    source = clean.text
    for operation in chosen:
        if operation == "missing_diacritics":
            source = _plain(source) if rng.random() < 0.7 else _partial_plain(source, rng)
        elif operation == "boundary":
            source = _join_boundary(source, rng, all_spaces=count >= 3 and rng.random() < 0.15)
        elif operation == "keyboard":
            source = _keyboard(source, rng)
        elif operation == "telex":
            source = encode_ime(source, "telex", rng, malformed=rng.random() < 0.3)
    return _row(clean, source, family, tuple(chosen))


def _lexical(clean: CleanQuery, family: str, rng: random.Random) -> Row | None:
    if family == "clean":
        return _row(clean, clean.text, family, ())
    if family == "acronym":
        candidates = [
            (source, short) for source, short in _acronym_candidates(clean.text)
            if source != short
        ]
        # Bare acronyms are unsafe to mine automatically from noisy OSM names.
        # They are admitted only through a reviewed mapping. Automatic
        # acronyms must retain context, e.g. ``đường ltk quận 1``.
        for short, expanded in CURATED_ACRONYMS.items():
            if clean.text == expanded:
                candidates.append((short, short))
        if not candidates:
            return None
        source, short = rng.choice(candidates)
        return _row(clean, source, family, (f"acronym:{short}",))
    if family == "contextual_abbreviation":
        source = clean.text
        for _ in range(rng.randint(1, 3)):
            updated = abbreviate(source, rng)
            if updated == source:
                break
            source = updated
        return _row(clean, source, family, ("abbreviation",))
    if family == "lexical_typo":
        source = _keyboard(clean.text, rng)
        source = _keyboard(source, rng) if rng.random() < 0.35 else source
        return _row(clean, source, family, ("lexical_typo_synthetic",))
    if family == "mixed_replay":
        if rng.random() < 0.60:
            source = clean.text
            for _ in range(rng.randint(1, 3)):
                source = abbreviate(source, rng)
            if source != clean.text:
                operations = ["abbreviation"]
                source = _join_boundary(source, rng)
                operations.append("boundary")
                if rng.random() < 0.5:
                    source = _partial_plain(source, rng)
                    operations.append("missing_diacritics")
                return _row(clean, source, family, tuple(operations))
        return _composition(clean, 2, rng, family)
    if family == "primitive_replay":
        return _primitive(clean, rng.choice(("missing_diacritics", "boundary", "telex", "keyboard")), rng)
    if family == "address_symbol":
        return _primitive(clean, family, rng)
    raise ValueError(f"unknown lexical family: {family}")


def _reservoir(
    clean_rows: list[CleanQuery], family: str, count: int, seed: int,
    factory: Callable[[CleanQuery, random.Random], Row | None],
) -> list[Row]:
    result: list[Row] = []
    seen = 0
    # Multiple deterministic passes permit capped oversampling without
    # materializing every possible corruption in memory.
    passes = max(20, (count // max(len(clean_rows), 1)) + 2)
    for pass_index in range(passes):
        rng = random.Random(_seed(seed, family, pass_index))
        for clean in clean_rows:
            row = factory(clean, rng)
            if row is None:
                continue
            seen += 1
            if len(result) < count:
                result.append(row)
            else:
                replacement = rng.randrange(seen)
                if replacement < count:
                    result[replacement] = row
        if len(result) >= count:
            break
    if len(result) < count:
        raise ValueError(f"family {family!r} produced {len(result)} rows, requested {count}")
    return result


def _allocate(total: int, weights: dict[str, int]) -> dict[str, int]:
    names = list(weights)
    counts = {name: total * weights[name] // sum(weights.values()) for name in names}
    for name in names[: total - sum(counts.values())]:
        counts[name] += 1
    return counts


def _curated_acronym_rows(required: int) -> list[Row]:
    templates = (
        ("{short}", "{target}"),
        ("đ {short}", "đường {target}"),
        ("đ{short}", "đường {target}"),
        ("đường {short}", "đường {target}"),
        ("{short} q1", "{target} quận 1"),
        ("đ {short} q1", "đường {target} quận 1"),
        ("đ{short} q1", "đường {target} quận 1"),
    )
    rows: list[Row] = []
    base: list[Row] = []
    for source, expected in CURATED_QUERY_EXPANSIONS.items():
        for variant in dict.fromkeys((source, source.replace(" ", ""))):
            base.append(Row(
                variant, expected, _source_query_group(expected),
                "curated-query-expansion", "acronym", "acronym",
                (f"curated_query:{source}",),
            ))
    for short, expected in CURATED_UNIVERSITY_ALIASES.items():
        for source in (
            short, f"đh {short}", f"dh {short}", f"trường {short}",
            f"truong {short}", f"đh{short}", f"dh{short}", f"truong{short}",
        ):
            base.append(Row(
                source, expected, _source_query_group(expected),
                f"curated-university-alias:{short}", "acronym",
                "university_abbreviation", (f"curated_university:{short}",),
            ))
    for short, target in CURATED_ACRONYMS.items():
        for source_template, target_template in templates:
            expected = target_template.format(target=target)
            base.append(Row(
                source_template.format(short=short), expected,
                _source_query_group(expected), f"curated-acronym:{short}",
                "acronym", "acronym", (f"curated_acronym:{short}",),
            ))
    while len(rows) < required:
        for row in base:
            rows.append(row)
            if len(rows) >= required:
                break
    return rows


def _write(root: Path, stage: str, split: str, rows: list[Row], seed: int) -> dict[str, object]:
    folder = root / stage
    folder.mkdir(parents=True, exist_ok=True)
    random.Random(_seed(seed, stage, split, "shuffle")).shuffle(rows)
    src_path, tgt_path, meta_path = (folder / f"{split}.{suffix}" for suffix in ("src", "tgt", "meta.jsonl"))
    with src_path.open("w", encoding="utf-8", newline="\n") as src, tgt_path.open("w", encoding="utf-8", newline="\n") as tgt, meta_path.open("w", encoding="utf-8", newline="\n") as meta:
        for index, row in enumerate(rows):
            src.write(row.source + "\n"); tgt.write(row.target + "\n")
            meta.write(json.dumps({
                "example_id": f"{stage}:{split}:{index}", "stage": stage,
                "split": split, "query_group": row.query_group,
                "source_group_id": row.source_group_id, "error_type": row.error_type,
                "source_family": row.family, "operations": row.operations,
                "is_clean": row.source == row.target,
            }, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "families": dict(sorted(Counter(row.family for row in rows).items()))}


def prepare_curriculum_v2(source: str | Path, output: str | Path, profile_name: str, seed: int = 2026) -> dict[str, object]:
    if profile_name not in PROFILES:
        raise ValueError(f"unknown profile {profile_name!r}; choose from {sorted(PROFILES)}")
    profile, source_root, output_root = PROFILES[profile_name], Path(source), Path(output)
    clean = {split: _load_clean(source_root, split) for split in ("train", "validation")}
    # Curated dictionary knowledge belongs to training. Remove exact matching
    # targets from validation and inject a canonical row if OSM did not contain
    # one in train. External user-centric benchmarks evaluate these mappings.
    curated_targets = (
        set(CURATED_ACRONYMS.values())
        | set(CURATED_QUERY_EXPANSIONS.values())
        | set(CURATED_UNIVERSITY_ALIASES.values())
    )
    clean["validation"] = [row for row in clean["validation"] if row.text not in curated_targets]
    train_targets = {row.text for row in clean["train"]}
    for short, target in CURATED_ACRONYMS.items():
        if target not in train_targets:
            clean["train"].append(CleanQuery(target, _source_query_group(target), f"curated-acronym:{short}"))
            train_targets.add(target)
    for short, target in CURATED_UNIVERSITY_ALIASES.items():
        if target not in train_targets:
            clean["train"].append(CleanQuery(
                target, _source_query_group(target), f"curated-university-alias:{short}"
            ))
            train_targets.add(target)
    overlap = {row.query_group for row in clean["train"]} & {row.query_group for row in clean["validation"]}
    if overlap:
        raise ValueError(f"query-group leakage between train and validation: {len(overlap)}")
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "profile": asdict(profile), "seed": seed,
        "source": str(source_root.resolve()), "test_policy": "external frozen benchmarks only",
        "acronym_policy": {
            "bare": "reviewed curated mappings only; never mined automatically",
            "bare_source": "reviewed CURATED_ACRONYMS only",
            "automatic": "contextual phrase replacement only",
            "curated": CURATED_ACRONYMS,
            "curated_university_aliases": CURATED_UNIVERSITY_ALIASES,
            "examples": {"ltk": "lý thường kiệt", "d_ltk_q1": "đường lý thường kiệt quận 1"},
        },
        "stages": {},
    }
    for stage, weights in STAGE_WEIGHTS.items():
        summary = {"weights": weights, "splits": {}}
        for split, clean_rows in clean.items():
            total = profile.train_rows[stage] if split == "train" else profile.validation_rows
            allocation = _allocate(total, weights)
            selected: list[Row] = []
            for family, count in allocation.items():
                if stage == "stage3-lexical" and family == "acronym":
                    if split == "train":
                        selected.extend(_curated_acronym_rows(count))
                    else:
                        # Known dictionary mappings are evaluated by the frozen
                        # external user benchmark. Repeating them in validation
                        # would leak identical target groups across splits.
                        factory = lambda item, rng: _lexical(item, "contextual_abbreviation", rng)
                        selected.extend(_reservoir(
                            clean_rows, "contextual_abbreviation", count,
                            _seed(seed, profile_name, stage, split, "acronym-reallocated"), factory,
                        ))
                    continue
                if stage == "stage1-primitives":
                    factory = lambda item, rng, f=family: _primitive(item, f, rng)
                elif stage == "stage2-composition":
                    if family == "clean": factory = lambda item, rng: _primitive(item, "clean", rng)
                    elif family == "primitive_replay": factory = lambda item, rng: _primitive(item, rng.choice(("missing_diacritics", "boundary", "keyboard", "telex", "vni")), rng)
                    elif family == "two_operation": factory = lambda item, rng: _composition(item, 2, rng, "two_operation")
                    else: factory = lambda item, rng: _composition(item, 3, rng, "three_operation")
                else:
                    factory = lambda item, rng, f=family: _lexical(item, f, rng)
                selected.extend(_reservoir(clean_rows, family, count, _seed(seed, profile_name, stage, split), factory))
            summary["splits"][split] = _write(output_root, stage, split, selected, seed)
        manifest["stages"][stage] = summary
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "curriculum-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
