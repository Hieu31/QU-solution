from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from reparos import base_v2 as core
from reparos.base_v2 import CleanSeed
from reparos.base_v2_production import (
    protection_reason,
    repeated_phrase,
    target_rejection_reason,
)
from reparos.data import _source_query_group
from reparos.typing_curriculum import encode_ime

# -----------------------------------------------------------------------------
# 1. DISJOINT ENTITY LISTS FOR ZERO-LEAK PROTECTION (GROUP C)
# -----------------------------------------------------------------------------

# SEEN_ENTITIES: Available for train/ and eval/protection_seen/
SEEN_PROTECTED_ENTITIES = [
    # Banking & Finance
    "BV Bank", "Techcombank", "Vietcombank", "MB Bank", "VPBank", "Agribank", "BIDV", "TPBank", "Sacombank",
    # Retail & Technology
    "Shopee", "Vincom", "VinFast", "The Manor", "Thế Giới Di Động", "Điện Máy Xanh", "Bách Hóa Xanh",
    # F&B & Lifestyle
    "Starbucks", "Starbucks Reserve", "McDonald's", "Highlands Coffee", "KFC", "Lotteria", "Jollibee",
    # Transport & Services
    "Grab", "Gojek", "Be", "Mai Linh", "Vinasun",
    # Major POIs & Proper Nouns
    "Hồ Gươm", "Sân bay Nội Bài", "Sân bay Tân Sơn Nhất", "Cầu Giấy", "Đống Đa", "Hoàn Kiếm", "Bến Nghé", "Bình Thạnh",
    # Standard Road & Highway Codes
    "quốc lộ 1A", "quốc lộ 13", "quốc lộ 51", "đường D2", "hẻm 42/21 đường số 5", "tỉnh lộ 10", "vành đai 3",
]

# HELDOUT_ENTITIES: STRICTLY RESERVED for eval/protection_heldout/ (NEVER IN TRAIN)
HELDOUT_PROTECTED_ENTITIES = [
    # Banking & Finance
    "ACB Bank", "VIB Bank", "MSB Bank", "SeABank", "Nam A Bank", "Eximbank", "HDBank", "OCB Bank", "VietinBank",
    # Retail & Malls
    "Tiki", "Lazada", "GigaMall", "Aeon Mall", "Emart", "Lotte Mart", "Annam Gourmet", "Co.opmart", "Circle K", "FamilyMart", "Biti's",
    # F&B & Chains
    "Katinat", "Phê La", "The Coffee House", "Gong Cha", "Ding Tea", "Pizza 4P's", "Golden Gate", "Haidilao", "Dookki", "Texas Chicken",
    # Logistics & Healthcare
    "Viettel Post", "J&T Express", "Giao Hàng Nhanh", "ShopeeFood", "Giao Hàng Tiết Kiệm", "Pharmacity", "Nhà thuốc Long Châu",
    # Landmarks & POIs
    "Landmark 81", "Chợ Bến Thành", "Bà Nà Hills", "Cầu Rồng", "Suối Tiên", "Đầm Sen", "Hồ Tây", "Phố cổ Hội An", "Chợ nổi Cái Răng",
    # Ambiguous abbreviations in no-edit context
    "nv", "cb", "tc", "kd", "bql", "kts", "gđ",
]

# Assert mathematical mutual exclusivity
assert set(SEEN_PROTECTED_ENTITIES).isdisjoint(set(HELDOUT_PROTECTED_ENTITIES)), "Seen and Held-out entities must be completely disjoint!"


# -----------------------------------------------------------------------------
# 2. CURATED ACRONYMS & SENTENCE TEMPLATES (GROUP A)
# -----------------------------------------------------------------------------

CURATED_ACRONYMS = {
    "thpt clhp": "trung học phổ thông chuyên lê hồng phong",
    "thpt tdn": "trung học phổ thông trần đại nghĩa",
    "dhbk": "đại học bách khoa",
    "bv bm": "bệnh viện bạch mai",
    "ubnd q1": "ủy ban nhân dân quận 1",
    "kcn tb": "khu công nghiệp tân bình",
}

CONTEXTUAL_TEMPLATES = [
    ("ở đoạn {short} giao với", "ở đoạn {full} giao với"),
    ("gần ngã tư {short}", "gần ngã tư {full}"),
    ("đoạn đường {short} {district}", "đoạn đường {full} {district}"),
    ("số 15 {short} {district}", "số 15 {full} {district}"),
    ("địa chỉ {short} {city}", "địa chỉ {full} {city}"),
    ("đi qua đường {short}", "đi qua đường {full}"),
    ("gần {short}", "gần {full}"),
    ("tại khu vực {short}", "tại khu vực {full}"),
]

SAMPLE_DISTRICTS = ["quận 1", "quận 3", "quận 10", "quận đống đa", "quận ba đình", "quận bình thạnh"]
SAMPLE_CITIES = ["thành phố hồ chí minh", "hà nội", "đà nẵng", "hải phòng"]


# -----------------------------------------------------------------------------
# 3. RECTIFIED ADDRESS ABBREVIATION GENERATOR (GROUP A)
# -----------------------------------------------------------------------------

# Strictly match "đường", NEVER "phố", to enforce canonical d/đ -> đường contract
_STREET_PREFIX_RE = re.compile(r"(?<!\bthành\s)\b(?:đường)\s+", re.IGNORECASE)
_DISTRICT_PREFIX_RE = re.compile(r"\bquận\s+(\d+|[a-zA-Z\s\u00C0-\u024F]+)", re.IGNORECASE)
_WARD_PREFIX_RE = re.compile(r"\bphường\s+(\d+|[a-zA-Z\s\u00C0-\u024F]+)", re.IGNORECASE)
_CITY_PREFIX_RE = re.compile(r"\bthành phố\s+", re.IGNORECASE)
_TOWNSHIP_PREFIX_RE = re.compile(r"\bthị xã\s+", re.IGNORECASE)


def generate_address_abbreviation(text: str, rng: random.Random) -> str | None:
    """Generates rectified address abbreviations according to the new specification.
    
    Transforms:
      đường X   -> d. X | đ. X | d X | đ X
      quận Y    -> q. Y | q Y | qY (if number)
      phường Z  -> p. Z | p Z | pZ (if number)
      thành phố W -> tp. W | tp W
      thị xã V  -> tx. V | tx V
    """
    modified = text
    applied = False

    # 1. Street prefix replacement
    if _STREET_PREFIX_RE.search(modified):
        rep = rng.choice(["d. ", "đ. ", "d ", "đ "])
        modified = _STREET_PREFIX_RE.sub(rep, modified, count=1)
        applied = True

    # 2. District prefix replacement
    match_d = _DISTRICT_PREFIX_RE.search(modified)
    if match_d and rng.random() < 0.8:
        val = match_d.group(1).strip()
        rep = rng.choice(["q.", "q", "q "]) + ("" if val.isdigit() and rng.random() < 0.5 else " ") + val
        modified = modified[:match_d.start()] + rep + modified[match_d.end():]
        applied = True

    # 3. Ward prefix replacement
    match_w = _WARD_PREFIX_RE.search(modified)
    if match_w and rng.random() < 0.8:
        val = match_w.group(1).strip()
        rep = rng.choice(["p.", "p", "p "]) + ("" if val.isdigit() and rng.random() < 0.5 else " ") + val
        modified = modified[:match_w.start()] + rep + modified[match_w.end():]
        applied = True

    # 4. City prefix replacement
    if _CITY_PREFIX_RE.search(modified) and rng.random() < 0.7:
        rep = rng.choice(["tp. ", "tp "])
        modified = _CITY_PREFIX_RE.sub(rep, modified, count=1)
        applied = True

    # 5. Township prefix replacement
    if _TOWNSHIP_PREFIX_RE.search(modified) and rng.random() < 0.7:
        rep = rng.choice(["tx. ", "tx "])
        modified = _TOWNSHIP_PREFIX_RE.sub(rep, modified, count=1)
        applied = True

    clean_mod = " ".join(modified.split())
    if applied and clean_mod != " ".join(text.split()):
        return clean_mod
    return None


def generate_contextual_acronym(rng: random.Random) -> tuple[str, str]:
    """Generates natural sentence pairs embedding acronyms in realistic Vietnamese frames."""
    short = rng.choice(list(CURATED_ACRONYMS.keys()))
    full = CURATED_ACRONYMS[short]
    tmpl_src, tmpl_tgt = rng.choice(CONTEXTUAL_TEMPLATES)
    district = rng.choice(SAMPLE_DISTRICTS)
    city = rng.choice(SAMPLE_CITIES)

    src = tmpl_src.format(short=short, district=district, city=city, street="nguyễn trãi")
    tgt = tmpl_tgt.format(full=full, district=district, city=city, street="nguyễn trãi")
    return " ".join(src.split()), " ".join(tgt.split())


# -----------------------------------------------------------------------------
# 4. COMPOSITION GENERATOR (GROUP A)
# -----------------------------------------------------------------------------

def generate_composition(clean_text: str, rng: random.Random) -> str | None:
    """Generates realistic compound noise (combining 2-3 primitives from Base V2)."""
    val = clean_text
    ops = []

    # Maybe apply address abbreviation first
    abbrev = generate_address_abbreviation(val, rng)
    if abbrev and abbrev != val:
        val = abbrev
        ops.append("abbrev")

    # Apply boundary join or split
    if rng.random() < 0.6:
        val = core._join_boundary(val, rng)
        ops.append("boundary")

    # Apply missing diacritics or telex or keyboard typo
    choice = rng.choice(["missing", "keyboard", "wrong_dia", "telex"])
    if choice == "missing":
        val = core._partial_plain(val, rng) if rng.random() < 0.6 else core._plain(val)
        ops.append("missing")
    elif choice == "keyboard":
        val = core._keyboard(val, rng, rng.choice(["drop", "duplicate", "transpose"]))
        ops.append("keyboard")
    elif choice == "wrong_dia":
        val = core._wrong_diacritic(val, rng)
        ops.append("wrong_dia")
    elif choice == "telex":
        val = encode_ime(val, "telex", rng, malformed=True)
        ops.append("telex")

    clean_mod = " ".join(val.split())
    if clean_mod != " ".join(clean_text.split()):
        return clean_mod
    return None


# -----------------------------------------------------------------------------
# 5. RETENTION PRIMITIVES (GROUP B - REUSING BASE V2 LOGIC DIRECTLY)
# -----------------------------------------------------------------------------

def generate_retention_variant(clean_text: str, family: str, rng: random.Random) -> str | None:
    """Reuses the exact slot implementations from Base V2."""
    if family == "vni":
        res = encode_ime(clean_text, "vni", rng, malformed=rng.random() < 0.25)
    elif family == "telex":
        res = encode_ime(clean_text, "telex", rng, malformed=rng.random() < 0.25)
    elif family == "wrong_diacritic":
        res = core._wrong_diacritic(clean_text, rng)
    elif family == "keyboard":
        op = rng.choice(["drop", "duplicate", "transpose", "insert"])
        res = core._keyboard(clean_text, rng, op)
    elif family == "boundary":
        res = core._join_boundary(clean_text, rng) if rng.random() < 0.7 else core._wrong_split(clean_text, rng)
    elif family == "missing_diacritics":
        res = core._partial_plain(clean_text, rng) if rng.random() < 0.5 else core._plain(clean_text)
    else:
        raise ValueError(f"Unknown retention family: {family}")

    clean_mod = " ".join(res.split())
    if clean_mod != " ".join(clean_text.split()):
        return clean_mod
    return None
