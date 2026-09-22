from __future__ import annotations

import re

# -----------------------------------------------------------------------------
# 1. DISJOINT BRAND & PROPER NOUN REGISTRIES (CANONICAL LOWERCASE)
# -----------------------------------------------------------------------------

# SEEN_BRANDS: Authorized for training and eval/protection_seen/
SEEN_BRANDS = [
    # Banking & Finance
    "techcombank", "vietcombank", "mb bank", "vpbank", "agribank", "bidv", "tpbank", "sacombank", "bv bank",
    # Retail & Technology
    "shopee", "vincom", "vinfast", "the manor", "thế giới di động", "điện máy xanh", "viettel post",
    "winmart", "emart", "lotte mart", "aeon mall", "crescent mall", "sc vivocity", "takashimaya",
    # F&B & Chains
    "starbucks", "starbucks reserve", "mcdonald's", "highlands coffee", "kfc", "lotteria", "jollibee",
    "phúc long", "circle k", "ministop", "gs25", "golden gate", "redsun", "hải xồm",
    # Transport & Services
    "grab", "gojek", "be", "mai linh", "vinasun",
    # Major Landmarks & POIs
    "hồ gươm", "sân bay nội bài", "sân bay tân sơn nhất", "cầu giấy", "đống đa", "hoàn kiếm", "bến nghé", "bình thạnh",
    "landmark 81", "bitexco", "thu cúc", "tâm anh",
    # Road / Transport codes
    "quốc lộ 1a", "quốc lộ 13", "quốc lộ 51", "đường d2", "tỉnh lộ 10", "vành đai 3",
]

# HELDOUT_BRANDS: STRICTLY RESERVED for eval/protection_heldout/ (NEVER IN TRAIN)
HELDOUT_BRANDS = [
    # Special Character Brands (Key verification targets)
    "biti's", "pizza 4p's", "j&t express", "co.opmart", "l'oréal", "d&g", "h&m", "p&g",
    # Banking & Finance
    "acb bank", "vib bank", "msb bank", "seabank", "nam a bank", "eximbank", "hdbank", "ocb bank", "vietinbank",
    # Retail & Shopping
    "tiki", "lazada", "gigamall", "annam gourmet", "family mart", "bách hóa xanh",
    # F&B & Chains
    "katinat", "phê la", "the coffee house", "gong cha", "ding tea", "haidilao", "dookki", "texas chicken",
    "shopeefood", "manwah", "gogi house", "kichi kichi", "chut chut", "cheese coffee", "cộng cà phê",
    # Logistics & Delivery
    "giao hàng nhanh", "giao hàng tiết kiệm", "ahamove", "ninja van", "shopee express",
    # Healthcare & Pharmacies
    "pharmacity", "nhà thuốc long châu", "medlatec", "hoàn mỹ", "hồng ngọc", "vinmec",
    # Airlines & Travel
    "vietnam airlines", "vietjet air", "bamboo airways",
    # Major Landmarks
    "chợ bến thành", "bà nà hills", "cầu rồng", "suối tiên", "đầm sen", "hồ tây", "phố cổ hội an", "chợ nổi cái răng",
]

# Guarantee mathematical disjointness
_seen_set = set(SEEN_BRANDS)
_heldout_set = set(HELDOUT_BRANDS)
assert _seen_set.isdisjoint(_heldout_set), f"Overlap detected: {_seen_set.intersection(_heldout_set)}"


# -----------------------------------------------------------------------------
# 2. BRAND CONTEXT TEMPLATES
# -----------------------------------------------------------------------------

BRAND_CONTEXT_TEMPLATES = [
    "chi nhánh {brand} quận 1",
    "chi nhánh {brand} hà nội",
    "uống cà phê tại {brand}",
    "mua hàng ở {brand}",
    "siêu thị {brand} gần nhất",
    "cửa hàng {brand} đường nguyễn trãi",
    "địa chỉ {brand} thành phố hồ chí minh",
    "đi đến {brand}",
    "gần {brand} quận đống đa",
    "tìm {brand} gần đây",
]


# -----------------------------------------------------------------------------
# 3. CURATED ACRONYMS & EXPANSIONS (LANE 2)
# -----------------------------------------------------------------------------

CURATED_ACRONYMS = {
    "ltk": "lý thường kiệt",
    "nct": "nguyễn chí thanh",
    "hbt": "hai bà trưng",
    "dbp": "điện biên phủ",
    "pvh": "phạm văn hai",
    "ntmk": "nguyễn thị minh khai",
    "nvl": "nguyễn văn linh",
    "thpt clhp": "trung học phổ thông chuyên lê hồng phong",
    "thpt tdn": "trung học phổ thông trần đại nghĩa",
    "dhbk": "đại học bách khoa",
    "bv bm": "bệnh viện bạch mai",
    "bv chợ rẫy": "bệnh viện chợ rẫy",
    "bv cr": "bệnh viện chợ rẫy",
    "ubnd q1": "ủy ban nhân dân quận 1",
    "kcn tb": "khu công nghiệp tân bình",
    "tp hcm": "thành phố hồ chí minh",
    "tphcm": "thành phố hồ chí minh",
    "hcm": "thành phố hồ chí minh",
    "hn": "hà nội",
    "đn": "đà nẵng",
    "hp": "hải phòng",
    "ct": "cần thơ",
}

# Negative list of tokens that CANNOT be blindly copied in Lane 4 identity
LANE2_EXPANSION_TRIGGERS = {
    "bv", "bv.", "d.", "đ.", "d", "đ", "q.", "q", "p.", "p", "tp.", "tp", "tx.", "tx", "tt.", "tt",
    "ql.", "ql", "tl.", "tl", "kdc.", "kdc", "kcn.", "kcn", "tp hcm", "tphcm", "hcm", "hn", "đn", "hp", "ct",
    "dhbk", "ubnd", "thpt"
}

# -----------------------------------------------------------------------------
# 4. ADDRESS REGEX RULES WITH NEGATIVE LOOKBEHINDS
# -----------------------------------------------------------------------------

# Strictly match "đường", NEVER "phố", to enforce canonical d/đ -> đường contract
STREET_PREFIX_RE = re.compile(r"(?<!\bthành\s)\b(?:đường)\s+", re.IGNORECASE)
DISTRICT_PREFIX_RE = re.compile(r"\bquận\s+(\d+|[a-zA-Z\s\u00C0-\u024F]+)", re.IGNORECASE)
WARD_PREFIX_RE = re.compile(r"\bphường\s+(\d+|[a-zA-Z\s\u00C0-\u024F]+)", re.IGNORECASE)
CITY_PREFIX_RE = re.compile(r"\bthành phố\s+", re.IGNORECASE)
TOWNSHIP_PREFIX_RE = re.compile(r"\bthị xã\s+", re.IGNORECASE)
