from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple


def normalize_nfc(text: str) -> str:
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text).strip().lower())
    return " ".join(norm.split())


# -----------------------------------------------------------------------------
# 1. CANONICAL ADMINISTRATIVE CONTRACT
# -----------------------------------------------------------------------------
# Maps any administrative abbreviation, unit prefix, or common POI prefix
# to its single, authoritative expanded canonical representation.
CANONICAL_ADMIN_MAP: Dict[str, str] = {
    # Street prefix: d, đ, dg exclusively map to "đường", NEVER "phố"
    "d": "đường",
    "đ": "đường",
    "dg": "đường",
    # Administrative units
    "p": "phường",
    "f": "phường",
    "q": "quận",
    "tp": "thành phố",
    "tx": "thị xã",
    "tt": "thị trấn",
    "h": "huyện",
    "t": "tỉnh",
    # Public & institutional entities
    "ubnd": "ủy ban nhân dân",
    "bv": "bệnh viện",
    "kcn": "khu công nghiệp",
    "đh": "đại học",
    "dh": "đại học",
    "thpt": "trung học phổ thông",
    "thcs": "trung học cơ sở",
}

# Reverse pattern for generating valid abbreviations from canonical targets
CANONICAL_REVERSE_ABBREVIATIONS: Dict[str, List[str]] = {
    "đường": ["d", "đ", "dg"],
    "phường": ["p", "f"],
    "quận": ["q"],
    "thành phố": ["tp"],
    "thị xã": ["tx"],
    "thị trấn": ["tt"],
    "huyện": ["h"],
    "tỉnh": ["t"],
    "ủy ban nhân dân": ["ubnd"],
    "bệnh viện": ["bv"],
    "khu công nghiệp": ["kcn"],
    "đại học": ["đh", "dh"],
    "trung học phổ thông": ["thpt"],
    "trung học cơ sở": ["thcs"],
}

# Words that MUST NOT be duplicated consecutively under any circumstances
DISALLOWED_CONSECUTIVE_WORDS = {
    "đường", "phố", "phường", "quận", "huyện", "thị", "xã", "thành", "số",
    "nhà", "ngõ", "ngách", "hẻm", "kiệt", "căn", "khu", "lô", "tòa",
    "bệnh", "viện", "ủy", "ban", "nhân", "dân", "đại", "học", "trung", "học",
    "công", "nghiệp", "thương", "mại", "vinfast", "hanoi", "vietnam",
}

# Target regex to detect unexpanded abbreviations that violate the contract
UNEXPANDED_TARGET_REGEX = re.compile(
    r"\b(?:ubnd|bv|kcn|thpt|thcs|kđt|tttm)\b|(?<!\w)[dđdgpfqht]\.(?=\s|$)|(?<!\w)[dđdgpfqht](?=\s+\d|\s+[a-zà-ỹ])",
    re.IGNORECASE
)


def has_consecutive_duplicates(text: str) -> bool:
    """Checks if text has any consecutive identical words."""
    words = normalize_nfc(text).split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            return True
    return False


def remove_consecutive_duplicates(text: str) -> str:
    """Removes any consecutive duplicate words while preserving single instances."""
    words = normalize_nfc(text).split()
    if not words:
        return ""
    cleaned = [words[0]]
    for w in words[1:]:
        if w != cleaned[-1]:
            cleaned.append(w)
    return " ".join(cleaned)


def remove_consecutive_phrase_duplicates(text: str) -> str:
    """Deduplicates consecutive 1-word, 2-word, 3-word, 4-word, and 5-word phrases.
    E.g.
      'thành phố bắc ninh bắc ninh' -> 'thành phố bắc ninh'
      'thành phố bắc ninh bắc ninh bac ninh' -> 'thành phố bắc ninh'
      'hà nội hà nội' -> 'hà nội'
      'quận 7 thành phố hồ chí minh thành phố hồ chí minh' -> 'quận 7 thành phố hồ chí minh'
    """
    cleaned = normalize_nfc(text)
    if not cleaned:
        return ""

    # 1. Specific administrative collision patterns from raw OSM dumps
    cleaned = re.sub(r"\bthành phố bắc ninh(?:\s+bắc ninh|\s+bac ninh)+\b", "thành phố bắc ninh", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bbắc ninh(?:\s+bắc ninh|\s+bac ninh)+\b", "bắc ninh", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhà nội(?:\s+hà nội|\s+ha noi)+\b", "hà nội", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhồ chí minh(?:\s+hồ chí minh|\s+ho chi minh)+\b", "thành phố hồ chí minh", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bđà nẵng(?:\s+đà nẵng|\s+da nang)+\b", "đà nẵng", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcần thơ(?:\s+cần thơ|\s+can tho)+\b", "cần thơ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhải phòng(?:\s+hải phòng|\s+hai phong)+\b", "hải phòng", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bbến tre(?:\s+bến tre|\s+ben tre)+\b", "bến tre", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhà tĩnh(?:\s+hà tĩnh|\s+ha tinh)+\b", "hà tĩnh", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhưng yên(?:\s+hưng yên|\s+hung yen)+\b", "hưng yên", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bdĩ an(?:\s+dĩ an|\s+di an)+\b", "dĩ an", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bquy nhơn(?:\s+quy nhơn|\s+quy nhon)+\b", "quy nhơn", cleaned, flags=re.IGNORECASE)

    # 2. General multi-word consecutive phrase loop
    words = cleaned.split()
    if len(words) >= 4:
        changed = True
        while changed:
            changed = False
            k_max = min(5, len(words) // 2)
            for k in range(k_max, 1, -1):
                i = 0
                new_words = []
                while i < len(words):
                    if i + 2 * k <= len(words):
                        p1 = words[i : i + k]
                        p2 = words[i + k : i + 2 * k]
                        if p1 == p2 and not all(w.isdigit() for w in p1):
                            new_words.extend(p1)
                            i += 2 * k
                            changed = True
                            continue
                    new_words.append(words[i])
                    i += 1
                words = new_words
                if changed:
                    break

    return remove_consecutive_duplicates(" ".join(words))


def canonicalize_target(text: str) -> str:
    """Transforms a target string to strictly adhere to the canonical contract.
    
    Ensures:
      1. All administrative acronyms (ubnd, bv, kcn, đh, thpt) are expanded.
      2. Business/POI abbreviations (cty, kđt, tttm) are expanded.
      3. Standalone administrative prefixes in target (p 12, q 1, đ...) are expanded.
      4. Phrasal repetitions (bắc ninh bắc ninh, hà nội hà nội) are deduplicated.
      5. Punctuation cleanup (strip leading/trailing punctuation).
      6. Normalizes NFC.
    """
    cleaned = normalize_nfc(text)
    
    # 1. Expand institutional & business acronyms
    cleaned = re.sub(r"\bubnd\b", "ủy ban nhân dân", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bbv\b", "bệnh viện", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bkcn\b", "khu công nghiệp", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bđh\b", "đại học", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bdh\b", "đại học", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bthpt\b", "trung học phổ thông", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bthcs\b", "trung học cơ sở", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btp\s+hcm\b", "thành phố hồ chí minh", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btphcm\b", "thành phố hồ chí minh", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcty\b", "công ty", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:kđt|kdt)\b", "khu đô thị", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btttm\b", "trung tâm thương mại", cleaned, flags=re.IGNORECASE)

    # 2. Expand standalone prefixes in target
    cleaned = re.sub(r"(?<!\w)(?:đ|d|dg)\.\s*", "đường ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<!\w)(?:p|f)\.\s*", "phường ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<!\w)q\.\s*", "quận ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<!\w)tp\.\s*", "thành phố ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<!\w)tx\.\s*", "thị xã ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<!\w)tt\.\s*", "thị trấn ", cleaned, flags=re.IGNORECASE)

    # Contextual expansion: e.g. 'p 12' -> 'phường 12', 'q 1' -> 'quận 1'
    cleaned = re.sub(r"(?<!\w)(?:p|f)\s+(\d+)\b", r"phường \1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<!\w)q\s+(\d+)\b", r"quận \1", cleaned, flags=re.IGNORECASE)

    # 3. Fix prefix collisions e.g. 'đ đường' -> 'đường'
    cleaned = re.sub(r"\b(?:đ|d|dg)\s+đường\b", "đường", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:p|f)\s+phường\b", "phường", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bq\s+quận\b", "quận", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btp\s+thành phố\b", "thành phố", cleaned, flags=re.IGNORECASE)

    # 4. Phrasal & single-word deduplication
    cleaned = remove_consecutive_phrase_duplicates(cleaned)

    # 5. Strip leading/trailing punctuation
    cleaned = re.sub(r"^[,\.\-\/\?\!\:\;\s]+", "", cleaned)
    cleaned = re.sub(r"[,\-\/\:\;\s]+$", "", cleaned)

    return normalize_nfc(cleaned)


def is_valid_vietnamese_script(text: str) -> bool:
    """Checks that text contains only valid Vietnamese/Latin/Numeric characters.
    Rejects Cyrillic (Russian), Thai, Japanese, Chinese, Korean, Emojis, and Stylized fake fonts.
    """
    if not text:
        return False
    for ch in text:
        code = ord(ch)
        # Cyrillic (Russian)
        if 0x0400 <= code <= 0x04FF:
            return False
        # Thai
        if 0x0E00 <= code <= 0x0E7F:
            return False
        # Japanese (Hiragana/Katakana)
        if 0x3040 <= code <= 0x30FF:
            return False
        # CJK (Chinese)
        if 0x4E00 <= code <= 0x9FFF:
            return False
        # Korean (Hangul)
        if (0xAC00 <= code <= 0xD7AF) or (0x1100 <= code <= 0x11FF) or (0x3130 <= code <= 0x318F):
            return False
        # Stylized phonetics / small capitals (e.g. ᴅ, ɪ, ᴀ, ᴄ, ʜ, ᴢ in chat spam)
        if 0x1D00 <= code <= 0x1D7F:
            return False
        # Emojis and miscellaneous symbols
        if (0x1F000 <= code <= 0x1FFFF) or (0x2600 <= code <= 0x27FF) or (0xFE00 <= code <= 0xFE0F):
            return False
    return True


BAD_CONTEXTLESS_ACRONYMS = {"ltk", "dbp", "nct", "hbt", "pvh", "ntmk", "nvl"}


def is_canonical_valid_pair(source: str, target: str) -> Tuple[bool, str]:
    """Validates whether a (source, target) pair complies with canonical contract.
    
    Returns (True, '') if valid, or (False, reason) if violating the contract.
    """
    s_norm = normalize_nfc(source)
    t_norm = normalize_nfc(target)

    if not s_norm or not t_norm:
        return False, "empty_source_or_target"

    # Rule 0: Must be valid Vietnamese script (no Cyrillic, Thai, Chinese, Japanese, Korean, Emojis, Stylized fake fonts)
    if not is_valid_vietnamese_script(s_norm) or not is_valid_vietnamese_script(t_norm):
        return False, "invalid_script"

    # Rule 1: No consecutive duplicate words in target
    if has_consecutive_duplicates(t_norm):
        return False, "target_has_duplicate_consecutive_words"

    # Rule 2: Target must not contain unexpanded institutional abbreviations
    for bad_tok in ["ubnd", "bv", "kcn", "thpt", "thcs"]:
        if re.search(rf"\b{bad_tok}\b", t_norm, re.IGNORECASE):
            return False, f"target_contains_unexpanded_{bad_tok}"

    # Rule 3: Target must not map d/đ to phố
    s_words = s_norm.split()
    t_words = t_norm.split()
    if s_words and t_words:
        if s_words[0] in ["d", "đ", "dg"] and t_words[0] == "phố":
            return False, "ambiguous_d_mapped_to_pho"

    # Rule 4: No ill-posed digit merging (e.g. 772 -> 77/2, 1834 -> 183/4)
    if "/" in t_norm and "/" not in s_norm:
        slashes = re.findall(r"\b(\d+)/(\d+(?:/\d+)*[a-zA-Z]?)\b", t_norm)
        for p1, p2 in slashes:
            merged = f"{p1}{p2}"
            if merged in s_norm:
                return False, "ill_posed_digit_merging"

    # Rule 5: No contextless proper-name acronyms in source
    for w in s_words:
        if w in BAD_CONTEXTLESS_ACRONYMS:
            return False, f"contextless_acronym_{w}"

    # Rule 6: Target must not have unexpanded standalone prefixes
    if re.search(r"(?<!\w)(?:d|đ|p|q|tp|tx|tt)\.(?=\s|$)|(?<!\w)(?:p|q)\s+\d+\b", t_norm):
        return False, "target_has_unexpanded_prefix"

    # Rule 7: Target must not contain residual unexpanded business abbreviations
    if re.search(r"\b(?:cty|kđt|kdt|tttm)\b", t_norm):
        return False, "target_has_unexpanded_business_abbrev"

    # Rule 8: Phrasal repetitions check in target
    words = t_norm.split()
    if len(words) >= 4:
        k_max = min(5, len(words) // 2)
        for k in range(k_max, 1, -1):
            for i in range(len(words) - 2 * k + 1):
                p1 = words[i : i + k]
                p2 = words[i + k : i + 2 * k]
                if p1 == p2 and not all(w.isdigit() for w in p1):
                    return False, f"target_has_phrasal_duplicate_{' '.join(p1)}"

    # Rule 9: Reject chat spam / customer service / phone numbers in target
    if len(t_words) > 25:
        return False, "target_too_long_chat_spam"
    if re.search(r"\b0\d{9,10}\b|\b0\d{3}\s+\d{3}\s+\d{3,4}\b", t_norm):
        return False, "target_contains_phone_number"
    if re.search(r"\b(?:zalo|viber|inbox|nhắn tin|chào anh|chào chị|em chào|liên hệ|sđt|hotline)\b", t_norm, re.IGNORECASE) and len(t_words) > 8:
        return False, "target_chat_spam"

    return True, ""



