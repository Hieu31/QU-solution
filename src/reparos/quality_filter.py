from __future__ import annotations

import random
import re
import unicodedata
from typing import Optional, Tuple

from reparos import base_v2 as core
from reparos.data_registry import LANE2_EXPANSION_TRIGGERS
from reparos.typing_curriculum import encode_ime

# Common Vietnamese vowels with diacritics
VIETNAMESE_VOWELS_REGEX = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.IGNORECASE)

# Incomplete endings: ends with slash, dash, dot, comma, colon
INCOMPLETE_ENDING_PUNCT_REGEX = re.compile(r"[/,.\-:]$", re.IGNORECASE)

# Category words that should not appear alone at the end of a multi-word address query
INCOMPLETE_TRAILING_WORDS = {
    "ngõ", "ngách", "hẻm", "số", "đường", "phố", "khu", "chợ", "bến", "trường",
    "bệnh", "và", "ở", "gần", "tại", "cổng", "nhà", "khách", "tầng", "thôn", "xóm"
}

# Suspicious keystroke prefixes: e.g. "199 hồ", "23 lê" (house number + single word)
HOUSE_NUMBER_SINGLE_WORD_REGEX = re.compile(r"^\d+[a-zA-Z\d/]*\s+[a-zà-ỹ]+$", re.IGNORECASE)

# Known incomplete street names from zero_click logs
KNOWN_INCOMPLETE_PREFIXES = {
    "199 hồ tùng",
    "khu nhà ở và trường",
    "khu nhà ở và trường học",
}

# Known common spelling mistakes in search queries
KNOWN_TYPOS = {
    "bên xe": "bến xe",
    "khách san": "khách sạn",
    "quan an": "quán ăn",
    "benh vien": "bệnh viện",
    "truong hoc": "trường học",
    "nha hang": "nhà hàng",
    "sieu thi": "siêu thị",
    "shoppe": "shopee",
    "trasua": "trà sữa",
    "cafe": "cà phê",
    "nhà gas": "nhà ga",
}


class ZeroClickQualityFilter:
    """3-Tier Confidence-Gated Quality Filter for real search queries from zero_click.csv."""

    def __init__(self):
        self.lane2_triggers = set(LANE2_EXPANSION_TRIGGERS)

    def normalize(self, query: str) -> str:
        """Applies canonical normalization (NFC + lowercase + collapse spaces)."""
        if not query:
            return ""
        norm = unicodedata.normalize("NFC", str(query).strip().lower())
        return " ".join(norm.split())

    def classify(self, raw_query: str) -> Tuple[str, str]:
        """Classifies a query into one of:
        - HIGH_CONFIDENCE_CLEAN: Valid, complete query with proper Vietnamese diacritics.
        - AMBIGUOUS_PREFIX: Keystroke increment, trailing symbol, or truncated prefix.
        - CONTRADICTION_LANE2: Contains tokens that Lane 2 is supposed to expand.
        - NOISY_TYPO: Obvious spelling or unaccented mistake.
        - REJECTED_GARBAGE: Too short, empty, or corrupted.
        
        Returns: (category, reason)
        """
        q = self.normalize(raw_query)
        if not q or len(q) < 3:
            return "REJECTED_GARBAGE", "too_short_or_empty"

        tokens = q.split()

        # Check 0a: Chat spam, long customer service text, or phone numbers
        if len(tokens) > 20:
            return "REJECTED_GARBAGE", "query_too_long_chat_spam"
        if re.search(r"\b0\d{9,10}\b|\b0\d{3}\s+\d{3}\s+\d{3,4}\b", q):
            return "REJECTED_GARBAGE", "contains_phone_number"
        if re.search(r"\b(?:zalo|viber|inbox|nhắn tin|chào anh|chào chị|em chào|liên hệ|sđt|hotline)\b", q, re.IGNORECASE) and len(tokens) > 8:
            return "REJECTED_GARBAGE", "chat_customer_service_message"

        # Check 1: Trailing punctuation
        if INCOMPLETE_ENDING_PUNCT_REGEX.search(q):
            return "AMBIGUOUS_PREFIX", "incomplete_trailing_punctuation"

        # Check 2: Single tokens
        if len(tokens) < 2:
            if not VIETNAMESE_VOWELS_REGEX.search(q):
                return "AMBIGUOUS_PREFIX", "single_unaccented_token"
            return "AMBIGUOUS_PREFIX", "single_bare_token"

        # Check 3: Known truncated prefixes
        if q in KNOWN_INCOMPLETE_PREFIXES:
            return "AMBIGUOUS_PREFIX", "known_incomplete_prefix"

        # Check 4: House number followed by only a single word (e.g. "199 hồ")
        if HOUSE_NUMBER_SINGLE_WORD_REGEX.match(q):
            return "AMBIGUOUS_PREFIX", "house_number_single_word_prefix"

        # Check 5: Trailing category word (e.g. "36 ngõ", "gần chợ")
        if tokens[-1] in INCOMPLETE_TRAILING_WORDS and not tokens[-1].isdigit():
            return "AMBIGUOUS_PREFIX", f"incomplete_trailing_word_{tokens[-1]}"

        # Check 6: Trailing single-letter token (e.g. "nhà gas t", "ngõ 23 t")
        if len(tokens[-1]) == 1 and not tokens[-1].isdigit():
            return "AMBIGUOUS_PREFIX", "trailing_single_letter"

        # Check 7: Contradiction with Lane 2 (Addresses & Acronyms)
        # Any query containing 'bv', 'd.', 'q.', 'tp hcm' must NOT be auto-labeled as identity!
        for token in tokens:
            cleaned_token = token.strip(".,/-")
            if cleaned_token in self.lane2_triggers or token in self.lane2_triggers:
                return "CONTRADICTION_LANE2", f"contains_lane2_trigger_{token}"

        for trigger in ["tp hcm", "tphcm", "bv ", "d. ", "q. ", "p. ", "ql. ", "tx. ", "tt. "]:
            if trigger in q:
                return "CONTRADICTION_LANE2", f"contains_lane2_trigger_{trigger.strip()}"

        # Check 8: Known common typos
        for typo in KNOWN_TYPOS:
            if typo in q:
                return "NOISY_TYPO", f"contains_known_typo_{typo}"

        # Check 9: Diacritics presence
        # If a query has 2+ words and ZERO Vietnamese diacritics, it's an unaccented query!
        has_diacritics = bool(VIETNAMESE_VOWELS_REGEX.search(q))
        if not has_diacritics and not any(c.isdigit() for c in q):
            return "NOISY_TYPO", "unaccented_query"

        # Check 10: Excessively long tokens
        for t in tokens:
            if len(t) > 15 and not t.startswith("http"):
                return "NOISY_TYPO", "excessively_long_token"

        return "HIGH_CONFIDENCE_CLEAN", "valid_vietnamese_query"

    def generate_dae_pair(self, clean_query: str, rng: random.Random) -> Tuple[str, str]:
        """Generates a Self-Supervised Denoising Autoencoder pair:
        Target Y = clean_query
        Input X = corrupt(clean_query)
        """
        target = self.normalize(clean_query)
        val = target
        op = rng.choice(["missing", "telex", "vni", "boundary", "keyboard"])

        if op == "missing":
            val = core._partial_plain(val, rng) if rng.random() < 0.6 else core._plain(val)
        elif op == "telex":
            val = encode_ime(val, "telex", rng, malformed=True)
        elif op == "vni":
            val = encode_ime(val, "vni", rng, malformed=True)
        elif op == "boundary":
            val = core._join_boundary(val, rng) if rng.random() < 0.7 else core._wrong_split(val, rng)
        elif op == "keyboard":
            val = core._keyboard(val, rng, rng.choice(["drop", "duplicate", "transpose"]))

        source = self.normalize(val)
        if source == target:
            # Fallback: force unaccenting if corrupt resulted in identity
            source = self.normalize(core._plain(target))

        return source, target

    def generate_identity_pair(self, clean_query: str) -> Tuple[str, str]:
        """Generates a Self-Supervised Identity pair:
        Target Y = clean_query
        Input X = clean_query
        """
        clean = self.normalize(clean_query)
        return clean, clean
