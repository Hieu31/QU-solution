from __future__ import annotations

import unicodedata


_TONE_ROWS = (
    "aáàảãạ", "ăắằẳẵặ", "âấầẩẫậ", "eéèẻẽẹ", "êếềểễệ",
    "iíìỉĩị", "oóòỏõọ", "ôốồổỗộ", "ơớờởỡợ", "uúùủũụ",
    "ưứừửữự", "yýỳỷỹỵ",
)
_TONE_KEYS = {"s": 1, "f": 2, "r": 3, "x": 4, "j": 5}
_VNI_TONES = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
_BASE_REPLACEMENTS = (
    ("uow", "ươ"), ("dd", "đ"), ("aw", "ă"), ("aa", "â"), ("ee", "ê"),
    ("oo", "ô"), ("ow", "ơ"), ("uw", "ư"),
)
_VNI_REPLACEMENTS = (
    ("d9", "đ"), ("a8", "ă"), ("a6", "â"), ("e6", "ê"),
    ("o6", "ô"), ("o7", "ơ"), ("u7", "ư"),
)

_TONE_LOOKUP: dict[str, tuple[str, int]] = {}
for row in _TONE_ROWS:
    for tone, character in enumerate(row):
        _TONE_LOOKUP[character] = (row[0], tone)


def strip_tone(text: str) -> str:
    return "".join(_TONE_LOOKUP.get(character, (character, 0))[0] for character in text)


def _tone_target(text: str) -> int | None:
    vowels = [index for index, char in enumerate(text) if char in _TONE_LOOKUP]
    if not vowels:
        return None
    marked = [index for index in vowels if text[index] in "ăâêôơư"]
    if marked:
        return marked[-1]
    if len(vowels) == 1:
        return vowels[0]
    cluster = "".join(text[index] for index in vowels)
    if cluster.startswith(("oa", "oe", "uy")):
        return vowels[0]
    return vowels[-2] if len(vowels) >= 2 else vowels[-1]


def apply_tone(text: str, tone: int) -> str:
    plain = strip_tone(text)
    target = _tone_target(plain)
    if target is None:
        return plain
    base = plain[target]
    row = next(row for row in _TONE_ROWS if row[0] == base)
    return plain[:target] + row[tone] + plain[target + 1 :]


def _replace_all(text: str, replacements: tuple[tuple[str, str], ...]) -> str:
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def decode_telex(token: str) -> str:
    normalized = unicodedata.normalize("NFC", token)
    lowered = normalized.lower()
    tone = 0
    if lowered and lowered[-1] in _TONE_KEYS:
        tone = _TONE_KEYS[lowered[-1]]
        lowered = lowered[:-1]
    decoded = apply_tone(_replace_all(lowered, _BASE_REPLACEMENTS), tone)
    return _restore_case(normalized, decoded)


def decode_vni(token: str) -> str:
    normalized = unicodedata.normalize("NFC", token)
    lowered = normalized.lower()
    tone = 0
    if lowered and lowered[-1] in _VNI_TONES:
        tone = _VNI_TONES[lowered[-1]]
        lowered = lowered[:-1]
    decoded = apply_tone(_replace_all(lowered, _VNI_REPLACEMENTS), tone)
    return _restore_case(normalized, decoded)


def _restore_case(source: str, decoded: str) -> str:
    if source.isupper():
        return decoded.upper()
    if source.istitle():
        return decoded.title()
    return decoded


class VietnameseVariantGenerator:
    """Generate canonical Unicode hypotheses from Unicode, Telex, or VNI input."""

    def __call__(self, token: str) -> tuple[str, ...]:
        normalized = unicodedata.normalize("NFC", token)
        variants = {normalized, decode_telex(normalized), decode_vni(normalized)}
        return tuple(sorted(variant for variant in variants if variant))
