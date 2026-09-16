from __future__ import annotations

import random
import re
import unicodedata
from typing import Mapping

DEFAULT_NOISE_WEIGHTS = {
    "missing_diacritics_full": 18, "missing_diacritics_partial": 14,
    "wrong_diacritic": 10, "telex_leak": 8, "vni_leak": 5,
    "keyboard_edit": 20, "word_boundary": 8, "address_abbreviation": 6,
    "address_symbol": 4, "combined": 7,
}
KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm",
    "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
    "b": "vghn", "n": "bhjm", "m": "njk",
}
ABBREVIATIONS = (("thanh pho", "tp"), ("tinh", "t"), ("quan", "q"), ("huyen", "h"), ("phuong", "p"), ("xa", "x"), ("thi tran", "tt"), ("thi xa", "tx"), ("duong", "d"))
SHAPE_KEYS = {774: ("w", "8"), 770: ("a", "6"), 795: ("w", "7")}
TONE_KEYS = {769: ("s", "1"), 768: ("f", "2"), 777: ("r", "3"), 771: ("x", "4"), 803: ("j", "5")}

def strip_diacritics(text: str) -> str:
    text = text.replace(chr(273), "d").replace(chr(272), "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

def _partial_strip(term: str, rng: random.Random) -> str | None:
    tokens = term.split()
    eligible = [i for i, token in enumerate(tokens) if strip_diacritics(token) != token]
    if not eligible:
        return None
    amount = rng.randint(1, max(1, len(eligible) - 1)) if len(eligible) > 1 else 1
    for index in rng.sample(eligible, amount):
        tokens[index] = strip_diacritics(tokens[index])
    return " ".join(tokens)

def _wrong_diacritic(term: str, rng: random.Random) -> str | None:
    tones = tuple(TONE_KEYS)
    chars = list(unicodedata.normalize("NFD", term))
    vowel_indexes = [i for i, c in enumerate(chars) if c.casefold() in "aeiouy"]
    if not vowel_indexes:
        return None
    index = rng.choice(vowel_indexes)
    following = index + 1
    while following < len(chars) and unicodedata.category(chars[following]) == "Mn":
        following += 1
    existing = next((ord(c) for c in chars[index + 1:following] if ord(c) in TONE_KEYS), None)
    replacement = rng.choice([tone for tone in tones if tone != existing])
    chars[index + 1:following] = [c for c in chars[index + 1:following] if ord(c) not in TONE_KEYS] + [chr(replacement)]
    return unicodedata.normalize("NFC", "".join(chars))

def _input_method(term: str, mode: int) -> str | None:
    output = []
    changed = False
    for part in re.split(r"(\s+)", term):
        if not part or part.isspace():
            output.append(part); continue
        encoded_part = []
        tone_suffix = []
        for char in part:
            decomposed = unicodedata.normalize("NFD", char.casefold())
            if char.casefold() == chr(273):
                encoded = "dd" if mode == 0 else "d9"
            else:
                base = decomposed[0]
                shape = "".join((base if mode == 0 and ord(c) == 770 else SHAPE_KEYS[ord(c)][mode]) for c in decomposed[1:] if ord(c) in SHAPE_KEYS)
                tone_suffix.extend(TONE_KEYS[ord(c)][mode] for c in decomposed[1:] if ord(c) in TONE_KEYS)
                encoded = base + shape
            encoded_part.append(encoded)
            changed |= encoded != char.casefold()
        encoded_part.extend(tone_suffix)
        changed |= bool(tone_suffix)
        output.append("".join(encoded_part))
    result = "".join(output)
    return result if changed else None

def _keyboard_edit(term: str, rng: random.Random) -> str | None:
    chars = list(term)
    eligible = [i for i, char in enumerate(chars) if not char.isspace()]
    if not eligible:
        return None
    index = rng.choice(eligible)
    operation = rng.choice(("delete", "swap", "insert", "replace"))
    if operation == "delete" and len(eligible) > 1:
        del chars[index]
    elif operation == "swap":
        adjacent = [i for i in (index - 1, index + 1) if 0 <= i < len(chars) and not chars[i].isspace()]
        if adjacent:
            other = rng.choice(adjacent); chars[index], chars[other] = chars[other], chars[index]
        else:
            chars.insert(index, rng.choice("abcdefghijklmnopqrstuvwxyz"))
    elif operation == "replace":
        chars[index] = rng.choice(KEYBOARD_NEIGHBORS.get(strip_diacritics(chars[index]).casefold(), "abcdefghijklmnopqrstuvwxyz"))
    else:
        chars.insert(index, rng.choice("abcdefghijklmnopqrstuvwxyz"))
    result = "".join(chars)
    return result if result != term else None

def _word_boundary(term: str, rng: random.Random) -> str | None:
    tokens = term.split()
    if len(tokens) > 1 and rng.random() < 0.7:
        index = rng.randrange(len(tokens) - 1)
        return " ".join(tokens[:index] + [tokens[index] + tokens[index + 1]] + tokens[index + 2:])
    eligible = [i for i, token in enumerate(tokens) if len(token) >= 6]
    if not eligible:
        return None
    index = rng.choice(eligible); cut = rng.randint(2, len(tokens[index]) - 2)
    tokens[index:index + 1] = (tokens[index][:cut], tokens[index][cut:])
    return " ".join(tokens)

def _abbreviate(term: str, rng: random.Random) -> str | None:
    shadow = strip_diacritics(term)
    matches = [(source, target, shadow.find(source)) for source, target in ABBREVIATIONS if re.search(rf"\b{source}\b", shadow)]
    if not matches:
        return None
    source, target, start = rng.choice(matches)
    return term[:start] + target + term[start + len(source):]

def _address_symbol(term: str, rng: random.Random) -> str | None:
    positions = [i for i, char in enumerate(term) if char in "/-"]
    if positions:
        chars = list(term); chars[rng.choice(positions)] = rng.choice((" ", ""))
        return re.sub(r"\s+", " ", "".join(chars)).strip()
    match = re.search(r"\b(\d+)\s+(\d+)\b", term)
    return term[:match.start()] + f"{match.group(1)}/{match.group(2)}" + term[match.end():] if match else None

def _single(term: str, kind: str, rng: random.Random) -> str | None:
    if kind == "missing_diacritics_full":
        value = strip_diacritics(term); return value if value != term else None
    if kind == "missing_diacritics_partial": return _partial_strip(term, rng)
    if kind == "wrong_diacritic": return _wrong_diacritic(term, rng)
    if kind == "telex_leak": return _input_method(term, 0)
    if kind == "vni_leak": return _input_method(term, 1)
    if kind == "keyboard_edit": return _keyboard_edit(term, rng)
    if kind == "word_boundary": return _word_boundary(term, rng)
    if kind == "address_abbreviation": return _abbreviate(term, rng)
    if kind == "address_symbol": return _address_symbol(term, rng)
    return None

def generate_noise(term: str, kind: str, rng: random.Random) -> tuple[str, str] | None:
    if kind != "combined":
        value = _single(term, kind, rng)
        return (value, kind) if value and value != term else None
    choices = tuple(name for name in DEFAULT_NOISE_WEIGHTS if name != "combined")
    first_kind = rng.choice(choices); first = _single(term, first_kind, rng)
    if not first or first == term: return None
    second_kind = rng.choice(tuple(name for name in choices if name != first_kind))
    second = _single(first, second_kind, rng)
    return (second, f"combined:{first_kind}+{second_kind}") if second and second != first else None

def location_query_variants(term: str, count: int, seed: int, weights: Mapping[str, int] | None = None) -> tuple[tuple[str, str, str], ...]:
    configured = dict(weights or DEFAULT_NOISE_WEIGHTS)
    names = tuple(name for name, weight in configured.items() if weight > 0)
    probabilities = tuple(configured[name] for name in names)
    variants = []
    for index in range(count):
        rng = random.Random(f"{seed}:{term}:{index}"); result = None
        for _ in range(20):
            result = generate_noise(term, rng.choices(names, weights=probabilities, k=1)[0], rng)
            if result: break
        noisy, kind = result or (_keyboard_edit(term, rng) or term + "a", "keyboard_edit")
        variants.append((noisy, kind, str(index)))
    return tuple(variants)
