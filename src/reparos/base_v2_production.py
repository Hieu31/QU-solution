from __future__ import annotations

import re
from pathlib import Path

from reparos import base_v2 as core
from reparos.base_v2 import BaseV2Config, CleanSeed, Variant
from reparos.base_v2_strict import generate_variants as strict_variants


PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)\d{8,10}(?!\d)")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def repeated_phrase(text: str) -> bool:
    tokens = text.casefold().split()
    for width in (4, 3, 2):
        seen: set[tuple[str, ...]] = set()
        for index in range(len(tokens) - width + 1):
            phrase = tuple(tokens[index : index + width])
            if phrase in seen:
                return True
            seen.add(phrase)
    return False


def target_rejection_reason(text: str) -> str | None:
    if len(text) > 100 or len(text.split()) > 16:
        return "too_long"
    if PHONE_RE.search(text.replace(" ", "")):
        return "phone_like"
    if URL_RE.search(text):
        return "url_like"
    if repeated_phrase(text):
        return "repeated_phrase"
    if not any(ch.isalpha() for ch in text):
        return "no_letters"
    return None


def protection_reason(text: str) -> str | None:
    compact = "".join(text.split())
    if len(compact) <= 3:
        return "short_ambiguous"
    tokens = text.split()
    if any(any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token) for token in tokens):
        return "alphanumeric_code"
    if text.isupper() and any(ch.isalpha() for ch in text):
        return "uppercase_name"
    return None


def generate_variants(seed: CleanSeed, global_seed: int = 2026) -> list[Variant]:
    if protection_reason(seed.text):
        return []
    return strict_variants(seed, global_seed)


def prepare_base_v2(
    source: str | Path,
    output: str | Path,
    config: BaseV2Config | None = None,
) -> dict[str, object]:
    previous_load = core._load_clean
    previous_generate = core.generate_variants
    rejection_counts: dict[str, dict[str, int]] = {}
    protected_counts: dict[str, dict[str, int]] = {}

    def load_filtered(source_root: Path, split: str, limit: int | None = None) -> list[CleanSeed]:
        # Apply limit after quality filtering so sample sizes are comparable.
        rows = previous_load(source_root, split, None)
        accepted: list[CleanSeed] = []
        rejected: dict[str, int] = {}
        protected: dict[str, int] = {}
        for row in rows:
            reason = target_rejection_reason(row.text)
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            guard = protection_reason(row.text)
            if guard:
                protected[guard] = protected.get(guard, 0) + 1
            accepted.append(row)
            if limit is not None and len(accepted) >= limit:
                break
        rejection_counts[split] = rejected
        protected_counts[split] = protected
        return accepted

    core._load_clean = load_filtered
    core.generate_variants = generate_variants
    try:
        report = core.prepare_base_v2(source, output, config)
    finally:
        core._load_clean = previous_load
        core.generate_variants = previous_generate
    report["quality_gate"] = {
        "rejected_targets": rejection_counts,
        "protected_targets": protected_counts,
        "policy": {
            "max_characters": 100,
            "max_tokens": 16,
            "reject_phone_url_no_letters": True,
            "reject_repeated_2_to_4_grams": True,
            "protect_short_or_alphanumeric": True,
        },
    }
    output_root = Path(output)
    name = "base-v2-manifest.json" if (config and config.materialize) else "base-v2-dry-run.json"
    (output_root / name).write_text(
        __import__("json").dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
