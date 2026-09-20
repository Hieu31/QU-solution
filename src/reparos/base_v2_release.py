from __future__ import annotations

import re
from pathlib import Path

from reparos.base_v2 import BaseV2Config
from reparos import base_v2_production as production


ADMIN_REDUNDANCY = (
    "hà nội thành phố hà nội",
    "hồ chí minh thành phố hồ chí minh",
    "thành phố hồ chí minh thành phố hồ chí minh",
)


def repeated_phrase(text: str) -> bool:
    tokens = text.casefold().split()
    # Reject only adjacent repeated phrases. Repetition at different positions
    # can be legitimate, e.g. a POI name followed by its street address.
    for width in range(1, min(8, len(tokens) // 2) + 1):
        for index in range(len(tokens) - 2 * width + 1):
            if tokens[index : index + width] == tokens[index + width : index + 2 * width]:
                return True
    normalized = " ".join(tokens)
    return any(pattern in normalized for pattern in ADMIN_REDUNDANCY)


def target_rejection_reason(text: str) -> str | None:
    if len(text) > 100 or len(text.split()) > 16:
        return "too_long"
    if production.PHONE_RE.search(text.replace(" ", "")):
        return "phone_like"
    if production.URL_RE.search(text):
        return "url_like"
    if repeated_phrase(text):
        return "repeated_phrase"
    return None


def protection_reason(text: str) -> str | None:
    compact = "".join(text.split())
    if not any(ch.isalpha() for ch in text):
        return "numeric_or_symbol"
    if len(compact) <= 3:
        return "short_ambiguous"
    if any(any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token) for token in text.split()):
        return "alphanumeric_code"
    if text.isupper():
        return "uppercase_name"
    return None


def prepare_base_v2(
    source: str | Path,
    output: str | Path,
    config: BaseV2Config | None = None,
) -> dict[str, object]:
    previous_reject = production.target_rejection_reason
    previous_protect = production.protection_reason
    production.target_rejection_reason = target_rejection_reason
    production.protection_reason = protection_reason
    try:
        report = production.prepare_base_v2(source, output, config)
    finally:
        production.target_rejection_reason = previous_reject
        production.protection_reason = previous_protect
    report["quality_gate"]["policy"] = {
        "max_characters": 100,
        "max_tokens": 16,
        "reject_phone_or_url": True,
        "reject_adjacent_repetition_and_admin_redundancy": True,
        "protect_short_alphanumeric_and_numeric": True,
    }
    name = "base-v2-manifest.json" if (config and config.materialize) else "base-v2-dry-run.json"
    (Path(output) / name).write_text(
        __import__("json").dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
