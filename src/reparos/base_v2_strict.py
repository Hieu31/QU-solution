from __future__ import annotations

import random
from pathlib import Path

from reparos import base_v2 as core
from reparos.base_v2 import BaseV2Config, CleanSeed, Variant
from reparos.data import _source_query_group


def generate_variants(seed: CleanSeed, global_seed: int = 2026) -> list[Variant]:
    """Generate only variants where every declared operation took effect."""
    seen: set[str] = set()
    result: list[Variant] = []
    target_key = _source_query_group(seed.text)
    for rank, (family, operations) in enumerate(core.SLOTS, start=1):
        rng = random.Random(core._stable_seed(global_seed, seed.split, seed.query_group, rank, family))
        source = seed.text
        valid = True
        for operation in operations:
            updated = core._apply(source, (operation,), rng)
            if updated == source:
                valid = False
                break
            source = updated
        source_key = _source_query_group(source)
        if not valid or not source_key or source_key == target_key or source_key in seen:
            continue
        seen.add(source_key)
        result.append(Variant(
            source, seed.text, seed.query_group, seed.source_group_id,
            seed.split, rank, family, operations,
        ))
    return result


def prepare_base_v2(
    source: str | Path,
    output: str | Path,
    config: BaseV2Config | None = None,
) -> dict[str, object]:
    # The core writer/auditor resolves this global at runtime. Override it only
    # for this call so the legacy experimental entry point remains reproducible.
    previous = core.generate_variants
    core.generate_variants = generate_variants
    try:
        return core.prepare_base_v2(source, output, config)
    finally:
        core.generate_variants = previous
