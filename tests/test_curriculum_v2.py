from __future__ import annotations

import csv
import random
from pathlib import Path

from reparos.curriculum_v2 import (
    CURATED_UNIVERSITY_ALIASES,
    CleanQuery,
    _acronym_candidates,
    _composition,
    _curated_acronym_rows,
    _lexical,
    _plain,
)


def test_plain_and_composition_keep_clean_target() -> None:
    clean = CleanQuery("sân bay nội bài", "sân bay nội bài", "poi-1")
    assert _plain(clean.text) == "san bay noi bai"
    row = _composition(clean, 2, random.Random(7), "two_operation")
    assert row is not None
    assert row.target == "sân bay nội bài"
    assert len(row.operations) == 2


def test_curated_code_is_not_mined_automatically() -> None:
    candidates = _acronym_candidates("đường lý thường kiệt quận một")
    assert not any("ltk" in source.split() for source, _ in candidates)


def test_automatic_acronym_requires_anchor_and_blocks_reserved_codes() -> None:
    assert not _acronym_candidates("nhà bảo vệ")
    assert not _acronym_candidates("31 bùi viện")
    assert not any(source.endswith(" tp") for source, _ in _acronym_candidates("hẻm 319 tân phước"))
    assert not any(source.endswith(" bv") for source, _ in _acronym_candidates("đường bùi viện"))
    assert not any(source.endswith(" ll") for source, _ in _acronym_candidates("đường lê lợi"))
    assert _acronym_candidates("đường phan văn trị quận một")


def test_no_acronym_for_only_generic_address_words() -> None:
    assert not _acronym_candidates("đường phường quận")


def test_bare_acronym_is_curated_not_automatic() -> None:
    wrong = CleanQuery("lê thị kính", "lê thị kính", "poi-wrong")
    wrong_rows = [_lexical(wrong, "acronym", random.Random(seed)) for seed in range(20)]
    assert all(row is None or row.source != "ltk" for row in wrong_rows)
    curated = CleanQuery("lý thường kiệt", "lý thường kiệt", "curated")
    rows = [_lexical(curated, "acronym", random.Random(seed)) for seed in range(20)]
    assert any(row is not None and row.source == "ltk" for row in rows)


def test_stage3_acronym_rows_are_only_curated() -> None:
    rows = _curated_acronym_rows(200)
    assert len(rows) == 200
    assert all(row.operations[0].startswith((
        "curated_acronym:", "curated_query:", "curated_university:"
    )) for row in rows)
    mappings = {(row.source, row.target) for row in rows}
    assert ("ltk", "lý thường kiệt") in mappings
    assert ("thpt clhp", "trung học phổ thông chuyên lê hồng phong") in mappings
    assert ("bv bm", "bệnh viện bạch mai") in mappings
    assert ("hust", "đại học bách khoa hà nội") in mappings
    assert ("đh hcmute", "đại học sư phạm kỹ thuật thành phố hồ chí minh") in mappings
    assert ("dh ptit", "học viện công nghệ bưu chính viễn thông") in mappings


def test_university_alias_dictionary_is_normalized() -> None:
    assert all(alias == alias.lower() and " " not in alias for alias in CURATED_UNIVERSITY_ALIASES)
