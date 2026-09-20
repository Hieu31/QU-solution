from reparos.base_v2 import CleanSeed
from reparos.base_v2_production import (
    generate_variants,
    protection_reason,
    repeated_phrase,
    target_rejection_reason,
)


def test_quality_gate_rejects_repeated_address_and_pii() -> None:
    assert repeated_phrase("34 hàng giấy hà nội thành phố hà nội")
    assert target_rejection_reason("34 hàng giấy hà nội thành phố hà nội") == "repeated_phrase"
    assert target_rejection_reason("gọi 0912345678") == "phone_like"


def test_short_and_alphanumeric_queries_are_protected() -> None:
    assert protection_reason("n36") == "alphanumeric_code"
    assert protection_reason("q1") == "short_ambiguous"
    for text in ("n36", "q1"):
        seed = CleanSeed(text, text, "g", "train")
        assert generate_variants(seed, 7) == []


def test_regular_location_still_gets_variants() -> None:
    seed = CleanSeed("bệnh viện bạch mai", "bệnh viện bạch mai", "g", "train")
    rows = generate_variants(seed, 7)
    assert rows
    assert any(row.family == "address_abbreviation" for row in rows)
