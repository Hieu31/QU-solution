from reparos.base_v2 import CleanSeed
from reparos.base_v2_strict import generate_variants


def test_composite_family_requires_every_operation_to_apply() -> None:
    seed = CleanSeed("bệnh viện bạch mai", "bệnh viện bạch mai", "g1", "train")
    rows = generate_variants(seed, 7)
    families = {row.family for row in rows}
    assert "address_symbol_diacritics" not in families
    assert "address_symbol" not in families
    assert "address_abbreviation" in families


def test_short_query_does_not_receive_fake_boundary_variants() -> None:
    seed = CleanSeed("cầu", "cầu", "g1", "train")
    rows = generate_variants(seed, 7)
    assert not any("boundary" in row.operations for row in rows)
