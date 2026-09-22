from reparos.canonical_contract import (
    CANONICAL_ADMIN_MAP,
    canonicalize_target,
    has_consecutive_duplicates,
    remove_consecutive_duplicates,
    is_canonical_valid_pair,
)


def test_consecutive_duplicates():
    assert has_consecutive_duplicates("đường đường số 6") is True
    assert has_consecutive_duplicates("quận quận 1") is True
    assert has_consecutive_duplicates("đường số 6") is False
    assert remove_consecutive_duplicates("đường đường số 6") == "đường số 6"
    assert remove_consecutive_duplicates("565 đường đường số 15") == "565 đường số 15"


def test_canonicalize_target():
    assert canonicalize_target("ubnd phường bến nghé") == "ủy ban nhân dân phường bến nghé"
    assert canonicalize_target("bv bạch mai") == "bệnh viện bạch mai"
    assert canonicalize_target("kcn tân bình") == "khu công nghiệp tân bình"
    assert canonicalize_target("đh bách khoa") == "đại học bách khoa"
    assert canonicalize_target("thpt chuyên lê hồng phong") == "trung học phổ thông chuyên lê hồng phong"
    assert canonicalize_target("đ đường số 6") == "đường số 6"
    assert canonicalize_target("đường đường số 6") == "đường số 6"
    assert canonicalize_target("cty sơn hà") == "công ty sơn hà"
    assert canonicalize_target("kđt the manor") == "khu đô thị the manor"
    assert canonicalize_target("tttm vincom") == "trung tâm thương mại vincom"
    assert canonicalize_target("p 12") == "phường 12"
    assert canonicalize_target("q 1") == "quận 1"
    assert canonicalize_target("thành phố bắc ninh bắc ninh bac ninh") == "thành phố bắc ninh"
    assert canonicalize_target("hai bà trưng hà nội hà nội") == "hai bà trưng hà nội"
    assert canonicalize_target("quận 7 thành phố hồ chí minh thành phố hồ chí minh") == "quận 7 thành phố hồ chí minh"


def test_is_canonical_valid_pair():
    # Valid pairs
    valid, _ = is_canonical_valid_pair("đ nguyễn trãi", "đường nguyễn trãi")
    assert valid is True

    valid, _ = is_canonical_valid_pair("bv bạch mai", "bệnh viện bạch mai")
    assert valid is True

    # Invalid: duplicate words in target
    valid, reason = is_canonical_valid_pair("đ đường số 6", "đường đường số 6")
    assert valid is False
    assert reason == "target_has_duplicate_consecutive_words"

    # Invalid: unexpanded abbreviation in target
    valid, reason = is_canonical_valid_pair("ubnd q1", "ubnd quận 1")
    assert valid is False
    assert "target_contains_unexpanded" in reason

    # Invalid: d/đ mapped to phố
    valid, reason = is_canonical_valid_pair("đ huế", "phố huế")
    assert valid is False
    assert reason == "ambiguous_d_mapped_to_pho"

    # Invalid: phrasal repetition in target
    valid, reason = is_canonical_valid_pair("hà nội", "hà nội hà nội")
    assert valid is False
    assert "phrasal_duplicate" in reason

    # Invalid: unexpanded business abbreviation in target
    valid, reason = is_canonical_valid_pair("cty tân á", "cty tân á")
    assert valid is False
    assert "unexpanded_business_abbrev" in reason


if __name__ == "__main__":
    test_consecutive_duplicates()
    test_canonicalize_target()
    test_is_canonical_valid_pair()
    print("ALL CANONICAL CONTRACT TESTS PASSED!")

