from reparos.base_v2_release import protection_reason, repeated_phrase, target_rejection_reason


def test_repetition_gate_keeps_legitimate_name_plus_address() -> None:
    assert not repeated_phrase("đài tưởng niệm khâm thiên - 45 khâm thiên")
    assert repeated_phrase("ngõ 481 ngọc lâm long biên hà nội hà nội")
    assert repeated_phrase("34 hàng giấy hà nội thành phố hà nội")


def test_numeric_query_is_protected_not_rejected() -> None:
    assert target_rejection_reason("40") is None
    assert protection_reason("40") == "numeric_or_symbol"
