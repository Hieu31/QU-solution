import sys
sys.stdout.reconfigure(encoding='utf-8')
from reparos.quality_filter import ZeroClickQualityFilter

qf = ZeroClickQualityFilter()

test_cases = [
    # Clean queries -> HIGH_CONFIDENCE_CLEAN
    ("Trường Đại Học Thương Mại", "HIGH_CONFIDENCE_CLEAN"),
    ("Chợ Hoa Quảng Bá", "HIGH_CONFIDENCE_CLEAN"),
    ("212/22 Đường Nguyễn Oanh", "HIGH_CONFIDENCE_CLEAN"),
    ("Bến Xe Miền Tây", "HIGH_CONFIDENCE_CLEAN"),
    ("Phố Tống Duy Tân", "HIGH_CONFIDENCE_CLEAN"),
    ("Bảo Minh Hotel - 305B - 308 Đường Tô Ký", "HIGH_CONFIDENCE_CLEAN"),
    ("quán ăn ngon quận 1", "HIGH_CONFIDENCE_CLEAN"),

    # Keystroke autocomplete increments / prefixes -> AMBIGUOUS_PREFIX
    ("199", "AMBIGUOUS_PREFIX"),
    ("199 hồ", "AMBIGUOUS_PREFIX"),
    ("199 hồ Tùng", "AMBIGUOUS_PREFIX"),
    ("71/", "AMBIGUOUS_PREFIX"),
    ("36 ngõ", "AMBIGUOUS_PREFIX"),
    ("nhà gas t", "AMBIGUOUS_PREFIX"),
    ("nhà gas t3", "NOISY_TYPO"),
    ("sân", "AMBIGUOUS_PREFIX"),
    ("bên", "AMBIGUOUS_PREFIX"),

    # Obvious typos / unaccented -> NOISY_TYPO
    ("bên xe", "NOISY_TYPO"),
    ("khách san phương", "NOISY_TYPO"),
    ("quan an ngon quan 1", "NOISY_TYPO"),
    ("benh vien cho ray", "NOISY_TYPO"),

    # Contradictions with Lane 2 -> CONTRADICTION_LANE2
    ("bv chợ rẫy", "CONTRADICTION_LANE2"),
    ("bv chợ ray", "CONTRADICTION_LANE2"),
    ("d. nguyễn trãi q1", "CONTRADICTION_LANE2"),
    ("tp hcm gần đây", "CONTRADICTION_LANE2"),
]

print("=== Running ZeroClickQualityFilter Verification ===")
passed = 0
failed = 0

for raw, expected in test_cases:
    cat, reason = qf.classify(raw)
    status = "PASS" if cat == expected else "FAIL"
    if cat == expected:
        passed += 1
    else:
        failed += 1
    print(f"[{status}] Query: '{raw}' -> Got: {cat} (Expected: {expected}, Reason: {reason})")

print(f"\nTotal Passed: {passed}/{len(test_cases)}")
assert failed == 0, f"{failed} test cases failed!"
print(">>> ALL TESTS PASSED SUCCESSFULLY! Quality filter is mathematically sound!")
