import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Load Arms A-F
with open('artifacts/pilot_scaling/benchmark_results.json', encoding='utf-8') as f:
    pilot_data = json.load(f)

# Load Arm G
with open('artifacts/arm_g_vit5_base/benchmark_results_arm_g.json', encoding='utf-8') as f:
    arm_g_data = json.load(f)

# Arm H from user prompt
arm_h_preds = {
    'cau vuot song than': 'cầu vượt sông thần',
    'cho ba chieu': 'chợ bà chiểu',
    'nga 4 hang xanh': 'ngã 4 hàng xanh',
    'nga 3 vung tau': 'ngã 3 vũng tàu',
    'bv cho ray': 'bệnh viện chợ ray',
    'Bv nhi đong': 'ben nhi đong',
    'benh vien 175': 'bệnh viện 175',
    'dh kinh te tphcm': 'đại học kinh tế thành phố hồ chí minh',
    '86 xo viet nghe tinh p19 binh thanh': '86 xô viết nghệ tĩnh phường 19 bình thạnh',
    'duong le van viet q9': 'đường lê văn việt quận 9',
    'hem 212 thoai ngoc hau phuong phu thanh': 'hẻm 212 thoại ngọc hầu phường phú thành',
    'kcn song than 1': 'khu công nghiệp sông thần 1',
    'ubnd xa phuoc thai': 'ban nhân dân xã phước thái',
    '158/16 binh quew': '158/16 bình quới',
    'chung cu ha': 'chung cư hà nội',
    'ngã 6 tahnhf': 'ngã 6 thành',
    'tttm aeon mall tan phu': 'trung tâm thương mại aeon mall tân phú',
    'dh bach khoa ha noi': 'đại học bách khoa hà nội'
}

old_targets = {
    'cau vuot song than': 'cầu vượt sóng thần',
    'cho ba chieu': 'chợ bà chiểu',
    'nga 4 hang xanh': 'ngã 4 hàng xanh',
    'nga 3 vung tau': 'ngã 3 vũng tàu',
    'bv cho ray': 'bệnh viện chợ rẫy',
    'Bv nhi đong': 'bệnh viện nhi đồng',
    'benh vien 175': 'bệnh viện 175',
    'dh kinh te tphcm': 'đại học kinh tế thành phố hồ chí minh',
    '86 xo viet nghe tinh p19 binh thanh': '86 xô viết nghệ tĩnh phường 19 bình thạnh',
    'duong le van viet q9': 'đường lê văn việt quận 9',
    'hem 212 thoai ngoc hau phuong phu thanh': 'hẻm 212 thoại ngọc hầu phường phú thạnh',
    'kcn song than 1': 'kcn sóng thần 1',
    'ubnd xa phuoc thai': 'ubnd xã phước thái',
    '158/16 binh quew': '158/16 bình quêw',
    'chung cu ha': 'chung cư ha',
    'ngã 6 tahnhf': 'ngã 6 thanhf',
    'tttm aeon mall tan phu': 'tttm aeon mall tân phú',
    'dh bach khoa ha noi': 'đại học bách khoa hà nội'
}

canonical_targets = {
    'cau vuot song than': 'cầu vượt sóng thần',
    'cho ba chieu': 'chợ bà chiểu',
    'nga 4 hang xanh': 'ngã 4 hàng xanh',
    'nga 3 vung tau': 'ngã 3 vũng tàu',
    'bv cho ray': 'bệnh viện chợ rẫy',
    'Bv nhi đong': 'bệnh viện nhi đồng',
    'benh vien 175': 'bệnh viện 175',
    'dh kinh te tphcm': 'đại học kinh tế thành phố hồ chí minh',
    '86 xo viet nghe tinh p19 binh thanh': '86 xô viết nghệ tĩnh phường 19 bình thạnh',
    'duong le van viet q9': 'đường lê văn việt quận 9',
    'hem 212 thoai ngoc hau phuong phu thanh': 'hẻm 212 thoại ngọc hầu phường phú thạnh',
    'kcn song than 1': 'khu công nghiệp sóng thần 1',
    'ubnd xa phuoc thai': 'ủy ban nhân dân xã phước thái',
    '158/16 binh quew': '158/16 bình quới',
    'chung cu ha': 'chung cư hà',
    'ngã 6 tahnhf': 'ngã 6 thành',
    'tttm aeon mall tan phu': 'trung tâm thương mại aeon mall tân phú',
    'dh bach khoa ha noi': 'đại học bách khoa hà nội'
}

all_arms = {}
for k, v in pilot_data.items():
    all_arms[v['name']] = v['zero_click_preds']
all_arms['Arm G (ViT5-base)'] = arm_g_data['zero_click']['preds_beam1']
all_arms['Arm H (BARTpho)'] = arm_h_preds

print(f"{'Mô hình':<25} | {'Điểm Cũ (Old)':<16} | {'Điểm Mới Chuẩn (Canonical)':<22} | {'Độ Chênh':<8}")
print("-" * 80)
for name, preds in all_arms.items():
    old_c = sum(1 for q, tgt in old_targets.items() if preds.get(q, '').strip().lower() == tgt.strip().lower())
    can_c = sum(1 for q, tgt in canonical_targets.items() if preds.get(q, '').strip().lower() == tgt.strip().lower())
    delta = (can_c - old_c) / 18 * 100
    print(f"{name:<25} | {old_c}/18 ({old_c/18*100:5.1f}%)   | {can_c}/18 ({can_c/18*100:5.1f}%)             | {delta:+5.1f}%")

print("\n" + "=" * 80)
print("CHI TIẾT DỰ ĐOÁN TỪNG CA TRÊN TẤT CẢ CÁC ARM:")
print("=" * 80)
for q, can_tgt in canonical_targets.items():
    old_tgt = old_targets[q]
    diff_marker = " [NHÃN CŨ KHÁC]" if old_tgt != can_tgt else ""
    print(f"\nTruy vấn: '{q}'")
    print(f"  Target chuẩn: '{can_tgt}'{diff_marker} (Cũ: '{old_tgt}')")
    for arm_name, preds in all_arms.items():
        p = preds.get(q, '')
        st = '✅' if p.strip().lower() == can_tgt.strip().lower() else '❌'
        print(f"    {st} {arm_name:<24}: '{p}'")
