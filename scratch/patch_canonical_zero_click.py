import json
from pathlib import Path

CANONICAL_CASES_BLOCK = [
    "ZERO_CLICK_CASES = [\n",
    "    (\"cau vuot song than\",                         \"cầu vượt sóng thần\"),\n",
    "    (\"cho ba chieu\",                               \"chợ bà chiểu\"),\n",
    "    (\"nga 4 hang xanh\",                            \"ngã 4 hàng xanh\"),\n",
    "    (\"nga 3 vung tau\",                             \"ngã 3 vũng tàu\"),\n",
    "    (\"bv cho ray\",                                 \"bệnh viện chợ rẫy\"),\n",
    "    (\"Bv nhi đong\",                                \"bệnh viện nhi đồng\"),\n",
    "    (\"benh vien 175\",                              \"bệnh viện 175\"),\n",
    "    (\"dh kinh te tphcm\",                           \"đại học kinh tế thành phố hồ chí minh\"),\n",
    "    (\"86 xo viet nghe tinh p19 binh thanh\",        \"86 xô viết nghệ tĩnh phường 19 bình thạnh\"),\n",
    "    (\"duong le van viet q9\",                       \"đường lê văn việt quận 9\"),\n",
    "    (\"hem 212 thoai ngoc hau phuong phu thanh\",    \"hẻm 212 thoại ngọc hầu phường phú thạnh\"),\n",
    "    (\"kcn song than 1\",                            \"khu công nghiệp sóng thần 1\"),\n",
    "    (\"ubnd xa phuoc thai\",                         \"ủy ban nhân dân xã phước thái\"),\n",
    "    (\"158/16 binh quew\",                           \"158/16 bình quới\"),\n",
    "    (\"chung cu ha\",                                \"chung cư hà\"),\n",
    "    (\"ngã 6 tahnhf\",                               \"ngã 6 thành\"),\n",
    "    (\"tttm aeon mall tan phu\",                     \"trung tâm thương mại aeon mall tân phú\"),\n",
    "    (\"dh bach khoa ha noi\",                        \"đại học bách khoa hà nội\"),\n",
    "]\n"
]

def patch_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        source = cell.get('source', [])
        joined = ''.join(source)
        if 'ZERO_CLICK_CASES = [' in joined:
            new_lines = []
            skipping = False
            for line in source:
                if 'ZERO_CLICK_CASES = [' in line:
                    new_lines.extend(CANONICAL_CASES_BLOCK)
                    skipping = True
                    continue
                if skipping:
                    if line.strip().startswith(']'):
                        skipping = False
                    continue
                new_lines.append(line)
            cell['source'] = new_lines
            print(f'Patched {nb_path}')
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

patch_notebook('notebook/benchmark_arm_h_bartpho_kaggle.ipynb')
patch_notebook('notebook/benchmark_arm_g_vit5_kaggle.ipynb')

# Also patch scripts/render_arm_h_bartpho_kaggle_notebook.py
render_path = Path('scripts/render_arm_h_bartpho_kaggle_notebook.py')
with open(render_path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('("kcn song than 1",                            "kcn sóng thần 1"),',
                    '("kcn song than 1",                            "khu công nghiệp sóng thần 1"),')
text = text.replace('("ubnd xa phuoc thai",                         "ubnd xã phước thái"),',
                    '("ubnd xa phuoc thai",                         "ủy ban nhân dân xã phước thái"),')
text = text.replace('("158/16 binh quew",                           "158/16 bình quêw"),',
                    '("158/16 binh quew",                           "158/16 bình quới"),')
text = text.replace('("chung cu ha",                                "chung cư ha"),',
                    '("chung cu ha",                                "chung cư hà"),')
text = text.replace('("ngã 6 tahnhf",                               "ngã 6 thanhf"),',
                    '("ngã 6 tahnhf",                               "ngã 6 thành"),')
text = text.replace('("tttm aeon mall tan phu",                     "tttm aeon mall tân phú"),',
                    '("tttm aeon mall tan phu",                     "trung tâm thương mại aeon mall tân phú"),')

with open(render_path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Patched scripts/render_arm_h_bartpho_kaggle_notebook.py')
