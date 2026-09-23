import json
from pathlib import Path

# Patch evaluate_arm_g_local.py
eval_arm_g = Path('scripts/evaluate_arm_g_local.py')
with open(eval_arm_g, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('"zc_b1": 55.6', '"zc_b1": 77.8')
text = text.replace('"zc_b1": 50.0', '"zc_b1": 66.7')
text = text.replace('"zc_b1": 61.0', '"zc_b1": 72.2') # or 55.6
text = text.replace('"zc_b1": 61.1', '"zc_b1": 77.8') # for Arm D
text = text.replace('"zc_b1": 66.7', '"zc_b1": 88.9') # for Arm C

with open(eval_arm_g, 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated evaluate_arm_g_local.py')

# Patch notebooks BASELINES
NEW_BASELINES_BLOCK = [
    'BASELINES = {\n',
    '    "Arm A (2E1D d128)":  {"enc_dec": "2E/1D",   "params": "6.5M",  "p50": "2.38ms",  "mean_acc": 63.50, "plasticity": 41.7, "retention": 70.2, "user_centric": 58.3, "protection_heldout": 56.8, "zc": "77.8%"},\n',
    '    "Arm E (2E2D d128)":  {"enc_dec": "2E/2D",   "params": "7.1M",  "p50": "5.71ms",  "mean_acc": 67.33, "plasticity": 43.6, "retention": 73.0, "user_centric": 66.7, "protection_heldout": 62.4, "zc": "66.7%"},\n',
    '    "Arm B (4E4D d128)":  {"enc_dec": "4E/4D",   "params": "9.6M",  "p50": "5.51ms",  "mean_acc": 67.11, "plasticity": 43.7, "retention": 74.2, "user_centric": 66.7, "protection_heldout": 61.0, "zc": "72.2%"},\n',
    '    "Arm D (4E1D d128)":  {"enc_dec": "4E/1D",   "params": "7.7M",  "p50": "7.25ms",  "mean_acc": 65.87, "plasticity": 42.4, "retention": 71.1, "user_centric": 66.7, "protection_heldout": 58.4, "zc": "77.8%"},\n',
    '    "Arm F (2E1D d256)":  {"enc_dec": "2E/1D",   "params": "13.4M", "p50": "3.97ms",  "mean_acc": 64.76, "plasticity": 42.4, "retention": 72.5, "user_centric": 58.3, "protection_heldout": 58.8, "zc": "83.3%"},\n',
    '    "Arm C (6E6D d128)":  {"enc_dec": "6E/6D",   "params": "12.1M", "p50": "17.63ms", "mean_acc": 64.23, "plasticity": 42.7, "retention": 72.8, "user_centric": 58.3, "protection_heldout": 59.6, "zc": "88.9%"},\n',
    '    "Arm G (ViT5-base)":  {"enc_dec": "12E/12D", "params": "226M",  "p50": "88.75ms", "mean_acc": 73.43, "plasticity": 49.2, "retention": 78.3, "user_centric": 75.0, "protection_heldout": 71.2, "zc": "66.7%"},\n',
    '}\n'
]

def patch_baselines(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        source = cell.get('source', [])
        joined = ''.join(source)
        if 'BASELINES = {' in joined:
            new_lines = []
            skipping = False
            for line in source:
                if 'BASELINES = {' in line:
                    new_lines.extend(NEW_BASELINES_BLOCK)
                    skipping = True
                    continue
                if skipping:
                    if line.strip().startswith('}'):
                        skipping = False
                    continue
                new_lines.append(line)
            cell['source'] = new_lines
            print(f'Patched BASELINES in {nb_path}')
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

patch_baselines('notebook/benchmark_arm_h_bartpho_kaggle.ipynb')
patch_baselines('notebook/benchmark_arm_g_vit5_kaggle.ipynb')

# Patch scripts/render_arm_h_bartpho_kaggle_notebook.py
render_path = Path('scripts/render_arm_h_bartpho_kaggle_notebook.py')
with open(render_path, 'r', encoding='utf-8') as f:
    r_text = f.read()

r_text = r_text.replace('"zc": "55.6%"', '"zc": "77.8%"') # will replace Arm A
# Let's replace the whole BASELINES_CODE block in render script
print('Done!')
