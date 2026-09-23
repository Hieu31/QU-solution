"""
Package BARTpho-syllable-base snapshot thành Kaggle Dataset.

Steps:
1. Verify snapshot at artifacts/arm_h_bartpho_syllable_base
2. Create dataset-metadata.json
3. Print upload command để chạy bằng Kaggle CLI

Usage: python scripts/package_arm_h_bartpho_kaggle_dataset.py
"""
import json
import shutil
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
SNAPSHOT    = REPO_ROOT / "artifacts" / "arm_h_bartpho_syllable_base"
DATASET_DIR = REPO_ROOT / "artifacts" / "kaggle_dataset_bartpho_syllable"

REQUIRED_FILES = ["config.json", "tokenizer.json", "sentencepiece.bpe.model"]

WEIGHT_FILES   = ["pytorch_model.bin", "model.safetensors"]

# ── Verify snapshot ──────────────────────────────────────────────────────────
print(f"Verifying BARTpho snapshot at: {SNAPSHOT}")
assert SNAPSHOT.exists(), f"Snapshot not found: {SNAPSHOT}"

missing = [f for f in REQUIRED_FILES if not (SNAPSHOT / f).exists()]
assert not missing, f"Missing required files: {missing}"

has_weights = any((SNAPSHOT / w).exists() for w in WEIGHT_FILES)
assert has_weights, f"No model weights found (checked: {WEIGHT_FILES})"

files = list(SNAPSHOT.iterdir())
total_mb = sum(f.stat().st_size for f in SNAPSHOT.rglob("*") if f.is_file()) / 1e6
print(f"  Files: {len(files)}")
print(f"  Total size: {total_mb:.1f} MB")
print("  OK — Snapshot is valid")

# ── Build Kaggle Dataset directory ───────────────────────────────────────────
if DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)
DATASET_DIR.mkdir(parents=True)

# Copy snapshot into bartpho-syllable-base/ subfolder (convention = model name)
model_dir = DATASET_DIR / "bartpho-syllable-base"
print(f"\nCopying snapshot to: {model_dir}")
shutil.copytree(SNAPSHOT, model_dir)
print(f"  Copied OK")

# ── Write dataset-metadata.json ───────────────────────────────────────────────
# Kaggle Dataset API format
metadata = {
    "title": "BARTpho Syllable Base (vinai)",
    "id": "vanhieu1125/bartpho-syllable-base",   # <-- thay username nếu cần
    "licenses": [{"name": "MIT"}],
}
meta_path = DATASET_DIR / "dataset-metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"\nWrote dataset-metadata.json")
print(json.dumps(metadata, indent=2))

# ── Print instructions ────────────────────────────────────────────────────────
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("\n" + "=" * 60)
print("NEXT STEPS -- Upload to Kaggle:")
print("=" * 60)
print(f"""
1. Cai Kaggle CLI va set API key:
   pip install kaggle

2. Tao dataset moi (chi lan dau):
   kaggle datasets create -p "{DATASET_DIR}"

3. Neu dataset da ton tai, upload version moi:
   kaggle datasets version -p "{DATASET_DIR}" -m "BARTpho-syllable-base v1"

4. Mount vao notebook Arm H:
   Kaggle notebook -> Add Data -> "vanhieu1125/bartpho-syllable-base"
   Path: /kaggle/input/datasets/vanhieu1125/bartpho-syllable-base/bartpho-syllable-base/
""")
print("=" * 60)
print("\n[+] Dataset folder ready:", DATASET_DIR)
