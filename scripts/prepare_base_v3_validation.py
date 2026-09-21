#!/usr/bin/env python
from __future__ import annotations

import csv
import random
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from reparos.data_registry import HELDOUT_BRANDS


def normalize(text: str) -> str:
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text).strip().lower())
    return " ".join(norm.split())


def is_valid_vietnamese_script(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (0x3040 <= code <= 0x30ff) or (0x4e00 <= code <= 0x9fff) or (0xac00 <= code <= 0xd7af) or (0x1100 <= code <= 0x11ff) or (0x3130 <= code <= 0x318f):
            return False
    return True


def contains_heldout_brand(text: str, precomputed_brands: List[Tuple[str, str, str]]) -> bool:
    clean_text = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split())
    padded = f" {clean_text} "
    for _, b_clean, b_padded in precomputed_brands:
        if b_padded in padded or clean_text == b_clean:
            return True
    return False


def load_eval_queries(eval_dir: Path) -> Set[str]:
    queries = set()
    for p in eval_dir.glob("*.src"):
        with open(p, "r", encoding="utf-8") as f:
            for l in f:
                queries.add(normalize(l))
    for p in eval_dir.glob("*.tgt"):
        with open(p, "r", encoding="utf-8") as f:
            for l in f:
                queries.add(normalize(l))
    return queries


def main():
    eval_dir = Path("data/base_v3_eval")
    eval_queries = load_eval_queries(eval_dir)
    print(f"Loaded {len(eval_queries):,} unique queries from frozen eval suite.")

    precomputed_brands = []
    for brand in HELDOUT_BRANDS:
        b_clean = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in brand.lower()).split())
        precomputed_brands.append((brand, b_clean, f" {b_clean} "))

    valid_csv = Path("data/osm/prepared-v4-leakfree/validation/noisy_pairs.csv")
    print(f"Sampling validation pairs from {valid_csv}...")

    candidates: List[Tuple[str, str]] = []
    with open(valid_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = normalize(row["noisy_query"])
            tgt = normalize(row["correct_query"])
            if not src or not tgt or src == tgt:
                continue
            if src in eval_queries or tgt in eval_queries:
                continue
            if contains_heldout_brand(src, precomputed_brands) or contains_heldout_brand(tgt, precomputed_brands):
                continue
            if not is_valid_vietnamese_script(src) or not is_valid_vietnamese_script(tgt):
                continue

            candidates.append((src, tgt))
            if len(candidates) >= 20_000:
                break

    rng = random.Random(2026)
    rng.shuffle(candidates)
    sampled = candidates[:5000]
    print(f"Selected {len(sampled):,} leak-free, clean validation pairs.")

    out_val_dir = Path("data/base_v3_ablation/validation")
    out_val_dir.mkdir(parents=True, exist_ok=True)
    val_src = out_val_dir / "valid.src"
    val_tgt = out_val_dir / "valid.tgt"

    with open(val_src, "w", encoding="utf-8") as f_s, open(val_tgt, "w", encoding="utf-8") as f_t:
        for s, t in sampled:
            f_s.write(s + "\n")
            f_t.write(t + "\n")
    print(f"Written validation files to {val_src} and {val_tgt}")

    for run_name in ["dataset_run_a", "dataset_run_b", "dataset_run_c"]:
        dest_dir = Path("data/base_v3_ablation") / run_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(val_src, dest_dir / "valid.src")
        shutil.copy2(val_tgt, dest_dir / "valid.tgt")
        print(f"Copied validation set into {dest_dir}")

    print("\nValidation set preparation COMPLETE!")


if __name__ == "__main__":
    main()
