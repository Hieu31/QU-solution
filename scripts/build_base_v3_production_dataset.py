#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from reparos.continual_dataset import (
    generate_address_abbreviation,
    generate_composition,
)
from reparos.data_registry import (
    HELDOUT_BRANDS,
    SEEN_BRANDS,
)
from reparos.quality_filter import ZeroClickQualityFilter


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


COMMON_EXPANSIONS = [
    (re.compile(r"\bbệnh viện\b", re.IGNORECASE), "bv"),
    (re.compile(r"\bủy ban nhân dân\b", re.IGNORECASE), "ubnd"),
    (re.compile(r"\bkhu công nghiệp\b", re.IGNORECASE), "kcn"),
    (re.compile(r"\btrung học phổ thông\b", re.IGNORECASE), "thpt"),
    (re.compile(r"\bđại học\b", re.IGNORECASE), "đh"),
    (re.compile(r"\bthành phố\b", re.IGNORECASE), "tp"),
    (re.compile(r"\bthị xã\b", re.IGNORECASE), "tx"),
    (re.compile(r"\bthị trấn\b", re.IGNORECASE), "tt"),
    (re.compile(r"\bquận\b", re.IGNORECASE), "q"),
    (re.compile(r"\bphường\b", re.IGNORECASE), "p"),
    (re.compile(r"\bđường\b", re.IGNORECASE), "đ"),
]

BRAND_PREFIXES = [
    "quán", "cửa hàng", "chi nhánh", "đại lý", "tiệm", "siêu thị",
    "mua hàng tại", "gần", "ở", "khu vực", "đến", "tìm", "địa chỉ",
]


def build_production_pools(
    eval_queries_set: Set[str],
    precomputed_heldout: List[Tuple[str, str, str]],
    rng: random.Random,
) -> Dict[str, List[Tuple[str, str]]]:
    pools: Dict[str, List[Tuple[str, str]]] = {
        "lane1": [],
        "lane2": [],
        "lane3_clean": [],
        "lane3_brand": [],
        "lane4_dae": [],
        "lane4_identity": [],
    }

    def contains_heldout(text: str) -> bool:
        clean = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split())
        padded = f" {clean} "
        for _, b_clean, b_padded in precomputed_heldout:
            if b_padded in padded or clean == b_clean:
                return True
        return False

    # -------------------------------------------------------------
    # 1. Lane 1: Typing & Diacritics (Quota: 1,600,000, Pool: 2,800,000)
    # -------------------------------------------------------------
    print("\n[1/4] Gathering Lane 1 Pool (Typing & Diacritics - Target ~2.8M pairs)...")
    t0 = time.time()
    train_noisy_src = Path("data/reparos/base-v2-production/v2-32/base/train.noisy.src")
    train_noisy_tgt = Path("data/reparos/base-v2-production/v2-32/base/train.noisy.tgt")

    with open(train_noisy_src, "r", encoding="utf-8") as f_s, open(train_noisy_tgt, "r", encoding="utf-8") as f_t:
        for s_raw, t_raw in zip(f_s, f_t):
            s = normalize(s_raw)
            t = normalize(t_raw)
            if not s or not t or s == t:
                continue
            if s in eval_queries_set or t in eval_queries_set:
                continue
            if not is_valid_vietnamese_script(s) or not is_valid_vietnamese_script(t):
                continue
            if contains_heldout(s) or contains_heldout(t):
                continue

            pools["lane1"].append((s, t))
            if len(pools["lane1"]) >= 2_800_000:
                break
    print(f"  -> Lane 1 gathered: {len(pools['lane1']):,} clean pairs in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------
    # 2. Lane 2: Address & Acronyms (Quota: 1,000,000, Pool: ~1.2M)
    # -------------------------------------------------------------
    print("\n[2/4] Gathering Lane 2 Pool (Address, Acronyms, Composition - Target ~1.2M pairs)...")
    t0 = time.time()
    corpus_file = Path("data/osm/prepared-v4-leakfree/train/corpus.txt")
    with open(corpus_file, "r", encoding="utf-8") as f:
        clean_osm_lines = [
            normalize(l) for l in f
            if len(l.strip()) > 8 and is_valid_vietnamese_script(l) and not contains_heldout(l)
        ]
    print(f"  Loaded {len(clean_osm_lines):,} clean OSM lines for synthesis.")

    # 2a. Address Abbreviations (Pass 1 & Pass 2 with random variation)
    print("  Generating Address Abbreviations...")
    for repeat in range(2):
        for line in clean_osm_lines:
            if line in eval_queries_set:
                continue
            abbrev = generate_address_abbreviation(line, rng)
            if abbrev and abbrev != line and abbrev not in eval_queries_set and is_valid_vietnamese_script(abbrev):
                pools["lane2"].append((abbrev, line))
                if len(pools["lane2"]) >= 500_000:
                    break
        if len(pools["lane2"]) >= 500_000:
            break

    # 2b. Compositional Variations (Pass 1 & Pass 2)
    print("  Generating Compositional variations...")
    for repeat in range(2):
        for line in clean_osm_lines:
            if line in eval_queries_set:
                continue
            comp = generate_composition(line, rng)
            if comp and comp != line and comp not in eval_queries_set and is_valid_vietnamese_script(comp):
                pools["lane2"].append((comp, line))
                if len(pools["lane2"]) >= 900_000:
                    break
        if len(pools["lane2"]) >= 900_000:
            break

    # 2c. POI / Administrative Acronym Expansions
    print("  Generating POI & Acronym expansions from real corpus...")
    for line in clean_osm_lines:
        if line in eval_queries_set:
            continue
        mod_line = line
        modified = False
        for pattern, replacement in COMMON_EXPANSIONS:
            if pattern.search(mod_line):
                if rng.random() < 0.7:
                    mod_line = pattern.sub(replacement, mod_line)
                    modified = True
        if modified and mod_line != line and mod_line not in eval_queries_set:
            pools["lane2"].append((mod_line, line))
            if len(pools["lane2"]) >= 1_300_000:
                break
    print(f"  -> Lane 2 gathered: {len(pools['lane2']):,} clean pairs in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------
    # 3. Lane 3: Clean OSM & Seen Brands (Quota: 1,000,000)
    # -------------------------------------------------------------
    print("\n[3/4] Gathering Lane 3 Pool (Clean Queries & Seen Brands - Target ~1.2M pairs)...")
    t0 = time.time()
    # 3a. Clean OSM + Clean V2
    for line in clean_osm_lines:
        if len(line.split()) >= 2 and line not in eval_queries_set and not contains_heldout(line):
            pools["lane3_clean"].append((line, line))
            if len(pools["lane3_clean"]) >= 350_000:
                break

    clean_v2_file = Path("data/reparos/base-v2-production/v2-32/base/train.clean.src")
    if clean_v2_file.exists():
        with open(clean_v2_file, "r", encoding="utf-8") as f:
            for l in f:
                c = normalize(l)
                if c and len(c.split()) >= 2 and c not in eval_queries_set and not contains_heldout(c) and is_valid_vietnamese_script(c):
                    pools["lane3_clean"].append((c, c))
                    if len(pools["lane3_clean"]) >= 600_000:
                        break

    # 3b. Seen Brands combined with real addresses (700,000 unique combinations)
    print("  Synthesizing Seen Brands with real street addresses...")
    seen_brands_list = list(SEEN_BRANDS)
    for repeat in range(2):
        for i, line in enumerate(clean_osm_lines):
            brand = seen_brands_list[(i + repeat * 19) % len(seen_brands_list)]
            prefix = rng.choice(BRAND_PREFIXES)
            choice = rng.randint(0, 2)
            if choice == 0:
                text = f"{brand} {line}"
            elif choice == 1:
                text = f"{prefix} {brand} {line}"
            else:
                text = f"{brand} tại {line}"
            text = normalize(text)
            if text not in eval_queries_set:
                pools["lane3_brand"].append((text, text))
                if len(pools["lane3_brand"]) >= 700_000:
                    break
        if len(pools["lane3_brand"]) >= 700_000:
            break
    print(f"  -> Lane 3 gathered: {len(pools['lane3_clean']):,} Clean + {len(pools['lane3_brand']):,} Brands in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------
    # 4. Lane 4: Real Query Adaptation (Quota: 400,000 - 280k DAE + 120k Id)
    # -------------------------------------------------------------
    print("\n[4/4] Gathering Lane 4 Pool from zero_click.csv (Confidence-Gated)...")
    t0 = time.time()
    qf = ZeroClickQualityFilter()
    zc_path = Path("data/zero_click.csv")
    high_conf_queries: List[str] = []

    with open(zc_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or not row[0]:
                continue
            cat, _ = qf.classify(row[0])
            if cat == "HIGH_CONFIDENCE_CLEAN":
                q = normalize(row[0])
                if q not in eval_queries_set and not contains_heldout(q) and is_valid_vietnamese_script(q):
                    high_conf_queries.append(q)
                    if len(high_conf_queries) >= 350_000:
                        break

    print(f"  Extracted {len(high_conf_queries):,} high-confidence clean queries from zero_click.")
    for q in high_conf_queries:
        s_dae, t_dae = qf.generate_dae_pair(q, rng)
        if s_dae not in eval_queries_set and is_valid_vietnamese_script(s_dae):
            pools["lane4_dae"].append((s_dae, t_dae))
        s_id, t_id = qf.generate_identity_pair(q)
        pools["lane4_identity"].append((s_id, t_id))
    print(f"  -> Lane 4 gathered: {len(pools['lane4_dae']):,} DAE + {len(pools['lane4_identity']):,} Identity in {time.time()-t0:.1f}s")

    return pools


def assemble_production_dataset(
    pools: Dict[str, List[Tuple[str, str]]],
    precomputed_heldout: List[Tuple[str, str, str]],
    output_dir: Path,
    rng: random.Random,
) -> Tuple[Path, Path]:
    print("\n==================================================================")
    print("   ASSEMBLING BASE V3 PRODUCTION DATASET (EXACTLY 4,000,000 PAIRS)")
    print("   Winning Recipe: 40% L1 + 25% L2 + 25% L3 + 7% DAE + 3% Identity")
    print("==================================================================")

    def is_heldout_leak(text: str) -> bool:
        clean_text = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split())
        padded = f" {clean_text} "
        for _, b_clean, b_padded in precomputed_heldout:
            if b_padded in padded or clean_text == b_clean:
                return True
        return False

    # Quotas:
    # Lane 1: 1,600,000 (40%)
    # Lane 2: 1,000,000 (25%)
    # Lane 3: 1,000,000 (25% -> 500k clean + 500k brand)
    # Lane 4: 400,000 (10% -> 280k DAE + 120k Identity)
    sources = [
        (pools["lane4_identity"], 120_000),
        (pools["lane3_clean"], 500_000),
        (pools["lane3_brand"], 500_000),
        (pools["lane4_dae"], 280_000),
        (pools["lane2"], 1_000_000),
        (pools["lane1"], 1_600_000),
    ]

    target_total = 4_000_000
    resolved_dict: Dict[str, str] = {}

    print("\nPass 1: Allocating quotas with 1-to-1 deterministic contradiction resolution...")
    for pool, quota in sources:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        added = 0
        for s, t in shuffled:
            if is_heldout_leak(s) or is_heldout_leak(t):
                continue
            if s not in resolved_dict:
                resolved_dict[s] = t
                added += 1
                if added >= quota:
                    break
        print(f"  - Allocated {added:,} / {quota:,} pairs (Current total: {len(resolved_dict):,})")

    # Pass 2: Top-up if duplicate elimination resulted in slightly fewer items
    if len(resolved_dict) < target_total:
        print(f"\nPass 2: Topping up {target_total - len(resolved_dict):,} pairs...")
        for pool, _ in sources:
            shuffled = list(pool)
            rng.shuffle(shuffled)
            for s, t in shuffled:
                if is_heldout_leak(s) or is_heldout_leak(t):
                    continue
                if s not in resolved_dict:
                    resolved_dict[s] = t
                    if len(resolved_dict) >= target_total:
                        break
            if len(resolved_dict) >= target_total:
                break

    pairs = list(resolved_dict.items())[:target_total]
    assert len(pairs) == target_total, f"Expected {target_total:,}, got {len(pairs):,}"
    rng.shuffle(pairs)

    output_dir.mkdir(parents=True, exist_ok=True)
    src_file = output_dir / "train.src"
    tgt_file = output_dir / "train.tgt"

    print(f"\nWriting {len(pairs):,} pairs to {src_file} and {tgt_file}...")
    with open(src_file, "w", encoding="utf-8") as f_s, open(tgt_file, "w", encoding="utf-8") as f_t:
        for s, t in pairs:
            f_s.write(s + "\n")
            f_t.write(t + "\n")

    # Copy validation set (5,000 pairs)
    val_src = Path("data/base_v3_ablation/validation/valid.src")
    val_tgt = Path("data/base_v3_ablation/validation/valid.tgt")
    shutil.copy2(val_src, output_dir / "valid.src")
    shutil.copy2(val_tgt, output_dir / "valid.tgt")
    print(f"Copied clean validation files into {output_dir}")

    # Write Manifest
    manifest = {
        "dataset_name": "reparos-base-v3-production",
        "total_train_pairs": target_total,
        "validation_pairs": 5000,
        "recipe": {
            "lane1_typing_diacritics": 1600000,
            "lane2_address_acronym_composition": 1000000,
            "lane3_clean_and_seen_brands": 1000000,
            "lane4_real_search_adaptation": {
                "dae": 280000,
                "identity": 120000,
            }
        },
        "zero_leakage_certified": True,
        "zero_contradiction_certified": True,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n>>> Base V3 Production Dataset (4,000,000 pairs) successfully built at {output_dir}!")
    return src_file, tgt_file


def main():
    parser = argparse.ArgumentParser(description="Build Base V3 Production 4,000,000 pairs dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/base_v3_production"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/base_v3_eval"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_queries = load_eval_queries(args.eval_dir)
    print(f"Loaded {len(eval_queries):,} unique queries from frozen eval suite.")

    precomputed_heldout = []
    for b in HELDOUT_BRANDS:
        b_clean = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in b.lower()).split())
        precomputed_heldout.append((b, b_clean, f" {b_clean} "))

    pools = build_production_pools(eval_queries, precomputed_heldout, rng)
    assemble_production_dataset(pools, precomputed_heldout, args.output_dir, rng)


if __name__ == "__main__":
    main()
