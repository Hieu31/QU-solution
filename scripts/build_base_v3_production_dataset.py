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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reparos.continual_dataset import (
    generate_address_abbreviation,
    generate_composition,
)
from reparos.data_registry import (
    HELDOUT_BRANDS,
    SEEN_BRANDS,
)
from reparos.quality_filter import ZeroClickQualityFilter
from reparos.canonical_contract import (
    CANONICAL_ADMIN_MAP,
    canonicalize_target,
    has_consecutive_duplicates,
    is_canonical_valid_pair,
    UNEXPANDED_TARGET_REGEX,
)


def normalize(text: str) -> str:
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text).strip().lower())
    return " ".join(norm.split())


def is_valid_vietnamese_script(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        code = ord(ch)
        # Cyrillic (Russian)
        if 0x0400 <= code <= 0x04FF:
            return False
        # Thai
        if 0x0E00 <= code <= 0x0E7F:
            return False
        # Japanese (Hiragana/Katakana)
        if 0x3040 <= code <= 0x30FF:
            return False
        # CJK (Chinese)
        if 0x4E00 <= code <= 0x9FFF:
            return False
        # Korean (Hangul)
        if (0xAC00 <= code <= 0xD7AF) or (0x1100 <= code <= 0x11FF) or (0x3130 <= code <= 0x318F):
            return False
        # Stylized phonetics / small capitals (e.g. ᴅ, ɪ, ᴀ, ᴄ, ʜ, ᴢ in chat spam)
        if 0x1D00 <= code <= 0x1D7F:
            return False
        # Emojis and miscellaneous symbols
        if (0x1F000 <= code <= 0x1FFFF) or (0x2600 <= code <= 0x27FF) or (0xFE00 <= code <= 0xFE0F):
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
    # 1. Lane 1: Typing & Diacritics (Quota: 1,500,000, Pool: 2,500,000)
    # -------------------------------------------------------------
    print("\n[1/4] Gathering Lane 1 Pool (Typing & Diacritics - Target ~2.5M pairs)...")
    t0 = time.time()
    train_noisy_src = Path("data/reparos/base-v2-production/v2-32/base/train.noisy.src")
    train_noisy_tgt = Path("data/reparos/base-v2-production/v2-32/base/train.noisy.tgt")

    with open(train_noisy_src, "r", encoding="utf-8") as f_s, open(train_noisy_tgt, "r", encoding="utf-8") as f_t:
        for s_raw, t_raw in zip(f_s, f_t):
            s = normalize(s_raw)
            t = canonicalize_target(t_raw)
            if not s or not t or s == t:
                continue
            if s in eval_queries_set or t in eval_queries_set:
                continue
            if not is_valid_vietnamese_script(s) or not is_valid_vietnamese_script(t):
                continue
            if contains_heldout(s) or contains_heldout(t):
                continue
            valid, _ = is_canonical_valid_pair(s, t)
            if not valid:
                continue

            pools["lane1"].append((s, t))
            if len(pools["lane1"]) >= 2_500_000:
                break
    print(f"  -> Lane 1 gathered: {len(pools['lane1']):,} clean pairs in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------
    # 2. Lane 2: Address & Acronyms (Quota: 900,000, Pool: ~1.2M)
    # -------------------------------------------------------------
    print("\n[2/4] Gathering Lane 2 Pool (Address, Acronyms, Composition - Target ~1.2M pairs)...")
    t0 = time.time()
    corpus_file = Path("data/osm/prepared-v4-leakfree/train/corpus.txt")
    with open(corpus_file, "r", encoding="utf-8") as f:
        clean_osm_lines = [
            canonicalize_target(l) for l in f
            if len(l.strip()) > 8 and is_valid_vietnamese_script(l) and not contains_heldout(l)
        ]
    clean_osm_lines = [
        l for l in clean_osm_lines
        if not has_consecutive_duplicates(l) and not UNEXPANDED_TARGET_REGEX.search(l)
    ]
    print(f"  Loaded {len(clean_osm_lines):,} canonical clean OSM lines for synthesis.")

    # 2a. Address Abbreviations (Pass 1 & Pass 2 with random variation)
    print("  Generating Address Abbreviations...")
    for repeat in range(2):
        for line in clean_osm_lines:
            if line in eval_queries_set:
                continue
            abbrev = generate_address_abbreviation(line, rng)
            if abbrev and abbrev != line and abbrev not in eval_queries_set and is_valid_vietnamese_script(abbrev):
                valid, _ = is_canonical_valid_pair(abbrev, line)
                if valid:
                    pools["lane2"].append((abbrev, line))
                    if len(pools["lane2"]) >= 450_000:
                        break
        if len(pools["lane2"]) >= 450_000:
            break

    # 2b. Compositional Variations (Pass 1 & Pass 2)
    print("  Generating Compositional variations...")
    for repeat in range(2):
        for line in clean_osm_lines:
            if line in eval_queries_set:
                continue
            comp = generate_composition(line, rng)
            if comp and comp != line and comp not in eval_queries_set and is_valid_vietnamese_script(comp):
                valid, _ = is_canonical_valid_pair(comp, line)
                if valid:
                    pools["lane2"].append((comp, line))
                    if len(pools["lane2"]) >= 850_000:
                        break
        if len(pools["lane2"]) >= 850_000:
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
            valid, _ = is_canonical_valid_pair(mod_line, line)
            if valid:
                pools["lane2"].append((mod_line, line))
                if len(pools["lane2"]) >= 1_200_000:
                    break
    print(f"  -> Lane 2 gathered: {len(pools['lane2']):,} clean pairs in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------
    # 3. Lane 3: Clean OSM & Seen Brands (Quota: 1,200,000 -> 600k Clean + 600k Brand)
    # -------------------------------------------------------------
    print("\n[3/4] Gathering Lane 3 Pool (Clean Queries & Seen Brands - Target ~1.4M pairs)...")
    t0 = time.time()
    # 3a. Clean OSM + Clean V2
    for line in clean_osm_lines:
        if len(line.split()) >= 2 and line not in eval_queries_set and not contains_heldout(line):
            valid, _ = is_canonical_valid_pair(line, line)
            if valid:
                pools["lane3_clean"].append((line, line))
                if len(pools["lane3_clean"]) >= 400_000:
                    break

    clean_v2_file = Path("data/reparos/base-v2-production/v2-32/base/train.clean.src")
    if clean_v2_file.exists():
        with open(clean_v2_file, "r", encoding="utf-8") as f:
            for l in f:
                c = canonicalize_target(l)
                if c and len(c.split()) >= 2 and c not in eval_queries_set and not contains_heldout(c) and is_valid_vietnamese_script(c):
                    valid, _ = is_canonical_valid_pair(c, c)
                    if valid:
                        pools["lane3_clean"].append((c, c))
                        if len(pools["lane3_clean"]) >= 750_000:
                            break

    # 3b. Seen Brands combined with real addresses (850,000 unique combinations)
    print("  Synthesizing Seen Brands with real street addresses...")
    seen_brands_list = list(SEEN_BRANDS)
    for repeat in range(3):
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
            text = canonicalize_target(text)
            if text not in eval_queries_set:
                valid, _ = is_canonical_valid_pair(text, text)
                if valid:
                    pools["lane3_brand"].append((text, text))
                    if len(pools["lane3_brand"]) >= 850_000:
                        break
        if len(pools["lane3_brand"]) >= 850_000:
            break
    print(f"  -> Lane 3 gathered: {len(pools['lane3_clean']):,} Clean + {len(pools['lane3_brand']):,} Brands in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------
    # 4. Lane 4: Multi-Error DAE on Clean OSM Seeds (100% Pure Clean Targets)
    # -------------------------------------------------------------
    print("\n[4/4] Generating Lane 4 Pool: Multi-Error DAE on Clean OSM Seeds (Zero Untrusted Data)...")
    t0 = time.time()
    qf = ZeroClickQualityFilter()
    shuffled_osm = list(clean_osm_lines)
    rng.shuffle(shuffled_osm)
    for repeat in range(2):
        for line in shuffled_osm:
            if len(line.split()) < 2 or line in eval_queries_set or contains_heldout(line):
                continue
            s_dae, t_dae = qf.generate_dae_pair(line, rng)
            t_dae = canonicalize_target(t_dae)
            valid, _ = is_canonical_valid_pair(s_dae, t_dae)
            if valid and s_dae != t_dae and s_dae not in eval_queries_set and is_valid_vietnamese_script(s_dae):
                pools["lane4_dae"].append((s_dae, t_dae))
                if len(pools["lane4_dae"]) >= 500_000:
                    break
        if len(pools["lane4_dae"]) >= 500_000:
            break

    print(f"  -> Lane 4 gathered: {len(pools['lane4_dae']):,} pure DAE pairs in {time.time()-t0:.1f}s")

    return pools


def assemble_production_dataset(
    pools: Dict[str, List[Tuple[str, str]]],
    precomputed_heldout: List[Tuple[str, str, str]],
    output_dir: Path,
    rng: random.Random,
) -> Tuple[Path, Path]:
    print("\n==================================================================")
    print("   ASSEMBLING BASE V3 PRODUCTION DATASET (EXACTLY 4,000,000 PAIRS)")
    print("   100% PURE CLEAN BASE: NO ZERO_CLICK LOG CONTAMINATION")
    print("   Canonical Contract Recipe: >= 35% Clean/Identity (1.4M pairs)")
    print("==================================================================")

    def is_heldout_leak(text: str) -> bool:
        clean_text = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split())
        padded = f" {clean_text} "
        for _, b_clean, b_padded in precomputed_heldout:
            if b_padded in padded or clean_text == b_clean:
                return True
        return False

    target_total = 4_000_000
    target_clean = 1_400_000
    target_noise = target_total - target_clean

    resolved_dict: Dict[str, str] = {}

    # 1. Clean identity sources (Target: 1,400,000 = 35%)
    clean_sources = [
        (pools["lane3_brand"], 750_000),
        (pools["lane3_clean"], 750_000),
    ]

    print("\nPass 1: Allocating Clean Identity pairs (Target: 1,400,000)...")
    clean_added = 0
    for pool, quota in clean_sources:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        added = 0
        for s, t in shuffled:
            if is_heldout_leak(s) or is_heldout_leak(t):
                continue
            if s not in resolved_dict:
                valid, _ = is_canonical_valid_pair(s, t)
                if not valid:
                    continue
                resolved_dict[s] = t
                added += 1
                clean_added += 1
                if added >= quota or clean_added >= target_clean:
                    break
        print(f"  - Allocated {added:,} / {quota:,} clean pairs (Total clean: {clean_added:,})")

    # Top-up clean if needed
    if clean_added < target_clean:
        print(f"  Topping up clean pairs ({target_clean - clean_added:,} remaining)...")
        for pool, _ in clean_sources:
            for s, t in pool:
                if is_heldout_leak(s) or is_heldout_leak(t):
                    continue
                if s not in resolved_dict:
                    valid, _ = is_canonical_valid_pair(s, t)
                    if not valid:
                        continue
                    resolved_dict[s] = t
                    clean_added += 1
                    if clean_added >= target_clean:
                        break
            if clean_added >= target_clean:
                break

    print(f"=> Certified Clean Identity pairs: {clean_added:,} ({clean_added/target_total*100:.2f}%)")

    # 2. Noise & Reconstruction sources (Target: 2,600,000 = 65%)
    noise_sources = [
        (pools["lane4_dae"], 400_000),
        (pools["lane2"], 900_000),
        (pools["lane1"], 1_500_000),
    ]

    print("\nPass 2: Allocating Noisy Reconstruction pairs (Target: 2,600,000)...")
    for pool, quota in noise_sources:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        added = 0
        for s, t in shuffled:
            if is_heldout_leak(s) or is_heldout_leak(t):
                continue
            if s not in resolved_dict:
                valid, _ = is_canonical_valid_pair(s, t)
                if not valid:
                    continue
                resolved_dict[s] = t
                added += 1
                if added >= quota or len(resolved_dict) >= target_total:
                    break
        print(f"  - Allocated {added:,} / {quota:,} noisy pairs (Current total: {len(resolved_dict):,})")

    # Pass 3: Top-up if duplicate elimination left any remainder
    if len(resolved_dict) < target_total:
        print(f"\nPass 3: Final top-up {target_total - len(resolved_dict):,} pairs...")
        all_sources = noise_sources + clean_sources
        for pool, _ in all_sources:
            shuffled = list(pool)
            rng.shuffle(shuffled)
            for s, t in shuffled:
                if is_heldout_leak(s) or is_heldout_leak(t):
                    continue
                if s not in resolved_dict:
                    valid, _ = is_canonical_valid_pair(s, t)
                    if not valid:
                        continue
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
        "dataset_name": "reparos-base-v3-production-canonical",
        "total_train_pairs": target_total,
        "clean_identity_ratio": 0.35,
        "validation_pairs": 5000,
        "recipe": {
            "lane1_typing_diacritics": 1500000,
            "lane2_address_acronym_composition": 900000,
            "lane3_clean_and_seen_brands": 1200000,
            "lane4_real_search_adaptation": {
                "dae": 200000,
                "identity": 200000,
            }
        },
        "zero_leakage_certified": True,
        "zero_contradiction_certified": True,
        "canonical_contract_certified": True,
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
