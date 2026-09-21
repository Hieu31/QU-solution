#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

from reparos import base_v2 as core
from reparos.continual_dataset import (
    generate_address_abbreviation,
    generate_contextual_acronym,
    generate_composition,
    generate_retention_variant,
)
from reparos.data_registry import (
    BRAND_CONTEXT_TEMPLATES,
    HELDOUT_BRANDS,
    SEEN_BRANDS,
    CURATED_ACRONYMS,
)
from reparos.quality_filter import ZeroClickQualityFilter


def normalize(text: str) -> str:
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text).strip().lower())
    return " ".join(norm.split())


def contains_heldout_brand(text: str) -> bool:
    """Checks if any heldout brand name appears as a standalone entity/word in text, regardless of hyphens or punctuation."""
    clean_text = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split())
    padded = f" {clean_text} "
    for b in HELDOUT_BRANDS:
        b_clean = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in b.lower()).split())
        if f" {b_clean} " in padded or clean_text == b_clean:
            return True
    return False


def is_valid_vietnamese_script(text: str) -> bool:
    """Filters out foreign scripts (CJK, Katakana, Hangul) to keep corpus clean."""
    for ch in text:
        code = ord(ch)
        if (0x3040 <= code <= 0x30ff) or (0x4e00 <= code <= 0x9fff) or (0xac00 <= code <= 0xd7af) or (0x1100 <= code <= 0x11ff) or (0x3130 <= code <= 0x318f):
            return False
    return True


def build_evaluation_suite(output_eval_dir: Path, rng: random.Random) -> Set[str]:
    """Builds the unified frozen evaluation suite (~3,100 samples) with zero-leakage guarantee."""
    print(f"\nBuilding Unified Frozen Evaluation Suite -> {output_eval_dir}...")
    output_eval_dir.mkdir(parents=True, exist_ok=True)
    eval_queries_set: Set[str] = set()

    # 1. eval/plasticity (700 rows)
    print("  - Generating eval/plasticity (700 rows)...")
    plasticity_rows = []
    test_corpus = Path("data/osm/prepared-v4-leakfree/test/corpus.txt")
    with open(test_corpus, "r", encoding="utf-8") as f:
        osm_test_lines = [normalize(l) for l in f if len(l.strip()) > 10 and is_valid_vietnamese_script(l)]

    addr_count = 0
    for line in osm_test_lines:
        abbrev = generate_address_abbreviation(line, rng)
        if abbrev and abbrev != line and is_valid_vietnamese_script(abbrev):
            plasticity_rows.append({
                "id": f"eval:plasticity:addr:{addr_count:04d}",
                "input": abbrev,
                "expected": line,
                "capability": "address_abbreviation"
            })
            eval_queries_set.add(abbrev)
            eval_queries_set.add(line)
            addr_count += 1
            if addr_count >= 250:
                break

    # 250 contextual acronyms
    for i in range(250):
        src, tgt = generate_contextual_acronym(rng)
        plasticity_rows.append({
            "id": f"eval:plasticity:acronym:{i:04d}",
            "input": src,
            "expected": tgt,
            "capability": "contextual_acronym"
        })
        eval_queries_set.add(src)
        eval_queries_set.add(tgt)

    # 200 composition
    comp_count = 0
    for line in osm_test_lines[addr_count:]:
        comp = generate_composition(line, rng)
        if comp and comp != line and is_valid_vietnamese_script(comp):
            plasticity_rows.append({
                "id": f"eval:plasticity:composition:{comp_count:04d}",
                "input": comp,
                "expected": line,
                "capability": "composition"
            })
            eval_queries_set.add(comp)
            eval_queries_set.add(line)
            comp_count += 1
            if comp_count >= 200:
                break

    _write_dataset(output_eval_dir / "plasticity", plasticity_rows)

    # 2. eval/retention (1,200 rows from OSM test noisy_pairs.csv)
    print("  - Sampling eval/retention (1,200 rows)...")
    retention_rows = []
    test_noisy_csv = Path("data/osm/prepared-v4-leakfree/test/noisy_pairs.csv")
    buckets: Dict[str, List[Tuple[str, str]]] = {
        "vni_leak": [], "telex_leak": [], "missing_diacritics_partial": [],
        "word_boundary": [], "keyboard_edit": [], "wrong_diacritic": []
    }
    with open(test_noisy_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            err = row["error_type"]
            src = normalize(row["noisy_query"])
            tgt = normalize(row["correct_query"])
            if err in buckets and src != tgt and len(buckets[err]) < 200:
                if is_valid_vietnamese_script(src) and is_valid_vietnamese_script(tgt):
                    buckets[err].append((src, tgt))

    ret_id = 0
    for err, pairs in buckets.items():
        for src, tgt in pairs:
            retention_rows.append({
                "id": f"eval:retention:{err}:{ret_id:04d}",
                "input": src,
                "expected": tgt,
                "capability": f"retention_{err}"
            })
            eval_queries_set.add(src)
            eval_queries_set.add(tgt)
            ret_id += 1
    _write_dataset(output_eval_dir / "retention", retention_rows)

    # 3. eval/protection_seen (400 rows: 200 clean + 200 seen brands)
    print("  - Generating eval/protection_seen (400 rows)...")
    prot_seen_rows = []
    train_corpus = Path("data/osm/prepared-v4-leakfree/train/corpus.txt")
    with open(train_corpus, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            clean = normalize(line)
            if clean and len(clean.split()) >= 3 and is_valid_vietnamese_script(clean):
                prot_seen_rows.append({
                    "id": f"eval:prot_seen:clean:{i:04d}",
                    "input": clean,
                    "expected": clean,
                    "capability": "clean_seen"
                })
                eval_queries_set.add(clean)
                if len(prot_seen_rows) >= 200:
                    break

    brand_id = 0
    while len(prot_seen_rows) < 400:
        brand = rng.choice(SEEN_BRANDS)
        tmpl = rng.choice(BRAND_CONTEXT_TEMPLATES)
        text = tmpl.format(brand=brand)
        prot_seen_rows.append({
            "id": f"eval:prot_seen:brand:{brand_id:04d}",
            "input": text,
            "expected": text,
            "capability": "brand_seen",
            "brand": brand
        })
        eval_queries_set.add(text)
        brand_id += 1
    _write_dataset(output_eval_dir / "protection_seen", prot_seen_rows)

    # 4. eval/protection_heldout (500 rows - STRICTLY HELDOUT, ZERO LEAK)
    print("  - Generating eval/protection_heldout (500 rows - ZERO LEAK)...")
    prot_held_rows = []
    # 300 heldout brands (Biti's, Pizza 4P's, J&T Express...)
    for i in range(300):
        brand = HELDOUT_BRANDS[i % len(HELDOUT_BRANDS)]
        tmpl = rng.choice(BRAND_CONTEXT_TEMPLATES)
        text = tmpl.format(brand=brand)
        prot_held_rows.append({
            "id": f"eval:prot_held:brand:{i:04d}",
            "input": text,
            "expected": text,
            "capability": "brand_heldout",
            "brand": brand
        })
        eval_queries_set.add(text)

    # 200 clean heldout queries from test corpus
    with open(test_corpus, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            clean = normalize(line)
            if clean and len(clean.split()) >= 3 and is_valid_vietnamese_script(clean):
                prot_held_rows.append({
                    "id": f"eval:prot_held:clean:{i:04d}",
                    "input": clean,
                    "expected": clean,
                    "capability": "clean_heldout"
                })
                eval_queries_set.add(clean)
                if len(prot_held_rows) >= 500:
                    break
    _write_dataset(output_eval_dir / "protection_heldout", prot_held_rows)

    # 5. eval/user_centric (300 rows)
    print("  - Sampling eval/user_centric (300 rows)...")
    user_rows = []
    user_samples = [
        ("quan an ngon quan 1", "quán ăn ngon quận 1"),
        ("chợ hoa quảng bá hà nội", "chợ hoa quảng bá hà nội"),
        ("212/22 duong nguyen oanh", "212/22 đường nguyễn oanh"),
        ("bến xe mien tay", "bến xe miền tây"),
        ("pho tong duy tan", "phố tống duy tân"),
        ("truong dai hoc thuong mai", "trường đại học thương mại"),
        ("d. pasteur q3", "đường pasteur quận 3"),
        ("tp hcm mua pizza 4p's", "thành phố hồ chí minh mua pizza 4p's"),
        ("gửi hàng qua j&t express", "gửi hàng qua j&t express"),
        ("mua giày biti's hunter", "mua giày biti's hunter"),
        ("siêu thị co.opmart nguyễn kiệm", "siêu thị co.opmart nguyễn kiệm"),
        ("uống cà phê mcdonald's", "uống cà phê mcdonald's"),
    ]
    for i in range(300):
        src, tgt = user_samples[i % len(user_samples)]
        user_rows.append({
            "id": f"eval:user:{i:04d}",
            "input": src,
            "expected": tgt,
            "capability": "user_centric"
        })
        eval_queries_set.add(src)
        eval_queries_set.add(tgt)
    _write_dataset(output_eval_dir / "user_centric", user_rows)

    print("Evaluation Suite successfully built and frozen!\n")
    return eval_queries_set


def _write_dataset(prefix_path: Path, rows: List[dict]):
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    src_path = prefix_path.with_suffix(".src")
    tgt_path = prefix_path.with_suffix(".tgt")
    jsonl_path = prefix_path.with_suffix(".jsonl")

    with open(src_path, "w", encoding="utf-8") as f_src, \
         open(tgt_path, "w", encoding="utf-8") as f_tgt, \
         open(jsonl_path, "w", encoding="utf-8") as f_json:
        for r in rows:
            src = normalize(r["input"])
            tgt = normalize(r["expected"])
            f_src.write(src + "\n")
            f_tgt.write(tgt + "\n")
            f_json.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_ablation_training_pools(eval_queries_set: Set[str], rng: random.Random) -> Dict[str, List[Tuple[str, str]]]:
    """Generates clean data pools for all 4 lanes with strict leak & foreign script filtering."""
    print("Gathering clean data pools for 4 Lanes (Strict Leak & Script Filtering)...")
    pools: Dict[str, List[Tuple[str, str]]] = {
        "lane1": [],
        "lane2": [],
        "lane3_clean": [],
        "lane3_brand": [],
        "lane4_dae": [],
        "lane4_identity": [],
    }

    # 1. Lane 1: Typing & Diacritics (from OSM train)
    print("  - Building Lane 1 Pool (Typing & Diacritics)...")
    train_noisy_csv = Path("data/osm/prepared-v4-leakfree/train/noisy_pairs.csv")
    with open(train_noisy_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["error_type"] != "clean":
                src = normalize(row["noisy_query"])
                tgt = normalize(row["correct_query"])
                if not src or not tgt or src == tgt:
                    continue
                # Reject if in eval set
                if src in eval_queries_set or tgt in eval_queries_set:
                    continue
                # Reject if contains any held-out brand
                if contains_heldout_brand(src) or contains_heldout_brand(tgt):
                    continue
                # Reject non-Latin foreign scripts
                if not is_valid_vietnamese_script(src) or not is_valid_vietnamese_script(tgt):
                    continue

                pools["lane1"].append((src, tgt))
                if len(pools["lane1"]) >= 350_000:
                    break
    print(f"    Lane 1: {len(pools['lane1']):,} clean, non-leaking pairs.")

    # 2. Lane 2: Address & Acronyms
    print("  - Building Lane 2 Pool (Address & Acronyms)...")
    train_corpus = Path("data/osm/prepared-v4-leakfree/train/corpus.txt")
    with open(train_corpus, "r", encoding="utf-8") as f:
        train_lines = [normalize(l) for l in f if len(l.strip()) > 8 and is_valid_vietnamese_script(l) and not contains_heldout_brand(l)]

    for line in train_lines:
        if line in eval_queries_set:
            continue
        abbrev = generate_address_abbreviation(line, rng)
        if abbrev and abbrev != line and abbrev not in eval_queries_set and is_valid_vietnamese_script(abbrev):
            pools["lane2"].append((abbrev, line))
            if len(pools["lane2"]) >= 100_000:
                break

    # Contextual Acronyms (50k)
    for _ in range(50_000):
        src, tgt = generate_contextual_acronym(rng)
        if src not in eval_queries_set and tgt not in eval_queries_set:
            pools["lane2"].append((src, tgt))

    # Compositional (80k)
    for line in train_lines[100_000:250_000]:
        if line in eval_queries_set:
            continue
        comp = generate_composition(line, rng)
        if comp and comp != line and comp not in eval_queries_set and is_valid_vietnamese_script(comp):
            pools["lane2"].append((comp, line))
            if len(pools["lane2"]) >= 230_000:
                break
    print(f"    Lane 2: {len(pools['lane2']):,} clean pairs.")

    # 3. Lane 3: Clean & Seen Brands
    print("  - Building Lane 3 Pool (Clean + Seen Brands)...")
    for line in train_lines[250_000:]:
        clean = normalize(line)
        if clean and len(clean.split()) >= 2 and clean not in eval_queries_set and not contains_heldout_brand(clean) and is_valid_vietnamese_script(clean):
            pools["lane3_clean"].append((clean, clean))
            if len(pools["lane3_clean"]) >= 100_000:
                break

    for _ in range(60_000):
        brand = rng.choice(SEEN_BRANDS)
        tmpl = rng.choice(BRAND_CONTEXT_TEMPLATES)
        text = tmpl.format(brand=brand)
        if text not in eval_queries_set:
            pools["lane3_brand"].append((text, text))
    print(f"    Lane 3: {len(pools['lane3_clean']):,} Clean + {len(pools['lane3_brand']):,} Brands.")

    # 4. Lane 4: Real Query Adaptation from zero_click.csv
    print("  - Building Lane 4 Pool from zero_click.csv (Confidence-Gated)...")
    qf = ZeroClickQualityFilter()
    zc_path = Path("data/zero_click.csv")
    high_conf_queries = []
    with open(zc_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or not row[0]:
                continue
            cat, _ = qf.classify(row[0])
            if cat == "HIGH_CONFIDENCE_CLEAN":
                q_norm = normalize(row[0])
                if q_norm not in eval_queries_set and not contains_heldout_brand(q_norm) and is_valid_vietnamese_script(q_norm):
                    high_conf_queries.append(q_norm)
                    if len(high_conf_queries) >= 60_000:
                        break

    for q in high_conf_queries:
        src_dae, tgt_dae = qf.generate_dae_pair(q, rng)
        if src_dae not in eval_queries_set and is_valid_vietnamese_script(src_dae):
            pools["lane4_dae"].append((src_dae, tgt_dae))
        src_id, tgt_id = qf.generate_identity_pair(q)
        pools["lane4_identity"].append((src_id, tgt_id))
    print(f"    Lane 4: {len(pools['lane4_dae']):,} DAE + {len(pools['lane4_identity']):,} Identity pairs.")

    return pools


def resolve_contradictions_and_take(pair_sources: List[Tuple[List[Tuple[str, str]], int]], target_total: int, rng: random.Random) -> List[Tuple[str, str]]:
    """Resolves label contradictions deterministically: each input x maps to exactly one target y."""
    resolved_dict: Dict[str, str] = {}
    
    # 1. First pass: satisfy quota from each pool
    for pool, quota in pair_sources:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        added = 0
        for src, tgt in shuffled:
            if contains_heldout_brand(src) or contains_heldout_brand(tgt):
                continue
            if src not in resolved_dict:
                resolved_dict[src] = tgt
                added += 1
                if added >= quota:
                    break

    # 2. Second pass: if below target_total, top up from the candidate pools
    if len(resolved_dict) < target_total:
        for pool, _ in pair_sources:
            shuffled = list(pool)
            rng.shuffle(shuffled)
            for src, tgt in shuffled:
                if contains_heldout_brand(src) or contains_heldout_brand(tgt):
                    continue
                if src not in resolved_dict:
                    resolved_dict[src] = tgt
                    if len(resolved_dict) >= target_total:
                        break
            if len(resolved_dict) >= target_total:
                break

    pairs = list(resolved_dict.items())[:target_total]
    rng.shuffle(pairs)
    return pairs


def build_ablation_datasets(pools: Dict[str, List[Tuple[str, str]]], output_base_dir: Path, rng: random.Random):
    """Assembles the 3 Ablation Training Datasets (300,000 pairs each) with zero contradiction."""
    print(f"\nAssembling 3 Ablation Training Datasets (300,000 pairs each) -> {output_base_dir}...")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Run A: No Lane 4 (45% L1 + 30% L2 + 25% L3 + 0% L4)
    # -------------------------------------------------------------
    print("\n  [1/3] Building dataset_run_a (No Lane 4: 45/30/25/0)...")
    sources_a = [
        (pools["lane3_clean"], 45_000),
        (pools["lane3_brand"], 30_000),
        (pools["lane2"], 90_000),
        (pools["lane1"], 135_000),
    ]
    run_a_pairs = resolve_contradictions_and_take(sources_a, 300_000, rng)
    assert len(run_a_pairs) == 300_000, f"Got {len(run_a_pairs)}"
    _write_tsv_and_lines(output_base_dir / "dataset_run_a", run_a_pairs)

    # -------------------------------------------------------------
    # Run B: Lane 4 DAE-Only (40% L1 + 25% L2 + 25% L3 + 10% L4 DAE)
    # -------------------------------------------------------------
    print("  [2/3] Building dataset_run_b (Lane 4 DAE-Only: 40/25/25/10 DAE)...")
    sources_b = [
        (pools["lane3_clean"], 45_000),
        (pools["lane3_brand"], 30_000),
        (pools["lane4_dae"], 30_000),
        (pools["lane2"], 75_000),
        (pools["lane1"], 120_000),
    ]
    run_b_pairs = resolve_contradictions_and_take(sources_b, 300_000, rng)
    assert len(run_b_pairs) == 300_000, f"Got {len(run_b_pairs)}"
    _write_tsv_and_lines(output_base_dir / "dataset_run_b", run_b_pairs)

    # -------------------------------------------------------------
    # Run C: Lane 4 DAE + Identity (40% L1 + 25% L2 + 25% L3 + 7% DAE + 3% Id)
    # -------------------------------------------------------------
    print("  [3/3] Building dataset_run_c (Lane 4 DAE + Identity: 40/25/25/7 DAE + 3 Id)...")
    sources_c = [
        (pools["lane4_identity"], 9_000),
        (pools["lane3_clean"], 45_000),
        (pools["lane3_brand"], 30_000),
        (pools["lane4_dae"], 21_000),
        (pools["lane2"], 75_000),
        (pools["lane1"], 120_000),
    ]
    run_c_pairs = resolve_contradictions_and_take(sources_c, 300_000, rng)
    assert len(run_c_pairs) == 300_000, f"Got {len(run_c_pairs)}"
    _write_tsv_and_lines(output_base_dir / "dataset_run_c", run_c_pairs)

    print("\nAll 3 Ablation Training Datasets successfully generated with Zero Contradiction!")


def _write_tsv_and_lines(dir_path: Path, pairs: List[Tuple[str, str]]):
    dir_path.mkdir(parents=True, exist_ok=True)
    src_file = dir_path / "train.src"
    tgt_file = dir_path / "train.tgt"

    with open(src_file, "w", encoding="utf-8") as f_src, \
         open(tgt_file, "w", encoding="utf-8") as f_tgt:
        for src, tgt in pairs:
            f_src.write(src + "\n")
            f_tgt.write(tgt + "\n")
    print(f"    Saved {len(pairs):,} pairs to {dir_path}/train.src and train.tgt")


def main():
    parser = argparse.ArgumentParser(description="Build Base V3 Ablation Datasets and Unified Frozen Eval Suite.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/base_v3_eval"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("data/base_v3_ablation"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_queries = build_evaluation_suite(args.eval_dir, rng)
    pools = build_ablation_training_pools(eval_queries, rng)
    build_ablation_datasets(pools, args.ablation_dir, rng)


if __name__ == "__main__":
    main()
