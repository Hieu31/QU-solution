from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from reparos import base_v2 as core
from reparos.base_v2 import CleanSeed
from reparos.base_v2_production import (
    protection_reason,
    target_rejection_reason,
)
from reparos.continual_dataset import (
    HELDOUT_PROTECTED_ENTITIES,
    SAMPLE_CITIES,
    SAMPLE_DISTRICTS,
    SEEN_PROTECTED_ENTITIES,
    generate_address_abbreviation,
    generate_composition,
    generate_contextual_acronym,
    generate_retention_variant,
)

BRAND_CONTEXT_TEMPLATES = [
    ("chi nhánh {entity} {district}", "chi nhánh {entity} {district}"),
    ("phòng giao dịch {entity} {city}", "phòng giao dịch {entity} {city}"),
    ("cửa hàng {entity} {district}", "cửa hàng {entity} {district}"),
    ("uống cà phê tại {entity}", "uống cà phê tại {entity}"),
    ("mua đồ ở siêu thị {entity}", "mua đồ ở siêu thị {entity}"),
    ("đến {entity} mua sắm", "đến {entity} mua sắm"),
    ("hẹn gặp ở {entity}", "hẹn gặp ở {entity}"),
    ("địa chỉ {entity} {district}", "địa chỉ {entity} {district}"),
    ("{entity}", "{entity}"),
]


def load_clean_seeds(source_root: Path, split: str, limit: int | None = None) -> list[CleanSeed]:
    """Loads clean seeds from OSM source that pass quality rejection."""
    raw = core._load_clean(source_root, split, None)
    accepted = []
    for s in raw:
        if not target_rejection_reason(s.text):
            accepted.append(s)
            if limit and len(accepted) >= limit:
                break
    return accepted


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_row(
    qid: str,
    inp: str,
    exp: str,
    group: str,
    capability: str,
    is_clean: bool,
    sub_type: str = "",
) -> dict:
    return {
        "id": qid,
        "input": " ".join(inp.split()),
        "expected": " ".join(exp.split()),
        "capability_group": group,
        "capability": capability,
        "sub_type": sub_type or capability,
        "is_clean": is_clean,
    }


def build_dataset(source_osm: Path, output_dir: Path, seed: int = 2026) -> dict:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clean seeds from OSM train...")
    train_seeds = load_clean_seeds(source_osm, "train", limit=80000)
    print(f"Loaded {len(train_seeds)} train clean seeds")

    print("Loading clean seeds from OSM test (leak-free from train)...")
    test_seeds = load_clean_seeds(source_osm, "test", limit=20000)
    print(f"Loaded {len(test_seeds)} test clean seeds")

    heldout_regexes = [re.compile(r"\b" + re.escape(e) + r"\b", re.IGNORECASE) for e in HELDOUT_PROTECTED_ENTITIES]
    heldout_compact = [e.replace(" ", "").casefold() for e in HELDOUT_PROTECTED_ENTITIES if len(e.replace(" ", "")) >= 4]

    def is_safe_train(inp: str, exp: str) -> bool:
        for text in (inp, exp):
            for rgx in heldout_regexes:
                if rgx.search(text):
                    return False
            compact = text.replace(" ", "").casefold()
            for c in heldout_compact:
                if c in compact:
                    return False
        return True

    clean_train_seeds = [s for s in train_seeds if is_safe_train(s.text, s.text)]
    print(f"Filtered {len(clean_train_seeds)} strictly leak-safe train seeds")

    train_seen_queries: set[str] = set()

    # -------------------------------------------------------------------------
    # 1. BUILD TRAIN SPLITS
    # -------------------------------------------------------------------------
    train_dir = output_dir / "train"

    # A. Plasticity
    print("Generating train/plasticity...")
    rows_addr = []
    idx = 0
    while len(rows_addr) < 12000:
        s = clean_train_seeds[idx % len(clean_train_seeds)]
        idx += 1
        src = generate_address_abbreviation(s.text, rng)
        if src and src != s.text:
            exp = s.text
        else:
            pfx = rng.choice(["d. ", "đ. ", "d ", "đ "])
            src = f"{pfx}{s.text}"
            exp = f"đường {s.text}"
        if is_safe_train(src, exp):
            row = make_row(f"train:plas:addr:{len(rows_addr):05d}", src, exp, "plasticity", "address_abbreviation", False)
            rows_addr.append(row)
            train_seen_queries.add(row["input"].casefold())
            train_seen_queries.add(row["expected"].casefold())
    write_jsonl(train_dir / "plasticity" / "address_abbreviation.jsonl", rows_addr)

    rows_acro = []
    while len(rows_acro) < 10000:
        src, tgt = generate_contextual_acronym(rng)
        if is_safe_train(src, tgt):
            row = make_row(f"train:plas:acro:{len(rows_acro):05d}", src, tgt, "plasticity", "contextual_acronym", False)
            rows_acro.append(row)
            train_seen_queries.add(row["input"].casefold())
            train_seen_queries.add(row["expected"].casefold())
    write_jsonl(train_dir / "plasticity" / "contextual_acronym.jsonl", rows_acro)

    rows_comp = []
    while len(rows_comp) < 10000:
        s = clean_train_seeds[idx % len(clean_train_seeds)]
        idx += 1
        src = generate_composition(s.text, rng) or s.text
        if is_safe_train(src, s.text):
            row = make_row(f"train:plas:comp:{len(rows_comp):05d}", src, s.text, "plasticity", "composition", False)
            rows_comp.append(row)
            train_seen_queries.add(row["input"].casefold())
            train_seen_queries.add(row["expected"].casefold())
    write_jsonl(train_dir / "plasticity" / "composition.jsonl", rows_comp)

    # B. Retention
    print("Generating train/retention...")
    retention_families = {
        "vni": 10000,
        "wrong_diacritic": 10000,
        "missing_diacritics": 8000,
        "telex": 6000,
        "keyboard": 6000,
        "boundary": 6000,
    }
    for fam, count in retention_families.items():
        rows_fam = []
        while len(rows_fam) < count:
            s = clean_train_seeds[idx % len(clean_train_seeds)]
            idx += 1
            src = generate_retention_variant(s.text, fam, rng) or s.text
            if is_safe_train(src, s.text):
                row = make_row(f"train:ret:{fam}:{len(rows_fam):05d}", src, s.text, "retention", fam, False)
                rows_fam.append(row)
                train_seen_queries.add(row["input"].casefold())
                train_seen_queries.add(row["expected"].casefold())
        write_jsonl(train_dir / "retention" / f"{fam}.jsonl", rows_fam)

    # C. Protection Seen
    print("Generating train/protection_seen...")
    rows_brand_seen = []
    while len(rows_brand_seen) < 6000:
        ent = rng.choice(SEEN_PROTECTED_ENTITIES)
        tmpl_src, tmpl_tgt = rng.choice(BRAND_CONTEXT_TEMPLATES)
        district = rng.choice(SAMPLE_DISTRICTS)
        city = rng.choice(SAMPLE_CITIES)
        src = tmpl_src.format(entity=ent, district=district, city=city)
        tgt = tmpl_tgt.format(entity=ent, district=district, city=city)
        if is_safe_train(src, tgt):
            row = make_row(f"train:prot_seen:brand:{len(rows_brand_seen):05d}", src, tgt, "protection_seen", "brand_entity", True)
            rows_brand_seen.append(row)
            train_seen_queries.add(row["input"].casefold())
            train_seen_queries.add(row["expected"].casefold())
    write_jsonl(train_dir / "protection_seen" / "brand_entity_seen.jsonl", rows_brand_seen)

    rows_clean_seen = []
    while len(rows_clean_seen) < 16000:
        s = clean_train_seeds[idx % len(clean_train_seeds)]
        idx += 1
        if is_safe_train(s.text, s.text):
            row = make_row(f"train:prot_seen:clean:{len(rows_clean_seen):05d}", s.text, s.text, "protection_seen", "clean_query", True)
            rows_clean_seen.append(row)
            train_seen_queries.add(row["input"].casefold())
            train_seen_queries.add(row["expected"].casefold())
    write_jsonl(train_dir / "protection_seen" / "clean_query_seen.jsonl", rows_clean_seen)

    # -------------------------------------------------------------------------
    # 2. BUILD EVAL SPLITS
    # -------------------------------------------------------------------------
    eval_dir = output_dir / "eval"

    # A. Plasticity Eval
    print("Generating eval/plasticity...")
    eval_addr = []
    for i in range(250):
        s = test_seeds[i % len(test_seeds)]
        src = generate_address_abbreviation(s.text, rng)
        if src and src != s.text:
            exp = s.text
        else:
            pfx = rng.choice(["d. ", "đ. ", "d ", "đ "])
            src = f"{pfx}{s.text}"
            exp = f"đường {s.text}"
        eval_addr.append(make_row(f"eval:plas:addr:{i:04d}", src, exp, "plasticity", "address_abbreviation", False))
    write_jsonl(eval_dir / "plasticity" / "address_abbreviation.jsonl", eval_addr)

    eval_acro = []
    for i in range(250):
        src, tgt = generate_contextual_acronym(rng)
        eval_acro.append(make_row(f"eval:plas:acro:{i:04d}", src, tgt, "plasticity", "contextual_acronym", False))
    write_jsonl(eval_dir / "plasticity" / "contextual_acronym.jsonl", eval_acro)

    eval_comp = []
    for i in range(200):
        s = test_seeds[(i + 1000) % len(test_seeds)]
        src = generate_composition(s.text, rng) or s.text
        eval_comp.append(make_row(f"eval:plas:comp:{i:04d}", src, s.text, "plasticity", "composition", False))
    write_jsonl(eval_dir / "plasticity" / "composition.jsonl", eval_comp)

    # B. Retention Eval
    print("Generating eval/retention...")
    for fam in ("vni", "wrong_diacritic", "telex", "keyboard", "boundary", "missing_diacritics"):
        eval_fam = []
        for i in range(200):
            s = test_seeds[(i + 2000) % len(test_seeds)]
            src = generate_retention_variant(s.text, fam, rng) or s.text
            eval_fam.append(make_row(f"eval:ret:{fam}:{i:04d}", src, s.text, "retention", fam, False))
        write_jsonl(eval_dir / "retention" / f"{fam}.jsonl", eval_fam)

    # C. Protection Seen Eval
    print("Generating eval/protection_seen...")
    eval_brand_seen = []
    for i in range(200):
        ent = rng.choice(SEEN_PROTECTED_ENTITIES)
        tmpl_src, tmpl_tgt = rng.choice(BRAND_CONTEXT_TEMPLATES)
        district = rng.choice(SAMPLE_DISTRICTS)
        city = rng.choice(SAMPLE_CITIES)
        src = tmpl_src.format(entity=ent, district=district, city=city)
        tgt = tmpl_tgt.format(entity=ent, district=district, city=city)
        eval_brand_seen.append(make_row(f"eval:prot_seen:brand:{i:04d}", src, tgt, "protection_seen", "brand_entity", True))
    write_jsonl(eval_dir / "protection_seen" / "brand_entity_seen.jsonl", eval_brand_seen)

    eval_clean_seen = []
    for i in range(200):
        s = clean_train_seeds[i]  # seen in train
        eval_clean_seen.append(make_row(f"eval:prot_seen:clean:{i:04d}", s.text, s.text, "protection_seen", "clean_query", True))
    write_jsonl(eval_dir / "protection_seen" / "clean_query_seen.jsonl", eval_clean_seen)

    # D. Protection Held-Out Eval (STRICT ZERO LEAK)
    print("Generating eval/protection_heldout (Strictly Disjoint)...")
    eval_brand_heldout = []
    idx_h = 0
    while len(eval_brand_heldout) < 250:
        ent = HELDOUT_PROTECTED_ENTITIES[idx_h % len(HELDOUT_PROTECTED_ENTITIES)]
        idx_h += 1
        tmpl_src, tmpl_tgt = rng.choice(BRAND_CONTEXT_TEMPLATES)
        district = rng.choice(SAMPLE_DISTRICTS)
        city = rng.choice(SAMPLE_CITIES)
        src = tmpl_src.format(entity=ent, district=district, city=city)
        tgt = tmpl_tgt.format(entity=ent, district=district, city=city)
        # MUST NEVER match any query in train
        if src.casefold() not in train_seen_queries and tgt.casefold() not in train_seen_queries:
            eval_brand_heldout.append(make_row(f"eval:prot_heldout:brand:{len(eval_brand_heldout):04d}", src, tgt, "protection_heldout", "brand_entity_heldout", True))
    write_jsonl(eval_dir / "protection_heldout" / "brand_entity_heldout.jsonl", eval_brand_heldout)

    eval_clean_heldout = []
    idx_t = 5000
    while len(eval_clean_heldout) < 250:
        s = test_seeds[idx_t % len(test_seeds)]
        idx_t += 1
        # MUST NEVER match any query in train and must not contain any heldout entity
        if s.text.casefold() not in train_seen_queries and not any(rgx.search(s.text) for rgx in heldout_regexes):
            eval_clean_heldout.append(make_row(f"eval:prot_heldout:clean:{len(eval_clean_heldout):04d}", s.text, s.text, "protection_heldout", "clean_query_heldout", True))
    write_jsonl(eval_dir / "protection_heldout" / "clean_query_heldout.jsonl", eval_clean_heldout)

    # -------------------------------------------------------------------------
    # 3. WRITE MANIFEST
    # -------------------------------------------------------------------------
    manifest = {
        "schema_version": "reparos-interleaved-continual/v1",
        "seed": seed,
        "source_osm": str(source_osm),
        "train": {
            "plasticity": 12000 + 10000 + 10000,
            "retention": sum(retention_families.values()),
            "protection_seen": 6000 + 16000,
            "total": 32000 + 46000 + 22000,
        },
        "eval": {
            "plasticity": 250 + 250 + 200,
            "retention": 1200,
            "protection_seen": 400,
            "protection_heldout": 500,
            "total": 700 + 1200 + 400 + 500,
        },
        "seen_entities_count": len(SEEN_PROTECTED_ENTITIES),
        "heldout_entities_count": len(HELDOUT_PROTECTED_ENTITIES),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nDataset generation completed successfully!")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Interleaved Continual Fine-Tuning Dataset")
    parser.add_argument("--source-osm", default="data/osm/prepared-v4-leakfree", help="Root of prepared-v4-leakfree")
    parser.add_argument("--output", default="data/reparos/interleaved-continual-v1", help="Output directory")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    build_dataset(Path(args.source_osm), Path(args.output), seed=args.seed)


if __name__ == "__main__":
    main()
