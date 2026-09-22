"""
Build Base V3.1 Production Fine-tuning Dataset (1,000,000 pairs)
Composition:
  1. Replay Anchor (600,000 pairs - 60%):
     Sampled from data/base_v3_production/train.src & train.tgt to preserve
     VNI (93%), Telex (92%), Clean Preservation (>90%), and baseline capabilities.
  2. Address Symbols, Slashes & Complex Numbers (200,000 pairs - 20%):
     Targeting the 18% accuracy bottleneck in address_symbol (slashes, hyphens, ngõ, hẻm, ngách).
  3. Nested Acronyms & Administrative Abbreviations (150,000 pairs - 15%):
     Targeting address abbreviations (d/đ, q, p, tp, h, tx, tt, bv, ubnd) & nested combinations (q1 tp hcm, ubnd p...).
  4. Hard User-Centric & POI DAE (50,000 pairs - 5%):
     Confidence-gated real search queries from zero_click.csv with multi-error DAE corruptions.
"""

import os
import sys
import csv
import json
import time
import random
import re
import shutil
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Set

# UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from reparos.data_registry import HELDOUT_BRANDS, SEEN_BRANDS
from reparos.quality_filter import ZeroClickQualityFilter
from reparos.typing_curriculum import encode_ime
from reparos.base_v2 import _plain, _partial_plain, _keyboard, _join_boundary
from reparos.canonical_contract import (
    canonicalize_target,
    is_canonical_valid_pair,
    has_consecutive_duplicates,
)


def normalize(text: str) -> str:
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text).strip().lower())
    return " ".join(norm.split())


def load_all_eval_queries(repo_root: Path) -> Set[str]:
    """Loads all eval queries from benchmark and eval suites to guarantee 0% leakage."""
    eval_set = set()

    # 1. Legacy benchmarks
    benchmarks = [
        repo_root / "benchmark/reparos-user-centric-v2/gold.jsonl",
        repo_root / "benchmark/reparos-diagnostic-10k/gold.jsonl",
        repo_root / "benchmark/reparos-compositional-4k/gold.jsonl",
    ]
    for b in benchmarks:
        if b.exists():
            with open(b, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    eval_set.add(normalize(row.get("input", "")))
                    eval_set.add(normalize(row.get("expected", "")))

    # 2. Frozen eval suite
    eval_dir = repo_root / "data/base_v3_eval"
    if eval_dir.exists():
        for p in eval_dir.glob("*.src"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    eval_set.add(normalize(line))
        for p in eval_dir.glob("*.tgt"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    eval_set.add(normalize(line))

    eval_set.discard("")
    return eval_set


# ----------------------------------------------------------------------
# Generator 1: Address Symbols & Slashes Generator
# ----------------------------------------------------------------------
class AddressSymbolGenerator:
    """Generates complex address number patterns with missing/altered slashes, hyphens, and unit codes."""

    PREFIXES = [
        "hẻm", "ngõ", "ngách", "kiệt", "số", "nhà số", "căn", "khu", "lô",
    ]
    STREET_NAMES = [
        "lê quang định", "nguyễn ái quốc", "yên thường", "trần phú", "sao biển 1",
        "tương mai", "số 6", "số 15", "nguyễn oanh", "nguyễn trãi",
        "hoàng diệu", "cách mạng tháng 8", "lý thường kiệt", "điện biên phủ",
        "võ thị sáu", "bạch đằng", "phan đăng lưu", "nguyễn thị minh khai",
        "tô ký", "hồ tùng mậu", "hoàng hoa thám", "nguyễn văn cừ", "quang trung",
        "trần hưng đạo", "lê lợi", "nguyễn huệ", "phạm văn đồng", "giải phóng",
    ]

    def __init__(self, rng: random.Random):
        self.rng = rng

    def generate_slash_pattern(self) -> Tuple[str, str]:
        """e.g. 195/9C -> 195 9c, 494/1/5 -> 494/1 5, 41/60 -> 41 60"""
        depth = self.rng.choice([1, 2, 3])
        parts = [str(self.rng.randint(1, 999)) for _ in range(depth + 1)]
        if self.rng.random() < 0.4:
            parts[-1] += self.rng.choice(["a", "b", "c", "d", "e", "l", "f"])

        clean_number = "/".join(parts)
        
        # Corrupt slash: space, spaced slash, partial space, or wrong slash (never omit/merge digits!)
        mode = self.rng.choice(["space", "spaced_slash", "partial_space", "wrong_slash"])
        if mode == "space":
            noisy_number = " ".join(parts)
        elif mode == "spaced_slash":
            noisy_number = " / ".join(parts)
        elif mode == "wrong_slash":
            wrong_char = self.rng.choice(["\\", ".", "-", " "])
            noisy_number = wrong_char.join(parts)
        else:
            # Partial space: e.g. 494/1 5
            noisy_parts = []
            for i, p in enumerate(parts):
                if i == 0:
                    noisy_parts.append(p)
                else:
                    sep = " " if self.rng.random() < 0.5 else "/"
                    noisy_parts.append(sep + p)
            noisy_number = "".join(noisy_parts)

        # Context wrapper
        street = self.rng.choice(self.STREET_NAMES)
        prefix = self.rng.choice(self.PREFIXES) if self.rng.random() < 0.7 else ""

        if prefix:
            clean_tgt = f"{prefix} {clean_number} {street}".strip()
            # Noisy prefix variation: missing diacritics on prefix
            p_noisy = _plain(prefix) if self.rng.random() < 0.3 else prefix
            clean_src = f"{p_noisy} {noisy_number} {street}".strip()
        else:
            clean_tgt = f"{clean_number} {street}".strip()
            clean_src = f"{noisy_number} {street}".strip()

        clean_tgt = canonicalize_target(clean_tgt)
        return normalize(clean_src), normalize(clean_tgt)

    def generate_hyphen_pattern(self) -> Tuple[str, str]:
        """e.g. sb01-37 -> sb01 37 or sb01 - 37, ql1a - trần phú -> ql1a trần phú"""
        unit = self.rng.choice(["sb", "sh", "lk", "bt", "tt", "no", "ct"])
        num1 = f"{self.rng.randint(1, 99):02d}"
        num2 = f"{self.rng.randint(1, 99):02d}"
        clean_code = f"{unit}{num1}-{num2}"

        mode = self.rng.choice(["space", "spaced_hyphen", "dot"])
        if mode == "space":
            noisy_code = f"{unit}{num1} {num2}"
        elif mode == "spaced_hyphen":
            noisy_code = f"{unit}{num1} - {num2}"
        else:
            noisy_code = f"{unit}{num1}.{num2}"

        street = self.rng.choice(self.STREET_NAMES)
        clean_tgt = f"{clean_code} {street}"
        clean_src = f"{noisy_code} {street}"
        clean_tgt = canonicalize_target(clean_tgt)
        return normalize(clean_src), normalize(clean_tgt)


# ----------------------------------------------------------------------
# Generator 2: Nested Acronyms & Administrative Expansions Generator
# ----------------------------------------------------------------------
class NestedAcronymGenerator:
    """Generates administrative acronym expansions and multi-level nested acronyms."""

    EXPANSIONS = {
        "d": "đường",
        "đ": "đường",
        "dg": "đường",
        "tx": "thị xã",
        "h": "huyện",
        "p": "phường",
        "q": "quận",
        "tp": "thành phố",
        "tt": "thị trấn",
        "bv": "bệnh viện",
        "ubnd": "ủy ban nhân dân",
        "kcn": "khu công nghiệp",
        "đh": "đại học",
        "dh": "đại học",
        "thpt": "trung học phổ thông",
        "thcs": "trung học cơ sở",
        "tttm": "trung tâm thương mại",
        "ql": "quốc lộ",
        "tl": "tỉnh lộ",
    }

    CITIES = [
        ("hcm", "hồ chí minh"),
        ("tp hcm", "thành phố hồ chí minh"),
        ("tphcm", "thành phố hồ chí minh"),
        ("hn", "hà nội"),
        ("tp hà nội", "thành phố hà nội"),
        ("đà nẵng", "đà nẵng"),
        ("hải phòng", "hải phòng"),
        ("cần thơ", "cần thơ"),
        ("bình dương", "bình dương"),
        ("đồng nai", "đồng nai"),
    ]

    DISTRICTS = [
        ("q1", "quận 1"), ("q2", "quận 2"), ("q3", "quận 3"), ("q4", "quận 4"),
        ("q5", "quận 5"), ("q7", "quận 7"), ("q8", "quận 8"), ("q10", "quận 10"),
        ("q11", "quận 11"), ("q12", "quận 12"),
        ("q bình thạnh", "quận bình thạnh"), ("q tân bình", "quận tân bình"),
        ("q gò vấp", "quận gò vấp"), ("q phú nhuận", "quận phú nhuận"),
        ("h cầu giấy", "huyện cầu giấy"), ("h đan phượng", "huyện đan phượng"),
        ("tx quảng yên", "thị xã quảng yên"), ("tx bến cát", "thị xã bến cát"),
    ]

    WARDS = [
        ("p bến nghé", "phường bến nghé"),
        ("p bến thành", "phường bến thành"),
        ("p đa kao", "phường đa kao"),
        ("p phạm ngũ lão", "phường phạm ngũ lão"),
        ("p1", "phường 1"), ("p2", "phường 2"), ("p5", "phường 5"),
        ("p12", "phường 12"), ("p15", "phường 15"),
        ("p dịch vọng", "phường dịch vọng"),
        ("p yên hòa", "phường yên hòa"),
    ]

    POIS = [
        ("bạch mai", "bệnh viện bạch mai", "bv bạch mai"),
        ("chợ rẫy", "bệnh viện chợ rẫy", "bv chợ rẫy"),
        ("từ dũ", "bệnh viện từ dũ", "bv từ dũ"),
        ("việt đức", "bệnh viện việt đức", "bv việt đức"),
        ("bách khoa", "đại học bách khoa", "đh bách khoa"),
        ("kinh tế quốc dân", "đại học kinh tế quốc dân", "đh kinh tế quốc dân"),
        ("y dược", "đại học y dược", "đh y dược"),
        ("tân bình", "khu công nghiệp tân bình", "kcn tân bình"),
        ("sóng thần", "khu công nghiệp sóng thần", "kcn sóng thần"),
        ("lê hồng phong", "trung học phổ thông chuyên lê hồng phong", "thpt lê hồng phong"),
    ]

    def __init__(self, rng: random.Random):
        self.rng = rng

    def generate_street_acronym(self) -> Tuple[str, str]:
        """e.g. d trần hưng đạo -> đường trần hưng đạo, d số 15 -> đường số 15"""
        st = self.rng.choice(AddressSymbolGenerator.STREET_NAMES)
        st_clean = re.sub(r"^(?:đường|phố)\s+", "", st, flags=re.IGNORECASE).strip()
        trigger = self.rng.choice(["d", "đ", "dg"])
        
        num = f"{self.rng.randint(1, 999)} " if self.rng.random() < 0.4 else ""
        clean_tgt = f"{num}đường {st_clean}".strip()
        clean_src = f"{num}{trigger} {st_clean}".strip()

        # Add optional district/city tail
        if self.rng.random() < 0.4:
            q_src, q_tgt = self.rng.choice(self.DISTRICTS)
            clean_src += f" {q_src}"
            clean_tgt += f" {q_tgt}"

        clean_tgt = canonicalize_target(clean_tgt)
        return normalize(clean_src), normalize(clean_tgt)

    def generate_nested_admin(self) -> Tuple[str, str]:
        """e.g. ubnd p bến nghé q1 tp hcm -> ủy ban nhân dân phường bến nghé quận 1 thành phố hồ chí minh"""
        w_src, w_tgt = self.rng.choice(self.WARDS)
        q_src, q_tgt = self.rng.choice(self.DISTRICTS)
        c_src, c_tgt = self.rng.choice(self.CITIES)

        prefix_type = self.rng.choice(["ubnd", "trạm y tế", "công an", ""])
        if prefix_type == "ubnd":
            pre_src = "ubnd"
            pre_tgt = "ủy ban nhân dân"
        elif prefix_type == "trạm y tế":
            pre_src = "tyt" if self.rng.random() < 0.5 else "trạm y tế"
            pre_tgt = "trạm y tế"
        elif prefix_type == "công an":
            pre_src = "ca" if self.rng.random() < 0.5 else "công an"
            pre_tgt = "công an"
        else:
            pre_src = pre_tgt = ""

        # Assemble
        src_parts = [p for p in [pre_src, w_src, q_src, c_src if self.rng.random() < 0.5 else ""] if p]
        tgt_parts = [p for p in [pre_tgt, w_tgt, q_tgt, c_tgt if len(src_parts) == 4 or (len(src_parts) == 3 and not pre_src) else ""] if p]

        clean_src = " ".join(src_parts)
        clean_tgt = canonicalize_target(" ".join(tgt_parts))
        return normalize(clean_src), normalize(clean_tgt)

    def generate_poi_expansion(self) -> Tuple[str, str]:
        """e.g. bv chợ rẫy q5 -> bệnh viện chợ rẫy quận 5"""
        name, clean_full, abbrev_full = self.rng.choice(self.POIS)
        clean_src = abbrev_full
        clean_tgt = clean_full

        if self.rng.random() < 0.6:
            q_src, q_tgt = self.rng.choice(self.DISTRICTS)
            clean_src += f" {q_src}"
            clean_tgt += f" {q_tgt}"
        elif self.rng.random() < 0.5:
            c_src, c_tgt = self.rng.choice(self.CITIES)
            clean_src += f" {c_src}"
            clean_tgt += f" {c_tgt}"

        clean_tgt = canonicalize_target(clean_tgt)
        return normalize(clean_src), normalize(clean_tgt)


def main():
    print("=" * 80)
    print("BUILDING BASE V3.1 FINE-TUNING DATASET (1,000,000 PAIRS)")
    print("Recipe: 60% Replay Anchor + 20% Address Symbol + 15% Nested Acronyms + 5% Real User")
    print("=" * 80)

    rng = random.Random(2026)
    out_dir = REPO_ROOT / "data/base_v3_finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading all evaluation queries to strictly forbid leakage...")
    eval_queries_set = load_all_eval_queries(REPO_ROOT)
    print(f"Loaded {len(eval_queries_set):,} forbidden evaluation queries.")

    heldout_brands_clean = [
        " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in b.lower()).split())
        for b in HELDOUT_BRANDS
    ]

    def is_forbidden(s: str, t: str) -> bool:
        if s in eval_queries_set or t in eval_queries_set:
            return True
        clean_s = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in s).split())
        clean_t = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in t).split())
        for b in heldout_brands_clean:
            padded_b = f" {b} "
            if padded_b in f" {clean_s} " or padded_b in f" {clean_t} " or clean_s == b or clean_t == b:
                return True
        return False

    pools: Dict[str, List[Tuple[str, str]]] = {
        "replay_anchor": [],
        "address_symbol": [],
        "nested_acronym": [],
        "real_user": [],
    }

    # ------------------------------------------------------------------
    # Step 1: 600,000 Replay Anchor from data/base_v3_production_canonical/
    # ------------------------------------------------------------------
    print("\n[2/5] Sampling 600,000 Replay Anchor pairs from Canonical Base dataset...")
    base_dir = REPO_ROOT / "data/base_v3_production_canonical"
    if not base_dir.exists():
        base_dir = REPO_ROOT / "data/base_v3_production"
    base_src_path = base_dir / "train.src"
    base_tgt_path = base_dir / "train.tgt"

    t0 = time.time()
    sampled_count = 0
    with open(base_src_path, "r", encoding="utf-8") as f_s, open(base_tgt_path, "r", encoding="utf-8") as f_t:
        lines_s = f_s.readlines()
        lines_t = f_t.readlines()

    assert len(lines_s) == len(lines_t), "Base V3 train.src and train.tgt length mismatch!"
    indices = list(range(len(lines_s)))
    rng.shuffle(indices)

    for idx in indices:
        s = normalize(lines_s[idx])
        t = canonicalize_target(lines_t[idx])
        if not s or not t:
            continue
        valid, _ = is_canonical_valid_pair(s, t)
        if not valid:
            continue
        if is_forbidden(s, t):
            continue
        pools["replay_anchor"].append((s, t))
        sampled_count += 1
        if sampled_count >= 650_000:
            break

    print(f"  Sampled {len(pools['replay_anchor']):,} Replay Anchor pairs in {time.time()-t0:.1f}s.")

    # ------------------------------------------------------------------
    # Step 2: 200,000 Address Symbol & Slashes
    # ------------------------------------------------------------------
    print("\n[3/5] Generating 200,000 Address Symbol & Slashes pairs...")
    t0 = time.time()
    sym_gen = AddressSymbolGenerator(rng)
    seen_sym = set()

    while len(pools["address_symbol"]) < 220_000:
        if rng.random() < 0.75:
            s, t = sym_gen.generate_slash_pattern()
        else:
            s, t = sym_gen.generate_hyphen_pattern()

        if s == t or (s, t) in seen_sym or is_forbidden(s, t):
            continue
        valid, _ = is_canonical_valid_pair(s, t)
        if not valid:
            continue
        seen_sym.add((s, t))
        pools["address_symbol"].append((s, t))

    print(f"  Generated {len(pools['address_symbol']):,} Address Symbol pairs in {time.time()-t0:.1f}s.")

    # ------------------------------------------------------------------
    # Step 3: 150,000 Nested Acronyms & Admin Expansions
    # ------------------------------------------------------------------
    print("\n[4/5] Generating 150,000 Nested Acronyms & Admin Expansions...")
    t0 = time.time()
    acro_gen = NestedAcronymGenerator(rng)
    seen_acro = set()

    while len(pools["nested_acronym"]) < 170_000:
        r = rng.random()
        if r < 0.45:
            s, t = acro_gen.generate_street_acronym()
        elif r < 0.75:
            s, t = acro_gen.generate_nested_admin()
        else:
            s, t = acro_gen.generate_poi_expansion()

        if s == t or (s, t) in seen_acro or is_forbidden(s, t):
            continue
        valid, _ = is_canonical_valid_pair(s, t)
        if not valid:
            continue
        seen_acro.add((s, t))
        pools["nested_acronym"].append((s, t))

    print(f"  Generated {len(pools['nested_acronym']):,} Nested Acronym pairs in {time.time()-t0:.1f}s.")

    # ------------------------------------------------------------------
    # Step 5: 50,000 Multi-Error DAE Pairs from Clean Seeds
    # ------------------------------------------------------------------
    print("\n[5/5] Synthesizing 50,000 Multi-Error DAE Pairs from Clean Seeds (Zero Untrusted Data)...")
    t0 = time.time()
    qf = ZeroClickQualityFilter()
    shuffled_seeds = list(seeds)
    rng.shuffle(shuffled_seeds)

    for q in shuffled_seeds:
        if len(q.split()) >= 3 and not is_forbidden(q, q):
            s_dae, t_dae = qf.generate_dae_pair(q, rng)
            t_dae = canonicalize_target(t_dae)
            valid, _ = is_canonical_valid_pair(s_dae, t_dae)
            if not valid:
                continue
            if s_dae != t_dae and not is_forbidden(s_dae, t_dae):
                pools["real_user"].append((s_dae, t_dae))
                if len(pools["real_user"]) >= 60_000:
                    break

    print(f"  Synthesized {len(pools['real_user']):,} Multi-Error DAE pairs in {time.time()-t0:.1f}s.")

    # ------------------------------------------------------------------
    # Assembly & Deterministic Contradiction Resolution
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ASSEMBLING EXACTLY 1,000,000 FINE-TUNING PAIRS")
    print("=" * 80)

    quotas = [
        ("real_user", 50_000),
        ("nested_acronym", 150_000),
        ("address_symbol", 200_000),
        ("replay_anchor", 600_000),
    ]

    target_total = 1_000_000
    resolved_dict: Dict[str, str] = {}

    for pool_name, quota in quotas:
        pool = pools[pool_name]
        rng.shuffle(pool)
        added = 0
        for s, t in pool:
            if s not in resolved_dict:
                resolved_dict[s] = t
                added += 1
                if added >= quota:
                    break
        print(f"  - Allocated {added:,} / {quota:,} pairs from {pool_name} (Current total: {len(resolved_dict):,})")

    # Top-up if needed
    if len(resolved_dict) < target_total:
        print(f"\nTop-up {target_total - len(resolved_dict):,} pairs...")
        for pool_name, _ in quotas:
            for s, t in pools[pool_name]:
                if s not in resolved_dict:
                    resolved_dict[s] = t
                    if len(resolved_dict) >= target_total:
                        break
            if len(resolved_dict) >= target_total:
                break

    pairs = list(resolved_dict.items())[:target_total]
    assert len(pairs) == target_total, f"Expected {target_total:,}, got {len(pairs):,}"
    rng.shuffle(pairs)

    src_file = out_dir / "train.src"
    tgt_file = out_dir / "train.tgt"

    print(f"\nWriting exactly {len(pairs):,} pairs to {src_file} and {tgt_file}...")
    with open(src_file, "w", encoding="utf-8") as f_s, open(tgt_file, "w", encoding="utf-8") as f_t:
        for s, t in pairs:
            f_s.write(s + "\n")
            f_t.write(t + "\n")

    # Copy clean validation set
    val_src = base_dir / "valid.src"
    val_tgt = base_dir / "valid.tgt"
    shutil.copy2(val_src, out_dir / "valid.src")
    shutil.copy2(val_tgt, out_dir / "valid.tgt")
    print(f"Copied clean validation files into {out_dir}")

    # Write Manifest
    manifest = {
        "dataset_name": "reparos-base-v3-1-finetune",
        "total_train_pairs": target_total,
        "validation_pairs": 5000,
        "recipe": {
            "replay_anchor": 600000,
            "address_symbol_slashes": 200000,
            "nested_acronyms_admin": 150000,
            "hard_real_user_queries": 50000,
        },
        "zero_leakage_certified": True,
        "zero_contradiction_certified": True,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nFine-tuning dataset created successfully at {out_dir}!")


if __name__ == "__main__":
    main()
