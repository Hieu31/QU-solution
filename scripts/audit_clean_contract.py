#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

repo_root = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(".")
sys.path.insert(0, str(repo_root / "src"))

from reparos.canonical_contract import (
    CANONICAL_ADMIN_MAP,
    has_consecutive_duplicates,
    is_canonical_valid_pair,
)


def audit_parallel_files(src_path: Path, tgt_path: Path) -> dict:
    print("=" * 80)
    print(f"AUDITING DATASET: {src_path.parent.name}")
    print(f"SRC: {src_path}")
    print(f"TGT: {tgt_path}")
    print("=" * 80)

    total = 0
    identity_count = 0
    dup_lines_tgt = 0
    dup_words_counter = Counter()
    unexpanded_counter = Counter()
    d_to_pho_counter = 0
    digit_merged_slashes = 0
    contextless_acronyms = Counter()
    phrasal_rep_counter = Counter()
    unexpanded_prefixes = Counter()
    business_abbrevs = Counter()
    invalid_script_count = 0
    chat_spam_count = 0

    bad_acronyms = {"ltk", "dbp", "nct", "hbt", "pvh", "ntmk", "nvl"}
    unexpanded_terms = ["ubnd", "bv", "kcn", "thpt", "thcs"]

    with open(src_path, "r", encoding="utf-8") as fs, open(tgt_path, "r", encoding="utf-8") as ft:
        for idx, (s_line, t_line) in enumerate(zip(fs, ft)):
            total += 1
            s = " ".join(unicodedata.normalize("NFC", s_line.strip().lower()).split())
            t = " ".join(unicodedata.normalize("NFC", t_line.strip().lower()).split())

            if s == t:
                identity_count += 1

            s_words = s.split()
            t_words = t.split()

            # 1. Consecutive duplicates in target
            if has_consecutive_duplicates(t):
                dup_lines_tgt += 1
                for i in range(len(t_words) - 1):
                    if t_words[i] == t_words[i + 1]:
                        dup_words_counter[f"{t_words[i]} {t_words[i+1]}"] += 1

            # 2. Unexpanded institutional abbreviations in target
            for term in unexpanded_terms:
                if re.search(rf"\b{term}\b", t):
                    unexpanded_counter[term] += 1

            # 3. d/đ mapped to phố
            if s_words and t_words:
                if s_words[0] in ["d", "đ", "dg"] and t_words[0] == "phố":
                    d_to_pho_counter += 1

            # 4. Ill-posed digit-merging in slashes
            if "/" in t and "/" not in s:
                slashes = re.findall(r"\b(\d+)/(\d+(?:/\d+)*[a-zA-Z]?)\b", t)
                for p1, p2 in slashes:
                    merged = f"{p1}{p2}"
                    if merged in s:
                        digit_merged_slashes += 1
                        break

            # 5. Contextless acronyms in source
            for w in s_words:
                if w in bad_acronyms:
                    contextless_acronyms[w] += 1

            # 6. Phrasal multi-word repetitions in target
            if len(t_words) >= 4:
                k_max = min(5, len(t_words) // 2)
                for k in range(k_max, 1, -1):
                    for i in range(len(t_words) - 2 * k + 1):
                        p1 = t_words[i : i + k]
                        p2 = t_words[i + k : i + 2 * k]
                        if p1 == p2 and not all(w.isdigit() for w in p1):
                            phrasal_rep_counter[" ".join(p1)] += 1
                            break

            # 7. Unexpanded standalone prefixes in target
            prefix_match = re.search(r"(?<!\w)(?:d|đ|p|q|tp|tx|tt)\.(?=\s|$)|(?<!\w)(?:p|q)\s+\d+\b", t)
            if prefix_match:
                unexpanded_prefixes[prefix_match.group(0).strip()] += 1

            # 8. Unexpanded business abbreviations in target
            biz_match = re.search(r"\b(?:cty|kđt|kdt|tttm)\b", t)
            if biz_match:
                business_abbrevs[biz_match.group(0)] += 1

            # 9. Invalid script (Cyrillic, Thai, Chinese, Emojis, Stylized fake fonts)
            from reparos.canonical_contract import is_valid_vietnamese_script
            if not is_valid_vietnamese_script(s) or not is_valid_vietnamese_script(t):
                invalid_script_count += 1

            # 10. Chat spam / phone numbers
            if len(t_words) > 25 or re.search(r"\b0\d{9,10}\b|\b0\d{3}\s+\d{3}\s+\d{3,4}\b", t):
                chat_spam_count += 1

            if total % 1_000_000 == 0:
                print(f"  Scanned {total:,} pairs...")

    print(f"\nAudit completed for {total:,} pairs:")
    print(f"  1. Total pairs:                     {total:,}")
    print(f"  2. Identity pairs:                  {identity_count:,} ({identity_count/max(total,1)*100:.2f}%)")
    print(f"  3. Duplicate word lines in TGT:     {dup_lines_tgt:,}")
    print(f"  4. Unexpanded acronyms in TGT:      {sum(unexpanded_counter.values()):,}")
    print(f"  5. Ambiguous d/đ -> phố:            {d_to_pho_counter:,}")
    print(f"  6. Ill-posed digit-merging:         {digit_merged_slashes:,}")
    print(f"  7. Contextless acronyms in SRC:     {sum(contextless_acronyms.values()):,}")
    print(f"  8. Phrasal multi-word repetitions:  {sum(phrasal_rep_counter.values()):,}")
    if phrasal_rep_counter:
        print(f"     Top phrasal: {phrasal_rep_counter.most_common(5)}")
    print(f"  9. Unexpanded prefixes in TGT:      {sum(unexpanded_prefixes.values()):,}")
    if unexpanded_prefixes:
        print(f"     Top prefixes: {unexpanded_prefixes.most_common(5)}")
    print(f" 10. Business abbreviations in TGT:   {sum(business_abbrevs.values()):,}")
    if business_abbrevs:
        print(f"     Top business abbrevs: {business_abbrevs.most_common(5)}")
    print(f" 11. Invalid script / Foreign text:   {invalid_script_count:,}")
    print(f" 12. Chat spam / Phone numbers:       {chat_spam_count:,}")

    passed = (
        dup_lines_tgt == 0
        and sum(unexpanded_counter.values()) == 0
        and d_to_pho_counter == 0
        and digit_merged_slashes == 0
        and sum(contextless_acronyms.values()) == 0
        and sum(phrasal_rep_counter.values()) == 0
        and sum(unexpanded_prefixes.values()) == 0
        and sum(business_abbrevs.values()) == 0
        and invalid_script_count == 0
        and chat_spam_count == 0
    )
    print(f"\n=> AUDIT RESULT: {'[PASSED] 100% CLEAN' if passed else '[FAILED] CONTAINS VIOLATIONS'}")
    return {
        "total": total,
        "identity_ratio": identity_count / max(total, 1),
        "passed": passed,
    }


def audit_benchmark_jsonl(benchmark_path: Path) -> dict:
    print("=" * 80)
    print(f"AUDITING BENCHMARK: {benchmark_path.name}")
    print(f"Path: {benchmark_path}")
    print("=" * 80)

    rows = [json.loads(l) for l in benchmark_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(rows)

    bad_acronyms = {"ltk", "dbp", "nct", "hbt", "pvh", "ntmk", "nvl"}
    unexpanded_terms = ["ubnd", "bv", "kcn", "thpt", "thcs"]
    
    digit_merged = 0
    d_to_pho = 0
    contextless_acr = 0
    dup_lines = 0

    for r in rows:
        s = " ".join(unicodedata.normalize("NFC", str(r["input"]).strip().lower()).split())
        t = " ".join(unicodedata.normalize("NFC", str(r["expected"]).strip().lower()).split())

        if has_consecutive_duplicates(t):
            dup_lines += 1

        s_words = s.split()
        t_words = t.split()

        if s_words and t_words:
            if s_words[0] in ["d", "đ", "dg"] and t_words[0] == "phố":
                d_to_pho += 1

        for w in s_words:
            if w in bad_acronyms:
                contextless_acr += 1

        if "/" in t and "/" not in s:
            slashes = re.findall(r"\b(\d+)/(\d+(?:/\d+)*[a-zA-Z]?)\b", t)
            for p1, p2 in slashes:
                merged = f"{p1}{p2}"
                if merged in s:
                    digit_merged += 1
                    break

    print(f"Audit completed for {total:,} benchmark queries:")
    print(f"  1. Duplicate word lines in TGT: {dup_lines}")
    print(f"  2. Ambiguous d/đ -> phố:        {d_to_pho}")
    print(f"  3. Ill-posed digit-merging:     {digit_merged}")
    print(f"  4. Contextless acronyms:        {contextless_acr}")

    passed = (dup_lines == 0 and d_to_pho == 0 and digit_merged == 0 and contextless_acr == 0)
    print(f"\n=> BENCHMARK RESULT: {'[PASSED] 100% CLEAN' if passed else '[FAILED] CONTAINS VIOLATIONS'}")
    return {"total": total, "passed": passed}


def check_generators():
    import random
    from build_base_v3_finetune_dataset import AddressSymbolGenerator, NestedAcronymGenerator
    from reparos.continual_dataset import generate_address_abbreviation, generate_composition

    print("=" * 80)
    print("AUDITING ACTIVE GENERATORS (10,000 SAMPLES EACH)")
    print("=" * 80)

    rng = random.Random(42)
    sym_gen = AddressSymbolGenerator(rng)
    acro_gen = NestedAcronymGenerator(rng)

    sample_osm = [
        "đường lê quang định", "đường trần phú", "đường nguyễn trãi", "đường số 6",
        "bệnh viện bạch mai", "ủy ban nhân dân phường 12", "đại học bách khoa",
        "khu công nghiệp tân bình", "trung học phổ thông lê hồng phong", "phố huế",
    ]

    # Test 1: AddressSymbolGenerator
    print("Testing AddressSymbolGenerator (10,000 samples)...")
    sym_violations = 0
    for _ in range(10_000):
        s, t = sym_gen.generate_slash_pattern() if rng.random() < 0.75 else sym_gen.generate_hyphen_pattern()
        valid, reason = is_canonical_valid_pair(s, t)
        if not valid:
            sym_violations += 1
            print(f"  [VIOLATION] ({reason}) SRC: '{s}' -> TGT: '{t}'")
            if sym_violations >= 5:
                break
    print(f"  -> AddressSymbolGenerator violations: {sym_violations}")

    # Test 2: NestedAcronymGenerator
    print("Testing NestedAcronymGenerator (10,000 samples)...")
    acro_violations = 0
    for _ in range(10_000):
        r = rng.random()
        if r < 0.45:
            s, t = acro_gen.generate_street_acronym()
        elif r < 0.75:
            s, t = acro_gen.generate_nested_admin()
        else:
            s, t = acro_gen.generate_poi_expansion()
        valid, reason = is_canonical_valid_pair(s, t)
        if not valid:
            acro_violations += 1
            print(f"  [VIOLATION] ({reason}) SRC: '{s}' -> TGT: '{t}'")
            if acro_violations >= 5:
                break
    print(f"  -> NestedAcronymGenerator violations: {acro_violations}")

    # Test 3: Continual dataset address abbreviation
    print("Testing generate_address_abbreviation & composition (10,000 samples)...")
    abbrev_violations = 0
    for _ in range(10_000):
        line = rng.choice(sample_osm)
        abbrev = generate_address_abbreviation(line, rng)
        if abbrev:
            valid, reason = is_canonical_valid_pair(abbrev, line)
            if not valid:
                abbrev_violations += 1
                print(f"  [VIOLATION] ({reason}) SRC: '{abbrev}' -> TGT: '{line}'")
                if abbrev_violations >= 5:
                    break
    print(f"  -> generate_address_abbreviation violations: {abbrev_violations}")

    total_violations = sym_violations + acro_violations + abbrev_violations
    print(f"\n=> GENERATOR AUDIT RESULT: {'[PASSED] 100% CLEAN' if total_violations == 0 else '[FAILED] CONTAINS VIOLATIONS'}")
    return total_violations == 0


def main():
    parser = argparse.ArgumentParser(description="Audit Canonical Contract across datasets and benchmarks")
    parser.add_argument("--dataset", help="Directory containing train.src and train.tgt")
    parser.add_argument("--benchmark", help="Path to gold.jsonl benchmark file")
    parser.add_argument("--check-generators", action="store_true", help="Audit dataset generators for contract violations")
    args = parser.parse_args()

    if args.check_generators:
        check_generators()
    elif args.dataset:
        d = Path(args.dataset)
        audit_parallel_files(d / "train.src", d / "train.tgt")
    elif args.benchmark:
        b = Path(args.benchmark)
        audit_benchmark_jsonl(b)
    else:
        # Default audit both benchmarks
        audit_benchmark_jsonl(repo_root / "benchmark/reparos-user-centric-v2/gold.jsonl")
        audit_benchmark_jsonl(repo_root / "benchmark/reparos-diagnostic-10k/gold.jsonl")


if __name__ == "__main__":
    main()

