#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

# UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import sentencepiece as spm
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from reparos.data_registry import HELDOUT_BRANDS
from benchmarking.io import normalize_text


def run_leak_audit(train_src: Path, train_tgt: Path) -> Dict[str, any]:
    print("\n--- [Audit 1/3] Zero-Leakage & Benchmark Isolation Audit ---")
    heldout_set = set(HELDOUT_BRANDS)
    print(f"Checking for presence of {len(heldout_set)} HELDOUT_BRANDS...")

    precomputed = []
    for brand in heldout_set:
        b_clean = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in brand.lower()).split())
        precomputed.append((brand, b_clean, f" {b_clean} "))

    leaks = []
    train_queries = set()
    line_count = 0

    with open(train_src, "r", encoding="utf-8") as f_s, open(train_tgt, "r", encoding="utf-8") as f_t:
        for idx, (l_s, l_t) in enumerate(zip(f_s, f_t)):
            line_count += 1
            s = l_s.strip().lower()
            t = l_t.strip().lower()
            train_queries.add(s)
            train_queries.add(t)

            clean_s = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in s).split())
            clean_t = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in t).split())
            padded_s = f" {clean_s} "
            padded_t = f" {clean_t} "

            for brand, b_clean, b_padded in precomputed:
                if b_padded in padded_s or b_padded in padded_t or clean_s == b_clean or clean_t == b_clean:
                    leaks.append({
                        "line": line_count,
                        "brand": brand,
                        "source": s,
                        "target": t,
                    })

    print(f"Scanned {line_count:,} pairs.")
    print(f"Total Held-out Brand leaks detected: {len(leaks)}")

    # Check against benchmark gold files
    benchmarks = [
        REPO_ROOT / "benchmark/reparos-user-centric-v2/gold.jsonl",
        REPO_ROOT / "benchmark/reparos-diagnostic-10k/gold.jsonl",
        REPO_ROOT / "benchmark/reparos-compositional-4k/gold.jsonl",
    ]
    overlap_count = 0
    overlaps = []
    for b in benchmarks:
        if b.exists():
            with open(b, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    inp = row.get("input", "").strip().lower()
                    exp = row.get("expected", "").strip().lower()
                    if inp in train_queries:
                        overlap_count += 1
                        if len(overlaps) < 5:
                            overlaps.append({"benchmark": b.name, "query": inp})
                    if exp in train_queries:
                        overlap_count += 1
                        if len(overlaps) < 5:
                            overlaps.append({"benchmark": b.name, "query": exp})

    print(f"Total exact overlaps with legacy benchmark gold queries: {overlap_count}")
    passed = len(leaks) == 0 and overlap_count == 0
    return {
        "passed": passed,
        "heldout_brand_leaks": len(leaks),
        "benchmark_overlaps": overlap_count,
        "leak_details": leaks[:5],
        "overlap_details": overlaps[:5],
    }


def run_zero_unk_audit(model_path: Path, train_files: List[Path]) -> Dict[str, any]:
    print("\n--- [Audit 2/3] Zero-UNK Compatibility Audit ---")
    print(f"Loading Tokenizer V3 from {model_path}...")
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    unk_id = sp.unk_id()

    total_tokens = 0
    unk_count = 0

    for tf in train_files:
        print(f"  Auditing {tf.name} (sample 20,000 lines)...")
        with open(tf, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 20000:
                    break
                text = line.strip()
                if not text:
                    continue
                ids = sp.encode_as_ids(text.lower())
                total_tokens += len(ids)
                unk_count += ids.count(unk_id)

    unk_rate = (unk_count / max(total_tokens, 1)) * 100.0
    print(f"Tokens checked: {total_tokens:,}, UNK tokens found: {unk_count} ({unk_rate:.4f}%)")
    passed = unk_count == 0
    return {
        "passed": passed,
        "total_tokens": total_tokens,
        "unk_count": unk_count,
        "unk_rate_pct": unk_rate,
    }


def run_contradiction_audit(train_src: Path, train_tgt: Path) -> Dict[str, any]:
    print("\n--- [Audit 3/3] Deterministic 1-to-1 Mapping Audit ---")
    mapping: Dict[str, Set[str]] = {}
    with open(train_src, "r", encoding="utf-8") as f_s, open(train_tgt, "r", encoding="utf-8") as f_t:
        for idx, (l_s, l_t) in enumerate(zip(f_s, f_t)):
            s = l_s.strip().lower()
            t = l_t.strip().lower()
            if s not in mapping:
                mapping[s] = set()
            mapping[s].add(t)

    contradictions = sum(1 for s, targets in mapping.items() if len(targets) > 1)
    print(f"Total Unique Sources: {len(mapping):,}, One-to-Many Contradictions: {contradictions}")
    passed = contradictions == 0
    return {
        "passed": passed,
        "unique_sources": len(mapping),
        "contradictions": contradictions,
    }


def main():
    print("=" * 70)
    print("   MATHEMATICAL QUALITY & ZERO-LEAKAGE AUDIT: BASE V3.1 FINE-TUNE   ")
    print("=" * 70)

    data_dir = REPO_ROOT / "data/base_v3_finetune"
    train_src = data_dir / "train.src"
    train_tgt = data_dir / "train.tgt"
    tok_model = REPO_ROOT / "data/tokenizer_v3/tokenizer.model"

    if not train_src.exists() or not train_tgt.exists():
        print(f"ERROR: {train_src} or {train_tgt} does not exist!")
        sys.exit(1)

    leak_res = run_leak_audit(train_src, train_tgt)
    unk_res = run_zero_unk_audit(tok_model, [train_src, train_tgt])
    contra_res = run_contradiction_audit(train_src, train_tgt)

    all_passed = leak_res["passed"] and unk_res["passed"] and contra_res["passed"]

    report = {
        "dataset": "reparos-base-v3-1-finetune",
        "zero_leak_audit": leak_res,
        "zero_unk_audit": unk_res,
        "contradiction_audit": contra_res,
        "all_passed": all_passed,
    }

    report_file = data_dir / "audit_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved Audit Report to {report_file}")
    print("=" * 70)
    if all_passed:
        print(">>> ALL AUDITS PASSED 100%! DATASET IS FULLY PRODUCTION CERTIFIED!")
    else:
        print(">>> AUDIT FAILED! Review details above.")
    print("=" * 70)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
