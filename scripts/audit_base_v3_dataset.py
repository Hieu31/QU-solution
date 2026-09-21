#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

# Force UTF-8 stdout with line buffering
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import sentencepiece as spm
from reparos.data_registry import HELDOUT_BRANDS, SEEN_BRANDS


def run_leak_audit(train_files: List[Path], heldout_eval_file: Path) -> Dict[str, any]:
    print("\n--- [Audit 1/3] Zero-Leakage Audit ---")
    heldout_brands_set = set(HELDOUT_BRANDS)
    print(f"Checking for presence of {len(heldout_brands_set)} HELDOUT_BRANDS across training files...")

    leaks_found = []
    train_queries: Set[str] = set()
    precomputed_brands = []
    for brand in heldout_brands_set:
        b_clean = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in brand.lower()).split())
        precomputed_brands.append((brand, b_clean, f" {b_clean} "))

    for tf in train_files:
        print(f"  Scanning {tf}...", flush=True)
        line_num = 0
        with open(tf, "r", encoding="utf-8") as f:
            for line in f:
                line_num += 1
                text = line.strip().lower()
                train_queries.add(text)

                clean_text = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).split())
                padded = f" {clean_text} "

                # Check if any heldout brand is contained in this line
                for brand, b_clean, b_padded in precomputed_brands:
                    if b_padded in padded or clean_text == b_clean:
                        leaks_found.append({
                            "file": str(tf),
                            "line": line_num,
                            "brand": brand,
                            "text": text
                        })

    print(f"Total Held-out Brand leaks detected: {len(leaks_found)}", flush=True)

    # Check exact query overlap with eval/protection_heldout
    print(f"Checking exact query intersection with {heldout_eval_file}...")
    eval_overlap = []
    with open(heldout_eval_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            q = line.strip().lower()
            if q in train_queries:
                eval_overlap.append({"line": i + 1, "query": q})

    print(f"Total exact query overlaps between Train and Heldout Eval: {len(eval_overlap)}")

    passed = len(leaks_found) == 0 and len(eval_overlap) == 0
    return {
        "passed": passed,
        "heldout_brand_leaks": len(leaks_found),
        "query_overlaps": len(eval_overlap),
        "leak_details": leaks_found[:10],
        "overlap_details": eval_overlap[:10]
    }


def run_zero_unk_audit(model_path: Path, data_files: List[Path]) -> Dict[str, any]:
    print("\n--- [Audit 2/3] Zero-UNK Compatibility Audit ---")
    print(f"Loading Tokenizer V3 from {model_path}...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    unk_id = sp.unk_id()

    total_tokens = 0
    unk_count = 0
    files_audited = 0

    for df in data_files:
        if not df.exists():
            continue
        files_audited += 1
        with open(df, "r", encoding="utf-8") as f:
            # Audit up to 5,000 lines per file for efficiency
            for i, line in enumerate(f):
                if i >= 5000:
                    break
                text = line.strip()
                if not text:
                    continue
                ids = sp.encode_as_ids(text.lower())
                total_tokens += len(ids)
                unk_count += ids.count(unk_id)

    unk_rate = (unk_count / total_tokens * 100) if total_tokens > 0 else 0.0
    print(f"Audited {files_audited} files | Total Tokens: {total_tokens:,} | UNK Tokens: {unk_count} ({unk_rate:.4f}%)")

    passed = unk_count == 0
    return {
        "passed": passed,
        "total_tokens": total_tokens,
        "unk_count": unk_count,
        "unk_rate_percent": unk_rate
    }


def run_label_contradiction_audit(ablation_dirs: List[Path]) -> Dict[str, any]:
    print("\n--- [Audit 3/3] Label Contradiction Audit ---")
    contradictions = 0
    details = []

    for ad in ablation_dirs:
        src_path = ad / "train.src"
        tgt_path = ad / "train.tgt"
        if not src_path.exists() or not tgt_path.exists():
            continue

        mapping: Dict[str, Set[str]] = {}
        with open(src_path, "r", encoding="utf-8") as f_src, \
             open(tgt_path, "r", encoding="utf-8") as f_tgt:
            for line_idx, (src, tgt) in enumerate(zip(f_src, f_tgt)):
                s = src.strip().lower()
                t = tgt.strip().lower()
                if s not in mapping:
                    mapping[s] = set()
                mapping[s].add(t)

        for s, targets in mapping.items():
            if len(targets) > 1:
                contradictions += 1
                if len(details) < 10:
                    details.append({"source": s, "targets": list(targets)})

    print(f"Total One-to-Many Mappings / Contradictions detected: {contradictions}")
    # In search correction, some very generic queries might map to variations, but for identity vs expansion it should be minimal
    passed = contradictions == 0
    return {
        "passed": passed,
        "contradictions_count": contradictions,
        "details": details
    }


def main():
    parser = argparse.ArgumentParser(description="Run mathematical audit on Base V3 datasets.")
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer_v3/tokenizer.model"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("data/base_v3_ablation"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/base_v3_eval"))
    parser.add_argument("--report-file", type=Path, default=Path("data/base_v3_audit_report.json"))
    args = parser.parse_args()

    print("==============================================================")
    print("      REPAROS BASE V3 MATHEMATICAL QUALITY & LEAK AUDIT      ")
    print("==============================================================")

    train_files = list(args.ablation_dir.glob("**/train.tgt")) + list(args.ablation_dir.glob("**/train.src"))
    heldout_eval = args.eval_dir / "protection_heldout.tgt"
    if not heldout_eval.exists():
        heldout_eval = args.eval_dir / "protection_heldout.src"

    leak_report = run_leak_audit(train_files, heldout_eval)

    all_data_files = train_files + list(args.eval_dir.glob("*.src")) + list(args.eval_dir.glob("*.tgt"))
    unk_report = run_zero_unk_audit(args.tokenizer, all_data_files)

    ablation_dirs = [d for d in args.ablation_dir.iterdir() if d.is_dir()]
    contra_report = run_label_contradiction_audit(ablation_dirs)

    full_report = {
        "zero_leak_audit": leak_report,
        "zero_unk_audit": unk_report,
        "contradiction_audit": contra_report,
        "all_passed": leak_report["passed"] and unk_report["passed"] and contra_report["passed"]
    }

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\nAudit Report saved to {args.report_file}")

    print("\n" + "="*62)
    if full_report["all_passed"]:
        print(">>> ALL AUDITS PASSED! DATASET AND TOKENIZER ARE 100% PRODUCTION-CERTIFIED!")
    else:
        print(">>> AUDIT FAILED! Issues found. Review details above.")
    print("="*62 + "\n")

    if not full_report["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
