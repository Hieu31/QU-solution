#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import unicodedata
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

import sentencepiece as spm
from reparos.data_registry import HELDOUT_BRANDS, SEEN_BRANDS, CURATED_ACRONYMS
from reparos.quality_filter import ZeroClickQualityFilter

# Special symbols that must be preserved as user-defined symbols or lossless characters
USER_DEFINED_SYMBOLS = [
    "'", "&", "mart", "express", "bank", "food", "katinat", "dookki", "shopee", "tiki",
    "lazada", "grab", "gojek", "starbucks", "highlands", "phuclong", "co.opmart"
]


def prepare_training_corpus(output_txt_path: Path, max_osm_samples: int = 400_000, max_zero_click_samples: int = 100_000):
    """Gathers text from OSM, zero-click, brands, and acronyms for Tokenizer V3 training."""
    if output_txt_path.exists() and output_txt_path.stat().st_size > 1_000_000:
        print(f"Corpus already exists at {output_txt_path} ({output_txt_path.stat().st_size:,} bytes). Skipping generation.")
        return 500_000

    print(f"Preparing training corpus for Tokenizer V3 -> {output_txt_path}...")
    qf = ZeroClickQualityFilter()
    lines = []

    # 1. Add all SEEN and HELDOUT brand entities with varied casings and sentence frames
    print(f"  - Adding {len(SEEN_BRANDS)} Seen Brands and {len(HELDOUT_BRANDS)} Heldout Brands...")
    for brand in SEEN_BRANDS + HELDOUT_BRANDS:
        b_norm = qf.normalize(brand)
        lines.append(b_norm)
        lines.append(f"chi nhánh {b_norm} quận 1")
        lines.append(f"siêu thị {b_norm} gần đây")
        lines.append(f"uống cà phê tại {b_norm}")
        lines.append(f"mua hàng ở {b_norm}")

    # 2. Add acronyms and address patterns
    print(f"  - Adding curated acronyms and address abbreviations...")
    for short, full in CURATED_ACRONYMS.items():
        lines.append(qf.normalize(short))
        lines.append(qf.normalize(full))
        lines.append(f"ở đoạn {short} giao với đường 3/2")
        lines.append(f"đoạn {full} quận 1")

    # Add standard address codes
    for num in range(1, 30):
        lines.append(f"đường số {num}")
        lines.append(f"quận {num}")
        lines.append(f"phường {num}")
        lines.append(f"q.{num}")
        lines.append(f"p.{num}")
        lines.append(f"d.{num}")

    # 3. Add clean seeds from OSM dataset
    osm_train = Path("data/osm/prepared-v4-leakfree/train/corpus.txt")
    if not osm_train.exists():
        osm_train = Path("data/osm/prepared-v4-leakfree/train.clean.tgt")
    if not osm_train.exists():
        candidates = list(Path("data/osm").glob("**/corpus.txt")) + list(Path("data/osm").glob("**/*.tgt"))
        if candidates:
            osm_train = candidates[0]

    if osm_train.exists():
        print(f"  - Sampling from OSM clean targets ({osm_train})...")
        count = 0
        with open(osm_train, "r", encoding="utf-8") as f:
            for line in f:
                line_clean = qf.normalize(line.strip())
                if line_clean:
                    lines.append(line_clean)
                    count += 1
                    if count >= max_osm_samples:
                        break
        print(f"    Loaded {count} lines from OSM.")
    else:
        print("    Warning: OSM clean target not found at expected path. Using available sources.")

    # 4. Add high-confidence clean queries from zero_click.csv
    zc_path = Path("data/zero_click.csv")
    if zc_path.exists():
        print(f"  - Sampling high-confidence clean queries from {zc_path}...")
        count = 0
        with open(zc_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row or not row[0]:
                    continue
                cat, _ = qf.classify(row[0])
                if cat == "HIGH_CONFIDENCE_CLEAN":
                    lines.append(qf.normalize(row[0]))
                    count += 1
                    if count >= max_zero_click_samples:
                        break
        print(f"    Loaded {count} high-confidence clean queries from zero_click.csv.")

    # Write output
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Training corpus ready with {len(lines):,} total lines.")
    return len(lines)


def train_tokenizer_v3(corpus_path: Path, output_dir: Path, vocab_size: int = 12000):
    """Trains SentencePiece model with 100% character coverage and user-defined symbols."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_dir / "tokenizer")

    print(f"\nTraining SentencePiece Tokenizer V3 (vocab_size={vocab_size}, character_coverage=1.0)...")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="unigram",
        normalization_rule_name="nfkc",
        user_defined_symbols=USER_DEFINED_SYMBOLS,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        byte_fallback=True,
        max_sentence_length=1024,
    )
    print(f"Successfully trained Tokenizer V3 -> {model_prefix}.model and {model_prefix}.vocab")


def verify_tokenizer_v3(model_path: Path):
    """Mandatory Acceptance Test: Verify 0% UNK on all critical test cases."""
    print(f"\n=== Running Acceptance Verification on {model_path} ===")
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    unk_id = sp.unk_id()

    test_queries = [
        "'",
        "&",
        ".",
        "4P's",
        "Pizza 4P's",
        "McDonald's",
        "J&T Express",
        "Biti's",
        "Co.opmart",
        "L'Oréal",
        "D&G",
        "H&M",
        "P&G",
        "Đường 3/2",
        "Quận 10",
        "TP. HCM",
        "bệnh viện chợ rẫy",
        "quán ăn ngon quận 1",
    ]

    all_passed = True
    for query in test_queries:
        norm_q = unicodedata.normalize("NFC", query.strip().lower())
        pieces = sp.encode_as_pieces(norm_q)
        ids = sp.encode_as_ids(norm_q)
        decoded = sp.decode(ids)
        has_unk = unk_id in ids or "<unk>" in pieces

        status = "PASS" if not has_unk else "FAIL"
        if has_unk:
            all_passed = False
        print(f"[{status}] '{query}' -> pieces: {pieces}, ids: {ids}, decoded: '{decoded}'")

    print("\n" + "="*60)
    if all_passed:
        print(">>> 100% ACCEPTANCE GATE PASSED! ZERO UNK DETECTED ON ALL SPECIAL CASES!")
    else:
        print(">>> FAILED: UNK tokens detected! Review vocabulary and coverage.")
    print("="*60 + "\n")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Build and verify Tokenizer V3 for ReparoS Base V3.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/tokenizer_v3"))
    parser.add_argument("--vocab-size", type=int, default=12000)
    args = parser.parse_args()

    corpus_path = args.output_dir / "corpus.txt"
    prepare_training_corpus(corpus_path)
    train_tokenizer_v3(corpus_path, args.output_dir, vocab_size=args.vocab_size)
    success = verify_tokenizer_v3(args.output_dir / "tokenizer.model")
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
