#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from reparos.data_registry import HELDOUT_BRANDS


def main():
    prod_dir = Path("data/base_v3_production")
    src_file = prod_dir / "train.src"
    tgt_file = prod_dir / "train.tgt"

    precomputed_brands = []
    for brand in HELDOUT_BRANDS:
        b_clean = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in brand.lower()).split())
        precomputed_brands.append((brand, b_clean, f" {b_clean} "))

    def is_leak(text: str) -> bool:
        clean = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split())
        padded = f" {clean} "
        for _, b_clean, b_padded in precomputed_brands:
            if b_padded in padded or clean == b_clean:
                return True
        return False

    print("Loading existing 4M dataset...")
    with open(src_file, "r", encoding="utf-8") as f_s, open(tgt_file, "r", encoding="utf-8") as f_t:
        src_lines = [l.strip() for l in f_s]
        tgt_lines = [l.strip() for l in f_t]

    assert len(src_lines) == len(tgt_lines) == 4_000_000, f"Expected 4M pairs, got {len(src_lines)}"

    existing_src = set(src_lines)
    leaks_indices = []
    for idx, (s, t) in enumerate(zip(src_lines, tgt_lines)):
        if is_leak(s) or is_leak(t):
            leaks_indices.append(idx)

    print(f"Detected {len(leaks_indices)} leaky lines out of 4,000,000.")
    if not leaks_indices:
        print("No leaks to fix! Dataset is already clean.")
        return

    # Find replacement lines from train.noisy
    replacements = []
    noisy_src_path = Path("data/reparos/base-v2-production/v2-32/base/train.noisy.src")
    noisy_tgt_path = Path("data/reparos/base-v2-production/v2-32/base/train.noisy.tgt")

    with open(noisy_src_path, "r", encoding="utf-8") as f_s, open(noisy_tgt_path, "r", encoding="utf-8") as f_t:
        for s_raw, t_raw in zip(f_s, f_t):
            s = s_raw.strip()
            t = t_raw.strip()
            if not s or not t or s == t:
                continue
            if is_leak(s) or is_leak(t):
                continue
            if s in existing_src:
                continue
            replacements.append((s, t))
            existing_src.add(s)
            if len(replacements) >= len(leaks_indices):
                break

    print(f"Found {len(replacements)} clean replacement pairs.")
    for idx, (rep_s, rep_t) in zip(leaks_indices, replacements):
        src_lines[idx] = rep_s
        tgt_lines[idx] = rep_t

    # Re-verify
    still_leaking = sum(1 for s, t in zip(src_lines, tgt_lines) if is_leak(s) or is_leak(t))
    assert still_leaking == 0, f"Still found {still_leaking} leaks after replacement!"

    print(f"Writing updated 4,000,000 lines back to {src_file} and {tgt_file}...")
    with open(src_file, "w", encoding="utf-8") as f_s, open(tgt_file, "w", encoding="utf-8") as f_t:
        for s in src_lines:
            f_s.write(s + "\n")
        for t in tgt_lines:
            f_t.write(t + "\n")

    print(">>> All leaks successfully replaced and dataset is 100% clean!")


if __name__ == "__main__":
    main()
