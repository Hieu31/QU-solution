#!/usr/bin/env python
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def main():
    repo_root = Path.cwd()
    dist_dir = repo_root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    out_zip = dist_dir / "reparos-base-v3-1-finetune-bundle.zip"

    if out_zip.exists():
        out_zip.unlink()

    print("==================================================================")
    print("      PACKAGING REPAROS BASE V3.1 FINE-TUNING BUNDLE (1M)         ")
    print("      Includes: Checkpoint Step 50k, 1M Dataset, Tokenizer V3     ")
    print("==================================================================")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add Tokenizer V3
        print("\n[1/6] Adding Tokenizer V3...")
        tok_dir = repo_root / "data/tokenizer_v3"
        for f in tok_dir.glob("*"):
            if f.is_file() and not f.name.endswith(".txt"):  # skip 13MB corpus.txt to save space
                zf.write(f, f"data/tokenizer_v3/{f.name}")
                print(f"  + data/tokenizer_v3/{f.name}")

        # 2. Add Pretrained Checkpoint Step 50,000 & Base Vocab
        print("\n[2/6] Adding Pretrained Base V3 Checkpoint (Step 50,000)...")
        ckpt_src = repo_root / "artifacts/checkpoints/base_v3_production/reparos_base_v3_production_step_50000.pt"
        if ckpt_src.exists():
            print(f"  + checkpoints/base_v3_production/{ckpt_src.name} ({ckpt_src.stat().st_size / (1024*1024):.1f} MB)")
            zf.write(ckpt_src, f"checkpoints/base_v3_production/{ckpt_src.name}")

        for v in ["vocab.src", "vocab.tgt"]:
            v_path = repo_root / f"data/base_v3_production/{v}"
            if v_path.exists():
                zf.write(v_path, f"data/base_v3_production/{v}")
                print(f"  + data/base_v3_production/{v}")

        # 3. Add Fine-tuning Dataset (1M pairs)
        print("\n[3/6] Adding Fine-tuning Dataset (data/base_v3_finetune/)...")
        ft_dir = repo_root / "data/base_v3_finetune"
        for f in ft_dir.glob("*"):
            if f.is_file():
                print(f"  + data/base_v3_finetune/{f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")
                zf.write(f, f"data/base_v3_finetune/{f.name}")

        # 4. Add Legacy Benchmarks & Frozen Eval Suites
        print("\n[4/6] Adding Benchmark Suites...")
        for bench_name in ["reparos-user-centric-v2", "reparos-diagnostic-10k", "reparos-compositional-4k"]:
            b_dir = repo_root / f"benchmark/{bench_name}"
            if b_dir.exists():
                for f in b_dir.glob("gold.*"):
                    zf.write(f, f"benchmark/{bench_name}/{f.name}")
                    print(f"  + benchmark/{bench_name}/{f.name}")

        eval_dir = repo_root / "data/base_v3_eval"
        if eval_dir.exists():
            for f in eval_dir.glob("*"):
                if f.is_file():
                    zf.write(f, f"data/base_v3_eval/{f.name}")
                    print(f"  + data/base_v3_eval/{f.name}")

        # 5. Add Source Code
        print("\n[5/6] Adding Source Code & Scripts...")
        src_dir = repo_root / "src"
        for root, _, files in os.walk(src_dir):
            for file in files:
                if not file.endswith((".pyc", ".pyo")) and "__pycache__" not in root:
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(repo_root)
                    zf.write(full_path, str(rel_path).replace("\\", "/"))

        scripts_dir = repo_root / "scripts"
        for root, _, files in os.walk(scripts_dir):
            for file in files:
                if not file.endswith((".pyc", ".pyo")) and "__pycache__" not in root:
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(repo_root)
                    zf.write(full_path, str(rel_path).replace("\\", "/"))

        # 6. Extra config files
        print("\n[6/6] Adding Root Configurations...")
        for extra in ["pyproject.toml", "README.md"]:
            p = repo_root / extra
            if p.is_file():
                zf.write(p, extra)

    zip_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"\nSuccessfully packaged bundle to: {out_zip} ({zip_mb:.1f} MB)")


if __name__ == "__main__":
    main()
