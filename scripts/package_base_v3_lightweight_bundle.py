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
    out_zip = dist_dir / "reparos-base-v3-data.zip"

    if out_zip.exists():
        out_zip.unlink()

    print("==================================================================")
    print("      PACKAGING LIGHTWEIGHT REPAROS BASE V3 (NO WHEELS)           ")
    print("==================================================================")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add Tokenizer V3
        print("\n[1/4] Adding Tokenizer V3...")
        tok_dir = repo_root / "data/tokenizer_v3"
        for f in tok_dir.glob("*"):
            if f.is_file():
                zf.write(f, f"data/tokenizer_v3/{f.name}")
                print(f"  + data/tokenizer_v3/{f.name}")

        # 2. Add Frozen Evaluation Suite
        print("\n[2/4] Adding Frozen Evaluation Suite...")
        eval_dir = repo_root / "data/base_v3_eval"
        for f in eval_dir.glob("*"):
            if f.is_file():
                zf.write(f, f"data/base_v3_eval/{f.name}")
                print(f"  + data/base_v3_eval/{f.name}")

        # 3. Add Ablation Datasets (Run A, Run B, Run C + Validation)
        print("\n[3/4] Adding Ablation Datasets (Run A, B, C)...")
        ablation_dir = repo_root / "data/base_v3_ablation"
        for root, _, files in os.walk(ablation_dir):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(repo_root)
                zf.write(full_path, str(rel_path).replace("\\", "/"))
        print("  + data/base_v3_ablation/ (all files added)")

        # 4. Add Project Source Code (src, scripts, configs)
        print("\n[4/4] Adding Project Source Code...")
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

        for extra in ["pyproject.toml", "README.md"]:
            p = repo_root / extra
            if p.is_file():
                zf.write(p, extra)

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 66)
    print(f">>> HOÀN THÀNH: {out_zip} ({size_mb:.2f} MB)")
    print("==================================================================")


if __name__ == "__main__":
    main()
