#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def main():
    parser = argparse.ArgumentParser(description="Package Base V3 data, tokenizer, and code for Kaggle offline dataset.")
    parser.add_argument("--output-zip", type=Path, default=Path("dist/kaggle-bundle-base-v3.zip"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("kaggle-bundle"))
    args = parser.parse_args()

    repo_root = Path.cwd()
    bundle_dir = args.bundle_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      PACKAGING REPAROS BASE V3 BUNDLE FOR KAGGLE DATASET         ")
    print("==================================================================")

    # 1. Sync project source code into kaggle-bundle/project
    project_dest = bundle_dir / "project"
    print(f"\n[1/4] Syncing project source code -> {project_dest}...")
    excluded_dirs = {".git", ".venv", ".pytest_cache", "data", "artifacts", "logs", "tmp", "dist", "checkpoints", "__pycache__", "kaggle-bundle", "deliverables", "unused"}
    
    if project_dest.exists():
        shutil.rmtree(project_dest)
    project_dest.mkdir(parents=True, exist_ok=True)

    for item in repo_root.iterdir():
        if item.name not in excluded_dirs and not item.name.startswith("."):
            dest_item = project_dest / item.name
            if item.is_dir():
                shutil.copytree(item, dest_item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(item, dest_item)
    print("  - Project code synced.")

    # 2. Sync Tokenizer V3 into kaggle-bundle/data/tokenizer_v3
    tok_src = repo_root / "data/tokenizer_v3"
    tok_dest = bundle_dir / "data/tokenizer_v3"
    print(f"\n[2/4] Syncing Tokenizer V3 -> {tok_dest}...")
    tok_dest.mkdir(parents=True, exist_ok=True)
    for f in tok_src.glob("*"):
        shutil.copy2(f, tok_dest / f.name)
    print("  - Tokenizer V3 synced.")

    # 3. Sync Frozen Eval Suite into kaggle-bundle/data/base_v3_eval
    eval_src = repo_root / "data/base_v3_eval"
    eval_dest = bundle_dir / "data/base_v3_eval"
    print(f"\n[3/4] Syncing Frozen Eval Suite -> {eval_dest}...")
    eval_dest.mkdir(parents=True, exist_ok=True)
    for f in eval_src.glob("*"):
        shutil.copy2(f, eval_dest / f.name)
    print("  - Frozen Eval Suite synced.")

    # 4. Sync Base V3 Ablation Datasets into kaggle-bundle/data/base_v3_ablation
    ablation_src = repo_root / "data/base_v3_ablation"
    ablation_dest = bundle_dir / "data/base_v3_ablation"
    print(f"\n[4/4] Syncing Ablation Datasets -> {ablation_dest}...")
    if ablation_dest.exists():
        shutil.rmtree(ablation_dest)
    shutil.copytree(ablation_src, ablation_dest)
    print("  - Ablation Datasets (Run A, Run B, Run C) synced.")

    # 5. Create compressed zip archive for uploading to Kaggle Dataset
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    if args.output_zip.exists():
        args.output_zip.unlink()

    print(f"\nCompressing Kaggle bundle -> {args.output_zip}...")
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(bundle_dir):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(bundle_dir)
                zf.write(full_path, str(rel_path))

    size_mb = args.output_zip.stat().st_size / (1024 * 1024)
    print(f"\n>>> SUCCESS! Base V3 Kaggle Bundle created: {args.output_zip} ({size_mb:.2f} MB)")
    print("You can upload this zip file to your Kaggle Dataset (e.g. vanhieu1125/reparos-haha).")


if __name__ == "__main__":
    main()
