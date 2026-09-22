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
    output_zip = repo_root / "dist" / "kaggle-bundle-pilot-scaling.zip"
    temp_bundle_dir = repo_root / "dist" / "temp_pilot_bundle"
    
    print("==================================================================")
    print("   PACKAGING REPAROS PILOT SCALING BUNDLE FOR KAGGLE DATASET      ")
    print("==================================================================")
    
    if temp_bundle_dir.exists():
        shutil.rmtree(temp_bundle_dir)
    temp_bundle_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Sync source code
    print("\n[1/6] Syncing src/...")
    shutil.copytree(repo_root / "src", temp_bundle_dir / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    
    # 2. Sync scripts (exclude huge temp scripts)
    print("[2/6] Syncing scripts/...")
    shutil.copytree(repo_root / "scripts", temp_bundle_dir / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    
    # 3. Sync experiments/pilot_scaling
    print("[3/6] Syncing experiments/pilot_scaling/...")
    shutil.copytree(repo_root / "experiments", temp_bundle_dir / "experiments", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    
    # 4. Sync benchmark/reparos-diagnostic-10k
    print("[4/6] Syncing benchmark/...")
    (temp_bundle_dir / "benchmark").mkdir(parents=True, exist_ok=True)
    if (repo_root / "benchmark" / "reparos-diagnostic-10k").exists():
        shutil.copytree(
            repo_root / "benchmark" / "reparos-diagnostic-10k",
            temp_bundle_dir / "benchmark" / "reparos-diagnostic-10k",
            ignore=shutil.ignore_patterns("*.orig.bak", "ctranslate2.jsonl")
        )
        
    # 5. Sync Data: Pilot dataset, Tokenizer V3, Frozen Eval Suite
    print("[5/6] Syncing data/ (Pilot 300k, Tokenizer V3, Frozen Eval Suite)...")
    data_dest = temp_bundle_dir / "data"
    data_dest.mkdir(parents=True, exist_ok=True)
    
    # 5a. base_v3_pilot
    shutil.copytree(repo_root / "data" / "base_v3_pilot", data_dest / "base_v3_pilot")
    
    # 5b. tokenizer_v3 (only model and vocab, skip raw 13MB corpus.txt)
    tok_dest = data_dest / "tokenizer_v3"
    tok_dest.mkdir(parents=True, exist_ok=True)
    for f in (repo_root / "data" / "tokenizer_v3").glob("tokenizer.*"):
        shutil.copy2(f, tok_dest / f.name)
        
    # 5c. base_v3_eval
    shutil.copytree(repo_root / "data" / "base_v3_eval", data_dest / "base_v3_eval")
    
    # 6. Create Zip archive
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
        
    print(f"\n[6/6] Compressing to {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(temp_bundle_dir):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(temp_bundle_dir)
                zf.write(full_path, str(rel_path))
                
    # Cleanup temp dir
    shutil.rmtree(temp_bundle_dir)
    
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print("="*66)
    print(f">>> SUCCESS! Created Kaggle Pilot Bundle:")
    print(f"    Path: {output_zip.resolve()}")
    print(f"    Size: {size_mb:.2f} MB")
    print("="*66)

if __name__ == "__main__":
    main()
