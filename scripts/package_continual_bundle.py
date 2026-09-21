from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def add_benchmarks(zf: zipfile.ZipFile, repo_root: Path) -> None:
    """Adds the three frozen standard benchmarks into evaluation/ in the archive."""
    bench_sources = {
        "user-centric-v2.jsonl": repo_root / "benchmark/reparos-user-centric-v2/gold.jsonl",
        "composition-4k.jsonl": repo_root / "benchmark/reparos-compositional-4k/gold.jsonl",
        "diagnostic-10k.jsonl": repo_root / "benchmark/reparos-diagnostic-10k/gold.jsonl",
    }
    for dest_name, src_path in bench_sources.items():
        if src_path.is_file():
            zf.write(src_path, f"evaluation/{dest_name}")
            print(f"  + Added benchmark: evaluation/{dest_name}")
        else:
            print(f"  ! Warning: missing benchmark source {src_path}")


def package_dataset(dataset_dir: Path, repo_root: Path, output_zip: Path) -> Path:
    """Zips interleaved-continual-v1 dataset and evaluation benchmarks."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    print(f"Creating dataset zip at {output_zip} ...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(dataset_dir)
                archive_name = Path("interleaved-continual-v1") / rel_path
                zf.write(full_path, str(archive_name))

        add_benchmarks(zf, repo_root)

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Dataset zip created successfully: {output_zip} ({size_mb:.2f} MB)")
    return output_zip


def package_all_in_one(dataset_dir: Path, base_artifact_dir: Path, repo_root: Path, output_zip: Path) -> Path:
    """Zips dataset + Base V2 checkpoint artifacts + benchmarks into a single all-in-one bundle."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    print(f"Creating all-in-one bundle at {output_zip} ...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add dataset
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(dataset_dir)
                archive_name = Path("interleaved-continual-v1") / rel_path
                zf.write(full_path, str(archive_name))

        # 2. Add Base V2 artifacts
        base_files = [
            "reparos_base_v2_step_10000.pt",
            "tokenizer.model",
            "vocab.src",
            "vocab.tgt",
            "opennmt-base-v2.json",
            "decoding-config.json",
        ]
        for name in base_files:
            file_path = base_artifact_dir / name
            if not file_path.is_file():
                raise FileNotFoundError(f"Missing base artifact file: {file_path}")
            archive_name = Path("base_v2_checkpoint") / name
            zf.write(file_path, str(archive_name))

        # 3. Add benchmarks
        add_benchmarks(zf, repo_root)

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"All-in-one bundle created successfully: {output_zip} ({size_mb:.2f} MB)")
    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Interleaved Continual Fine-Tuning Bundle for Kaggle")
    parser.add_argument("--dataset", default="data/reparos/interleaved-continual-v1", help="Path to continual dataset")
    parser.add_argument("--base-artifact", default="artifacts/reparos-base-v2-32-final-1", help="Path to Base V2 checkpoint artifacts")
    parser.add_argument("--output-dir", default="dist", help="Output directory for zip files")
    parser.add_argument("--mode", choices=("dataset", "all-in-one", "both"), default="both")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = Path(args.dataset)
    base_path = Path(args.base_artifact)
    out_dir = Path(args.output_dir)

    if args.mode in ("dataset", "both"):
        package_dataset(dataset_path, repo_root, out_dir / "reparos-continual-v1-dataset.zip")

    if args.mode in ("all-in-one", "both"):
        package_all_in_one(dataset_path, base_path, repo_root, out_dir / "reparos-continual-all-in-one.zip")


if __name__ == "__main__":
    main()
