from __future__ import annotations

import csv
import json
from pathlib import Path

from reparos.base_v2 import BaseV2Config, CleanSeed, generate_variants, prepare_base_v2


def _source(root: Path) -> Path:
    rows = {
        "train": (("g1", "bệnh viện bạch mai"), ("g2", "đường nguyễn trãi quận 1")),
        "validation": (("g3", "đại học bách khoa hà nội"),),
        "test": (("g4", "thành phố hồ chí minh quận 1"),),
    }
    for split, values in rows.items():
        folder = root / split
        folder.mkdir(parents=True)
        with (folder / "noisy_pairs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("noisy_query", "correct_query", "group_id", "error_type", "noise_source"))
            writer.writeheader()
            for group, text in values:
                writer.writerow({"noisy_query": text, "correct_query": text, "group_id": group, "error_type": "clean", "noise_source": "clean"})
    return root


def test_variants_are_deterministic_unique_and_ranked() -> None:
    seed = CleanSeed("bệnh viện bạch mai", "bệnh viện bạch mai", "g1", "train")
    first = generate_variants(seed, 7)
    second = generate_variants(seed, 7)
    assert first == second
    assert len({row.source for row in first}) == len(first)
    assert all(row.source != row.target for row in first)
    assert all(1 <= row.rank <= 32 for row in first)
    assert any(row.family == "telex" for row in first)
    assert any(row.family == "vni" for row in first)
    assert any(row.family == "address_abbreviation" for row in first)


def test_profiles_are_nested_and_dry_run_does_not_materialize(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    output = tmp_path / "output"
    report = prepare_base_v2(source, output, BaseV2Config(seed=7))
    train = report["splits"]["train"]["profiles"]
    assert train["8"]["noisy"] <= train["16"]["noisy"] <= train["32"]["noisy"]
    assert (output / "base-v2-dry-run.json").is_file()
    assert not (output / "v2-8").exists()
    assert report["leakage"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}


def test_materialization_writes_separate_clean_and_noisy_lanes(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    output = tmp_path / "output"
    prepare_base_v2(source, output, BaseV2Config(seed=7, profiles=(8,), materialize=True))
    base = output / "v2-8" / "base"
    assert (base / "train.noisy.src").is_file()
    assert (base / "train.clean.src").is_file()
    clean = (base / "train.clean.src").read_text(encoding="utf-8").splitlines()
    noisy = (base / "train.noisy.src").read_text(encoding="utf-8").splitlines()
    assert len(clean) == 2
    assert noisy
    metadata = [json.loads(line) for line in (base / "train.noisy.meta.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(not row["is_clean"] and row["rank"] <= 8 for row in metadata)
