from __future__ import annotations

import csv
import json
from pathlib import Path

from reparos.typing_curriculum import CurriculumConfig, abbreviate, encode_ime, prepare_typing_curriculum


def _source(root: Path) -> Path:
    rows = {
        "train": [("g1", "bệnh viện bạch mai"), ("g2", "đường nguyễn trãi quận 1")],
        "validation": [("g3", "trường đại học bách khoa")],
        "test": [("g4", "thành phố hồ chí minh")],
    }
    for split, values in rows.items():
        folder = root / split
        folder.mkdir(parents=True)
        with (folder / "noisy_pairs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("noisy_query", "correct_query", "group_id", "error_type"))
            writer.writeheader()
            for group_id, target in values:
                writer.writerow({"noisy_query": target, "correct_query": target, "group_id": group_id, "error_type": "clean"})
    return root


def test_ime_encoder_and_abbreviation_are_explicit() -> None:
    import random
    assert encode_ime("trường học", "telex", random.Random(1)) == "truowngf hocj"
    assert encode_ime("trường học", "vni", random.Random(1)) == "tru7o7ng2 hoc5"
    assert abbreviate("bệnh viện bạch mai", random.Random(1)).startswith("bv")


def test_curriculum_writes_four_stages_without_cross_split_groups(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    output = tmp_path / "output"
    manifest = prepare_typing_curriculum(source, output, CurriculumConfig(seed=7, ime_variants_per_query=2, abbreviation_variants_per_query=2, mixed_variants_per_query=2))
    assert set(manifest["stages"]) == {"stage1-ime", "stage2-abbreviation", "stage3-mixed", "stage4-domain"}
    assert manifest["leakage"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    for stage in manifest["stages"]:
        for split in ("train", "validation", "test"):
            assert (output / stage / f"{split}.src").is_file()
            assert (output / stage / f"{split}.tgt").is_file()
            metadata = [json.loads(line) for line in (output / stage / f"{split}.meta.jsonl").read_text(encoding="utf-8").splitlines()]
            assert all(row["split"] == split for row in metadata)


def test_curriculum_is_deterministic(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    first, second = tmp_path / "first", tmp_path / "second"
    prepare_typing_curriculum(source, first, CurriculumConfig(seed=99))
    prepare_typing_curriculum(source, second, CurriculumConfig(seed=99))
    assert (first / "stage3-mixed" / "train.src").read_bytes() == (second / "stage3-mixed" / "train.src").read_bytes()


def test_source_poi_group_may_cross_splits_when_query_groups_differ(tmp_path: Path) -> None:
    source = tmp_path / "source"
    values = {
        "train": "phường lào cai",
        "validation": "lào cai",
        "test": "ủy ban nhân dân phường lào cai",
    }
    for split, target in values.items():
        folder = source / split
        folder.mkdir(parents=True)
        with (folder / "noisy_pairs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("noisy_query", "correct_query", "group_id", "error_type"))
            writer.writeheader()
            writer.writerow({"noisy_query": target, "correct_query": target, "group_id": "phường lào cai", "error_type": "clean"})
    output = tmp_path / "output"
    prepare_typing_curriculum(source, output, CurriculumConfig(seed=7))
    metadata = json.loads((output / "stage1-ime" / "validation.meta.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metadata["source_group_id"] == "phường lào cai"
    assert metadata["query_group"] == "lào cai"
