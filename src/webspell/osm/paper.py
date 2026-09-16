from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


PROFILE = "webspell-2009-osm-substitute/v1"
CHARACTER_ERROR_RATE = 0.02


def _seed(seed: int, split: str, document_id: str, trial: int) -> int:
    raw = f"{seed}:{split}:{document_id}:{trial}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=16).digest(), "big")


def corrupt_document(
    text: str,
    seed: int,
    character_error_rate: float = CHARACTER_ERROR_RATE,
) -> tuple[str, tuple[str, ...]]:
    """Paper Section 3.4.1 corruption, with explicitly documented choices.

    The paper does not state whether whitespace participates. We preserve it so
    token-level labels remain alignable; the manifest records this approximation.
    Each non-whitespace source character receives at most one independent trial.
    """
    if not 0.0 <= character_error_rate <= 1.0:
        raise ValueError("character_error_rate must be between 0 and 1")
    rng = random.Random(seed)
    alphabet = [char for char in text if not char.isspace()]
    if not alphabet:
        return text, ()
    output: list[str] = []
    operations: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace() or rng.random() >= character_error_rate:
            output.append(char)
            index += 1
            continue
        operation = rng.choice(("deletion", "transposition", "insertion"))
        if operation == "deletion":
            operations.append(operation)
            index += 1
        elif operation == "transposition" and index + 1 < len(text) and not text[index + 1].isspace():
            output.extend((text[index + 1], char))
            operations.append(operation)
            index += 2
        elif operation == "insertion":
            output.extend((rng.choice(alphabet), char))
            operations.append(operation)
            index += 1
        else:
            # A transposition at a boundary cannot be applied. Keeping the
            # character is less biased than silently converting it to deletion.
            output.append(char)
            index += 1
    return "".join(output), tuple(operations)


def prepare_artificial_data(
    prepared_data: str | Path,
    output: str | Path,
    *,
    variants_per_document: int = 1,
    seed: int = 2026,
    character_error_rate: float = CHARACTER_ERROR_RATE,
) -> dict[str, object]:
    """Create paper-style artificial splits from already group-split clean data."""
    if variants_per_document < 1:
        raise ValueError("variants_per_document must be positive")
    source_root, output_root = Path(prepared_data), Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        source_corpus = source_root / split / "corpus.txt"
        if not source_corpus.is_file():
            raise FileNotFoundError(source_corpus)
        split_root = output_root / split
        split_root.mkdir(exist_ok=True)
        shutil.copy2(source_corpus, split_root / "corpus.txt")
        pair_path = split_root / "noisy_pairs.csv"
        with source_corpus.open(encoding="utf-8") as source, pair_path.open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            writer = csv.writer(destination)
            writer.writerow((
                "noisy_query", "correct_query", "entity_id", "group_id",
                "term_role", "noise_source", "error_type", "variant_id",
            ))
            for line_number, line in enumerate(source, 1):
                clean = line.strip()
                if not clean:
                    continue
                document_id = f"{split}:{line_number}"
                for trial in range(variants_per_document):
                    noisy, operations = corrupt_document(
                        clean, _seed(seed, split, document_id, trial), character_error_rate
                    )
                    error_type = "paper_" + "+".join(sorted(set(operations))) if operations else "clean"
                    writer.writerow((
                        noisy, clean, document_id, document_id, "document",
                        "paper_artificial", error_type, f"trial-{trial}",
                    ))
                    counts[f"{split}_documents"] += 1
                    counts["pairs"] += 1
                    counts["changed_pairs" if noisy != clean else "unchanged_pairs"] += 1
                    operation_counts.update(operations)

    profile = {
        "profile": PROFILE,
        "purpose": "paper reproduction baseline with OSM as substitute corpus",
        "lanes": {
            "web_corpus": "split/corpus.txt; vocabulary, frequency, LM and Section 3.2 mining",
            "artificial": "split/noisy_pairs.csv; lambda, confidence training/tuning and artificial evaluation",
            "typed": "external held-out human-typed CSV; evaluation only and never consumed here",
        },
        "artificial_generation": {
            "character_error_rate": character_error_rate,
            "operations": ["deletion", "transposition", "insertion"],
            "operation_probability": "uniform",
            "insertion_alphabet": "characters from the same input line",
            "variants_per_document": variants_per_document,
            "seed": seed,
        },
        "known_approximations": [
            "OSM names/address fragments replace news documents and public Web pages.",
            "Each corpus line is treated as a document for insertion-character sampling.",
            "Whitespace is preserved; the paper does not disclose whether it was mutable.",
            "A source character receives at most one mutation trial; the paper does not disclose the sampler.",
        ],
        "counts": dict(sorted(counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
    }
    (output_root / "reproduction-profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return profile


def validate_typed_test(path: str | Path) -> dict[str, int]:
    """Validate the immutable human-typed evaluation interchange format."""
    required = {
        "noisy_query", "correct_query", "query_id", "participant_id",
        "source_text_id", "collection_protocol", "review_status",
    }
    counts: Counter[str] = Counter()
    groups: dict[str, str] = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"typed test is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            if not row["noisy_query"].strip() or not row["correct_query"].strip():
                raise ValueError(f"empty query at row {row_number}")
            if row["review_status"] != "adjudicated":
                raise ValueError(f"row {row_number} is not adjudicated")
            query_id = row["query_id"]
            if query_id in groups:
                raise ValueError(f"duplicate query_id: {query_id}")
            groups[query_id] = row["source_text_id"]
            counts["queries"] += 1
            counts["changed_queries" if row["noisy_query"] != row["correct_query"] else "clean_queries"] += 1
            counts[f"protocol:{row['collection_protocol']}"] += 1
    if not counts["queries"]:
        raise ValueError("typed test is empty")
    return dict(sorted(counts.items()))
