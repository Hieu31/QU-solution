from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from webspell.osm.noise import DEFAULT_NOISE_WEIGHTS, location_query_variants


NAME_TAGS = ("name:vi", "name", "official_name", "short_name", "alt_name", "old_name", "loc_name", "brand", "operator")
ADDRESS_TAGS = ("addr:housenumber", "addr:street", "addr:suburb", "addr:district", "addr:city", "addr:province")
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s\-/]", re.UNICODE)


@dataclass(frozen=True)
class OSMPreparationConfig:
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    seed: int = 2026
    noisy_variants_per_term: int = 3
    minimum_length: int = 2
    character_error_rate: float = 0.02
    clean_variants_per_term: int = 1
    noise_weights: tuple[tuple[str, int], ...] = tuple(DEFAULT_NOISE_WEIGHTS.items())

    def validate(self) -> None:
        if not (0 < self.train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1")
        if not (0 <= self.validation_ratio < 1):
            raise ValueError("validation_ratio must be between 0 and 1")
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("train_ratio + validation_ratio must be below 1")
        if self.noisy_variants_per_term < 0:
            raise ValueError("noisy_variants_per_term cannot be negative")
        if not (0.0 <= self.character_error_rate <= 1.0):
            raise ValueError("character_error_rate must be between 0 and 1")
        if self.clean_variants_per_term < 0:
            raise ValueError("clean_variants_per_term cannot be negative")
        weights = dict(self.noise_weights)
        unknown = set(weights).difference(DEFAULT_NOISE_WEIGHTS)
        if unknown or not weights or any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError(f"invalid noise_weights; unknown={sorted(unknown)}")


@dataclass(frozen=True)
class OSMEntity:
    entity_id: str
    entity_type: str
    canonical: str
    aliases: tuple[str, ...]


def normalize_term(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold().replace("_", " ")
    value = PUNCT_RE.sub(" ", value)
    return SPACE_RE.sub(" ", value).strip(" -/")


def _values(tags: Mapping[str, str], keys: Iterable[str]) -> Iterator[str]:
    for key in keys:
        for value in tags.get(key, "").split(";"):
            normalized = normalize_term(value)
            if normalized:
                yield normalized


def entity_from_tags(entity_id: str, tags: Mapping[str, str]) -> OSMEntity | None:
    names = list(dict.fromkeys(_values(tags, NAME_TAGS)))
    address = " ".join(_values(tags, ADDRESS_TAGS))
    if address:
        names.append(address)
    names = list(dict.fromkeys(term for term in names if len(term) >= 2))
    if not names:
        return None
    canonical = normalize_term(tags.get("name:vi") or tags.get("name") or names[0])
    if not canonical:
        canonical = names[0]
    entity_type = _entity_type(tags)
    return OSMEntity(entity_id, entity_type, canonical, tuple(names))


def _entity_type(tags: Mapping[str, str]) -> str:
    for key in ("amenity", "shop", "tourism", "leisure", "office", "highway", "boundary", "place"):
        if tags.get(key):
            return f"{key}:{tags[key]}"
    return "named_object"


def iter_osm_entities(path: str | Path) -> Iterator[OSMEntity]:
    source = Path(path)
    if source.suffix.lower() != ".pbf" and not source.name.endswith(".osm.pbf"):
        raise ValueError("input must be an .osm.pbf file")
    try:
        import osmium  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PBF support requires the optional dependency: pip install 'qu-solution[osm]'") from exc

    # KeyFilter executes in libosmium before Python iteration. This avoids a
    # Python callback for every unnamed node in the country extract.
    keys = (*NAME_TAGS, *ADDRESS_TAGS)
    processor = osmium.FileProcessor(str(source)).with_filter(osmium.filter.KeyFilter(*keys))
    prefixes = {"n": "node", "w": "way", "r": "relation"}
    for obj in processor:
        tags = {tag.k: tag.v for tag in obj.tags}
        prefix = prefixes.get(obj.type_str(), obj.type_str())
        entity = entity_from_tags(f"{prefix}/{obj.id}", tags)
        if entity is not None:
            yield entity


def split_for_entity(entity_id: str, config: OSMPreparationConfig) -> str:
    digest = hashlib.blake2b(f"{config.seed}:{entity_id}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / 2**64
    if value < config.train_ratio:
        return "train"
    if value < config.train_ratio + config.validation_ratio:
        return "validation"
    return "test"


def _strip_diacritics(text: str) -> str:
    text = text.replace("đ", "d")
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def synthetic_query_variants(
    term: str,
    count: int,
    seed: int,
    character_error_rate: float = 0.02,
    noise_weights: Mapping[str, int] | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Generate deterministic noisy queries for Vietnamese location search."""
    del character_error_rate  # kept for compatibility with existing commands
    return tuple(
        (noisy, f"synthetic_{error_type}", variant_id)
        for noisy, error_type, variant_id in location_query_variants(
            term, count, seed, noise_weights
        )
    )


def synthetic_noisy_variants(term: str, count: int, seed: int) -> tuple[str, ...]:
    """Backward-compatible text-only view of synthetic query trials."""
    return tuple(item[0] for item in synthetic_query_variants(term, count, seed))


def prepare_entities(entities: Iterable[OSMEntity], output: str | Path, config: OSMPreparationConfig | None = None) -> dict[str, int]:
    config = config or OSMPreparationConfig()
    config.validate()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    handles: dict[str, tuple[object, object, object, csv.writer]] = {}
    counts = {"entities": 0, "terms": 0, "noisy_pairs": 0, "clean_pairs": 0, "misspelled_pairs": 0, "train_entities": 0, "validation_entities": 0, "test_entities": 0}
    error_type_counts: dict[str, int] = {}
    try:
        for split in ("train", "validation", "test"):
            directory = root / split
            directory.mkdir(exist_ok=True)
            corpus = (directory / "corpus.txt").open("w", encoding="utf-8", newline="\n")
            entity_file = (directory / "entities.jsonl").open("w", encoding="utf-8", newline="\n")
            pair_file = (directory / "noisy_pairs.csv").open("w", encoding="utf-8", newline="")
            writer = csv.writer(pair_file)
            writer.writerow(("noisy_query", "correct_query", "entity_id", "group_id", "term_role", "noise_source", "error_type", "variant_id"))
            handles[split] = (corpus, entity_file, pair_file, writer)
        seen: set[str] = set()
        for entity in entities:
            if entity.entity_id in seen:
                continue
            seen.add(entity.entity_id)
            # Duplicate OSM objects for the same canonical name stay together.
            group_id = entity.canonical
            split = split_for_entity(group_id, config)
            corpus, entity_file, pair_file, writer = handles[split]
            aliases = tuple(dict.fromkeys(normalize_term(x) for x in entity.aliases if len(normalize_term(x)) >= config.minimum_length))
            if not aliases:
                continue
            entity_file.write(json.dumps({"entity_id": entity.entity_id, "group_id": group_id, "type": entity.entity_type, "canonical": entity.canonical, "aliases": aliases}, ensure_ascii=False) + "\n")
            counts["entities"] += 1
            counts[f"{split}_entities"] += 1
            for term in aliases:
                term_role = "canonical" if term == entity.canonical else "alias"
                corpus.write(term + "\n")
                counts["terms"] += 1
                for clean_index in range(config.clean_variants_per_term):
                    writer.writerow((term, term, entity.entity_id, group_id, term_role, "clean", "clean", f"clean-{clean_index}"))
                    counts["noisy_pairs"] += 1
                    counts["clean_pairs"] += 1
                    error_type_counts["clean"] = error_type_counts.get("clean", 0) + 1
                for noisy, noise_source, variant_id in synthetic_query_variants(
                    term, config.noisy_variants_per_term, config.seed,
                    config.character_error_rate, dict(config.noise_weights),
                ):
                    error_type = noise_source.removeprefix("synthetic_")
                    writer.writerow((noisy, term, entity.entity_id, group_id, term_role, noise_source, error_type, variant_id))
                    counts["noisy_pairs"] += 1
                    counts["misspelled_pairs"] += 1
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
    finally:
        for corpus, entity_file, pair_file, writer in handles.values():
            corpus.close()
            entity_file.close()
            pair_file.close()
    manifest_config = {**config.__dict__, "noise_weights": dict(config.noise_weights)}
    manifest = {"config": manifest_config, "counts": counts, "error_type_counts": dict(sorted(error_type_counts.items()))}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return counts


def prepare_osm(source: str | Path, output: str | Path, config: OSMPreparationConfig | None = None) -> dict[str, int]:
    return prepare_entities(iter_osm_entities(source), output, config)


def resplit_prepared_queries(
    source: str | Path,
    output: str | Path,
    config: OSMPreparationConfig | None = None,
) -> dict[str, object]:
    """Repartition prepared query pairs by normalized correct query.

    This preserves the generated rows and their metadata while preventing an
    alias shared by different OSM entities from crossing data splits.
    """
    config = config or OSMPreparationConfig()
    config.validate()
    source_root, output_root = Path(source), Path(output)
    inputs = [source_root / split / 'noisy_pairs.csv' for split in ('train', 'validation', 'test')]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing prepared query pairs: {missing}')
    fieldnames = (
        'noisy_query', 'correct_query', 'entity_id', 'group_id',
        'term_role', 'noise_source', 'error_type', 'variant_id',
    )
    handles: dict[str, tuple[object, object, csv.DictWriter]] = {}
    counts: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    groups: dict[str, set[str]] = {split: set() for split in ('train', 'validation', 'test')}
    try:
        for split in groups:
            directory = output_root / split
            directory.mkdir(parents=True, exist_ok=True)
            corpus = (directory / 'corpus.txt').open('w', encoding='utf-8', newline='\n')
            pair_file = (directory / 'noisy_pairs.csv').open('w', encoding='utf-8', newline='')
            writer = csv.DictWriter(pair_file, fieldnames=fieldnames)
            writer.writeheader()
            handles[split] = (corpus, pair_file, writer)
        for input_path in inputs:
            with input_path.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                if not set(fieldnames).issubset(reader.fieldnames or ()):
                    raise ValueError(f'invalid noisy-pair schema: {input_path}')
                for row in reader:
                    correct = normalize_term(row['correct_query'])
                    if not correct:
                        continue
                    split = split_for_entity(correct, config)
                    corpus, _, writer = handles[split]
                    writer.writerow({name: row.get(name, '') for name in fieldnames})
                    counts[f'{split}_pairs'] += 1
                    error_types[row['error_type']] += 1
                    groups[split].add(correct)
                    if row['noise_source'] == 'clean':
                        corpus.write(correct + '\n')
                        counts[f'{split}_corpus_rows'] += 1
    finally:
        for corpus, pair_file, _ in handles.values():
            corpus.close()
            pair_file.close()
    overlaps = {
        'train_validation': len(groups['train'] & groups['validation']),
        'train_test': len(groups['train'] & groups['test']),
        'validation_test': len(groups['validation'] & groups['test']),
    }
    if any(overlaps.values()):
        raise AssertionError(f'query split leaked: {overlaps}')
    manifest: dict[str, object] = {
        'profile': 'webspell-prepared-query-group-split/v1',
        'source': str(source_root.resolve()),
        'config': {
            'train_ratio': config.train_ratio,
            'validation_ratio': config.validation_ratio,
            'seed': config.seed,
            'group_key': 'normalized correct_query',
        },
        'counts': {
            split: {
                'pairs': counts[f'{split}_pairs'],
                'corpus_rows': counts[f'{split}_corpus_rows'],
                'unique_query_groups': len(groups[split]),
            }
            for split in ('train', 'validation', 'test')
        },
        'overlaps': overlaps,
        'error_type_counts': dict(sorted(error_types.items())),
    }
    (output_root / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return manifest
