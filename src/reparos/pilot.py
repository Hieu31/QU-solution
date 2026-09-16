from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from reparos.data import _rng, noisy_query


PILOT_PROFILE = 'xanhsm-vietnamese-search-pilot/v1'
ERROR_CLASSES = ('missing_diacritics_full', 'edit', 'compounding', 'clean')


def remove_vietnamese_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize('NFD', text)
    plain = ''.join(character for character in decomposed if unicodedata.category(character) != 'Mn')
    return plain.replace('đ', 'd').replace('Đ', 'D')


def _group_id(clean: str) -> str:
    normalized = ' '.join(clean.casefold().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _variant(clean: str, error_class: str, seed: int, split: str, line_number: int) -> str:
    if error_class == 'missing_diacritics_full':
        return remove_vietnamese_diacritics(clean)
    if error_class == 'clean':
        return clean
    return noisy_query(clean, error_class, _rng(seed, 'vietnamese-pilot', split, line_number), {})


def _reservoir_by_class(
    corpus: Path,
    split: str,
    limits: dict[str, int],
    seed: int,
    excluded_groups: set[str] | None = None,
) -> dict[str, list[tuple[str, str, str, str]]]:
    samples: dict[str, list[tuple[str, str, str, str]]] = {
        error_class: [] for error_class in ERROR_CLASSES
    }
    seen: Counter[str] = Counter()
    rngs = {
        error_class: random.Random(f'{seed}:{split}:{error_class}')
        for error_class in ERROR_CLASSES
    }
    with corpus.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            clean = ' '.join(line.split())
            if not clean:
                continue
            group_id = _group_id(clean)
            if excluded_groups and group_id in excluded_groups:
                continue
            for error_class in ERROR_CLASSES:
                source = _variant(clean, error_class, seed, split, line_number)
                if error_class != 'clean' and source == clean:
                    continue
                seen[error_class] += 1
                record = (source, clean, error_class, group_id)
                bucket = samples[error_class]
                limit = limits[error_class]
                if len(bucket) < limit:
                    bucket.append(record)
                else:
                    position = rngs[error_class].randrange(seen[error_class])
                    if position < limit:
                        bucket[position] = record
    missing = {
        error_class: {'requested': limits[error_class], 'available': seen[error_class]}
        for error_class in ERROR_CLASSES
        if len(samples[error_class]) != limits[error_class]
    }
    if missing:
        raise ValueError(f'not enough eligible pilot examples: {missing}')
    return samples


def _write_split(
    output: Path,
    split: str,
    samples: dict[str, list[tuple[str, str, str, str]]],
    seed: int,
) -> set[str]:
    records = [record for error_class in ERROR_CLASSES for record in samples[error_class]]
    random.Random(f'{seed}:{split}:shuffle').shuffle(records)
    base = output / 'base'
    base.mkdir(parents=True, exist_ok=True)
    (base / f'{split}.src').write_text(
        ''.join(source + '\n' for source, _, _, _ in records), encoding='utf-8'
    )
    (base / f'{split}.tgt').write_text(
        ''.join(target + '\n' for _, target, _, _ in records), encoding='utf-8'
    )
    with (base / f'{split}.meta.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for _, _, error_class, group_id in records:
            stream.write(json.dumps({
                'error_class': error_class,
                'group_id': group_id,
                'profile': PILOT_PROFILE,
            }) + '\n')
    return {group_id for _, _, _, group_id in records}


def prepare_vietnamese_search_pilot(
    clean_corpus_root: str | Path,
    output: str | Path,
    *,
    train_per_class: int = 4096,
    validation_per_class: int = 512,
    train_clean_multiplier: int = 1,
    seed: int = 2026,
) -> dict[str, object]:
    if train_per_class < 1 or validation_per_class < 1 or train_clean_multiplier < 1:
        raise ValueError('per-class limits and clean multiplier must be positive')
    source_root, output_root = Path(clean_corpus_root), Path(output)
    limits = {
        'train': {
            error_class: (
                train_per_class * train_clean_multiplier
                if error_class == 'clean' else train_per_class
            )
            for error_class in ERROR_CLASSES
        },
        'validation': {error_class: validation_per_class for error_class in ERROR_CLASSES},
    }
    groups: dict[str, set[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    corpora = {split: source_root / split / 'corpus.txt' for split in limits}
    missing = [str(path) for path in corpora.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing pilot corpora: {missing}')
    selected: dict[str, dict[str, list[tuple[str, str, str, str]]]] = {}
    selected['train'] = _reservoir_by_class(
        corpora['train'], 'train', limits['train'], seed,
    )
    train_groups = {
        record[3] for records in selected['train'].values() for record in records
    }
    selected['validation'] = _reservoir_by_class(
        corpora['validation'], 'validation', limits['validation'], seed,
        excluded_groups=train_groups,
    )
    for split in ('train', 'validation'):
        groups[split] = {
            record[3] for records in selected[split].values() for record in records
        }
        counts[split] = {
            error_class: len(selected[split][error_class]) for error_class in ERROR_CLASSES
        }
    leakage = groups['train'].intersection(groups['validation'])
    if leakage:
        raise ValueError(f'group leakage between train and validation: {len(leakage)} groups')
    for split in ('train', 'validation'):
        _write_split(output_root, split, selected[split], seed)
    manifest: dict[str, object] = {
        'profile': PILOT_PROFILE,
        'claim': 'local Vietnamese search augmentation pilot; not ReparoS paper baseline',
        'source': str(source_root.resolve()),
        'seed': seed,
        'train_clean_multiplier': train_clean_multiplier,
        'error_classes': list(ERROR_CLASSES),
        'counts': counts,
        'group_split_verified': True,
        'test_split_used': False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'pilot-manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return manifest
