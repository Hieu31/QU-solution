from __future__ import annotations

import csv
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


PROFILE = 'reparos-2023-osm-substitute/v1'


def normalized_query_group(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def prepare_query_group_split(
    prepared_data: str | Path,
    output: str | Path,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 2026,
) -> dict[str, object]:
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError('invalid train or validation ratio')
    if train_ratio + validation_ratio >= 1:
        raise ValueError('train_ratio + validation_ratio must be below 1')
    source_root, output_root = Path(prepared_data), Path(output)
    sources = [source_root / split / 'corpus.txt' for split in ('train', 'validation', 'test')]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing source corpora: {missing}')
    for split in ('train', 'validation', 'test'):
        (output_root / split).mkdir(parents=True, exist_ok=True)
    streams = {
        split: (output_root / split / 'corpus.txt').open('w', encoding='utf-8', newline='\n')
        for split in ('train', 'validation', 'test')
    }
    counts: Counter[str] = Counter()
    groups: dict[str, set[bytes]] = {split: set() for split in streams}
    train_cutoff = int(train_ratio * 1_000_000)
    validation_cutoff = int((train_ratio + validation_ratio) * 1_000_000)
    try:
        for source in sources:
            with source.open(encoding='utf-8') as input_stream:
                for line in input_stream:
                    clean = ' '.join(line.split())
                    if not clean:
                        continue
                    normalized = normalized_query_group(clean)
                    digest = hashlib.blake2b(
                        f'{seed}:{normalized}'.encode('utf-8'), digest_size=16,
                    ).digest()
                    bucket = int.from_bytes(digest[:8], 'big') % 1_000_000
                    split = (
                        'train' if bucket < train_cutoff else
                        'validation' if bucket < validation_cutoff else
                        'test'
                    )
                    streams[split].write(clean + '\n')
                    counts[f'{split}_rows'] += 1
                    groups[split].add(digest)
    finally:
        for stream in streams.values():
            stream.close()
    overlaps = {
        'train_validation': len(groups['train'] & groups['validation']),
        'train_test': len(groups['train'] & groups['test']),
        'validation_test': len(groups['validation'] & groups['test']),
    }
    if any(overlaps.values()):
        raise AssertionError(f'query-group split leaked: {overlaps}')
    manifest: dict[str, object] = {
        'profile': 'normalized-query-group-split/v1',
        'source': str(source_root.resolve()),
        'seed': seed,
        'ratios': {
            'train': train_ratio,
            'validation': validation_ratio,
            'test': 1 - train_ratio - validation_ratio,
        },
        'normalization': 'Unicode NFKC + casefold + collapsed whitespace',
        'counts': {
            split: {
                'rows': counts[f'{split}_rows'],
                'unique_query_groups': len(groups[split]),
            }
            for split in ('train', 'validation', 'test')
        },
        'overlaps': overlaps,
    }
    (output_root / 'group-split-manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return manifest


def _rng(seed: int, *parts: object) -> random.Random:
    raw = ':'.join((str(seed), *(str(part) for part in parts))).encode('utf-8')
    return random.Random(int.from_bytes(hashlib.blake2b(raw, digest_size=16).digest(), 'big'))


def _edit_word(word: str, rng: random.Random) -> str:
    if not word:
        return word
    operation = rng.choice(('delete', 'swap', 'replace', 'insert'))
    index = rng.randrange(len(word))
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
    if operation == 'delete' and len(word) > 1:
        return word[:index] + word[index + 1:]
    if operation == 'swap' and len(word) > 1:
        index = min(index, len(word) - 2)
        return word[:index] + word[index + 1] + word[index] + word[index + 2:]
    if operation == 'replace':
        return word[:index] + rng.choice(alphabet) + word[index + 1:]
    return word[:index] + rng.choice(alphabet) + word[index:]


def _compound(words: list[str], rng: random.Random) -> list[str]:
    if len(words) > 1:
        index = rng.randrange(len(words) - 1)
        return words[:index] + [words[index] + words[index + 1]] + words[index + 2:]
    word = words[0]
    if len(word) > 3:
        index = rng.randrange(1, len(word))
        return [word[:index], word[index:]]
    return words


def load_phonetic_pairs(path: str | Path | None) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    values: dict[str, set[str]] = defaultdict(set)
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'correct_word', 'phonetic_variant'}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f'phonetic pairs missing columns: {sorted(missing)}')
        for row in reader:
            correct, variant = row['correct_word'].strip(), row['phonetic_variant'].strip()
            if correct and variant and correct != variant:
                values[correct].add(variant)
    return {word: tuple(sorted(variants)) for word, variants in values.items()}


def noisy_query(
    clean: str,
    error_class: str,
    rng: random.Random,
    phonetic: dict[str, tuple[str, ...]],
    max_error_words: int = 2,
) -> str:
    words = clean.split()
    if not words:
        return clean
    complex_error = error_class in {'edit_compounding', 'phonetic_compounding'}
    base = error_class.removesuffix('_compounding')
    eligible = list(range(len(words)))
    if base == 'phonetic':
        eligible = [index for index, word in enumerate(words) if word in phonetic]
    if eligible:
        count = rng.randint(1, min(max_error_words, len(eligible)))
        for index in rng.sample(eligible, count):
            if base == 'edit':
                words[index] = _edit_word(words[index], rng)
            elif base == 'phonetic':
                words[index] = rng.choice(phonetic[words[index]])
    if error_class == 'compounding' or complex_error:
        words = _compound(words, rng)
    return ' '.join(words)


def _write_pairs(path: Path, pairs: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for source, _, _ in pairs:
            stream.write(source + '\n')
    with path.with_suffix('.tgt').open('w', encoding='utf-8', newline='\n') as stream:
        for _, target, _ in pairs:
            stream.write(target + '\n')
    with path.with_suffix('.meta.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for _, _, error_class in pairs:
            stream.write(json.dumps({'error_class': error_class}) + '\n')


def _read_feedback(path: str | Path, minimum_ctr: float) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'user_query', 'corrected_query', 'corrected_query_ctr'}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f'weak feedback missing columns: {sorted(missing)}')
        for row in reader:
            if float(row['corrected_query_ctr']) >= minimum_ctr:
                source, target = row['user_query'].strip(), row['corrected_query'].strip()
                if source and target:
                    pairs.append((source, target, 'weak_click_feedback'))
    return pairs


def prepare_reparos_data(
    prepared_data: str | Path,
    output: str | Path,
    *,
    variants_per_query: int = 4,
    clean_ratio: float = 1.0,
    seed: int = 2026,
    phonetic_pairs: str | Path | None = None,
    weak_feedback: str | Path | None = None,
    minimum_feedback_ctr: float = 0.1,
) -> dict[str, object]:
    if variants_per_query < 1 or clean_ratio < 0:
        raise ValueError('variants_per_query must be positive and clean_ratio non-negative')
    source_root, output_root = Path(prepared_data), Path(output)
    phonetic = load_phonetic_pairs(phonetic_pairs)
    counts: Counter[str] = Counter()
    base_classes = ['edit', 'compounding'] + (['phonetic'] if phonetic else [])
    for split in ('train', 'validation', 'test'):
        corpus = source_root / split / 'corpus.txt'
        if not corpus.is_file():
            raise FileNotFoundError(corpus)
        base_pairs: list[tuple[str, str, str]] = []
        c1_pairs: list[tuple[str, str, str]] = []
        with corpus.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                clean = line.strip()
                if not clean:
                    continue
                for trial in range(variants_per_query):
                    error_class = base_classes[trial % len(base_classes)]
                    noisy = noisy_query(clean, error_class, _rng(seed, split, line_number, trial), phonetic)
                    if noisy != clean:
                        base_pairs.append((noisy, clean, error_class))
                    complex_class = ('phonetic_compounding' if phonetic and trial % 2 else 'edit_compounding')
                    complex_noisy = noisy_query(clean, complex_class, _rng(seed, 'c1', split, line_number, trial), phonetic)
                    if complex_noisy != clean:
                        c1_pairs.append((complex_noisy, clean, complex_class))
                clean_copies = round(variants_per_query * clean_ratio)
                base_pairs.extend((clean, clean, 'clean') for _ in range(clean_copies))
        _write_pairs(output_root / 'base' / f'{split}.src', base_pairs)
        _write_pairs(output_root / 'c1' / f'{split}.src', c1_pairs)
        counts[f'base_{split}'] = len(base_pairs)
        counts[f'c1_{split}'] = len(c1_pairs)

    feedback_pairs: list[tuple[str, str, str]] = []
    if weak_feedback is not None:
        feedback_pairs = _read_feedback(weak_feedback, minimum_feedback_ctr)
        # Deterministic 90/10 train/validation split; never use feedback as test.
        train, validation = [], []
        for source, target, kind in feedback_pairs:
            digest = hashlib.blake2b(f'{seed}:{source}:{target}'.encode(), digest_size=8).digest()
            (validation if int.from_bytes(digest, 'big') % 10 == 0 else train).append((source, target, kind))
        _write_pairs(output_root / 'c2' / 'train.src', train)
        _write_pairs(output_root / 'c2' / 'validation.src', validation)
        counts['c2_train'], counts['c2_validation'] = len(train), len(validation)

    manifest = {
        'profile': PROFILE,
        'curriculum': ['base_synthetic', 'c1_complex_only', 'c2_weak_feedback'],
        'config': {
            'variants_per_query': variants_per_query,
            'clean_ratio': clean_ratio,
            'max_error_words': 2,
            'seed': seed,
            'minimum_feedback_ctr': minimum_feedback_ctr,
        },
        'available': {
            'phonetic_pairs': bool(phonetic),
            'weak_click_feedback': bool(feedback_pairs),
        },
        'missing_proprietary_inputs': [
            item for item, available in (
                ('Flipkart English-Hindi transliteration models and Hindi reformulation errors', bool(phonetic)),
                ('production click-feedback pairs', bool(feedback_pairs)),
                ('72K human-labelled Improvement/Regression evaluation set', False),
            ) if not available
        ],
        'counts': dict(sorted(counts.items())),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'reproduction-profile.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    write_opennmt_configs(output_root)
    return manifest


def write_opennmt_configs(root: str | Path) -> None:
    """Write disclosed ReparoS settings; mark undisclosed values explicitly."""
    root = Path(root)
    common = {
        'architecture': 'Transformer encoder-decoder',
        'encoder_layers': 1,
        'decoder_layers': 1,
        'attention_heads': 8,
        'hidden_size': 128,
        'subword': {'type': 'SentencePiece', 'vocabulary_size': 8000},
        'optimizer': {'name': 'Adam', 'learning_rate': 1.0, 'beta1': 0.8, 'beta2': 0.998, 'epsilon': 1e-8},
        'beam_width': 10,
        'fine_tuning_learning_rate': 0.0001,
        'update_all_parameters_on_fine_tuning': True,
        'toolkit': 'OpenNMT with CTranslate2 for serving',
        'undisclosed': ['epochs/steps', 'batch size', 'dropout', 'FFN size', 'warmup/schedule', 'length penalty', 'SentencePiece model type'],
    }
    stages = {
        'base': {'initialize_from': None, 'training_data': 'base/train', 'learning_rate': 1.0},
        'c1': {'initialize_from': 'base checkpoint', 'training_data': 'c1/train', 'learning_rate': 0.0001},
        'c2': {'initialize_from': 'c1 checkpoint', 'training_data': 'c2/train', 'learning_rate': 0.0001},
    }
    (root / 'opennmt-disclosed-config.json').write_text(
        json.dumps({'common': common, 'stages': stages}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )


def prepare_improvement_regression_sets(
    labeled_csv: str | Path,
    output: str | Path,
    *,
    examples_per_set: int = 0,
    seed: int = 2026,
) -> dict[str, int]:
    """Build the paper's 90/10 head-tail evaluation mixtures from human labels."""
    with Path(labeled_csv).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'noisy_query', 'correct_query', 'query_id', 'monthly_frequency', 'review_status'}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f'labeled evaluation data missing columns: {sorted(missing)}')
        rows = list(reader)
    if not rows or any(row['review_status'] != 'adjudicated' for row in rows):
        raise ValueError('evaluation rows must be non-empty and adjudicated')
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        if row['query_id'] in unique:
            raise ValueError(f"duplicate query_id: {row['query_id']}")
        float(row['monthly_frequency'])
        unique[row['query_id']] = row
    ordered = sorted(unique.values(), key=lambda row: (-float(row['monthly_frequency']), row['query_id']))
    midpoint = len(ordered) // 2
    head, tail = ordered[:midpoint], ordered[midpoint:]
    rng = _rng(seed, 'evaluation')
    rng.shuffle(head)
    rng.shuffle(tail)
    size = examples_per_set or min(len(head), len(tail))
    size = min(size, len(head), len(tail))
    if size < 2:
        raise ValueError('not enough labeled rows to form head/tail mixtures')

    def mixture(primary: list[dict[str, str]], secondary: list[dict[str, str]]) -> list[dict[str, str]]:
        primary_count = round(size * 0.9)
        return primary[:primary_count] + secondary[:size - primary_count]

    regression = mixture(head, tail)
    improvement = mixture(tail, head)
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) + ['frequency_bucket']
    head_ids = {row['query_id'] for row in head}
    for name, selected in (('regression', regression), ('improvement', improvement)):
        with (output_root / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                writer.writerow({**row, 'frequency_bucket': 'head' if row['query_id'] in head_ids else 'tail'})
    return {
        'labeled_queries': len(rows),
        'head_queries': len(head),
        'tail_queries': len(tail),
        'regression_queries': len(regression),
        'improvement_queries': len(improvement),
    }
