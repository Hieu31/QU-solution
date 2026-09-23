'''Build a production word-level labelled test set from ``data/zero_click.csv``.

The zero-click log is an unlabelled sensitive query log, not a correction
corpus. This builder derives word-level labels for real production queries from
two independent channels:

1. ``entity_anchor`` - the whole query matches exactly one clean OSM term after
   case folding and diacritic stripping, so every differing token has a unique,
   unambiguous correction.
2. ``token_lexicon`` - the query is not a known entity, but an unknown token
   resolves through plain-form match, Telex decode, VNI decode or a single
   substitution, and the winning candidate dominates every rival candidate.

The result is a held-out *test* artifact. Never use it for training.

Reference: docs/zero-click-data-audit.md (section 8.4, weak labelling).
'''

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

TONE_ROWS = (
    'a\u00e1\u00e0\u1ea3\u00e3\u1ea1',
    '\u0103\u1eaf\u1eb1\u1eb3\u1eb5\u1eb7',
    '\u00e2\u1ea5\u1ea7\u1ea9\u1eab\u1ead',
    'e\u00e9\u00e8\u1ebb\u1ebd\u1eb9',
    '\u00ea\u1ebf\u1ec1\u1ec3\u1ec5\u1ec7',
    'i\u00ed\u00ec\u1ec9\u0129\u1ecb',
    'o\u00f3\u00f2\u1ecf\u00f5\u1ecd',
    '\u00f4\u1ed1\u1ed3\u1ed5\u1ed7\u1ed9',
    '\u01a1\u1edb\u1edd\u1edf\u1ee1\u1ee3',
    'u\u00fa\u00f9\u1ee7\u0169\u1ee5',
    '\u01b0\u1ee9\u1eeb\u1eed\u1eef\u1ef1',
    'y\u00fd\u1ef3\u1ef7\u1ef9\u1ef5',
)
TONE_KEYS = {'s': 1, 'f': 2, 'r': 3, 'x': 4, 'j': 5}
VNI_TONES = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
BASE_REPLACEMENTS = (
    ('uow', '\u01b0\u01a1'), ('dd', '\u0111'), ('aw', '\u0103'), ('aa', '\u00e2'),
    ('ee', '\u00ea'), ('oo', '\u00f4'), ('ow', '\u01a1'), ('uw', '\u01b0'),
)
VNI_REPLACEMENTS = (
    ('d9', '\u0111'), ('a8', '\u0103'), ('a6', '\u00e2'), ('e6', '\u00ea'),
    ('o6', '\u00f4'), ('o7', '\u01a1'), ('u7', '\u01b0'),
)
TONE_LOOKUP: dict[str, tuple[str, int]] = {}
for _row in TONE_ROWS:
    for _tone, _char in enumerate(_row):
        TONE_LOOKUP[_char] = (_row[0], _tone)

PHONE = re.compile(r'(?<!\d)(?:\+?84|0)(?:[ .-]?\d){8,10}(?!\d)')
EMAIL = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
WORDLIKE = re.compile(r'[^\W\d_]', re.UNICODE)
DELETE_KEY = '\x00'
SPLITS = ('train', 'validation', 'test')

DEFAULT_QUOTAS = {
    'missing_diacritics_full': 2400,
    'telex_leak': 900,
    'vni_leak': 700,
    'wrong_diacritic': 500,
    'keyboard_edit': 400,
    'missing_diacritics_partial': 100,
}


def nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def plain(text: str) -> str:
    value = unicodedata.normalize('NFD', nfc(text).casefold())
    value = ''.join(char for char in value if unicodedata.category(char) != 'Mn')
    return nfc(value).replace('\u0111', 'd')


def plain_key(text: str) -> str:
    return ' '.join(plain(text).split())


def strip_tone(text: str) -> str:
    return ''.join(TONE_LOOKUP.get(char, (char, 0))[0] for char in text)


def tone_target(text: str) -> int | None:
    vowels = [index for index, char in enumerate(text) if char in TONE_LOOKUP]
    if not vowels:
        return None
    marked = [index for index in vowels if text[index] in '\u0103\u00e2\u00ea\u00f4\u01a1\u01b0']
    if marked:
        return marked[-1]
    if len(vowels) == 1:
        return vowels[0]
    cluster = ''.join(text[index] for index in vowels)
    if cluster.startswith(('oa', 'oe', 'uy')):
        return vowels[0]
    return vowels[-2] if len(vowels) >= 2 else vowels[-1]


def apply_tone(text: str, tone: int) -> str:
    base_form = strip_tone(text)
    target = tone_target(base_form)
    if target is None:
        return base_form
    tone_row = next(row for row in TONE_ROWS if row[0] == base_form[target])
    return base_form[:target] + tone_row[tone] + base_form[target + 1:]


def decode_input_method(token: str, replacements, tones) -> str:
    lowered = nfc(token).lower()
    tone = 0
    if lowered and lowered[-1] in tones:
        tone = tones[lowered[-1]]
        lowered = lowered[:-1]
    for source, target in replacements:
        lowered = lowered.replace(source, target)
    return apply_tone(lowered, tone)


def restore_case(source: str, decoded: str) -> str:
    if source.isupper() and len(source) > 1:
        return decoded.upper()
    if source.istitle():
        return decoded[:1].upper() + decoded[1:]
    return decoded


def safe_query(text: str, max_characters: int) -> bool:
    return bool(
        text
        and len(text) <= max_characters
        and not PHONE.search(text)
        and not EMAIL.search(text)
    )


def stable_id(key: str) -> str:
    return 'zero-click:' + hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]


def family_for_query_token(query_token: str, target_token: str) -> str:
    query = nfc(query_token).casefold()
    target = nfc(target_token).casefold()
    if plain(query) == plain(target):
        return (
            'missing_diacritics_full' if plain(query) == query
            else 'wrong_diacritic'
        )
    if decode_input_method(query, BASE_REPLACEMENTS, TONE_KEYS) == target:
        return 'telex_leak'
    if decode_input_method(query, VNI_REPLACEMENTS, VNI_TONES) == target:
        return 'vni_leak'
    if plain(query) == query:
        return 'missing_diacritics_full'
    return 'keyboard_edit'


KEYBOARD_NEIGHBORS = {
    'q': 'wa', 'w': 'qeas', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg',
    'y': 'tugh', 'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',
    'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
    'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}


def non_letters(text: str) -> str:
    return ''.join(char for char in text if not char.isalnum())


def vni_shape(token: str) -> bool:
    '''True when a token can plausibly carry VNI shape markers (6, 7, 8, 9).'''
    if plain(token) != token:
        return False
    body = token[:-1] if token and token[-1] in '12345' else token
    if any(char in '6789' for char in body):
        stripped = ''.join(char for char in body if not char.isdigit())
        return stripped.isalpha() and len(stripped) >= 2
    return len(body) >= 4 and body.isalpha() and plain(body) == body


def telex_shape(token: str) -> bool:
    '''True when an unaccented alphabetic token can plausibly be Telex input.'''
    if not token.isalpha() or token.isupper() or plain(token) != token:
        return False
    base = token[:-1] if token[-1] in TONE_KEYS else token
    return len(base) >= 3


def keyboard_adjacent(query: str, target: str) -> bool:
    '''True when the plain forms differ by one adjacent-key or neighbouring-key typo.'''
    left, right = plain(query), plain(target)
    if len(left) < 3 or len(left) != len(right) or left == right:
        return False
    differences = [(a, b) for a, b in zip(left, right) if a != b]
    if len(differences) != 1:
        return False
    source, replacement = differences[0]
    return (
        replacement in KEYBOARD_NEIGHBORS.get(source, '')
        or source in KEYBOARD_NEIGHBORS.get(replacement, '')
    )


class Resources:
    '''Frozen term lexicon and clean entity surfaces used by every channel.'''

    def __init__(self, database: Path, osm_root: Path) -> None:
        connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
        rows = connection.execute('select term, frequency from terms').fetchall()
        connection.close()
        self.frequency: dict[str, int] = {}
        for term, frequency in rows:
            term = nfc(term)
            self.frequency[term] = max(self.frequency.get(term, 0), int(frequency))
        self.terms = set(self.frequency)
        self.by_plain: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.by_substitution: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for term, frequency in self.frequency.items():
            flattened = plain(term)
            self.by_plain[flattened].append((term, frequency))
            for index in range(len(flattened)):
                key = flattened[:index] + DELETE_KEY + flattened[index + 1:]
                self.by_substitution[key].append((term, frequency))
        self.entities = self._load_entities(osm_root)

    @staticmethod
    def _load_entities(osm_root: Path) -> dict[str, tuple[str, ...]]:
        surfaces: dict[str, set[str]] = defaultdict(set)
        for split in SPLITS:
            path = osm_root / split / 'corpus.txt'
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    surface = nfc(line.strip())
                    if surface:
                        surfaces[plain_key(surface)].add(surface)
        return {
            key: tuple(sorted(values))
            for key, values in surfaces.items()
            if len(values) == 1
        }

    def candidates(self, token: str) -> list[tuple[str, int, str]]:
        lowered = token.casefold()
        found: dict[str, tuple[str, int, str]] = {}

        def offer(term: str, frequency: int, family: str) -> None:
            if term == lowered or term not in self.terms:
                return
            current = found.get(term)
            if current is None or current[1] < frequency:
                found[term] = (term, frequency, family)

        plain_token = plain(lowered)
        diacritic_family = (
            'missing_diacritics_full' if plain_token == lowered
            else 'missing_diacritics_partial'
        )
        for term, frequency in self.by_plain.get(plain_token, []):
            offer(term, frequency, diacritic_family)

        telex = decode_input_method(lowered, BASE_REPLACEMENTS, TONE_KEYS)
        if telex in self.terms and telex_shape(lowered):
            offer(telex, self.frequency[telex], 'telex_leak')
        vni = decode_input_method(lowered, VNI_REPLACEMENTS, VNI_TONES)
        if vni in self.terms and vni_shape(lowered):
            offer(vni, self.frequency[vni], 'vni_leak')

        if lowered.isalpha():
            for index in range(len(plain_token)):
                key = plain_token[:index] + DELETE_KEY + plain_token[index + 1:]
                for term, frequency in self.by_substitution.get(key, []):
                    if (
                        term != lowered
                        and plain(term) != plain_token
                        and keyboard_adjacent(lowered, term)
                    ):
                        offer(term, frequency, 'keyboard_edit')
        return list(found.values())


def make_row(raw_query: str, labels: list[dict], channel: str, entity: str | None) -> dict:
    tokens = raw_query.split()
    corrected = list(tokens)
    for label in labels:
        corrected[label['index']] = label['expected']
    families = sorted({label['error_type'] for label in labels})
    return {
        'query_id': stable_id(' '.join(token.casefold() for token in tokens)),
        'input': ' '.join(tokens),
        'expected': ' '.join(corrected),
        'error_type': families[0] if len(families) == 1 else 'combined:' + '+'.join(families),
        'error_types': families,
        'word_labels': labels,
        'tokens': len(tokens),
        'label_channel': channel,
        'entity': entity,
        'review_status': 'weak_auto_labeled',
        'label_provenance': 'zero-click-x-osm',
        'split': 'zero-click-word-level',
    }


def entity_analysis(raw_query: str, resources: Resources) -> dict | None:
    surface = resources.entities.get(plain_key(raw_query))
    if surface is None:
        return None
    target = surface[0]
    tokens, target_tokens = raw_query.split(), target.split()
    if len(tokens) != len(target_tokens):
        return None
    labels = []
    for index, (token, target_token) in enumerate(zip(tokens, target_tokens)):
        if nfc(token).casefold() == nfc(target_token).casefold():
            continue
        if non_letters(token) != non_letters(target_token):
            continue
        labels.append({
            'index': index,
            'surface': token,
            'expected': restore_case(token, nfc(target_token)),
            'error_type': family_for_query_token(token, target_token),
            'lexicon_frequency': resources.frequency.get(nfc(target_token).casefold()),
            'margin': None,
            'alternatives': [],
        })
    if not labels:
        return None
    return make_row(raw_query, labels, 'entity_anchor', target)


def token_analysis(
    raw_query: str, resources: Resources, minimum_frequency: int, minimum_margin: float
) -> dict | None:
    labels = []
    for index, token in enumerate(raw_query.split()):
        lowered = nfc(token).casefold()
        if lowered in resources.terms or not WORDLIKE.search(lowered):
            continue
        alphabetic = lowered.isalpha()
        pool = [
            item for item in resources.candidates(lowered)
            if item[1] >= minimum_frequency
        ]
        if not pool:
            if alphabetic:
                return None
            continue
        pool.sort(key=lambda item: (-item[1], item[0]))
        best = pool[0]
        rival = max((item[1] for item in pool[1:]), default=1)
        if best[1] / rival < minimum_margin:
            if alphabetic:
                return None
            continue
        labels.append({
            'index': index,
            'surface': token,
            'expected': restore_case(token, best[0]),
            'error_type': best[2],
            'lexicon_frequency': best[1],
            'margin': round(best[1] / rival, 2),
            'alternatives': [item[0] for item in pool[1:4]],
        })
    if not labels:
        return None
    return make_row(raw_query, labels, 'token_lexicon', None)


def is_clean_query(raw_query: str, resources: Resources) -> bool:
    tokens = raw_query.split()
    return all(
        nfc(token).casefold() in resources.terms or not WORDLIKE.search(token)
        for token in tokens
    )


def collect(path: Path, resources: Resources, arguments):
    frequency: Counter[str] = Counter()
    representative: dict[str, str] = {}
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            raw = nfc(row['keyword'].strip())
            if not safe_query(raw, arguments.max_characters):
                continue
            key = ' '.join(raw.casefold().split())
            if not key:
                continue
            frequency[key] += 1
            representative.setdefault(key, raw)

    analysed: dict[str, dict] = {}
    clean: list[str] = []
    for key, raw in representative.items():
        if len(key.split()) > arguments.max_tokens:
            continue
        row = entity_analysis(raw, resources)
        if row is None:
            row = token_analysis(
                raw, resources, arguments.minimum_term_frequency, arguments.minimum_margin
            )
        if row is not None:
            row['frequency'] = frequency[key]
            analysed[key] = row
        elif is_clean_query(raw, resources):
            clean.append(key)
    return analysed, clean, frequency


def dominant_family(row: dict) -> str:
    counts = Counter(label['error_type'] for label in row['word_labels'])
    return counts.most_common(1)[0][0]


def sample(analysed: dict[str, dict], arguments) -> list[dict]:
    by_family: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in analysed.values():
        by_family[dominant_family(row)][row['label_channel']].append(row)

    rng = random.Random(arguments.seed)
    selected: list[dict] = []
    used: set[str] = set()
    for family, quota in arguments.quotas.items():
        groups = by_family.get(family, {})
        anchored = list(groups.get('entity_anchor', []))
        lexical = list(groups.get('token_lexicon', []))
        rng.shuffle(anchored)
        rng.shuffle(lexical)
        anchored_quota = min(len(anchored), -(-quota * arguments.entity_share // 100))
        picked = anchored[:anchored_quota]
        picked += lexical[:quota - len(picked)]
        if len(picked) < quota:
            remaining = anchored[anchored_quota:]
            picked += remaining[:quota - len(picked)]
        for row in picked:
            if row['query_id'] in used:
                continue
            used.add(row['query_id'])
            row['stratum'] = family
            selected.append(row)

    if len(selected) < arguments.target_rows:
        leftovers = [row for row in analysed.values() if row['query_id'] not in used]
        anchored = [row for row in leftovers if row['label_channel'] == 'entity_anchor']
        lexical = [row for row in leftovers if row['label_channel'] != 'entity_anchor']
        rng.shuffle(anchored)
        rng.shuffle(lexical)
        need = arguments.target_rows - len(selected)
        for pool in (anchored, lexical):
            for row in pool:
                if need <= 0:
                    break
                if row['query_id'] in used:
                    continue
                used.add(row['query_id'])
                row['stratum'] = dominant_family(row)
                selected.append(row)
                need -= 1
    return sorted(
        selected,
        key=lambda row: (dominant_family(row), row['label_channel'], -row['frequency'], row['query_id']),
    )


def build_clean_controls(
    clean: list[str], frequency: Counter, resources: Resources, limit: int, seed: int
) -> list[dict]:
    ordered = sorted(clean)
    random.Random(seed).shuffle(ordered)
    rows = []
    for key in ordered[:limit]:
        row = {
            'query_id': stable_id(key),
            'input': key,
            'expected': key,
            'error_type': 'clean',
            'error_types': ['clean'],
            'word_labels': [],
            'tokens': len(key.split()),
            'frequency': frequency[key],
            'label_channel': 'clean_control',
            'entity': (resources.entities.get(plain_key(key)) or (None,))[0],
            'stratum': 'clean_control',
            'review_status': 'weak_auto_labeled',
            'label_provenance': 'zero-click-x-osm',
            'split': 'zero-click-word-level',
        }
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def write_review_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'query_id', 'stratum', 'label_channel', 'error_type', 'frequency',
            'input', 'expected', 'wrong_word', 'correction', 'word_error_type',
            'margin', 'alternatives',
        ])
        for row in rows:
            for label in row['word_labels']:
                writer.writerow([
                    row['query_id'], row['stratum'], row['label_channel'],
                    row['error_type'], row['frequency'], row['input'], row['expected'],
                    label['surface'], label['expected'], label['error_type'],
                    label['margin'], '|'.join(label['alternatives']),
                ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zero-click', default='data/zero_click.csv')
    parser.add_argument(
        '--lexicon',
        default='artifacts/osm-model-production-v4-leakfree/statistics.sqlite3',
        help='frozen WebSpell statistics database holding the term lexicon',
    )
    parser.add_argument('--osm-root', default='data/osm/prepared-v4-leakfree')
    parser.add_argument('--output', default='benchmark/zero-click-word-level-5k')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--minimum-term-frequency', type=int, default=50)
    parser.add_argument('--minimum-margin', type=float, default=5.0)
    parser.add_argument('--entity-share', type=int, default=70, help='percent of each quota taken from entity-anchored rows')
    parser.add_argument('--max-characters', type=int, default=80)
    parser.add_argument('--max-tokens', type=int, default=12)
    parser.add_argument('--clean-controls', type=int, default=500)
    parser.add_argument('--review-sample', type=int, default=80)
    parser.add_argument('--target-rows', type=int, default=5000)
    arguments = parser.parse_args()
    arguments.quotas = dict(DEFAULT_QUOTAS)

    resources = Resources(Path(arguments.lexicon), Path(arguments.osm_root))
    analysed, clean, frequency = collect(Path(arguments.zero_click), resources, arguments)
    rows = sample(analysed, arguments)
    controls = build_clean_controls(
        clean, frequency, resources, arguments.clean_controls, arguments.seed
    )

    destination = Path(arguments.output)
    write_jsonl(destination / 'word-level-5k.jsonl', rows)
    write_jsonl(destination / 'clean-controls.jsonl', controls)
    write_review_csv(destination / 'word-level-5k.csv', rows)
    write_review_csv(
        destination / 'review-sample.csv',
        random.Random(arguments.seed).sample(
            rows, min(arguments.review_sample, len(rows))
        ),
    )

    manifest = {
        'schema_version': 1,
        'profile': 'zero-click-word-level-testset/v1',
        'purpose': 'held-out test set; never use for training',
        'source': str(arguments.zero_click),
        'lexicon': str(arguments.lexicon),
        'osm_root': str(arguments.osm_root),
        'lexicon_terms': len(resources.terms),
        'entity_surfaces': len(resources.entities),
        'channels': {
            'entity_anchor': 'whole query equals exactly one clean OSM term after case folding and diacritic stripping',
            'token_lexicon': 'unknown token resolves through plain-form, Telex, VNI or single substitution with a dominant candidate',
        },
        'thresholds': {
            'minimum_term_frequency': arguments.minimum_term_frequency,
            'minimum_margin': arguments.minimum_margin,
            'entity_share': arguments.entity_share,
            'max_characters': arguments.max_characters,
            'max_tokens': arguments.max_tokens,
        },
        'sampling': {
            'seed': arguments.seed,
            'quotas': arguments.quotas,
            'candidates_available': len(analysed),
            'candidate_pool_by_family': dict(
                Counter(dominant_family(row) for row in analysed.values())
            ),
            'candidate_pool_by_channel': dict(
                Counter(row['label_channel'] for row in analysed.values())
            ),
        },
        'rows': len(rows),
        'clean_control_rows': len(controls),
        'rows_by_channel': dict(Counter(row['label_channel'] for row in rows)),
        'query_families': dict(Counter(dominant_family(row) for row in rows)),
        'word_families': dict(
            Counter(label['error_type'] for row in rows for label in row['word_labels'])
        ),
        'wrong_words': sum(len(row['word_labels']) for row in rows),
        'privacy': {
            'coordinates_exported': False,
            'phone_email_filtered': True,
            'maximum_query_characters': arguments.max_characters,
        },
        'outputs': {
            'dataset': 'word-level-5k.jsonl',
            'clean_controls': 'clean-controls.jsonl',
            'review_csv': 'word-level-5k.csv',
            'review_sample': 'review-sample.csv',
        },
        'dataset_sha256': hashlib.sha256(
            (destination / 'word-level-5k.jsonl').read_bytes()
        ).hexdigest(),
    }
    (destination / 'word-level-5k.manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
