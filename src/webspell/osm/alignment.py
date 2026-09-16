from __future__ import annotations

import csv
import hashlib
import heapq
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from webspell.types import Context, ErrorTriple, LabeledToken
from webspell.vocabulary import damerau_levenshtein


TRAINING_ERROR_TYPE_WEIGHTS = {
    'keyboard_edit': 25,
    'missing_diacritics_full': 20,
    'missing_diacritics_partial': 15,
    'wrong_diacritic': 15,
    'telex_leak': 10,
    'vni_leak': 10,
    'combined': 5,
}


def error_type_bucket(error_type: str) -> str:
    if error_type.startswith('combined:'):
        return 'combined'
    return error_type if error_type in TRAINING_ERROR_TYPE_WEIGHTS else 'other'


def token_classifier_compatible(row: dict[str, str]) -> bool:
    """Whether a query pair can be represented as one correction per token."""
    error_type = row.get('error_type', '')
    if 'address_symbol' in error_type:
        return False
    noisy = row.get('noisy_query', '').strip().split()
    correct = row.get('correct_query', '').strip().split()
    return bool(noisy and correct and len(noisy) == len(correct))


def _sample_priority(seed: int, namespace: str, *parts: object) -> int:
    payload = '\x1f'.join((str(seed), namespace, *(str(part) for part in parts)))
    return int.from_bytes(
        hashlib.blake2b(payload.encode('utf-8'), digest_size=16).digest(), 'big'
    )


def _keep_bottom_k(
    heap: list[tuple[int, int, object]],
    priority: int,
    ordinal: int,
    value: object,
    limit: int,
) -> None:
    item = (-priority, -ordinal, value)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _ordered_sample(
    heap: Iterable[tuple[int, int, object]],
) -> list[tuple[int, int, object]]:
    return sorted((-priority, -ordinal, value) for priority, ordinal, value in heap)


def iter_sampled_query_rows(
    path: str | Path,
    limit: int = 0,
    seed: int = 2026,
    namespace: str = 'query',
) -> Iterator[dict[str, str]]:
    with Path(path).open(encoding='utf-8', newline='') as source:
        rows = csv.DictReader(source)
        if not limit:
            yield from rows
            return
        heap: list[tuple[int, int, object]] = []
        for ordinal, row in enumerate(rows):
            priority = _sample_priority(
                seed,
                namespace,
                row.get('entity_id', ''),
                row.get('noisy_query', ''),
                row.get('correct_query', ''),
                row.get('noise_source', ''),
                row.get('variant_id', ''),
            )
            _keep_bottom_k(heap, priority, ordinal, dict(row), limit)
    for _, _, row in _ordered_sample(heap):
        yield row  # type: ignore[misc]


def write_token_level_pairs(source_path: str | Path, output_path: str | Path) -> dict[str, int]:
    """Create a separate evaluation CSV containing only 1:1-token pairs."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError('output_path must differ from source_path')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = kept = dropped = 0
    with source_path.open(encoding='utf-8', newline='') as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f'missing CSV header: {source_path}')
        missing = {'noisy_query', 'correct_query'}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f'missing required columns: {sorted(missing)}')
        with output_path.open('w', encoding='utf-8', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                noisy_tokens = row['noisy_query'].strip().split()
                correct_tokens = row['correct_query'].strip().split()
                if noisy_tokens and correct_tokens and len(noisy_tokens) == len(correct_tokens):
                    writer.writerow(row)
                    kept += 1
                else:
                    dropped += 1
    return {'source_rows': total, 'kept_rows': kept, 'excluded_rows': dropped}


@dataclass
class AlignmentDiagnostics:
    total_query_pairs: int = 0
    identical_length_pairs: int = 0
    aligned_differing_pairs: int = 0
    unaligned_phrase_pairs: int = 0
    split_merge_pairs: int = 0
    misspelled_tokens_emitted: int = 0
    clean_tokens_emitted: int = 0
    unique_error_triples: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_query_pairs": self.total_query_pairs,
            "identical_length_pairs": self.identical_length_pairs,
            "aligned_differing_pairs": self.aligned_differing_pairs,
            "unaligned_phrase_pairs": self.unaligned_phrase_pairs,
            "split_merge_pairs": self.split_merge_pairs,
            "misspelled_tokens_emitted": self.misspelled_tokens_emitted,
            "clean_tokens_emitted": self.clean_tokens_emitted,
            "unique_error_triples": self.unique_error_triples,
        }


def _align_tokens(
    observed: Sequence[str], intended: Sequence[str]
) -> list[tuple[int | None, int | None]]:
    """Align two token sequences using Wagner-Fischer dynamic programming.

    Returns a list of (observed_index, intended_index) pairs, where None represents a gap.
    """
    n, m = len(observed), len(intended)
    if n == m:
        return [(i, i) for i in range(n)]

    # dp[i][j] = (min_cost, path)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * 1.0
    for j in range(1, m + 1):
        dp[0][j] = j * 1.0

    for i in range(1, n + 1):
        obs_tok = observed[i - 1]
        for j in range(1, m + 1):
            int_tok = intended[j - 1]
            if obs_tok == int_tok:
                sub_cost = 0.0
            else:
                dist = damerau_levenshtein(obs_tok, int_tok)
                max_len = max(len(obs_tok), len(int_tok), 1)
                sub_cost = min(1.0, dist / max_len) if dist <= 2 else 2.0

            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,  # substitution or match
                dp[i - 1][j] + 1.0,            # insertion in observed
                dp[i][j - 1] + 1.0,            # deletion from intended
            )

    # Backtrack to reconstruct alignment
    alignment: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            obs_tok = observed[i - 1]
            int_tok = intended[j - 1]
            dist = damerau_levenshtein(obs_tok, int_tok) if obs_tok != int_tok else 0
            max_len = max(len(obs_tok), len(int_tok), 1)
            sub_cost = (
                0.0
                if obs_tok == int_tok
                else (min(1.0, dist / max_len) if dist <= 2 else 2.0)
            )
            if abs(dp[i][j] - (dp[i - 1][j - 1] + sub_cost)) < 1e-6:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + 1.0)) < 1e-6:
            alignment.append((i - 1, None))
            i -= 1
        else:
            alignment.append((None, j - 1))
            j -= 1

    alignment.reverse()
    return alignment


class QueryAlignmentAdapter:
    """Adapts query/phrase pairs into token-level training examples and error triples.

    Handles exact-length pairs, differing-length pairs with sequence alignment,
    and identifies split/merge operations.
    """

    def __init__(self, order: int = 5) -> None:
        self.order = order

    def align_pair(
        self,
        noisy_query: str,
        correct_query: str,
        include_clean: bool = False,
        error_type: str = "unknown",
    ) -> tuple[list[LabeledToken], list[tuple[str, str]], bool]:
        """Align a single query pair.

        Returns:
            - list of LabeledToken instances
            - list of (intended, observed) error pairs
            - boolean indicating whether a split/merge was detected
        """
        observed = noisy_query.strip().split()
        intended = correct_query.strip().split()
        if not observed or not intended:
            return [], [], False

        compact_obs = "".join(observed)
        compact_int = "".join(intended)
        is_split_merge = (
            len(observed) != len(intended)
            and compact_obs == compact_int
        )

        tokens: list[LabeledToken] = []
        error_pairs: list[tuple[str, str]] = []
        width = self.order - 1

        if len(observed) == len(intended):
            for idx, (obs_tok, int_tok) in enumerate(zip(observed, intended, strict=True)):
                context = Context(
                    tuple(observed[max(0, idx - width) : idx]),
                    tuple(observed[idx + 1 : idx + 1 + width]),
                )
                if obs_tok != int_tok:
                    tokens.append(LabeledToken(obs_tok, int_tok, context, error_type))
                    error_pairs.append((int_tok, obs_tok))
                elif include_clean:
                    tokens.append(LabeledToken(int_tok, int_tok, context, "clean"))
            return tokens, error_pairs, is_split_merge

        # Differing lengths: run sequence alignment
        alignment = _align_tokens(observed, intended)
        for obs_idx, int_idx in alignment:
            if obs_idx is not None and int_idx is not None:
                obs_tok = observed[obs_idx]
                int_tok = intended[int_idx]
                context = Context(
                    tuple(observed[max(0, obs_idx - width) : obs_idx]),
                    tuple(observed[obs_idx + 1 : obs_idx + 1 + width]),
                )
                if obs_tok != int_tok:
                    tokens.append(LabeledToken(obs_tok, int_tok, context, error_type))
                    error_pairs.append((int_tok, obs_tok))
                elif include_clean:
                    tokens.append(LabeledToken(int_tok, int_tok, context, "clean"))

        return tokens, error_pairs, is_split_merge

    def iter_examples(
        self,
        path: str | Path,
        limit: int = 0,
        include_clean: bool = False,
        clean_ratio: float = 1.0,
        sampling_seed: int = 2026,
        compatible_only: bool = False,
        error_type_weights: dict[str, int] | None = None,
    ) -> Iterator[LabeledToken]:
        """Stream deterministic, optionally stratified token examples."""
        if limit:
            clean_target = math.floor(limit * clean_ratio / (1.0 + clean_ratio)) if include_clean and clean_ratio > 0 else 0
            error_target = limit - clean_target
            weights = dict(error_type_weights or TRAINING_ERROR_TYPE_WEIGHTS)
            active = [(name, weight) for name, weight in weights.items() if weight > 0]
            total_weight = sum(weight for _, weight in active)
            targets = {name: math.floor(error_target * weight / total_weight) for name, weight in active}
            for name, _ in active[:error_target - sum(targets.values())]:
                targets[name] += 1
            clean_heap: list[tuple[int, int, object]] = []
            error_heap: list[tuple[int, int, object]] = []
            category_heaps: dict[str, list[tuple[int, int, object]]] = {name: [] for name in targets}
            ordinal = 0
            for row in iter_sampled_query_rows(path):
                if compatible_only and not token_classifier_compatible(row):
                    continue
                row_error_type = row.get('error_type') or row.get('noise_source', 'other')
                tokens, _, _ = self.align_pair(
                    row['noisy_query'], row['correct_query'], include_clean=include_clean,
                    error_type=row_error_type,
                )
                for token_index, item in enumerate(tokens):
                    priority = _sample_priority(
                        sampling_seed,
                        'training-example',
                        row.get('entity_id', ''),
                        row['noisy_query'],
                        row['correct_query'],
                        row.get('noise_source', ''),
                        row.get('variant_id', ''),
                        token_index,
                        item.observed,
                        item.intended,
                    )
                    if item.observed == item.intended:
                        if clean_target:
                            _keep_bottom_k(clean_heap, priority, ordinal, item, clean_target)
                    else:
                        if error_target:
                            _keep_bottom_k(error_heap, priority, ordinal, item, error_target)
                        bucket = error_type_bucket(row_error_type)
                        if targets.get(bucket, 0):
                            _keep_bottom_k(category_heaps[bucket], priority, ordinal, item, targets[bucket])
                    ordinal += 1

            selected_errors = [item for heap in category_heaps.values() for item in _ordered_sample(heap)]
            selected_keys = {(priority, ordinal) for priority, ordinal, _ in selected_errors}
            for item in _ordered_sample(error_heap):
                if len(selected_errors) >= error_target:
                    break
                if (item[0], item[1]) not in selected_keys:
                    selected_errors.append(item)
                    selected_keys.add((item[0], item[1]))
            selected = selected_errors + _ordered_sample(clean_heap)
            for _, _, item in sorted(selected):
                yield item  # type: ignore[misc]
            return

        emitted = 0
        emitted_clean = 0
        emitted_misspelled = 0

        with Path(path).open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                if compatible_only and not token_classifier_compatible(row):
                    continue
                row_error_type = row.get('error_type') or row.get('noise_source', 'other')
                tokens, _, _ = self.align_pair(
                    row["noisy_query"], row["correct_query"], include_clean=include_clean,
                    error_type=row_error_type,
                )
                for item in tokens:
                    is_clean = item.observed == item.intended
                    if is_clean:
                        # Regulate clean-to-misspelled ratio
                        if emitted_clean >= emitted_misspelled * clean_ratio and emitted_misspelled > 0:
                            continue
                        emitted_clean += 1
                    else:
                        emitted_misspelled += 1

                    yield item
                    emitted += 1
                    if limit and emitted >= limit:
                        return

    def mine_error_triples(
        self,
        path: str | Path,
        max_examples: int = 0,
        sampling_seed: int = 2026,
    ) -> tuple[list[ErrorTriple], AlignmentDiagnostics]:
        """Mine and aggregate ErrorTriple instances from a noisy_pairs.csv file.

        Deduplicates (intended, observed) pairs and counts occurrences as required by Section 5.4.
        """
        counts: Counter[tuple[str, str]] = Counter()
        diagnostics = AlignmentDiagnostics()
        sampled: list[tuple[int, int, object]] = []
        occurrence = 0

        with Path(path).open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                diagnostics.total_query_pairs += 1
                observed = row["noisy_query"].strip().split()
                intended = row["correct_query"].strip().split()

                if len(observed) == len(intended):
                    diagnostics.identical_length_pairs += 1
                elif "".join(observed) == "".join(intended):
                    diagnostics.split_merge_pairs += 1
                else:
                    diagnostics.aligned_differing_pairs += 1

                tokens, error_pairs, _ = self.align_pair(
                    row["noisy_query"], row["correct_query"], include_clean=False
                )
                if not tokens and len(observed) != len(intended):
                    diagnostics.unaligned_phrase_pairs += 1

                for token_index, (int_tok, obs_tok) in enumerate(error_pairs):
                    if max_examples:
                        priority = _sample_priority(
                            sampling_seed,
                            'error-example',
                            row.get('entity_id', ''),
                            row['noisy_query'],
                            row['correct_query'],
                            token_index,
                            int_tok,
                            obs_tok,
                        )
                        _keep_bottom_k(
                            sampled,
                            priority,
                            occurrence,
                            (int_tok, obs_tok),
                            max_examples,
                        )
                    else:
                        counts[(int_tok, obs_tok)] += 1
                    occurrence += 1

                if not max_examples:
                    diagnostics.misspelled_tokens_emitted += len(tokens)

        if max_examples:
            selected = _ordered_sample(sampled)
            counts.update(pair for _, _, pair in selected)  # type: ignore[misc]
            diagnostics.misspelled_tokens_emitted = len(selected)

        triples = [
            ErrorTriple(intended=int_tok, observed=obs_tok, count=count)
            for (int_tok, obs_tok), count in sorted(counts.items())
        ]
        diagnostics.unique_error_triples = len(triples)
        return triples, diagnostics
