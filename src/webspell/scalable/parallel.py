from __future__ import annotations

import atexit
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor

from webspell.candidates import CandidateRanker, LambdaGrid
from webspell.confidence import TrainingRow
from webspell.config import CandidateConfig, ErrorModelConfig
from webspell.error_model import SubstringErrorModel
from webspell.pipeline import iter_training_rows
from webspell.scalable.language_model import SQLiteBidirectionalLanguageModel
from webspell.scalable.symspell import SQLiteSymSpellIndex
from webspell.text import VietnameseVariantGenerator
from webspell.types import LabeledToken


_worker_ranker: CandidateRanker | None = None
_worker_lambdas: LambdaGrid | None = None


def _close_worker() -> None:
    global _worker_ranker
    if _worker_ranker is not None:
        _worker_ranker.close()
        _worker_ranker = None


def _initialize_worker(
    database: str,
    error_config: dict[str, object],
    transitions: dict[str, dict[str, float]],
    targets: set[str],
    candidate_config: dict[str, object],
    lambda_values: dict[tuple[int, int], float],
    lambda_default: float,
    variant_generator: str,
) -> None:
    global _worker_ranker, _worker_lambdas
    error_model = SubstringErrorModel(ErrorModelConfig(**error_config))
    error_model._counts = defaultdict(Counter)
    for source, target_counts in transitions.items():
        error_model._counts[source] = Counter(target_counts)
    error_model._targets = set(targets)
    index = SQLiteSymSpellIndex(database)
    language_model = SQLiteBidirectionalLanguageModel(database)
    _worker_ranker = CandidateRanker(
        index,
        error_model,
        language_model,
        CandidateConfig(**candidate_config),
        VietnameseVariantGenerator() if variant_generator == 'vietnamese' else None,
    )
    _worker_lambdas = LambdaGrid(dict(lambda_values), lambda_default)
    atexit.register(_close_worker)


def _build_row_batch(examples: tuple[LabeledToken, ...]) -> list[TrainingRow]:
    if _worker_ranker is None or _worker_lambdas is None:
        raise RuntimeError('parallel training worker was not initialized')
    return list(iter_training_rows(examples, _worker_ranker, _worker_lambdas))


def _batches(
    examples: Sequence[LabeledToken], batch_size: int
) -> Iterator[tuple[LabeledToken, ...]]:
    for start in range(0, len(examples), batch_size):
        yield tuple(examples[start : start + batch_size])


def _flatten(row_batches: Iterable[list[TrainingRow]]) -> list[TrainingRow]:
    return [row for batch in row_batches for row in batch]


def build_training_rows_parallel(
    training_examples: Sequence[LabeledToken],
    development_examples: Sequence[LabeledToken],
    ranker: CandidateRanker,
    lambdas: LambdaGrid,
    database: str,
    workers: int,
    batch_size: int,
    variant_generator: str = 'vietnamese',
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    if workers < 2:
        raise ValueError('parallel feature generation requires at least two workers')
    if batch_size < 1:
        raise ValueError('worker batch size must be positive')
    error_model = ranker.error_model
    initializer_args = (
        database,
        {
            'max_source_substring_length': error_model.config.max_source_substring_length,
            'max_target_substring_length': error_model.config.max_target_substring_length,
            'smoothing_alpha': error_model.config.smoothing_alpha,
            'identity_prior': error_model.config.identity_prior,
            'alignment_iterations': error_model.config.alignment_iterations,
        },
        error_model.transitions(),
        set(error_model._targets),
        {'preselection_limit': ranker.config.preselection_limit},
        dict(lambdas.values),
        lambdas.default,
        variant_generator,
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=initializer_args,
    ) as executor:
        training_rows = _flatten(
            executor.map(_build_row_batch, _batches(training_examples, batch_size))
        )
        development_rows = _flatten(
            executor.map(_build_row_batch, _batches(development_examples, batch_size))
        )
    return training_rows, development_rows
