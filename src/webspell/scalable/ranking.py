from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable

from webspell.candidates import CandidateRanker, LambdaGrid
from webspell.config import LambdaConfig
from webspell.types import LabeledToken


ExamplesFactory = Callable[[], Iterable[LabeledToken]]


def tune_lambdas_streaming(
    examples_factory: ExamplesFactory,
    ranker: CandidateRanker,
    config: LambdaConfig | None = None,
) -> LambdaGrid:
    """One-pass, bounded-memory MRR optimization over a fixed lambda grid."""
    settings = config or LambdaConfig()
    steps = round((settings.maximum - settings.minimum) / settings.step)
    search = [settings.minimum + index * settings.step for index in range(steps + 1)]
    max_context = ranker.language_model.config.order - 1
    reciprocal_rank_sums: dict[tuple[int, int], list[float]] = defaultdict(
        lambda: [0.0] * len(search)
    )
    counts: dict[tuple[int, int], int] = defaultdict(int)

    for example in examples_factory():
        candidates = ranker.rank(example.observed, example.context, 0.0)
        if not any(candidate.term == example.intended for candidate in candidates):
            continue
        bucket = (
            min(len(example.context.left), max_context),
            min(len(example.context.right), max_context),
        )
        counts[bucket] += 1
        intended_candidate = next(
            candidate for candidate in candidates if candidate.term == example.intended
        )
        for lambda_index, lambda_value in enumerate(search):
            intended_key = (
                -(
                    intended_candidate.error_log_score
                    + lambda_value * float(intended_candidate.lm_log_score)
                ),
                intended_candidate.edit_distance,
                -intended_candidate.term_frequency,
                intended_candidate.term,
            )
            rank = 1 + sum(
                (
                    -(
                        candidate.error_log_score
                        + lambda_value * float(candidate.lm_log_score)
                    ),
                    candidate.edit_distance,
                    -candidate.term_frequency,
                    candidate.term,
                ) < intended_key
                for candidate in candidates
            )
            reciprocal_rank_sums[bucket][lambda_index] += 1.0 / rank

    values: dict[tuple[int, int], float] = {}
    for bucket, sums in reciprocal_rank_sums.items():
        best_index = max(range(len(search)), key=lambda index: (sums[index], -index))
        values[bucket] = search[best_index]
    return LambdaGrid(values, settings.default)
