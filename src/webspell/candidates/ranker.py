from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache

from webspell.config import CandidateConfig, LambdaConfig
from webspell.error_model import SubstringErrorModel
from webspell.language_model import BidirectionalLanguageModel
from webspell.types import Candidate, Context, LabeledToken
from webspell.vocabulary import Lexicon, damerau_levenshtein


@dataclass(frozen=True)
class LambdaGrid:
    values: dict[tuple[int, int], float]
    default: float = 1.0

    def for_context(self, context: Context, max_context: int) -> float:
        bucket = (min(len(context.left), max_context), min(len(context.right), max_context))
        return self.values.get(bucket, self.default)


class CandidateRanker:
    def __init__(
        self,
        lexicon: Lexicon,
        error_model: SubstringErrorModel,
        language_model: BidirectionalLanguageModel,
        config: CandidateConfig | None = None,
        variant_generator: Callable[[str], Iterable[str]] | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.error_model = error_model
        self.language_model = language_model
        self.config = config or CandidateConfig()
        self.variant_generator = variant_generator

    def close(self) -> None:
        self.clear_caches()
        self.error_model.word_logprob.cache_clear()
        self.error_model.transition_logprob.cache_clear()
        self.error_model._transition_denominator.cache_clear()
        if hasattr(self.lexicon, "close") and callable(self.lexicon.close):
            self.lexicon.close()
        if hasattr(self.language_model, "close") and callable(self.language_model.close):
            self.language_model.close()

    @lru_cache(maxsize=50_000)
    def _generate_cached(self, observed: str) -> tuple[Candidate, ...]:
        queries = {observed}
        if self.variant_generator is not None:
            queries.update(self.variant_generator(observed))
        matched_terms: dict[str, tuple[int, int]] = {}
        for query in sorted(queries):
            max_distance = self.config.max_edit_distance(len(query))
            for match in self.lexicon.close_terms(query, max_distance):
                raw_distance = damerau_levenshtein(observed, match.term)
                previous = matched_terms.get(match.term)
                if previous is None or raw_distance < previous[1]:
                    matched_terms[match.term] = (match.frequency, raw_distance)

        suggestions: list[Candidate] = []
        for term, (frequency, distance) in matched_terms.items():
            if term == observed:
                continue
            error_score = self.error_model.word_logprob(observed, term)
            suggestions.append(
                Candidate(term, distance, frequency, error_score)
            )
        suggestions.sort(
            key=lambda item: (
                -(item.error_log_score + math.log(item.term_frequency)),
                item.edit_distance,
                -item.term_frequency,
                item.term,
            )
        )
        suggestions = suggestions[: self.config.preselection_limit]
        original_frequency = max(1, self.lexicon.frequency(observed))
        suggestions.append(
            Candidate(
                observed,
                0,
                original_frequency,
                self.error_model.word_logprob(observed, observed),
            )
        )
        return tuple(suggestions)

    def generate(self, observed: str) -> list[Candidate]:
        return list(self._generate_cached(observed))

    @lru_cache(maxsize=50_000)
    def _rank_cached(
        self, observed: str, context: Context, lambda_value: float
    ) -> tuple[Candidate, ...]:
        ranked = []
        for candidate in self.generate(observed):
            lm_score = self.language_model.token_logscore(candidate.term, context)
            combined = candidate.error_log_score + lambda_value * lm_score
            ranked.append(
                Candidate(
                    term=candidate.term,
                    edit_distance=candidate.edit_distance,
                    term_frequency=candidate.term_frequency,
                    error_log_score=candidate.error_log_score,
                    lm_log_score=lm_score,
                    combined_log_score=combined,
                )
            )
        return tuple(
            sorted(
                ranked,
                key=lambda item: (
                    -float(item.combined_log_score),
                    item.edit_distance,
                    -item.term_frequency,
                    item.term,
                ),
            )
        )

    def rank(
        self, observed: str, context: Context, lambda_value: float
    ) -> list[Candidate]:
        return list(self._rank_cached(observed, context, lambda_value))

    def clear_caches(self) -> None:
        self._generate_cached.cache_clear()
        self._rank_cached.cache_clear()


def tune_lambdas(
    examples: Iterable[LabeledToken],
    ranker: CandidateRanker,
    config: LambdaConfig | None = None,
) -> LambdaGrid:
    settings = config or LambdaConfig()
    max_context = ranker.language_model.config.order - 1
    buckets: dict[
        tuple[int, int], list[tuple[str, list[Candidate]]]
    ] = defaultdict(list)
    for example in examples:
        bucket = (
            min(len(example.context.left), max_context),
            min(len(example.context.right), max_context),
        )
        candidates = ranker.rank(example.observed, example.context, 0.0)
        if any(candidate.term == example.intended for candidate in candidates):
            buckets[bucket].append((example.intended, candidates))

    values: dict[tuple[int, int], float] = {}
    steps = round((settings.maximum - settings.minimum) / settings.step)
    search = [settings.minimum + index * settings.step for index in range(steps + 1)]
    for bucket, prepared_records in buckets.items():
        best_lambda, best_mrr = settings.default, -1.0
        for lambda_value in search:
            reciprocal_ranks = []
            for intended, candidates in prepared_records:
                ranked = sorted(
                    candidates,
                    key=lambda item: (
                        -(item.error_log_score + lambda_value * float(item.lm_log_score)),
                        item.edit_distance,
                        -item.term_frequency,
                        item.term,
                    ),
                )
                rank = next(
                    (index for index, item in enumerate(ranked, 1) if item.term == intended),
                    None,
                )
                reciprocal_ranks.append(1.0 / int(rank))
            if not reciprocal_ranks:
                continue
            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
            if mrr > best_mrr or (mrr == best_mrr and lambda_value < best_lambda):
                best_lambda, best_mrr = lambda_value, mrr
        values[bucket] = best_lambda
    return LambdaGrid(values, settings.default)
