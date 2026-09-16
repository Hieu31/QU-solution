from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import groupby

from webspell.config import CandidateConfig
from webspell.types import ErrorTriple
from webspell.vocabulary import Lexicon

WordContext = tuple[str, str]

@dataclass(frozen=True)
class ClosePair:
    intended: str
    observed: str
    distance: int
    intended_frequency: int
    observed_frequency: int

class ContextStatistics:
    """Counts one-word-left/one-word-right contexts used by the paper."""

    def __init__(self, boundary: str = "<boundary>") -> None:
        self.boundary = boundary
        self.term_context: Counter[tuple[str, WordContext]] = Counter()
        self.context_total: Counter[WordContext] = Counter()

    @classmethod
    def from_sentences(
        cls, sentences: Iterable[Sequence[str]], boundary: str = "<boundary>"
    ) -> "ContextStatistics":
        statistics = cls(boundary)
        for sentence in sentences:
            words = list(sentence)
            for index, term in enumerate(words):
                left = words[index - 1] if index else boundary
                right = words[index + 1] if index + 1 < len(words) else boundary
                context = (left, right)
                statistics.term_context[(term, context)] += 1
                statistics.context_total[context] += 1
        return statistics

    def count(self, term: str, context: WordContext) -> int:
        return self.term_context.get((term, context), 0)

    def total(self, context: WordContext) -> int:
        return self.context_total.get(context, 0)

    def contexts_for(self, term: str) -> list[tuple[WordContext, int]]:
        found = [
            (context, count)
            for (candidate, context), count in self.term_context.items()
            if candidate == term
        ]
        return sorted(found, key=lambda item: item[0])


def mine_close_pairs(
    lexicon: Lexicon,
    frequency_ratio: float = 10.0,
    candidate_config: CandidateConfig | None = None,
) -> list[ClosePair]:
    return list(iter_close_pairs(lexicon, frequency_ratio, candidate_config))


def iter_close_pairs(
    lexicon: Lexicon,
    frequency_ratio: float = 10.0,
    candidate_config: CandidateConfig | None = None,
) -> Iterable[ClosePair]:
    config = candidate_config or CandidateConfig()
    for observed in lexicon.terms():
        observed_frequency = lexicon.frequency(observed)
        distance_limit = config.max_edit_distance(len(observed))
        for match in lexicon.close_terms(observed, distance_limit):
            if match.term == observed:
                continue
            if match.frequency < frequency_ratio * observed_frequency:
                continue
            yield ClosePair(
                intended=match.term,
                observed=observed,
                distance=match.distance,
                intended_frequency=match.frequency,
                observed_frequency=observed_frequency,
            )


def mine_error_triples(
    pairs: Iterable[ClosePair],
    contexts: ContextStatistics,
    minimum_context_frequency: int = 10,
) -> list[ErrorTriple]:
    return list(iter_error_triples(pairs, contexts, minimum_context_frequency))


def iter_error_triples(
    pairs: Iterable[ClosePair],
    contexts: ContextStatistics,
    minimum_context_frequency: int = 10,
) -> Iterable[ErrorTriple]:
    for observed, group in groupby(pairs, key=lambda pair: pair.observed):
        observed_pairs = list(group)
        candidates = {observed, *(pair.intended for pair in observed_pairs)}
        metadata = {pair.intended: pair for pair in observed_pairs}
        aggregate: Counter[str] = Counter()
        for context, observed_count in contexts.contexts_for(observed):
            if contexts.total(context) < minimum_context_frequency:
                continue

            contextual_counts = {
                candidate: contexts.count(candidate, context)
                for candidate in candidates
            }
            maximum = max(contextual_counts.values())
            winners = [
                candidate
                for candidate, count in contextual_counts.items()
                if count == maximum
            ]
            # The paper assigns the candidate that occurs most in this exact
            # context. A tie has no contextual winner; global term frequency
            # must not silently turn it into an error pair.
            if len(winners) != 1:
                continue
            intended = winners[0]
            if intended != observed and maximum > 0:
                aggregate[intended] += observed_count
        for intended, count in sorted(aggregate.items()):
            yield ErrorTriple(intended, observed, count)
