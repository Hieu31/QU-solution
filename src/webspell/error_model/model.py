from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache

from webspell.config import ErrorModelConfig
from webspell.types import ErrorTriple


Segment = tuple[str, str]


@dataclass(frozen=True)
class Alignment:
    segments: tuple[Segment, ...]
    score: float


class SubstringAligner:
    def __init__(self, max_source: int = 2, max_target: int = 2) -> None:
        self.max_source = max_source
        self.max_target = max_target

    def best_alignment(
        self,
        intended: str,
        observed: str,
        segment_score: Callable[[str, str], float],
    ) -> Alignment:
        best: dict[tuple[int, int], tuple[float, tuple[Segment, ...]]] = {
            (0, 0): (0.0, ())
        }
        for i in range(len(intended) + 1):
            for j in range(len(observed) + 1):
                current = best.get((i, j))
                if current is None:
                    continue
                current_score, path = current
                for source_len in range(self.max_source + 1):
                    for target_len in range(self.max_target + 1):
                        if source_len == target_len == 0:
                            continue
                        next_i, next_j = i + source_len, j + target_len
                        if next_i > len(intended) or next_j > len(observed):
                            continue
                        segment = (intended[i:next_i], observed[j:next_j])
                        new_path = path + (segment,)
                        new_score = current_score + segment_score(*segment)
                        previous = best.get((next_i, next_j))
                        if previous is None or self._better(new_score, new_path, previous):
                            best[(next_i, next_j)] = (new_score, new_path)
        score, segments = best[(len(intended), len(observed))]
        return Alignment(segments, score)

    def best_score(
        self,
        intended: str,
        observed: str,
        segment_score: Callable[[str, str], float],
    ) -> float:
        rows, columns = len(intended) + 1, len(observed) + 1
        unreachable = float('-inf')
        best = [[unreachable] * columns for _ in range(rows)]
        best[0][0] = 0.0
        for i in range(rows):
            for j in range(columns):
                current = best[i][j]
                if current == unreachable:
                    continue
                max_source = min(self.max_source, len(intended) - i)
                max_target = min(self.max_target, len(observed) - j)
                for source_len in range(max_source + 1):
                    for target_len in range(max_target + 1):
                        if source_len == target_len == 0:
                            continue
                        next_i, next_j = i + source_len, j + target_len
                        score = current + segment_score(
                            intended[i:next_i], observed[j:next_j]
                        )
                        if score > best[next_i][next_j]:
                            best[next_i][next_j] = score
        return best[-1][-1]

    @staticmethod
    def _better(
        score: float,
        path: tuple[Segment, ...],
        previous: tuple[float, tuple[Segment, ...]],
    ) -> bool:
        old_score, old_path = previous
        if score != old_score:
            return score > old_score
        if len(path) != len(old_path):
            return len(path) < len(old_path)
        return path < old_path


def _initial_score(source: str, target: str) -> float:
    if source == target:
        return 0.0
    if len(source) == len(target) == 2 and source == target[::-1]:
        return -1.0
    return -float(max(len(source), len(target), 1))


class SubstringErrorModel:
    def __init__(self, config: ErrorModelConfig | None = None) -> None:
        self.config = config or ErrorModelConfig()
        self.aligner = SubstringAligner(
            self.config.max_source_substring_length,
            self.config.max_target_substring_length,
        )
        self._counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._targets: set[str] = {""}

    @classmethod
    def fit(
        cls,
        triples: Iterable[ErrorTriple],
        config: ErrorModelConfig | None = None,
    ) -> "SubstringErrorModel":
        model = cls(config)
        positive = (triple for triple in triples if triple.count > 0)
        records: Iterable[ErrorTriple]
        if model.config.alignment_iterations > 1:
            records = list(positive)
        else:
            records = positive
        for triple in records:
            alignment = model.aligner.best_alignment(
                triple.intended, triple.observed, _initial_score
            )
            for source, target in alignment.segments:
                model._counts[source][target] += triple.count
                model._targets.add(target)
            for size in (1, 2):
                for start in range(len(triple.intended) - size + 1):
                    identity = triple.intended[start : start + size]
                    model._counts[identity][identity] += model.config.identity_prior
                    model._targets.add(identity)

        for _ in range(max(0, model.config.alignment_iterations - 1)):
            updated: dict[str, Counter[str]] = defaultdict(Counter)
            for triple in records:
                alignment = model.aligner.best_alignment(
                    triple.intended, triple.observed, model.transition_logprob
                )
                for source, target in alignment.segments:
                    updated[source][target] += triple.count
            model._counts = updated
            model.transition_logprob.cache_clear()
            model._transition_denominator.cache_clear()
        return model

    @lru_cache(maxsize=100_000)
    def _transition_denominator(self, source: str) -> float:
        counts = self._counts.get(source)
        total = sum(counts.values()) if counts is not None else 0.0
        support = max(1, len(self._targets) + 1)
        return total + self.config.smoothing_alpha * support

    @lru_cache(maxsize=300_000)
    def transition_logprob(self, source: str, target: str) -> float:
        alpha = self.config.smoothing_alpha
        counts = self._counts.get(source)
        count = counts.get(target, 0.0) if counts is not None else 0.0
        return math.log((count + alpha) / self._transition_denominator(source))

    @lru_cache(maxsize=500_000)
    def word_logprob(self, observed: str, intended: str) -> float:
        return self.aligner.best_score(
            intended, observed, self.transition_logprob
        )

    def transitions(self) -> dict[str, dict[str, float]]:
        return {
            source: {target: float(count) for target, count in sorted(targets.items())}
            for source, targets in sorted(self._counts.items())
        }
