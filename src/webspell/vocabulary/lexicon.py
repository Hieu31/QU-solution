from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol


def damerau_levenshtein(
    source: str, target: str, max_distance: int | None = None
) -> int:
    """Optimal-string-alignment Damerau-Levenshtein distance."""
    if max_distance is not None and abs(len(source) - len(target)) > max_distance:
        return max_distance + 1
    columns = len(target) + 1
    infinity = (max_distance + 1) if max_distance is not None else len(source) + len(target)
    previous_previous = [infinity] * columns
    previous = list(range(columns))
    for i in range(1, len(source) + 1):
        current = [infinity] * columns
        current[0] = i
        start = max(1, i - max_distance) if max_distance is not None else 1
        stop = min(len(target), i + max_distance) if max_distance is not None else len(target)
        for j in range(start, stop + 1):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            if (
                i > 1 and j > 1
                and source[i - 1] == target[j - 2]
                and source[i - 2] == target[j - 1]
            ):
                current[j] = min(current[j], previous_previous[j - 2] + 1)
        previous_previous, previous = previous, current
    result = previous[-1]
    if max_distance is not None and result > max_distance:
        return max_distance + 1
    return result


@dataclass(frozen=True)
class TermMatch:
    term: str
    frequency: int
    distance: int


class Lexicon(Protocol):
    def contains(self, term: str) -> bool: ...
    def frequency(self, term: str) -> int: ...
    def close_terms(self, term: str, max_distance: int) -> list[TermMatch]: ...
    def terms(self) -> Iterable[str]: ...


class TermLexicon:
    """In-memory index; production can replace it with a trie or FST."""

    def __init__(self, frequencies: Mapping[str, int]) -> None:
        self._frequencies = {
            term: int(frequency)
            for term, frequency in frequencies.items()
            if frequency > 0
        }

    @classmethod
    def from_sentences(cls, sentences: Iterable[Iterable[str]]) -> "TermLexicon":
        counts: Counter[str] = Counter()
        for sentence in sentences:
            counts.update(sentence)
        return cls(counts)

    def __len__(self) -> int:
        return len(self._frequencies)

    def contains(self, term: str) -> bool:
        return term in self._frequencies

    def frequency(self, term: str) -> int:
        return self._frequencies.get(term, 0)

    def close_terms(self, term: str, max_distance: int) -> list[TermMatch]:
        matches = []
        for candidate, frequency in self._frequencies.items():
            distance = damerau_levenshtein(term, candidate)
            if distance <= max_distance:
                matches.append(TermMatch(candidate, frequency, distance))
        return sorted(matches, key=lambda item: (item.distance, -item.frequency, item.term))

    def terms(self) -> tuple[str, ...]:
        return tuple(sorted(self._frequencies))

    def frequencies(self) -> dict[str, int]:
        return dict(self._frequencies)
