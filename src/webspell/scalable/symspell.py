from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from webspell.scalable.statistics import SQLiteCorpusStatistics, chunks, connect
from webspell.vocabulary import TermMatch, damerau_levenshtein


def deletion_variants(term: str, max_distance: int) -> set[str]:
    variants = {term}
    frontier = {term}
    for _ in range(max_distance):
        next_frontier = {
            candidate[:index] + candidate[index + 1 :]
            for candidate in frontier
            for index in range(len(candidate))
        }
        next_frontier -= variants
        variants.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return variants


class SQLiteSymSpellIndex:
    """Disk-backed delete index with exact Damerau distance verification."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Read-only model bundles are cached across Streamlit rerun threads.
        # Calls are serialized by the serving layer, so allow the connection
        # to follow the cached model between those threads.
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        metadata = SQLiteCorpusStatistics(self.path).metadata()
        if not metadata.get("symspell_complete"):
            raise ValueError("SymSpell index is missing or incomplete")
        self.max_distance = int(metadata["symspell_max_distance"])
        self.minimum_frequency = int(metadata["symspell_minimum_frequency"])
        self._terms_by_id = {
            int(term_id): (str(term), int(frequency))
            for term_id, term, frequency in self.connection.execute(
                'SELECT id, term, frequency FROM terms WHERE frequency >= ?',
                (self.minimum_frequency,),
            )
        }
        self._frequencies = {
            term: frequency for term, frequency in self._terms_by_id.values()
        }

    @classmethod
    def build(
        cls,
        path: str | Path,
        max_distance: int = 2,
        minimum_frequency: int = 2,
        batch_rows: int = 50_000,
    ) -> "SQLiteSymSpellIndex":
        database = Path(path)
        connection = connect(database)
        try:
            connection.execute(
                """
                CREATE TABLE symspell_deletes (
                    delete_key TEXT NOT NULL,
                    term_id INTEGER NOT NULL,
                    PRIMARY KEY (delete_key, term_id)
                ) WITHOUT ROWID
                """
            )
            statement = "INSERT OR IGNORE INTO symspell_deletes(delete_key, term_id) VALUES (?, ?)"
            pending: list[tuple[str, int]] = []
            cursor = connection.execute(
                "SELECT id, term FROM terms WHERE frequency >= ? ORDER BY id",
                (minimum_frequency,),
            )
            for term_id, term in cursor:
                pending.extend(
                    (variant, int(term_id))
                    for variant in deletion_variants(term, max_distance)
                )
                if len(pending) >= batch_rows:
                    connection.executemany(statement, pending)
                    connection.commit()
                    pending = []
            if pending:
                connection.executemany(statement, pending)
            SQLiteCorpusStatistics.write_metadata(
                connection,
                {
                    "symspell_complete": True,
                    "symspell_max_distance": max_distance,
                    "symspell_minimum_frequency": minimum_frequency,
                },
            )
            connection.commit()
        finally:
            connection.close()
        return cls(database)

    def __len__(self) -> int:
        return len(self._terms_by_id)

    def frequency(self, term: str) -> int:
        return self._frequencies.get(term, 0)

    def contains(self, term: str) -> bool:
        return self.frequency(term) > 0

    @lru_cache(maxsize=100_000)
    def _close_terms(self, term: str, max_distance: int) -> tuple[TermMatch, ...]:
        distance = min(max_distance, self.max_distance)
        variants = sorted(deletion_variants(term, distance))
        candidate_ids: set[int] = set()
        for chunk in chunks(variants):
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT DISTINCT term_id FROM symspell_deletes "
                f"WHERE delete_key IN ({placeholders})"
            )
            candidate_ids.update(
                row[0] for row in self.connection.execute(query, tuple(chunk))
            )
        matches: list[TermMatch] = []
        for term_id in candidate_ids:
            candidate, frequency = self._terms_by_id[term_id]
            actual = damerau_levenshtein(term, candidate, distance)
            if actual <= distance:
                matches.append(TermMatch(candidate, frequency, actual))
        return tuple(
            sorted(matches, key=lambda item: (item.distance, -item.frequency, item.term))
        )

    def close_terms(self, term: str, max_distance: int) -> list[TermMatch]:
        return list(self._close_terms(term, max_distance))

    def terms(self) -> Iterator[str]:
        yield from sorted(self._frequencies)

    def close(self) -> None:
        self.connection.close()
