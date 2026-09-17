from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

from webspell.mining.triples import WordContext


class SQLiteContextStatistics:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    @lru_cache(maxsize=200_000)
    def term_id(self, term: str) -> int | None:
        row = self.connection.execute(
            "SELECT id FROM terms WHERE term = ?", (term,)
        ).fetchone()
        return int(row[0]) if row else None

    @lru_cache(maxsize=200_000)
    def term(self, term_id: int) -> str:
        row = self.connection.execute(
            "SELECT term FROM terms WHERE id = ?", (term_id,)
        ).fetchone()
        if row is None:
            raise KeyError(term_id)
        return str(row[0])

    def _context_ids(self, context: WordContext) -> tuple[int, int] | None:
        left = self.term_id(context[0])
        right = self.term_id(context[1])
        if left is None or right is None:
            return None
        return left, right

    @lru_cache(maxsize=1_000_000)
    def count(self, term: str, context: WordContext) -> int:
        term_id = self.term_id(term)
        context_ids = self._context_ids(context)
        if term_id is None or context_ids is None:
            return 0
        row = self.connection.execute(
            """
            SELECT count FROM contexts
            WHERE term_id = ? AND left_id = ? AND right_id = ?
            """,
            (term_id, *context_ids),
        ).fetchone()
        return int(row[0]) if row else 0

    @lru_cache(maxsize=500_000)
    def total(self, context: WordContext) -> int:
        context_ids = self._context_ids(context)
        if context_ids is None:
            return 0
        row = self.connection.execute(
            "SELECT count FROM context_totals WHERE left_id = ? AND right_id = ?",
            context_ids,
        ).fetchone()
        return int(row[0]) if row else 0

    def contexts_for(self, term: str) -> list[tuple[WordContext, int]]:
        term_id = self.term_id(term)
        if term_id is None:
            return []
        rows = self.connection.execute(
            """
            SELECT left_id, right_id, count FROM contexts
            WHERE term_id = ? ORDER BY left_id, right_id
            """,
            (term_id,),
        )
        return [
            ((self.term(left_id), self.term(right_id)), int(count))
            for left_id, right_id, count in rows
        ]

    def close(self) -> None:
        self.connection.close()
