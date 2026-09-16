from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from webspell.config import LanguageModelConfig
from webspell.scalable.statistics import SQLiteCorpusStatistics, encode_ids
from webspell.types import Context


class SQLiteNGramModel:
    def __init__(self, path: str | Path, direction: int) -> None:
        self.path = Path(path)
        self.direction = direction
        self.connection = sqlite3.connect(self.path)
        self.term_ids = {
            str(term): int(term_id)
            for term, term_id in self.connection.execute('SELECT term, id FROM terms')
        }
        metadata = SQLiteCorpusStatistics(self.path).metadata()
        if not metadata.get("counting_complete"):
            raise ValueError("corpus statistics artifact is incomplete")
        self.config = LanguageModelConfig(order=int(metadata["order"]))
        key = "forward_unigrams" if direction == 0 else "backward_unigrams"
        self.total_unigrams = int(metadata[key])

    def term_id(self, term: str) -> int | None:
        return self.term_ids.get(term)

    @lru_cache(maxsize=1_000_000)
    def count(self, ids: tuple[int, ...]) -> int:
        row = self.connection.execute(
            "SELECT count FROM ngrams WHERE direction = ? AND n = ? AND key = ?",
            (self.direction, len(ids), encode_ids(ids)),
        ).fetchone()
        return int(row[0]) if row else 0

    def logscore(self, token: str, history: Sequence[str]) -> float:
        token_id = self.term_id(token)
        if token_id is None:
            return math.log(self.config.unknown_probability)
        history_ids = [self.term_id(term) for term in history[-(self.config.order - 1) :]]
        while history_ids and history_ids[0] is None:
            history_ids.pop(0)
        if any(term_id is None for term_id in history_ids):
            history_ids = []
        bounded = tuple(int(term_id) for term_id in history_ids)
        backoffs = 0
        while bounded:
            numerator = self.count(bounded + (token_id,))
            denominator = self.count(bounded)
            if numerator and denominator:
                return math.log(numerator / denominator) + backoffs * math.log(
                    self.config.backoff_alpha
                )
            bounded = bounded[1:]
            backoffs += 1
        unigram = self.count((token_id,))
        probability = (
            unigram / self.total_unigrams
            if unigram and self.total_unigrams
            else self.config.unknown_probability
        )
        return math.log(probability) + backoffs * math.log(self.config.backoff_alpha)

    def close(self) -> None:
        self.connection.close()


class SQLiteBidirectionalLanguageModel:
    def __init__(self, path: str | Path) -> None:
        self.forward = SQLiteNGramModel(path, 0)
        self.backward = SQLiteNGramModel(path, 1)
        self.config = self.forward.config

    @lru_cache(maxsize=250_000)
    def token_logscore(self, candidate: str, context: Context) -> float:
        return self.forward.logscore(candidate, context.left) + self.backward.logscore(
            candidate, tuple(reversed(context.right))
        )

    def close(self) -> None:
        self.token_logscore.cache_clear()
        self.forward.close()
        self.backward.close()
