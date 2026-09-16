from __future__ import annotations

import json
import sqlite3
import struct
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path


START_TOKEN = "<s>"
END_TOKEN = "</s>"
BOUNDARY_TOKEN = "<boundary>"


def encode_ids(ids: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(ids)}I", *ids)


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-262144")
    return connection


def chunks(values: Sequence[str], size: int = 800) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class SQLiteCorpusStatistics:
    """Disk-backed vocabulary, n-gram, and context count artifact."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE term_counts (term TEXT PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID;
            CREATE TABLE terms (
                id INTEGER PRIMARY KEY,
                term TEXT NOT NULL UNIQUE,
                frequency INTEGER NOT NULL
            );
            CREATE TABLE ngrams (
                direction INTEGER NOT NULL,
                n INTEGER NOT NULL,
                key BLOB NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (direction, n, key)
            ) WITHOUT ROWID;
            CREATE TABLE contexts (
                term_id INTEGER NOT NULL,
                left_id INTEGER NOT NULL,
                right_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (term_id, left_id, right_id)
            ) WITHOUT ROWID;
            CREATE TABLE context_totals (
                left_id INTEGER NOT NULL,
                right_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (left_id, right_id)
            ) WITHOUT ROWID;
            """
        )

    @staticmethod
    def sentence_batches(
        sentences: Iterable[Sequence[str]], batch_tokens: int
    ) -> Iterator[list[tuple[str, ...]]]:
        batch: list[tuple[str, ...]] = []
        tokens = 0
        for sentence in sentences:
            materialized = tuple(sentence)
            if not materialized:
                continue
            batch.append(materialized)
            tokens += len(materialized)
            if tokens >= batch_tokens:
                yield batch
                batch, tokens = [], 0
        if batch:
            yield batch

    @staticmethod
    def write_metadata(connection: sqlite3.Connection, values: dict[str, object]) -> None:
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in values.items()],
        )

    def metadata(self) -> dict[str, object]:
        connection = connect(self.path)
        try:
            return {
                key: json.loads(value)
                for key, value in connection.execute("SELECT key, value FROM metadata")
            }
        finally:
            connection.close()

    @classmethod
    def build_vocabulary(
        cls,
        sentences: Iterable[Sequence[str]],
        path: str | Path,
        order: int = 5,
        batch_tokens: int = 50_000,
    ) -> "SQLiteCorpusStatistics":
        artifact = cls(path)
        if artifact.path.exists():
            raise FileExistsError(f"statistics database already exists: {artifact.path}")
        artifact.path.parent.mkdir(parents=True, exist_ok=True)
        connection = connect(artifact.path)
        sentence_count = token_count = 0
        try:
            cls.create_schema(connection)
            statement = """
                INSERT INTO term_counts(term, count) VALUES (?, ?)
                ON CONFLICT(term) DO UPDATE SET count = count + excluded.count
            """
            for batch in cls.sentence_batches(sentences, batch_tokens):
                counts: Counter[str] = Counter()
                for sentence in batch:
                    counts.update(sentence)
                    sentence_count += 1
                    token_count += len(sentence)
                connection.executemany(statement, counts.items())
                connection.commit()
            cls._finalize_vocabulary(connection)
            cls.write_metadata(
                connection,
                {
                    "schema_version": cls.SCHEMA_VERSION,
                    "order": order,
                    "sentence_count": sentence_count,
                    "token_count": token_count,
                    "counting_complete": False,
                },
            )
            connection.commit()
        finally:
            connection.close()
        return artifact

    @staticmethod
    def _finalize_vocabulary(connection: sqlite3.Connection) -> None:
        cursor = connection.execute(
            "SELECT term, count FROM term_counts ORDER BY count DESC, term ASC"
        )
        next_id = 1
        batch: list[tuple[int, str, int]] = []
        statement = "INSERT INTO terms(id, term, frequency) VALUES (?, ?, ?)"
        for term, frequency in cursor:
            batch.append((next_id, term, frequency))
            next_id += 1
            if len(batch) >= 10_000:
                connection.executemany(statement, batch)
                batch = []
        if batch:
            connection.executemany(statement, batch)
        for special in (START_TOKEN, END_TOKEN, BOUNDARY_TOKEN):
            connection.execute(statement, (next_id, special, 0))
            next_id += 1
        connection.execute("DROP TABLE term_counts")

    def term_count(self) -> int:
        connection = connect(self.path)
        try:
            return int(connection.execute("SELECT COUNT(*) FROM terms").fetchone()[0])
        finally:
            connection.close()

    def term_records(self, include_special: bool = False) -> Iterator[tuple[int, str, int]]:
        connection = connect(self.path)
        try:
            query = "SELECT id, term, frequency FROM terms"
            parameters: tuple[object, ...] = ()
            if not include_special:
                query += " WHERE frequency > ?"
                parameters = (0,)
            query += " ORDER BY id"
            yield from connection.execute(query, parameters)
        finally:
            connection.close()

    @staticmethod
    def _resolve_ids(
        connection: sqlite3.Connection, terms: set[str]
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        ordered = sorted(terms)
        for chunk in chunks(ordered):
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT term, id FROM terms WHERE term IN ({placeholders})"
            result.update(connection.execute(query, tuple(chunk)))
        if len(result) != len(terms):
            missing = sorted(terms - result.keys())[:5]
            raise KeyError(f"terms missing from vocabulary: {missing}")
        return result

    def count_statistics(
        self,
        sentences: Iterable[Sequence[str]],
        batch_tokens: int = 50_000,
    ) -> None:
        metadata = self.metadata()
        if metadata.get("counting_complete"):
            raise RuntimeError("statistics have already been counted")
        order = int(metadata["order"])
        connection = connect(self.path)
        total_unigrams = [0, 0]
        try:
            special_ids = self._resolve_ids(
                connection, {START_TOKEN, END_TOKEN, BOUNDARY_TOKEN}
            )
            start_id = special_ids[START_TOKEN]
            end_id = special_ids[END_TOKEN]
            boundary_id = special_ids[BOUNDARY_TOKEN]
            for batch in self.sentence_batches(sentences, batch_tokens):
                vocabulary = {term for sentence in batch for term in sentence}
                term_ids = self._resolve_ids(connection, vocabulary)
                ngrams: Counter[tuple[int, int, bytes]] = Counter()
                contexts: Counter[tuple[int, int, int]] = Counter()
                context_totals: Counter[tuple[int, int]] = Counter()
                for sentence in batch:
                    ids = [term_ids[term] for term in sentence]
                    left_right = [boundary_id, *ids, boundary_id]
                    for index, term_id in enumerate(ids, 1):
                        context = (left_right[index - 1], left_right[index + 1])
                        contexts[(term_id, *context)] += 1
                        context_totals[context] += 1
                    for direction, sequence in enumerate((ids, list(reversed(ids)))):
                        padded = [start_id] * (order - 1) + sequence + [end_id]
                        for n in range(1, order + 1):
                            for offset in range(len(padded) - n + 1):
                                ngrams[(direction, n, encode_ids(padded[offset : offset + n]))] += 1
                                if n == 1:
                                    total_unigrams[direction] += 1
                self._flush_counts(connection, ngrams, contexts, context_totals)
                connection.commit()
            self.write_metadata(
                connection,
                {
                    "counting_complete": True,
                    "forward_unigrams": total_unigrams[0],
                    "backward_unigrams": total_unigrams[1],
                },
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _flush_counts(
        connection: sqlite3.Connection,
        ngrams: Counter[tuple[int, int, bytes]],
        contexts: Counter[tuple[int, int, int]],
        context_totals: Counter[tuple[int, int]],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO ngrams(direction, n, key, count) VALUES (?, ?, ?, ?)
            ON CONFLICT(direction, n, key) DO UPDATE SET count = count + excluded.count
            """,
            ((*key, count) for key, count in ngrams.items()),
        )
        connection.executemany(
            """
            INSERT INTO contexts(term_id, left_id, right_id, count) VALUES (?, ?, ?, ?)
            ON CONFLICT(term_id, left_id, right_id)
            DO UPDATE SET count = count + excluded.count
            """,
            ((*key, count) for key, count in contexts.items()),
        )
        connection.executemany(
            """
            INSERT INTO context_totals(left_id, right_id, count) VALUES (?, ?, ?)
            ON CONFLICT(left_id, right_id) DO UPDATE SET count = count + excluded.count
            """,
            ((*key, count) for key, count in context_totals.items()),
        )

    @classmethod
    def build(
        cls,
        sentences: Iterable[Sequence[str]],
        path: str | Path,
        order: int = 5,
        batch_tokens: int = 50_000,
    ) -> "SQLiteCorpusStatistics":
        if iter(sentences) is sentences:
            raise TypeError("build requires a replayable corpus, not a one-shot iterator")
        artifact = cls.build_vocabulary(sentences, path, order, batch_tokens)
        artifact.count_statistics(sentences, batch_tokens)
        return artifact
