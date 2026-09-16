from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from pathlib import Path

from webspell.text import SimpleTokenizer


def _is_lexical(token: str) -> bool:
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in token)


class LineCorpus:
    """Replayable streaming corpus with one document or sentence per line."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: SimpleTokenizer | None = None,
        encoding: str = "utf-8",
        lowercase: bool = False,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.encoding = encoding
        self.lowercase = lowercase

    def sentences(self) -> Iterator[tuple[str, ...]]:
        with self.path.open("r", encoding=self.encoding, errors="strict") as stream:
            for line in stream:
                tokens = self.tokenizer.tokenize(line.rstrip("\r\n"))
                current_sentence: list[str] = []
                current_id: int | None = None
                for token in tokens:
                    if current_id is not None and token.sentence_id != current_id:
                        if current_sentence:
                            yield tuple(current_sentence)
                        current_sentence = []
                    current_id = token.sentence_id
                    if _is_lexical(token.normalized):
                        value = token.normalized.lower() if self.lowercase else token.normalized
                        current_sentence.append(value)
                if current_sentence:
                    yield tuple(current_sentence)

    def __iter__(self) -> Iterator[tuple[str, ...]]:
        return self.sentences()
