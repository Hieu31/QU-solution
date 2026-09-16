from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from webspell.types import Context, Token


_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
_SENTENCE_END = frozenset({".", "!", "?"})


class SimpleTokenizer:
    """Deterministic Unicode tokenizer replaceable by a language adapter."""

    def __init__(self, normalization: str = "NFC") -> None:
        self.normalization = normalization

    @property
    def identity(self) -> str:
        return f"simple-unicode-tokenizer/{self.normalization}/v1"

    def normalize_token(self, token: str) -> str:
        return unicodedata.normalize(self.normalization, token)

    def tokenize(self, text: str) -> list[Token]:
        result: list[Token] = []
        sentence_id = 0
        position = 0
        for match in _TOKEN_RE.finditer(text):
            surface = match.group(0)
            result.append(
                Token(
                    surface=surface,
                    normalized=self.normalize_token(surface),
                    sentence_id=sentence_id,
                    position=position,
                    character_start=match.start(),
                    character_end=match.end(),
                )
            )
            position += 1
            if surface in _SENTENCE_END:
                sentence_id += 1
                position = 0
        return result


def context_for(
    tokens: Sequence[Token], token_index: int, max_left: int, max_right: int
) -> Context:
    current = tokens[token_index]
    left: list[str] = []
    right: list[str] = []
    cursor = token_index - 1
    while cursor >= 0 and len(left) < max_left:
        token = tokens[cursor]
        if token.sentence_id != current.sentence_id:
            break
        left.append(token.normalized)
        cursor -= 1
    left.reverse()
    cursor = token_index + 1
    while cursor < len(tokens) and len(right) < max_right:
        token = tokens[cursor]
        if token.sentence_id != current.sentence_id:
            break
        right.append(token.normalized)
        cursor += 1
    return Context(tuple(left), tuple(right))
