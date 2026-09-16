from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Token:
    surface: str
    normalized: str
    sentence_id: int
    position: int
    character_start: int
    character_end: int


@dataclass(frozen=True)
class Context:
    left: tuple[str, ...]
    right: tuple[str, ...]


@dataclass(frozen=True)
class ErrorTriple:
    intended: str
    observed: str
    count: int


@dataclass(frozen=True)
class Candidate:
    term: str
    edit_distance: int
    term_frequency: int
    error_log_score: float
    lm_log_score: float | None = None
    combined_log_score: float | None = None


Action = Literal["keep", "flag", "correct"]


@dataclass(frozen=True)
class Prediction:
    token: Token
    is_misspelled: bool
    action: Action
    correction: str | None
    candidates: tuple[Candidate, ...]
    spellcheck_confidence: float
    autocorrect_confidence: float | None


@dataclass(frozen=True)
class LabeledToken:
    observed: str
    intended: str
    context: Context
    error_type: str = "unknown"
