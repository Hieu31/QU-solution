from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from webspell.config import ClassifierConfig
from webspell.confidence.logistic import BinaryLogisticRegression, tune_f1_threshold
from webspell.types import Candidate, Context


def _case_signature(token: str) -> tuple[float, ...]:
    cased = [char for char in token if char.isalpha()]
    return (
        float(bool(cased) and token.islower()),
        float(bool(cased) and token.isupper()),
        float(bool(cased) and token.istitle()),
        float(bool(cased) and not (token.islower() or token.isupper() or token.istitle())),
        float(not cased),
    )


def non_original_candidates(observed: str, candidates: list[Candidate]) -> list[Candidate]:
    return [candidate for candidate in candidates if candidate.term != observed]


def extract_features(
    observed: str, context: Context, candidates: list[Candidate]
) -> tuple[float, ...]:
    original = next(candidate for candidate in candidates if candidate.term == observed)
    suggestions = non_original_candidates(observed, candidates)

    def deltas(index: int) -> tuple[float, float, float]:
        if index >= len(suggestions):
            return 0.0, 0.0, 0.0
        candidate = suggestions[index]
        error_delta = candidate.error_log_score - original.error_log_score
        lm_delta = float(candidate.lm_log_score) - float(original.lm_log_score)
        return 1.0, error_delta, lm_delta

    top1 = deltas(0)
    top2 = deltas(1)
    case = _case_signature(observed)
    case_match = float(
        bool(suggestions) and _case_signature(suggestions[0].term) == _case_signature(observed)
    )
    return (
        original.error_log_score,
        float(original.lm_log_score),
        *top1,
        *top2,
        *case,
        case_match,
        float(len(suggestions)),
        float(len(observed)),
        float(len(context.left)),
        float(len(context.right)),
    )


def is_blacklisted(token: str) -> bool:
    if len(token) <= 1 or token.isnumeric():
        return True
    return all(unicodedata.category(char)[0] in {"P", "S"} for char in token)


@dataclass(frozen=True)
class TrainingRow:
    features: tuple[float, ...]
    misspelled: int
    has_suggestions: bool
    top_is_correct: int
    error_type: str = "unknown"


class ConfidenceModel:
    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        self.with_suggestions = BinaryLogisticRegression(self.config)
        self.without_suggestions = BinaryLogisticRegression(self.config)
        self.autocorrect = BinaryLogisticRegression(self.config)
        self.with_threshold = 0.5
        self.without_threshold = 0.5
        self.autocorrect_threshold = 0.5

    @staticmethod
    def _fit_classifier(
        classifier: BinaryLogisticRegression,
        training_rows: list[TrainingRow],
        development_rows: list[TrainingRow],
    ) -> float:
        classifier.fit(
            [row.features for row in training_rows],
            [row.misspelled for row in training_rows],
        )
        probabilities = [classifier.predict_probability(row.features) for row in development_rows]
        return tune_f1_threshold(
            probabilities, [row.misspelled for row in development_rows]
        )

    def fit(
        self,
        rows: list[TrainingRow],
        development_rows: list[TrainingRow] | None = None,
    ) -> None:
        development = development_rows or rows
        with_rows = [row for row in rows if row.has_suggestions]
        without_rows = [row for row in rows if not row.has_suggestions]
        development_with = [row for row in development if row.has_suggestions]
        development_without = [row for row in development if not row.has_suggestions]
        if not with_rows or not without_rows or not development_with or not development_without:
            raise ValueError("confidence training needs rows both with and without suggestions")
        self.with_threshold = self._fit_classifier(
            self.with_suggestions, with_rows, development_with
        )
        self.without_threshold = self._fit_classifier(
            self.without_suggestions, without_rows, development_without
        )

        selected = [
            row
            for row in with_rows
            if self.with_suggestions.predict_probability(row.features) >= self.with_threshold
        ]
        if not selected:
            selected = with_rows
        autocorrect_labels = [row.top_is_correct for row in selected]
        self.autocorrect.fit([row.features for row in selected], autocorrect_labels)
        selected_development = [
            row
            for row in development_with
            if self.with_suggestions.predict_probability(row.features) >= self.with_threshold
        ] or development_with
        probabilities = [
            self.autocorrect.predict_probability(row.features)
            for row in selected_development
        ]
        self.autocorrect_threshold = tune_f1_threshold(
            probabilities, [row.top_is_correct for row in selected_development]
        )

    def spellcheck_probability(self, features: tuple[float, ...], has_suggestions: bool) -> float:
        classifier = self.with_suggestions if has_suggestions else self.without_suggestions
        return classifier.predict_probability(features)

    def spellcheck_threshold(self, has_suggestions: bool) -> float:
        return self.with_threshold if has_suggestions else self.without_threshold

    def autocorrect_probability(self, features: tuple[float, ...]) -> float:
        return self.autocorrect.predict_probability(features)
