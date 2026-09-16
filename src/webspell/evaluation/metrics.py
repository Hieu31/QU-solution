from __future__ import annotations

from dataclasses import dataclass

from webspell.types import Prediction


@dataclass(frozen=True)
class EvaluationMetrics:
    tokens: int
    e1: int
    e2: int
    e3: int
    e4: int
    e5: int
    no_good_suggestion: int
    misspelled_tokens: int

    @property
    def cer(self) -> float:
        return (self.e1 + self.e2 + self.e3 + self.e4) / self.tokens

    @property
    def fer(self) -> float:
        return (self.e3 + self.e5) / self.tokens

    @property
    def ter(self) -> float:
        return (self.e1 + self.e2 + self.e3 + self.e4 + self.e5) / self.tokens

    @property
    def ngs(self) -> float:
        return (
            self.no_good_suggestion / self.misspelled_tokens
            if self.misspelled_tokens
            else 0.0
        )


def evaluate_predictions(
    predictions: list[Prediction], intended_tokens: list[str]
) -> EvaluationMetrics:
    if len(predictions) != len(intended_tokens) or not predictions:
        raise ValueError("predictions and intended tokens must have equal non-zero length")
    errors = [0, 0, 0, 0, 0]
    no_good = 0
    misspelled_count = 0
    for prediction, intended in zip(predictions, intended_tokens, strict=True):
        observed = prediction.token.normalized
        misspelled = observed != intended
        if misspelled:
            misspelled_count += 1
            if not any(candidate.term == intended for candidate in prediction.candidates):
                no_good += 1
        if prediction.action == "correct":
            if misspelled and prediction.correction != intended:
                errors[0] += 1
            elif not misspelled:
                errors[3] += 1
        elif prediction.action == "flag":
            errors[1 if misspelled else 4] += 1
        elif misspelled:
            errors[2] += 1
    return EvaluationMetrics(
        len(predictions), *errors, no_good, misspelled_count
    )
