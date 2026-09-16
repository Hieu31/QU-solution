from __future__ import annotations

from collections.abc import Iterable, Iterator

from webspell.candidates import CandidateRanker, LambdaGrid
from webspell.confidence import ConfidenceModel, TrainingRow, extract_features
from webspell.confidence.model import is_blacklisted, non_original_candidates
from webspell.text import SimpleTokenizer, context_for
from webspell.types import Candidate, LabeledToken, Prediction


class WebSpellModel:
    def __init__(
        self,
        tokenizer: SimpleTokenizer,
        ranker: CandidateRanker,
        lambdas: LambdaGrid,
        confidence: ConfidenceModel,
    ) -> None:
        self.tokenizer = tokenizer
        self.ranker = ranker
        self.lambdas = lambdas
        self.confidence = confidence

    def predict(self, text: str) -> list[Prediction]:
        tokens = self.tokenizer.tokenize(text)
        max_context = self.ranker.language_model.config.order - 1
        predictions = []
        for index, token in enumerate(tokens):
            if is_blacklisted(token.surface):
                candidate = Candidate(token.normalized, 0, 1, 0.0, 0.0, 0.0)
                predictions.append(
                    Prediction(token, False, "keep", None, (candidate,), 0.0, None)
                )
                continue

            context = context_for(tokens, index, max_context, max_context)
            lambda_value = self.lambdas.for_context(context, max_context)
            candidates = self.ranker.rank(token.normalized, context, lambda_value)
            suggestions = non_original_candidates(token.normalized, candidates)
            features = extract_features(token.normalized, context, candidates)
            has_suggestions = bool(suggestions)
            spell_probability = self.confidence.spellcheck_probability(
                features, has_suggestions
            )
            if spell_probability < self.confidence.spellcheck_threshold(has_suggestions):
                predictions.append(
                    Prediction(
                        token, False, "keep", None, tuple(candidates), spell_probability, None
                    )
                )
                continue
            if not suggestions:
                predictions.append(
                    Prediction(
                        token, True, "flag", None, tuple(candidates), spell_probability, None
                    )
                )
                continue

            autocorrect_probability = self.confidence.autocorrect_probability(features)
            if autocorrect_probability >= self.confidence.autocorrect_threshold:
                predictions.append(
                    Prediction(
                        token,
                        True,
                        "correct",
                        suggestions[0].term,
                        tuple(candidates),
                        spell_probability,
                        autocorrect_probability,
                    )
                )
            else:
                predictions.append(
                    Prediction(
                        token,
                        True,
                        "flag",
                        None,
                        tuple(candidates),
                        spell_probability,
                        autocorrect_probability,
                    )
                )
        return predictions

    def save(self, directory: str) -> None:
        from webspell.serialization import save_model_bundle

        save_model_bundle(self, directory)

    @classmethod
    def load(cls, directory: str) -> "WebSpellModel":
        from webspell.serialization import load_model_bundle

        return load_model_bundle(directory)

    def close(self) -> None:
        if hasattr(self.ranker, "close") and callable(self.ranker.close):
            self.ranker.close()

    def __enter__(self) -> "WebSpellModel":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @staticmethod
    def corrected_text(text: str, predictions: list[Prediction]) -> str:
        pieces: list[str] = []
        cursor = 0
        for prediction in predictions:
            token = prediction.token
            pieces.append(text[cursor : token.character_start])
            pieces.append(prediction.correction or token.surface)
            cursor = token.character_end
        pieces.append(text[cursor:])
        return "".join(pieces)


def build_training_rows(
    examples: Iterable[LabeledToken],
    ranker: CandidateRanker,
    lambdas: LambdaGrid,
) -> list[TrainingRow]:
    return list(iter_training_rows(examples, ranker, lambdas))


def iter_training_rows(
    examples: Iterable[LabeledToken],
    ranker: CandidateRanker,
    lambdas: LambdaGrid,
) -> Iterator[TrainingRow]:
    max_context = ranker.language_model.config.order - 1
    for example in examples:
        lambda_value = lambdas.for_context(example.context, max_context)
        candidates = ranker.rank(example.observed, example.context, lambda_value)
        suggestions = non_original_candidates(example.observed, candidates)
        yield TrainingRow(
            features=extract_features(example.observed, example.context, candidates),
            misspelled=int(example.observed != example.intended),
            has_suggestions=bool(suggestions),
            top_is_correct=int(bool(suggestions) and suggestions[0].term == example.intended),
            error_type=example.error_type,
        )
