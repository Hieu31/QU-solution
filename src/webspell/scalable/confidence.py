from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from webspell.confidence import ConfidenceModel, TrainingRow
from webspell.confidence.logistic import BinaryLogisticRegression
from webspell.config import ClassifierConfig


RowsFactory = Callable[[], Iterable[TrainingRow]]
Predicate = Callable[[TrainingRow], bool]
Label = Callable[[TrainingRow], int]


@dataclass(frozen=True)
class StreamingClassifierConfig:
    epochs: int = 5
    learning_rate: float = 0.03
    l2: float = 1e-5
    threshold_bins: int = 1000
    feature_clip: float = 8.0


def _sigmoid(value: float) -> float:
    if value >= 0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def _normalization(
    rows_factory: RowsFactory, predicate: Predicate, label: Label
) -> tuple[int, list[float], list[float], set[int]]:
    count = 0
    means: list[float] = []
    m2: list[float] = []
    labels: set[int] = set()
    for row in rows_factory():
        if not predicate(row):
            continue
        if not means:
            means = [0.0] * len(row.features)
            m2 = [0.0] * len(row.features)
        count += 1
        labels.add(label(row))
        for index, value in enumerate(row.features):
            delta = value - means[index]
            means[index] += delta / count
            m2[index] += delta * (value - means[index])
    if count == 0:
        raise ValueError("streaming classifier received no matching rows")
    scales = [math.sqrt(value / count) or 1.0 for value in m2]
    return count, means, scales, labels


def _fit_model(
    rows_factory: RowsFactory,
    predicate: Predicate,
    label: Label,
    settings: StreamingClassifierConfig,
) -> BinaryLogisticRegression:
    count, means, scales, labels = _normalization(rows_factory, predicate, label)
    model = BinaryLogisticRegression(
        ClassifierConfig(
            learning_rate=settings.learning_rate,
            epochs=settings.epochs,
            l2=settings.l2,
        )
    )
    model.means = means
    model.scales = scales
    model.weights = [0.0] * len(means)
    if len(labels) == 1:
        model.constant_probability = float(next(iter(labels)))
        return model

    step = 0
    for _ in range(settings.epochs):
        for row in rows_factory():
            if not predicate(row):
                continue
            values = [
                max(-settings.feature_clip, min(settings.feature_clip, value))
                for value in model._standardize(row.features)
            ]
            probability = _sigmoid(
                model.bias + sum(weight * value for weight, value in zip(model.weights, values))
            )
            error = probability - label(row)
            step += 1
            rate = settings.learning_rate / math.sqrt(1.0 + step / max(count, 1))
            model.bias -= rate * error
            for index, value in enumerate(values):
                gradient = error * value + settings.l2 * model.weights[index]
                model.weights[index] -= rate * gradient
    return model


def _histogram_threshold(
    model: BinaryLogisticRegression,
    rows_factory: RowsFactory,
    predicate: Predicate,
    label: Label,
    bins: int,
) -> float:
    positives = [0] * (bins + 1)
    negatives = [0] * (bins + 1)
    total_positive = 0
    matched = 0
    for row in rows_factory():
        if not predicate(row):
            continue
        bucket = min(bins, int(model.predict_probability(row.features) * bins))
        target = label(row)
        positives[bucket] += target
        negatives[bucket] += 1 - target
        total_positive += target
        matched += 1
    if not matched:
        raise ValueError("threshold tuning received no matching rows")
    true_positive = false_positive = 0
    best_threshold, best_f1 = 0.5, -1.0
    for bucket in range(bins, -1, -1):
        true_positive += positives[bucket]
        false_positive += negatives[bucket]
        false_negative = total_positive - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        threshold = bucket / bins
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold


class StreamingConfidenceTrainer:
    def __init__(self, config: StreamingClassifierConfig | None = None) -> None:
        self.config = config or StreamingClassifierConfig()

    def fit(
        self,
        training_rows: RowsFactory,
        development_rows: RowsFactory,
    ) -> ConfidenceModel:
        with_suggestions = lambda row: row.has_suggestions
        without_suggestions = lambda row: not row.has_suggestions
        misspelled = lambda row: row.misspelled

        result = ConfidenceModel()
        result.with_suggestions = _fit_model(
            training_rows, with_suggestions, misspelled, self.config
        )
        result.without_suggestions = _fit_model(
            training_rows, without_suggestions, misspelled, self.config
        )
        result.with_threshold = _histogram_threshold(
            result.with_suggestions,
            development_rows,
            with_suggestions,
            misspelled,
            self.config.threshold_bins,
        )
        result.without_threshold = _histogram_threshold(
            result.without_suggestions,
            development_rows,
            without_suggestions,
            misspelled,
            self.config.threshold_bins,
        )

        def selected(row: TrainingRow) -> bool:
            return row.has_suggestions and (
                result.with_suggestions.predict_probability(row.features)
                >= result.with_threshold
            )

        top_correct = lambda row: row.top_is_correct
        try:
            result.autocorrect = _fit_model(
                training_rows, selected, top_correct, self.config
            )
            result.autocorrect_threshold = _histogram_threshold(
                result.autocorrect,
                development_rows,
                selected,
                top_correct,
                self.config.threshold_bins,
            )
        except ValueError:
            result.autocorrect = _fit_model(
                training_rows, with_suggestions, top_correct, self.config
            )
            result.autocorrect_threshold = _histogram_threshold(
                result.autocorrect,
                development_rows,
                with_suggestions,
                top_correct,
                self.config.threshold_bins,
            )
        return result
