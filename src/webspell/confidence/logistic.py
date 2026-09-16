from __future__ import annotations

import math
from collections.abc import Sequence

from webspell.config import ClassifierConfig


def _sigmoid(value: float) -> float:
    if value >= 0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


class BinaryLogisticRegression:
    """Small deterministic implementation used to keep the prototype dependency-free."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        self.weights: list[float] = []
        self.bias = 0.0
        self.means: list[float] = []
        self.scales: list[float] = []
        self.constant_probability: float | None = None

    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
        if len(features) != len(labels) or not features:
            raise ValueError("features and labels must have the same non-zero length")
        width = len(features[0])
        if any(len(row) != width for row in features):
            raise ValueError("all feature rows must have the same width")
        if len(set(labels)) == 1:
            self.constant_probability = float(labels[0])
            self.weights = [0.0] * width
            self.means = [0.0] * width
            self.scales = [1.0] * width
            return

        count = len(features)
        self.means = [sum(row[col] for row in features) / count for col in range(width)]
        self.scales = []
        for col in range(width):
            variance = sum((row[col] - self.means[col]) ** 2 for row in features) / count
            self.scales.append(math.sqrt(variance) or 1.0)
        standardized = [self._standardize(row) for row in features]
        self.weights = [0.0] * width
        self.bias = 0.0
        for _ in range(self.config.epochs):
            weight_gradient = [0.0] * width
            bias_gradient = 0.0
            for row, label in zip(standardized, labels, strict=True):
                probability = _sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, row)))
                error = probability - label
                bias_gradient += error
                for col, value in enumerate(row):
                    weight_gradient[col] += error * value
            rate = self.config.learning_rate
            self.bias -= rate * bias_gradient / count
            for col in range(width):
                regularized = weight_gradient[col] / count + self.config.l2 * self.weights[col]
                self.weights[col] -= rate * regularized

    def _standardize(self, row: Sequence[float]) -> list[float]:
        return [
            (value - mean) / scale
            for value, mean, scale in zip(row, self.means, self.scales, strict=True)
        ]

    def predict_probability(self, features: Sequence[float]) -> float:
        if self.constant_probability is not None:
            return self.constant_probability
        if not self.weights:
            raise RuntimeError("classifier has not been fitted")
        row = self._standardize(features)
        return _sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, row)))

    def to_state(self) -> dict[str, object]:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "means": self.means,
            "scales": self.scales,
            "constant_probability": self.constant_probability,
        }

    def load_state(self, state: dict[str, object]) -> None:
        self.weights = [float(value) for value in state["weights"]]  # type: ignore[arg-type]
        self.bias = float(state["bias"])  # type: ignore[arg-type]
        self.means = [float(value) for value in state["means"]]  # type: ignore[arg-type]
        self.scales = [float(value) for value in state["scales"]]  # type: ignore[arg-type]
        constant = state.get("constant_probability")
        self.constant_probability = None if constant is None else float(constant)


def tune_f1_threshold(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must have the same non-zero length")
    candidates = sorted({0.0, 1.0, *probabilities})
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        true_positive = false_positive = false_negative = 0
        for probability, label in zip(probabilities, labels, strict=True):
            prediction = probability >= threshold
            true_positive += int(prediction and label == 1)
            false_positive += int(prediction and label == 0)
            false_negative += int(not prediction and label == 1)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_threshold, best_f1 = threshold, f1
    return best_threshold
