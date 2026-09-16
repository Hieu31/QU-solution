from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

from webspell.config import LanguageModelConfig
from webspell.types import Context


class NGramModel:
    def __init__(self, config: LanguageModelConfig | None = None) -> None:
        self.config = config or LanguageModelConfig()
        self.counts: dict[int, Counter[tuple[str, ...]]] = {
            order: Counter() for order in range(1, self.config.order + 1)
        }
        self.total_unigrams = 0

    @classmethod
    def fit(
        cls,
        sentences: Iterable[Sequence[str]],
        config: LanguageModelConfig | None = None,
    ) -> "NGramModel":
        model = cls(config)
        for sentence in sentences:
            padded = ["<s>"] * (model.config.order - 1) + list(sentence) + ["</s>"]
            for order in range(1, model.config.order + 1):
                for start in range(len(padded) - order + 1):
                    model.counts[order][tuple(padded[start : start + order])] += 1
        model.total_unigrams = sum(model.counts[1].values())
        return model

    def logscore(self, token: str, history: Sequence[str]) -> float:
        bounded = tuple(history[-(self.config.order - 1) :])
        backoffs = 0
        while bounded:
            ngram = bounded + (token,)
            count = self.counts[len(ngram)].get(ngram, 0)
            if count:
                denominator = self.counts[len(bounded)].get(bounded, 0)
                if denominator:
                    return math.log(count / denominator) + backoffs * math.log(
                        self.config.backoff_alpha
                    )
            bounded = bounded[1:]
            backoffs += 1
        count = self.counts[1].get((token,), 0)
        probability = (
            count / self.total_unigrams
            if count and self.total_unigrams
            else self.config.unknown_probability
        )
        return math.log(probability) + backoffs * math.log(self.config.backoff_alpha)


class BidirectionalLanguageModel:
    def __init__(self, forward: NGramModel, backward: NGramModel) -> None:
        if forward.config != backward.config:
            raise ValueError("forward and backward language-model configs must match")
        self.forward = forward
        self.backward = backward
        self.config = forward.config

    @classmethod
    def fit(
        cls,
        sentences: Iterable[Sequence[str]],
        config: LanguageModelConfig | None = None,
    ) -> "BidirectionalLanguageModel":
        materialized = [tuple(sentence) for sentence in sentences]
        forward = NGramModel.fit(materialized, config)
        backward = NGramModel.fit((tuple(reversed(s)) for s in materialized), config)
        return cls(forward, backward)

    def token_logscore(self, candidate: str, context: Context) -> float:
        forward = self.forward.logscore(candidate, context.left)
        backward = self.backward.logscore(candidate, tuple(reversed(context.right)))
        return forward + backward
