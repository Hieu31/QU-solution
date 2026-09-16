from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ErrorModelConfig:
    max_source_substring_length: int = 2
    max_target_substring_length: int = 2
    smoothing_alpha: float = 0.1
    identity_prior: float = 10.0
    alignment_iterations: int = 1


@dataclass(frozen=True)
class LanguageModelConfig:
    order: int = 5
    backoff_alpha: float = 0.4
    unknown_probability: float = 1e-9


@dataclass(frozen=True)
class CandidateConfig:
    preselection_limit: int = 20
    maximum_edit_distance: int = 3

    def max_edit_distance(self, token_length: int) -> int:
        if token_length <= 4:
            distance = 1
        elif token_length <= 12:
            distance = 2
        else:
            distance = 3
        return min(distance, self.maximum_edit_distance)


@dataclass(frozen=True)
class LambdaConfig:
    minimum: float = 0.0
    maximum: float = 10.0
    step: float = 0.25
    default: float = 1.0


@dataclass(frozen=True)
class ClassifierConfig:
    learning_rate: float = 0.1
    epochs: int = 600
    l2: float = 1e-3


@dataclass(frozen=True)
class WebSpellConfig:
    unicode_normalization: str = "NFC"
    error_model: ErrorModelConfig = field(default_factory=ErrorModelConfig)
    language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    lambda_tuning: LambdaConfig = field(default_factory=LambdaConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
