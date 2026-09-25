"""Feature contract shared by confidence training and the live demo."""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher

import numpy as np

from train_reparos_candidate_confidence import FEATURE_NAMES, features, normalized


EXTRA_NAMES = (
    "log1p_margin", "log1p_gap", "gap_to_next", "gap_from_previous",
    "margin_x_rank1", "margin_x_rank2", "change_x_rank1", "score_x_rank1",
    "accent_fold_change", "score_squared",
)
ALL_NAMES = FEATURE_NAMES + EXTRA_NAMES
CHANGE_INDEX = FEATURE_NAMES.index("character_change_ratio")


def accent_fold(value: str) -> str:
    text = unicodedata.normalize("NFD", normalized(value).lower())
    return "".join(char for char in text if unicodedata.category(char) != "Mn").replace("\u0111", "d")


def candidate_feature_rows(query: str, hypotheses: list[str], scores: list[float]) -> np.ndarray:
    if len(hypotheses) != 10 or len(scores) != 10:
        raise ValueError("Confidence requires exactly 10 candidates and 10 sequence scores")
    top_score = float(scores[0])
    margin = top_score - float(scores[1])
    input_fold = accent_fold(query)
    rows = []
    for index, (hypothesis, score_value) in enumerate(zip(hypotheses, scores)):
        rank, score = index + 1, float(score_value)
        base = features(query, hypothesis, score, top_score, margin, rank)
        gap = top_score - score
        change = base[CHANGE_INDEX]
        accent_change = 1 - SequenceMatcher(
            None, input_fold, accent_fold(hypothesis), autojunk=False
        ).ratio()
        extras = [
            np.log1p(max(margin, 0)), np.log1p(max(gap, 0)),
            score - float(scores[index + 1]) if index < 9 else 0.0,
            float(scores[index - 1]) - score if index > 0 else 0.0,
            margin * (rank == 1), margin * (rank == 2),
            change * (rank == 1), score * (rank == 1),
            accent_change, score * score,
        ]
        rows.append(base + extras)
    return np.asarray(rows, dtype=np.float64)
