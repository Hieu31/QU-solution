from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from webspell.candidates import CandidateRanker, LambdaGrid
from webspell.confidence import ConfidenceModel
from webspell.config import (
    CandidateConfig,
    ClassifierConfig,
    ErrorModelConfig,
    LanguageModelConfig,
)
from webspell.error_model import SubstringErrorModel
from webspell.language_model import BidirectionalLanguageModel, NGramModel
from webspell.pipeline.model import WebSpellModel
from webspell.text import SimpleTokenizer, VietnameseVariantGenerator
from webspell.vocabulary import TermLexicon


SCHEMA_VERSION = 1
MODEL_FILENAME = "model.json"
MANIFEST_FILENAME = "manifest.json"


class BundleValidationError(ValueError):
    pass


def _ngram_state(model: NGramModel) -> dict[str, Any]:
    return {
        "counts": {
            str(order): [
                [list(ngram), count]
                for ngram, count in sorted(model.counts[order].items())
            ]
            for order in sorted(model.counts)
        },
        "total_unigrams": model.total_unigrams,
    }


def _load_ngram(state: dict[str, Any], config: LanguageModelConfig) -> NGramModel:
    model = NGramModel(config)
    model.counts = {
        int(order): Counter({tuple(ngram): int(count) for ngram, count in records})
        for order, records in state["counts"].items()
    }
    model.total_unigrams = int(state["total_unigrams"])
    return model


def _variant_identity(ranker: CandidateRanker) -> str | None:
    if ranker.variant_generator is None:
        return None
    if isinstance(ranker.variant_generator, VietnameseVariantGenerator):
        return "vietnamese-input/v1"
    raise TypeError("cannot serialize an unregistered candidate variant generator")


def _model_state(model: WebSpellModel) -> dict[str, Any]:
    ranker = model.ranker
    error_model = ranker.error_model
    language_model = ranker.language_model
    confidence = model.confidence

    from webspell.scalable.symspell import SQLiteSymSpellIndex
    is_sqlite = isinstance(ranker.lexicon, SQLiteSymSpellIndex)

    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "storage": "sqlite" if is_sqlite else "in_memory",
        "tokenizer": {
            "type": "simple-unicode-tokenizer",
            "normalization": model.tokenizer.normalization,
            "identity": model.tokenizer.identity,
        },
        "error_model": {
            "config": asdict(error_model.config),
            "counts": error_model.transitions(),
            "targets": sorted(error_model._targets),
        },
        "candidate_config": asdict(ranker.config),
        "variant_generator": _variant_identity(ranker),
        "lambdas": {
            "default": model.lambdas.default,
            "values": [
                {"left": left, "right": right, "value": value}
                for (left, right), value in sorted(model.lambdas.values.items())
            ],
        },
        "confidence": {
            "config": asdict(confidence.config),
            "with_suggestions": confidence.with_suggestions.to_state(),
            "without_suggestions": confidence.without_suggestions.to_state(),
            "autocorrect": confidence.autocorrect.to_state(),
            "thresholds": {
                "with_suggestions": confidence.with_threshold,
                "without_suggestions": confidence.without_threshold,
                "autocorrect": confidence.autocorrect_threshold,
            },
        },
    }
    if is_sqlite:
        base["sqlite_database"] = "statistics.sqlite3"
    else:
        base["lexicon"] = ranker.lexicon.frequencies()
        base["language_model"] = {
            "config": asdict(language_model.config),
            "forward": _ngram_state(language_model.forward),
            "backward": _ngram_state(language_model.backward),
        }
    return base


def _json_bytes(value: object) -> bytes:
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_model_bundle(model: WebSpellModel, directory: str | os.PathLike[str]) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    state = _model_state(model)
    model_bytes = _json_bytes(state)
    checksum = hashlib.sha256(model_bytes).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "model_file": MODEL_FILENAME,
        "model_sha256": checksum,
        "model_size": len(model_bytes),
        "tokenizer_identity": model.tokenizer.identity,
        "storage": state.get("storage", "in_memory"),
    }
    from webspell.scalable.symspell import SQLiteSymSpellIndex
    if isinstance(model.ranker.lexicon, SQLiteSymSpellIndex):
        sqlite_source = Path(model.ranker.lexicon.path)
        sqlite_target = destination / "statistics.sqlite3"
        if sqlite_source.resolve() != sqlite_target.resolve():
            shutil.copy2(sqlite_source, sqlite_target)
        manifest["sqlite_database"] = "statistics.sqlite3"

    _atomic_write(destination / MODEL_FILENAME, model_bytes)
    _atomic_write(destination / MANIFEST_FILENAME, _json_bytes(manifest))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError(f"cannot read valid JSON from {path.name}") from error
    if not isinstance(value, dict):
        raise BundleValidationError(f"{path.name} must contain a JSON object")
    return value


def _validate_bundle(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(directory / MANIFEST_FILENAME)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BundleValidationError("unsupported manifest schema version")
    if manifest.get("model_file") != MODEL_FILENAME:
        raise BundleValidationError("manifest references an unexpected model file")
    try:
        model_bytes = (directory / MODEL_FILENAME).read_bytes()
    except OSError as error:
        raise BundleValidationError("model file is missing or unreadable") from error
    if len(model_bytes) != manifest.get("model_size"):
        raise BundleValidationError("model file size does not match manifest")
    actual_checksum = hashlib.sha256(model_bytes).hexdigest()
    if actual_checksum != manifest.get("model_sha256"):
        raise BundleValidationError("model checksum does not match manifest")
    state = _read_json(directory / MODEL_FILENAME)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise BundleValidationError("unsupported model schema version")
    return manifest, state


def load_model_bundle(directory: str | os.PathLike[str]) -> WebSpellModel:
    manifest, state = _validate_bundle(Path(directory))
    tokenizer_state = state["tokenizer"]
    if tokenizer_state.get("type") != "simple-unicode-tokenizer":
        raise BundleValidationError("unsupported tokenizer type")
    tokenizer = SimpleTokenizer(str(tokenizer_state["normalization"]))
    if tokenizer.identity != tokenizer_state.get("identity"):
        raise BundleValidationError("tokenizer identity in model state is inconsistent")
    if tokenizer.identity != manifest.get("tokenizer_identity"):
        raise BundleValidationError("tokenizer identity does not match manifest")

    if state.get("storage") == "sqlite" or (Path(directory) / "statistics.sqlite3").is_file():
        from webspell.scalable.language_model import SQLiteBidirectionalLanguageModel
        from webspell.scalable.symspell import SQLiteSymSpellIndex
        db_name = state.get("sqlite_database", "statistics.sqlite3")
        db_path = Path(directory) / db_name
        if not db_path.is_file():
            raise BundleValidationError(f"SQLite database {db_name} is missing from bundle")
        lexicon = SQLiteSymSpellIndex(db_path)
        language_model = SQLiteBidirectionalLanguageModel(db_path)
    else:
        lexicon = TermLexicon({
            str(term): int(frequency) for term, frequency in state["lexicon"].items()
        })
        lm_state = state["language_model"]
        lm_config = LanguageModelConfig(**lm_state["config"])
        language_model = BidirectionalLanguageModel(
            _load_ngram(lm_state["forward"], lm_config),
            _load_ngram(lm_state["backward"], lm_config),
        )

    error_state = state["error_model"]
    error_model = SubstringErrorModel(ErrorModelConfig(**error_state["config"]))
    error_model._counts = defaultdict(Counter)
    for source, targets in error_state["counts"].items():
        error_model._counts[source] = Counter(
            {target: float(count) for target, count in targets.items()}
        )
    error_model._targets = set(error_state["targets"])

    variant_identity = state.get("variant_generator")
    if variant_identity is None:
        variant_generator = None
    elif variant_identity == "vietnamese-input/v1":
        variant_generator = VietnameseVariantGenerator()
    else:
        raise BundleValidationError("unsupported candidate variant generator")
    ranker = CandidateRanker(
        lexicon,
        error_model,
        language_model,
        CandidateConfig(**state["candidate_config"]),
        variant_generator,
    )

    lambda_state = state["lambdas"]
    lambdas = LambdaGrid(
        {
            (int(record["left"]), int(record["right"])): float(record["value"])
            for record in lambda_state["values"]
        },
        float(lambda_state["default"]),
    )

    confidence_state = state["confidence"]
    confidence = ConfidenceModel(ClassifierConfig(**confidence_state["config"]))
    confidence.with_suggestions.load_state(confidence_state["with_suggestions"])
    confidence.without_suggestions.load_state(confidence_state["without_suggestions"])
    confidence.autocorrect.load_state(confidence_state["autocorrect"])
    thresholds = confidence_state["thresholds"]
    confidence.with_threshold = float(thresholds["with_suggestions"])
    confidence.without_threshold = float(thresholds["without_suggestions"])
    confidence.autocorrect_threshold = float(thresholds["autocorrect"])
    return WebSpellModel(tokenizer, ranker, lambdas, confidence)
