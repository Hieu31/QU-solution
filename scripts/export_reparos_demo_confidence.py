"""Fit the selected nonlinear confidence model for the pinned V3 demo.

This fits only the development side of the benchmark split. The held-out test
queries remain excluded from the saved demo model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn

from compare_reparos_confidence_methods import DEFAULT_INPUT, group_hash
from compare_reparos_nonlinear_confidence import build_model
from reparos_confidence_features import ALL_NAMES, candidate_feature_rows


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/reparos_base_v3_confidence_14088/demo_hgb_deep"
V3_EXPORT = ROOT / "artifacts/reparos_base_v3_production_checkpoints/checkpoints/base_v3_production/ctranslate2_export"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    rows, labels = [], []
    dev_queries = 0
    with DEFAULT_INPUT.open("r", encoding="utf-8") as stream:
        for line in stream:
            query = json.loads(line)
            if group_hash(7193, query["expected"]) < 0.2:
                continue
            rows.append(candidate_feature_rows(
                query["input"], query["hypotheses"], query["sequence_scores"]
            ))
            labels.extend(query["strict_exact_labels"])
            dev_queries += 1
    x = np.vstack(rows)
    y = np.asarray(labels, dtype=np.int8)
    model = build_model("hgb_deep")
    model.fit(x, y)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT / "model.joblib"
    joblib.dump(model, model_path)
    manifest = {
        "schema_version": 1,
        "task": "candidate strict-gold confidence for ReparoS Base V3 demo",
        "model_kind": "hgb_deep",
        "sklearn_version": sklearn.__version__,
        "feature_names": ALL_NAMES,
        "training_queries": dev_queries,
        "training_candidates": int(len(y)),
        "data_sha256": digest(DEFAULT_INPUT),
        "v3_model_bin_sha256": digest(V3_EXPORT / "model.bin"),
        "v3_tokenizer_sha256": digest(V3_EXPORT / "tokenizer.model"),
        "decoding": {
            "beam_size": 10, "num_hypotheses": 10,
            "repetition_penalty": 1.0, "length_penalty": 0.0,
            "compute_type": "default", "preprocessing": "raw input",
        },
        "selection": "5-fold CV Brier among exploratory nonlinear models",
        "scope": "gold-exact on diagnostic suites; uncalibrated for production traffic",
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {model_path} from {dev_queries} development queries")


if __name__ == "__main__":
    main()
