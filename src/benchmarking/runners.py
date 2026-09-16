from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from benchmarking.io import normalize_text, write_jsonl
from reparos.architecture import DecodingConfig


def load_decoding(path: str | Path | None) -> DecodingConfig:
    if path is None:
        return DecodingConfig()
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = DecodingConfig.__dataclass_fields__.keys()
    return DecodingConfig(**{key: state[key] for key in allowed if key in state})


def run_webspell(gold: list[dict], model_path: str | Path, output: str | Path) -> Path:
    from webspell.pipeline.model import WebSpellModel
    rows = []
    with WebSpellModel.load(str(model_path)) as model:
        for item in gold:
            started = time.perf_counter()
            predictions = model.predict(item["input"])
            corrected = model.corrected_text(item["input"], predictions)
            token_candidates = [[candidate.term for candidate in prediction.candidates] for prediction in predictions]
            rows.append({
                "query_id": item["query_id"], "error_type": item["error_type"],
                "input": item["input"], "expected": item["expected"],
                "output": normalize_text(corrected), "hypotheses": [normalize_text(corrected)], "scores": [],
                "latency_ms": (time.perf_counter() - started) * 1000,
                "token_candidates": token_candidates, "backend": "webspell",
            })
    return write_jsonl(output, rows)


def run_ctranslate2(
    gold: list[dict], model_path: str | Path, output: str | Path, *,
    decoding: DecodingConfig, device: str = "cpu", compute_type: str = "default",
) -> Path:
    from reparos.serving.ctranslate2 import CTranslate2Predictor
    predictor = CTranslate2Predictor(model_path, device=device, compute_type=compute_type)
    rows = []
    for item in gold:
        result = predictor.predict(item["input"], decoding=decoding)
        hypotheses = [normalize_text(str(value)) for value in result["hypotheses"]]
        rows.append({
            "query_id": item["query_id"], "error_type": item["error_type"],
            "input": item["input"], "expected": item["expected"],
            "output": hypotheses[0], "hypotheses": hypotheses,
            "scores": list(result["sequence_scores"]), "latency_ms": result["latency_ms"],
            "backend": "ctranslate2",
        })
    return write_jsonl(output, rows)


def run_opennmt(
    gold: list[dict], checkpoint: str | Path, tokenizer: str | Path, output: str | Path, *,
    decoding: DecodingConfig,
) -> Path:
    from reparos.training.opennmt import translate_opennmt
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, destination = root / "queries.txt", root / "translations.txt"
        source.write_text("\n".join(item["input"] for item in gold) + "\n", encoding="utf-8")
        started = time.perf_counter()
        translate_opennmt(checkpoint, tokenizer, source, destination, decoding=decoding)
        elapsed = (time.perf_counter() - started) * 1000
        lines = destination.read_text(encoding="utf-8").splitlines()
    expected = len(gold) * decoding.num_hypotheses
    if len(lines) != expected:
        raise ValueError(f"OpenNMT emitted {len(lines)} lines; expected {expected}")
    per_query = elapsed / max(len(gold), 1)
    rows = []
    for index, item in enumerate(gold):
        start = index * decoding.num_hypotheses
        hypotheses = [normalize_text(value) for value in lines[start:start + decoding.num_hypotheses]]
        rows.append({
            "query_id": item["query_id"], "error_type": item["error_type"],
            "input": item["input"], "expected": item["expected"],
            "output": hypotheses[0], "hypotheses": hypotheses, "scores": [],
            "latency_ms": per_query, "latency_note": "amortized subprocess batch; not official batch=1 latency",
            "backend": "opennmt",
        })
    return write_jsonl(output, rows)


def parity_rows(reference: list[dict], converted: list[dict]) -> dict:
    right = {row["query_id"]: row for row in converted}
    counters = {"top1": 0, "ordered_topk": 0, "topk_set": 0}
    records = []
    for left in reference:
        other = right.get(left["query_id"])
        if other is None: raise ValueError(f"missing CTranslate2 prediction {left['query_id']}")
        lhs = [normalize_text(str(value)) for value in left.get("hypotheses", [])]
        rhs = [normalize_text(str(value)) for value in other.get("hypotheses", [])]
        top1 = bool(lhs and rhs and lhs[0] == rhs[0])
        ordered, as_set = lhs == rhs, set(lhs) == set(rhs)
        counters["top1"] += top1; counters["ordered_topk"] += ordered; counters["topk_set"] += as_set
        if not ordered:
            category = "top1_mismatch" if not top1 else "order_only" if as_set else "candidate_set_mismatch"
            records.append({"query_id": left["query_id"], "category": category, "opennmt": lhs, "ctranslate2": rhs})
    total = len(reference)
    return {"queries": total, **{f"{key}_parity": value / total if total else 0.0 for key, value in counters.items()}, "mismatches": records}
