"""Generate top-10 candidate data from the pinned Base V3 CTranslate2 export.

Run from anywhere with the project's Python environment:
    python scripts/build_reparos_base_v3_confidence_data.py

The exact-match label is a diagnostic proxy, not a production correctness label.
The three source suites are frozen diagnostics; using this output to fit a
confidence model consumes them for that purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reparos.architecture import DecodingConfig  # noqa: E402


SUITES = (
    ("reparos-diagnostic-10k", 10_000),
    ("reparos-compositional-4k", 4_000),
    ("reparos-user-centric-v2", 88),
)
DEFAULT_MODEL = (
    ROOT
    / "artifacts/reparos_base_v3_production_checkpoints"
    / "checkpoints/base_v3_production/ctranslate2_export"
)
DEFAULT_OUTPUT = ROOT / "artifacts/reparos_base_v3_confidence_14088"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_text(value: str) -> str:
    """Match the diagnostic policy: NFC + whitespace, preserving case."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def load_gold() -> tuple[list[tuple[str, dict]], dict[str, dict]]:
    rows: list[tuple[str, dict]] = []
    sources: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()
    for suite, expected_count in SUITES:
        path = ROOT / "benchmark" / suite / "gold.jsonl"
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("query_id", "input", "expected"):
                    if not isinstance(row.get(key), str):
                        raise ValueError(f"{path}:{line_number}: missing string {key}")
                identity = (suite, row["query_id"])
                if identity in seen:
                    raise ValueError(f"duplicate query_id in {suite}: {row['query_id']}")
                seen.add(identity)
                rows.append((suite, row))
                count += 1
        if count != expected_count:
            raise ValueError(f"{path}: expected {expected_count} rows, got {count}")
        sources[suite] = {"path": str(path.resolve()), "rows": count, "sha256": sha256(path)}
    return rows, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    model = args.model.resolve()
    required = ("model.bin", "config.json", "tokenizer.model", "conversion-manifest.json")
    if any(not (model / name).is_file() for name in required):
        parser.error(f"--model must be the CTranslate2 export containing {', '.join(required)}: {model}")
    conversion = json.loads((model / "conversion-manifest.json").read_text(encoding="utf-8"))
    tokenizer_hash = sha256(model / "tokenizer.model")
    if conversion.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("tokenizer.model does not match conversion-manifest.json")

    gold, sources = load_gold()
    decoding = DecodingConfig(beam_size=10, num_hypotheses=10)
    decoding.validate()

    # Use the same tokenizer and decoding options as CTranslate2Predictor.predict.
    import ctranslate2
    from reparos.tokenization import SentencePieceTokenizer

    tokenizer = SentencePieceTokenizer(model / "tokenizer.model").processor
    translator = ctranslate2.Translator(str(model), device=args.device, compute_type=args.compute_type)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "top10_predictions.jsonl"
    candidate_path = output / "candidate_labels.jsonl"
    prediction_tmp = output / "top10_predictions.jsonl.tmp"
    candidate_tmp = output / "candidate_labels.jsonl.tmp"
    exact_counts: Counter[str] = Counter()
    started = time.perf_counter()

    with prediction_tmp.open("w", encoding="utf-8") as predictions, candidate_tmp.open(
        "w", encoding="utf-8"
    ) as candidates:
        for offset in range(0, len(gold), args.batch_size):
            batch = gold[offset : offset + args.batch_size]
            inputs = [row["input"] for _, row in batch]
            tokens = [tokenizer.encode(text, out_type=str) for text in inputs]
            batch_started = time.perf_counter()
            results = translator.translate_batch(
                tokens,
                beam_size=decoding.beam_size,
                patience=decoding.patience,
                num_hypotheses=decoding.num_hypotheses,
                length_penalty=decoding.ctranslate2_length_penalty,
                coverage_penalty=decoding.coverage_penalty,
                no_repeat_ngram_size=decoding.no_repeat_ngram_size,
                disable_unk=decoding.disable_unk,
                min_decoding_length=decoding.min_decoding_length,
                max_decoding_length=decoding.max_decoding_length,
                return_scores=True,
            )
            if len(results) != len(batch):
                raise ValueError(f"batch at offset {offset}: expected {len(batch)} results, got {len(results)}")
            batch_ms_per_query = 1000 * (time.perf_counter() - batch_started) / len(batch)
            for (suite, gold_row), result in zip(batch, results):
                hypotheses = [tokenizer.decode(parts) for parts in result.hypotheses]
                scores = [float(score) for score in result.scores]
                if len(hypotheses) != 10 or len(scores) != 10:
                    raise ValueError(f"{suite}/{gold_row['query_id']}: expected exactly 10 hypotheses and scores")
                gold_text = strict_text(gold_row["expected"])
                labels = [int(strict_text(text) == gold_text) for text in hypotheses]
                exact_counts[suite] += sum(labels)
                record = {
                    "suite": suite,
                    "query_id": gold_row["query_id"],
                    "error_type": gold_row.get("error_type"),
                    "entity_id": gold_row.get("entity_id"),
                    "group_id": gold_row.get("group_id"),
                    "input": gold_row["input"],
                    "model_input": gold_row["input"],
                    "expected": gold_row["expected"],
                    "hypotheses": hypotheses,
                    "sequence_scores": scores,
                    "strict_exact_labels": labels,
                    "latency_ms_amortized_batch": batch_ms_per_query,
                }
                predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
                for rank, (hypothesis, score, label) in enumerate(zip(hypotheses, scores, labels), 1):
                    candidates.write(json.dumps({
                        "suite": suite,
                        "query_id": gold_row["query_id"],
                        "entity_id": gold_row.get("entity_id"),
                        "group_id": gold_row.get("group_id"),
                        "error_type": gold_row.get("error_type"),
                        "input": gold_row["input"],
                        "expected": gold_row["expected"],
                        "rank": rank,
                        "hypothesis": hypothesis,
                        "sequence_score": score,
                        "strict_exact_label": label,
                    }, ensure_ascii=False) + "\n")
            print(f"{min(offset + len(batch), len(gold))}/{len(gold)}", end="\r", flush=True)

    prediction_tmp.replace(prediction_path)
    candidate_tmp.replace(candidate_path)
    manifest = {
        "schema_version": 1,
        "purpose": "candidate-level relative confidence development",
        "label_policy": "NFC and whitespace-normalized exact match to gold; case and punctuation preserved; other valid rewrites may be labeled 0",
        "warning": "These diagnostic suites are used for confidence development and are not independent confidence test sets.",
        "model_export": str(model),
        "model_bin_sha256": sha256(model / "model.bin"),
        "config_sha256": sha256(model / "config.json"),
        "tokenizer_sha256": tokenizer_hash,
        "conversion_manifest": conversion,
        "device": args.device,
        "compute_type": args.compute_type,
        "ctranslate2_version": ctranslate2.__version__,
        "decoding": asdict(decoding),
        "preprocessing": "raw input exactly as stored in gold.jsonl; SentencePiece encodes it directly",
        "queries": len(gold),
        "candidate_rows": len(gold) * 10,
        "strict_exact_positive_candidates_by_suite": dict(exact_counts),
        "sources": sources,
        "outputs": {
            "top10_predictions": str(prediction_path),
            "candidate_labels": str(candidate_path),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {len(gold)} queries and {len(gold) * 10} candidate rows to {output}")


if __name__ == "__main__":
    main()
