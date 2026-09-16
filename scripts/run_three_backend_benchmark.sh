#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)-production-v1}"
RUN_DIR="benchmark/${RUN_ID}"

DATA_DIR="${DATA_DIR:-data/reparos/reparos-production-v1/base}"
WEBSPELL_MODEL="${WEBSPELL_MODEL:-artifacts/osm-model-production-v4-leakfree}"
OPENNMT_CHECKPOINT="${OPENNMT_CHECKPOINT:-artifacts/opennmt-production-kaggle-t4-v1/reparos_base_step_25000.pt}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-artifacts/tokenizer-production-v1/tokenizer.model}"
DECODING_CONFIG="${DECODING_CONFIG:-artifacts/opennmt-production-kaggle-t4-v1/decoding-config.json}"
CTRANSLATE2_MODEL="${CTRANSLATE2_MODEL:-artifacts/opennmt-production-kaggle-t4-v1-ctranslate2-float32}"
CTRANSLATE2_DEVICE="${CTRANSLATE2_DEVICE:-cpu}"
CTRANSLATE2_COMPUTE_TYPE="${CTRANSLATE2_COMPUTE_TYPE:-float32}"

echo "Benchmark run: ${RUN_ID}"
echo "Output: ${RUN_DIR}"

uv run benchmark-three prepare \
  --data "${DATA_DIR}" \
  --split test \
  --output "${RUN_DIR}/gold.jsonl"

uv run benchmark-three run-webspell \
  --gold "${RUN_DIR}/gold.jsonl" \
  --model "${WEBSPELL_MODEL}" \
  --output "${RUN_DIR}/predictions/webspell.jsonl"

uv run benchmark-three run-opennmt \
  --gold "${RUN_DIR}/gold.jsonl" \
  --checkpoint "${OPENNMT_CHECKPOINT}" \
  --tokenizer "${TOKENIZER_MODEL}" \
  --decoding-config "${DECODING_CONFIG}" \
  --output "${RUN_DIR}/predictions/opennmt.jsonl"

uv run benchmark-three run-ctranslate2 \
  --gold "${RUN_DIR}/gold.jsonl" \
  --model "${CTRANSLATE2_MODEL}" \
  --decoding-config "${DECODING_CONFIG}" \
  --device "${CTRANSLATE2_DEVICE}" \
  --compute-type "${CTRANSLATE2_COMPUTE_TYPE}" \
  --output "${RUN_DIR}/predictions/ctranslate2.jsonl"

uv run benchmark-three parity \
  --opennmt "${RUN_DIR}/predictions/opennmt.jsonl" \
  --ctranslate2 "${RUN_DIR}/predictions/ctranslate2.jsonl" \
  --output "${RUN_DIR}/parity.json"

uv run benchmark-three score \
  --gold "${RUN_DIR}/gold.jsonl" \
  --prediction "webspell=${RUN_DIR}/predictions/webspell.jsonl" \
  --prediction "opennmt=${RUN_DIR}/predictions/opennmt.jsonl" \
  --prediction "ctranslate2=${RUN_DIR}/predictions/ctranslate2.jsonl" \
  --output "${RUN_DIR}/quality.json"

uv run benchmark-three resources \
  --prediction "webspell=${RUN_DIR}/predictions/webspell.jsonl" \
  --prediction "opennmt=${RUN_DIR}/predictions/opennmt.jsonl" \
  --prediction "ctranslate2=${RUN_DIR}/predictions/ctranslate2.jsonl" \
  --artifact "webspell=${WEBSPELL_MODEL}" \
  --artifact "opennmt=${OPENNMT_CHECKPOINT}" \
  --artifact "ctranslate2=${CTRANSLATE2_MODEL}" \
  --output "${RUN_DIR}/resources.json"

echo
echo "Benchmark completed: ${RUN_DIR}"
echo "Report: ${RUN_DIR}/report.md"
cat "${RUN_DIR}/report.md"
