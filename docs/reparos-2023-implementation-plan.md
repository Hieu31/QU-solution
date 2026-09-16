# ReparoS 2023 - Implementation Plan

**Status:** Approved for implementation  
**Source spec:** `docs/reparos-2023-technical-specification.md`  
**Package boundary:** `src/reparos` only  
**Rule:** Do not start a full training job automatically. Each phase must pass its gate before the next phase begins.

## 1. Delivery strategy

Implement a thin end-to-end vertical slice first:

```text
tiny parallel data
→ SentencePiece
→ one-layer Transformer
→ train/checkpoint
→ beam inference
→ exact-accuracy evaluation
→ CTranslate2 export
```

After this path is reliable, expand synthetic generation, curriculum and production-scale data. This prevents data work from hiding model/runtime defects.

### Approved MVP scope: ReparoS-Base only

The first benchmark stops at the Base model. C1, C2 and the production ML reranker are deferred.

The initial comparison is:

```text
WebSpell paper baseline
vs
ReparoS-Base (NMT top-1 and beam top-10, no reranker)
```

Rationale:

- Table 1 reports ReparoS-Base independently before C1/C2 curriculum fine-tuning.
- The paper states that its production second-stage ML ranker was removed from the offline results discussion.
- Adding C1/C2 or a reranker before measuring Base would make it impossible to attribute gains to the core seq2seq method.
- C2 additionally requires genuine intervention/click-feedback data that are not currently available.

## 2. Phase overview

| Phase | Outcome | Blocking gate |
|---|---|---|
| 0 | Package foundation and dependency isolation | Both CLIs import independently |
| 1 | Versioned schemas, manifests and leakage checks | Invalid/leaking data is rejected |
| 2 | SentencePiece 8K pipeline | Deterministic encode/decode artifact |
| 3 | Runnable Transformer Base model | Tiny corpus can overfit |
| 4 | Checkpoint, resume and beam inference | Reloaded predictions are identical |
| 5 | Improvement/Regression evaluator | Metrics verified by fixtures |
| 6 | Base CTranslate2 serving | Conversion and parity pass |
| 7 | Base benchmark decision gate | Base compared fairly with WebSpell |
| 8 | Paper-style data generators | Distributions and provenance audited |
| 9 | Deferred curriculum Base→C1→C2 | Implement only after Base decision |
| 10 | Deferred ranker/Xanh SM production work | Separate experiment and report |

## 3. Phase 0 - Foundation

### Tasks

- Add `reparos.config`, `reparos.manifests` and subpackages from the spec.
- Define optional extras for training and serving separately.
- Keep base data utilities importable without Torch, SentencePiece or CTranslate2.
- Add CLI commands as stubs with actionable missing-dependency errors.
- Add an import-boundary test prohibiting runtime imports between `reparos` and `webspell`.

### Dependency groups

```toml
reparos-train = [torch, sentencepiece]
reparos-serve = [ctranslate2, sentencepiece]
reparos = [torch, sentencepiece, ctranslate2]
```

Exact compatible versions must be resolved and locked during implementation, not guessed in this plan.

### Gate

- `uv run webspell --help` works without ReparoS extras.
- `uv run reparos --help` works without importing WebSpell.
- Import-boundary and CLI smoke tests pass.

## 4. Phase 1 - Data contracts and manifests

### Tasks

- Split the current monolithic `data.py`.
- Implement typed records for clean queries, parallel examples, error triples, phonetic pairs, feedback and human labels.
- Normalize text with NFC while retaining raw text.
- Implement deterministic group split and SHA-256 dataset fingerprints.
- Detect duplicate IDs and train/dev/test group overlap.
- Add provenance tags: `published`, `inferred`, `local_default`, `data_derived`.
- Change implicit fallbacks into explicit `--allow-fallback` behavior.

### Outputs

- `dataset-manifest.json`
- Base/C1/C2 `.src`, `.tgt`, `.meta.jsonl`
- Validation and leakage report

### Gate

- Reordered source rows produce identical split membership.
- Leakage fixtures fail loudly.
- Manifest hashes change when any source row or config changes.

## 5. Phase 2 - SentencePiece

### Tasks

- Train a shared source/target SentencePiece model from training data only.
- Default vocabulary size to 8,000; allow smaller vocabulary for smoke tests.
- Fix PAD/BOS/EOS/UNK IDs.
- Implement batch encode/decode and maximum-length validation.
- Persist tokenizer config, checksums and corpus fingerprint.

### CLI

```powershell
uv run reparos train-tokenizer `
  --data data\reparos\prepared-v1 `
  --output artifacts\reparos\tokenizer-v1 `
  --vocab-size 8000
```

### Gate

- Vietnamese and ASCII round-trip fixtures pass.
- Same seed/data produce identical tokenizer artifacts where supported.
- Test/evaluation files are proven absent from tokenizer inputs.

## 6. Phase 3 - Transformer Base model

### Tasks

- Implement one-layer encoder/decoder, hidden 128, eight heads.
- Implement embeddings, positional encoding, padding masks and causal masks.
- Implement teacher forcing and PAD-masked cross-entropy.
- Implement Adam with published beta/epsilon values.
- Put FFN, dropout, scheduler, batch and clipping under local config.
- Support CPU and CUDA mixed precision.

### First milestone

Overfit a tiny 20-100 pair dataset. Do not optimize throughput before this succeeds.

### Gate

- Tensor/mask unit tests pass.
- Training loss decreases on a deterministic fixture.
- Tiny training query exact accuracy reaches the configured overfit threshold.
- No evaluation set is loaded by the trainer.

## 7. Phase 4 - Checkpoint and reference inference

### Tasks

- Save best/last checkpoints with optimizer, scheduler, tokenizer and dataset lineage.
- Implement resume with deterministic continuation.
- Implement greedy decoding followed by beam search width 10.
- Return hypotheses, scores, changed flag and latency.
- Reject incompatible tokenizer/model checkpoints.

### Gate

- Prediction before and after reload is identical.
- Resumed training matches uninterrupted training within tolerance.
- Beam search terminates correctly on EOS/max length.
- Beam size 1 agrees with greedy decoding.

## 8. Phase 5 - Evaluator

### Tasks

- Finalize deterministic Improvement/Regression construction.
- Implement query exact accuracy and top-10 oracle accuracy.
- Implement clean preservation and clean false-correction rate.
- Reuse definitions, not code dependencies, for E1-E5/CER/FER/TER.
- Report structural split/merge cases separately.
- Add latency, memory and artifact-size fields.

### Gate

- Hand-calculated metric fixtures match exactly.
- Regression set contains 90/10 head/tail; Improvement contains 10/90.
- Evaluator never mutates or samples the sealed input.

## 9. Phase 6 - Deferred curriculum Base→C1→C2

**MVP status:** Deferred until the Base benchmark is reviewed.

### Tasks

- Base trains from random initialization.
- C1 loads selected Base and trains only on complex+compounding data.
- C2 loads selected C1 and trains only on validated weak feedback.
- Fine-tuning uses published LR `0.0001` and updates all parameters.
- Record parent checkpoint hashes and dataset hashes.
- Report every stage on the same validation/evaluation IDs.

### Gate

- Invalid stage lineage is rejected.
- C2 is unavailable, not synthesized, when feedback is absent.
- Stage comparison report exposes improvement and regression deltas.
- Clean overcorrection changes are visible before model promotion.

## 10. Phase 7 - Base CTranslate2

### Tasks

- Convert the selected Base checkpoint without overwriting source artifacts.
- Package SentencePiece with the converted model.
- Support float32 first, then an approved CPU compute type such as int8/float32.
- Implement reference/CTranslate2 greedy and beam-10 parity suites.
- Add warm-up and p50/p95/p99 benchmark at concurrency 1 and configured higher concurrency.
- Record CPU, threads, compute type, CTranslate2 version, memory and throughput.

### CLI

```powershell
uv run reparos export-ctranslate2 `
  --model artifacts\reparos\base-v1 `
  --output artifacts\reparos\base-v1-ct2 `
  --compute-type int8_float32
```

### Gate

- Converted model loads in a fresh process.
- Float32 top-1 parity passes.
- Quantized ranking changes remain within approved regression tolerance.
- Latency is measured on target hardware; `<7 ms` from the paper is contextual, not a universal gate.

## 11. Phase 8 - Paper-style data generation

### Tasks

- Replace generic replacement with keyboard-neighbour maps.
- Implement Brill-Moore-weighted edit sampling from supplied triples.
- Implement evidence-based joins/splits.
- Implement query-chain mining with configurable LLR/PMI/CTR/edit guardrails.
- Implement phonetic resource ingestion and Soundex/Metaphone candidates.
- Add transliteration model interface; keep proprietary English-Hindi resources marked missing.
- Audit error-class, query-length and clean/error distributions.

### Gate

- Sampling distributions match configured weights within statistical tolerance.
- No generator reads evaluation data.
- Every output row has source, error class, generator version and seed.

## 12. Phase 9 - Xanh SM adaptation

### Tasks

- Add Vietnamese diacritic, wrong-tone, Telex/VNI, abbreviation and location-alias sources.
- Derive weights from real logs when available.
- Keep structural errors in seq2seq training.
- Add location-category and seen/unseen-POI evaluation slices.
- Never enable these sources in `paper-reproduction`.

### Gate

- Profile isolation tests pass.
- Synthetic distributions are documented.
- Typed-test improvement does not hide clean-query regression.

## 13. Phase 10 - Final benchmark

### Model matrix

- WebSpell paper baseline.
- ReparoS Base.
- ReparoS C1/C2 only in a later curriculum experiment.
- Xanh SM adaptations reported separately.

### Fairness rules

- Same sealed query IDs and gold corrections.
- No shared learned artifacts between methods.
- Same normalization contract at metric time.
- Same target hardware and concurrency for latency.
- Report accuracy, false positives, coverage, memory, size and latency together.

### Promotion gate

No production recommendation until the model meets explicit limits for clean false corrections, Improvement accuracy, Regression accuracy, p95 latency and memory on target hardware.

### Base decision gate

After the first benchmark, choose one of three actions:

1. **Stop ReparoS:** Base is materially worse than WebSpell and its top-10 oracle does not show recoverable headroom.
2. **Add a reranker experiment:** Base top-10 is strong but top-1 is weak, indicating candidate selection rather than generation is the bottleneck.
3. **Proceed to C1/C2:** Base generation/top-1 is viable, but hard combined errors or clean-query regression are the dominant remaining issues and appropriate curriculum data exist.

Do not implement both reranking and curriculum simultaneously; benchmark each contribution separately.

## 14. Execution batches

### Batch 1 - Runnable core

Phases 0-5. Deliver a tiny but complete model that trains, predicts and evaluates.

### Batch 2 - Base serving and benchmark

Deliver Base CTranslate2, parity, latency and the Base-vs-WebSpell decision report.

### Batch 3 - Data fidelity or reranker

Select based on the Base decision gate: improve paper-style data or test a standalone reranker when top-10 headroom supports it.

### Batch 4 - Optional curriculum and product adaptation

Only after Base analysis, add C1/C2 or Vietnamese production adaptations as isolated experiments.

## 15. Definition of done

ReparoS is complete only when:

- All 12 acceptance criteria in the technical spec pass.
- CPU Base→C1→C2 smoke training passes in CI.
- A saved model predicts through both reference and CTranslate2 backends.
- Improvement/Regression and sealed typed-test reports are generated from immutable data.
- Missing proprietary data and all local assumptions remain visible in artifacts.
- WebSpell remains independently installable and runnable.
