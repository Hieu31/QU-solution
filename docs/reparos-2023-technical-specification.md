# ReparoS 2023 - Technical Specification

**Status:** Draft for implementation  
**Target package:** `src/reparos`  
**Reference:** Kakkar et al., *Search Query Spell Correction with Weak Supervision in E-commerce*, ACL Industry 2023  
**Objective:** Build a runnable, isolated ReparoS-style sequence-to-sequence spell corrector and benchmark it fairly against WebSpell.

## 1. Scope

The system covers clean-query selection, synthetic and weakly supervised data, SentencePiece, Transformer training, Base→C1→C2 curriculum, beam inference, Improvement/Regression evaluation, artifacts and leakage controls.

ReparoS must remain independent from `webspell`. Both methods may consume the same immutable source corpus and sealed evaluation sets, but must not import each other's model or training modules.

## 2. Reproduction profiles

### 2.1 `paper-reproduction`

Implements behavior described by the paper. Missing proprietary inputs remain visibly missing. Vietnamese-specific rules must not silently enter this profile.

Defensible claim: **reproduction of the published ReparoS architecture, curriculum and data contracts using substitute data**. It is not a numerical reproduction of Flipkart's results.

### 2.2 `xanh-sm-adaptation`

Extends the architecture for Vietnamese location search with Vietnamese phonetic/diacritic/IME errors, location aliases and address data. Every extension must be recorded in the model manifest and must not be reported as paper reproduction.

## 3. Configuration provenance

Every configuration field must be tagged:

- `published`: explicitly stated in the paper.
- `inferred`: strongly implied but not fixed.
- `local_default`: selected to make the implementation runnable.
- `data_derived`: measured or learned from supplied data.

| Setting | Value | Provenance |
|---|---:|---|
| Architecture | Transformer encoder-decoder | published |
| Base encoder/decoder layers | 1 / 1 | published |
| Attention heads | 8 | published |
| Hidden dimension | 128 | published |
| Tokenizer/vocabulary | SentencePiece / 8,000 | published |
| Optimizer | Adam | published |
| Base learning rate | 1.0 | published |
| Adam beta1/beta2/epsilon | 0.8 / 0.998 / 1e-8 | published |
| Fine-tuning learning rate | 0.0001 | published |
| Fine-tuning parameters | Update all | published |
| Beam width | 10 | published |
| Training toolkit | OpenNMT | published |
| Production inference | CTranslate2 conversion/runtime | published |
| Reported 1-layer CPU latency | <7 ms/query at concurrency 1 | published on paper hardware |

The paper does not disclose batch size, steps/epochs, dropout, FFN dimension, warmup, gradient clipping, label smoothing, SentencePiece model type, shared vocabulary choice, length penalty or checkpoint selection. Runnable values must be labeled `local_default`, never paper values.

## 4. Architecture

```text
Raw sources
├── clean-query candidates + frequency/CTR/category
├── error triples and phonetic resources
├── query sessions and click feedback
└── adjudicated evaluation queries
        │
        ▼
Data preparation
├── Base synthetic data
├── C1 complex errors
├── C2 weak feedback
└── Improvement/Regression sets
        │
        ▼
Shared SentencePiece 8K
        │
        ▼
Base checkpoint → C1 checkpoint → C2 checkpoint
        │
        ▼
Beam-10 inference → offline evaluation
```

## 5. Package structure

```text
src/reparos/
├── cli.py
├── config.py
├── manifests.py
├── data/
│   ├── schemas.py
│   ├── selection.py
│   ├── edit_model.py
│   ├── edit_noise.py
│   ├── compounding.py
│   ├── phonetic.py
│   ├── query_chain.py
│   ├── feedback.py
│   ├── curriculum.py
│   └── evaluation_sets.py
├── tokenization/sentencepiece.py
├── model/
│   ├── transformer.py
│   ├── decoding.py
│   └── checkpoint.py
├── serving/
│   ├── ctranslate2_export.py
│   ├── ctranslate2_runtime.py
│   └── benchmark.py
├── training/
│   ├── trainer.py
│   └── curriculum.py
└── evaluation/
    ├── metrics.py
    ├── runner.py
    └── report.py
```

The current monolithic `src/reparos/data.py` is transitional.

## 6. Data contracts

All text is UTF-8 and Unicode NFC. Preserve raw surface strings and put normalized values in separate columns.

### 6.1 Clean-query candidates

`clean_query_candidates.csv`:

```text
query_id,query,category,monthly_frequency,ctr,group_id
```

Paper profile selection:

1. Partition by category.
2. Rank by frequency and CTR.
3. Select top 20% per category.
4. Deduplicate by normalized query.
5. Preserve `group_id` for splitting.

OSM fallback may map category from entity type and group from canonical POI identity. Since OSM has no CTR, record `substitute_selection=osm_without_ctr`.

### 6.2 Error triples

`error_triples.csv`:

```text
intended_word,observed_word,frequency,source
```

Edit candidates must be sampled proportional to learned Brill-Moore error score when triples exist. Uniform generation is an explicit fallback only.

### 6.3 Phonetic pairs

`phonetic_pairs.csv`:

```text
correct_word,phonetic_variant,frequency,source,language_path
```

`language_path` examples: `en-hi-en`, `vi-ime`, `soundex`, `metaphone`, `human_log`.

### 6.4 Query sessions

`query_events.csv`:

```text
session_id,event_index,query,timestamp,results_count,clicks,ctr
```

Mined reformulations:

```text
observed_query,intended_query,count,llr,pmi,edit_distance,source
```

Session IDs must be pseudonymous. LLR, PMI, CTR and edit thresholds are local because the paper does not disclose them.

### 6.5 Weak feedback

`weak_feedback.csv`:

```text
event_id,user_query,system_correction,correction_model_version,results_shown,clicks,corrected_query_ctr,clicked_back
```

Default positive filter: changed query, results shown, CTR above configured threshold, and no clickback when available. Threshold selection uses development data only.

### 6.6 Human-labelled evaluation

`labeled_queries.csv`:

```text
query_id,noisy_query,correct_query,monthly_frequency,participant_id,source_text_id,review_status
```

Only `review_status=adjudicated` is accepted. Evaluation rows are forbidden from tokenizer, vocabulary, training, fine-tuning and threshold selection.

## 7. Splitting and leakage

All variants of one clean query stay in one split. Group keys include canonical POI/address identity, normalized clean query and source entity. Human-typed groups also include participant and source location.

Default split is 80/10/10. Final human-typed and Improvement/Regression tests remain sealed. Store SHA-256, row counts and schema version in every model artifact.

## 8. Synthetic data

### 8.1 Edit errors

Support deletion, adjacent transposition, insertion and keyboard-neighbour replacement. Use a configured keyboard layout, not a generic alphabet. When error triples exist, sample proportional to error-model score; otherwise mark `edit_sampler=uniform_fallback`.

### 8.2 Compounding

Generate joins (`ball pen→ballpen`) and splits (`backpack→back pack`). Prefer observed boundary evidence or vocabulary frequency. Random splitting is fallback-only.

### 8.3 Phonetic errors

Paper profile:

1. Train source→native transliteration.
2. Transliterate to native script.
3. Apply native-script confusions from reformulations.
4. Back-transliterate.
5. Add Soundex/Metaphone candidates.
6. Deduplicate and frequency-weight.

Xanh SM adaptation may use Vietnamese diacritics, Telex/VNI leakage, data-supported regional confusions, aliases and user reformulations. These are adaptation sources, not paper-exact generation.

### 8.4 Combined errors and clean controls

C1 combines edit/phonetic errors with compounding. Corrupt at most two words per query. Base includes `(correct,correct)` pairs; the unpublished ratio is configured and selected using Regression development performance.

## 9. Curriculum

### Base

Train from random initialization on edit, compounding, phonetic, some combined errors and clean controls.

### C1

Fine-tune the selected Base checkpoint only on edit/phonetic+compounding examples. Update all parameters with LR `0.0001`.

### C2

Fine-tune the selected C1 checkpoint on weak click feedback. Update all parameters with LR `0.0001`. If feedback is unavailable, record C2 as `not_available`; never replace it silently with synthetic data.

## 10. SentencePiece

Train only from training-side source and target text. Use vocabulary size 8,000; explicit PAD/BOS/EOS/UNK IDs; deterministic normalization; and save model, vocabulary, command and corpus hashes. Evaluation text must never enter tokenizer training. Shared source/target vocabulary is a `local_default` because the paper does not specify it.

## 11. Transformer

### Architecture

- One-layer Transformer encoder and decoder.
- Hidden size 128 and eight heads.
- Autoregressive decoder with cross-entropy loss ignoring PAD.
- Configurable FFN, dropout, label smoothing and positional encoding, all marked local.

### Training

Support deterministic seeds, length-aware batching, gradient accumulation, clipping, mixed precision, validation loss/query accuracy, best/last checkpoints, resume and CPU smoke tests.

Published Base LR `1.0` likely relies on an OpenNMT schedule. Do not use constant LR 1.0 without an explicit experiment. Record the schedule as `local_default`.

Checkpoints store weights, optimizer/scheduler, step/epoch, resolved config, tokenizer/dataset checksums, git revision and seeds. Reject curriculum checkpoints with incompatible tokenizer or architecture.

## 12. Inference

Two inference backends are required:

1. `reference`: the training-framework decoder, used for correctness and checkpoint debugging.
2. `ctranslate2`: converted production artifact, used for CPU latency and deployment evaluation.

Both backends support greedy diagnostics and beam search width 10. Return:

```text
input_query,top1_query,hypotheses,sequence_scores,changed,latency_ms,model_stage
```

Maximum length, EOS, length penalty and duplicate handling are explicit config.

### 12.1 CTranslate2 export

The selected Base/C1/C2 checkpoint must be convertible into a standalone CTranslate2 model. Export stores:

- Converted model files.
- SentencePiece model and vocabulary.
- Source checkpoint SHA-256.
- Converter/CTranslate2 versions.
- Compute type (`float32`, `float16`, `int8_float32` or selected supported type).
- Beam and decoding defaults.

Conversion must fail when the checkpoint architecture or tokenizer is unsupported. The original training checkpoint remains the source of truth and must never be overwritten by conversion.

### 12.2 Parity requirements

Before accepting an exported model, run the same fixed query suite through both backends:

- Encoded source IDs must match.
- Greedy top-1 output must match exactly.
- Beam-10 top-1 output must match, except when documented numeric quantization changes ranking.
- Top-k hypotheses and scores must meet configured parity tolerances.
- Clean-query preservation decisions must not regress beyond the approved tolerance.

Quantized and non-quantized artifacts are evaluated separately.

### 12.3 Latency benchmark

Benchmark CTranslate2 on CPU with warm-up and report p50/p95/p99, throughput, concurrency, thread counts, compute type, CPU model and maximum memory. Run at least concurrency 1 because that is the paper's reported setting.

The paper reports reducing one-layer CPU inference from about 25 ms to below 7 ms/query on an Intel x86 64-bit 2.1 GHz VM. This is historical evidence, not a portable acceptance threshold. Local launch criteria must be based on the target Xanh SM serving hardware.

## 13. Evaluation

Sort unique monthly queries by frequency and split into equal head/tail quantiles:

- Regression: 90% head, 10% tail.
- Improvement: 10% head, 90% tail.

Create both deterministically from adjudicated labels. Report Base/C1/C2 separately. Primary paper metric is exact corrected queries divided by total queries.

For a fair WebSpell comparison, also report clean preservation, clean false-correction rate, E1-E5/CER/FER/TER where alignment is valid, top-k oracle accuracy, p50/p95/p99 latency, peak memory and artifact size. Report split/merge cases separately.

## 14. Optional production ranker

The deployed paper system has a second-stage ML ranker, but features and labels are undisclosed. Core offline reproduction ends at NMT beam output. A local ranker is a separate `xanh-sm-ranker`; always report results with and without it.

The initial milestone benchmarks ReparoS-Base without C1, C2 or the ranker. If Base top-10 accuracy is substantially stronger than top-1, a separately trained reranker becomes the next candidate experiment. It must not be bundled into the Base result because the paper excludes the production ranker from its offline comparison.

## 15. CLI contract

```powershell
uv run reparos prepare `
  --profile paper-reproduction `
  --clean-queries data\reparos\clean-query-candidates.csv `
  --error-triples data\reparos\error-triples.csv `
  --phonetic-pairs data\reparos\phonetic-pairs.csv `
  --weak-feedback data\reparos\weak-feedback.csv `
  --output data\reparos\prepared-v1

uv run reparos train-tokenizer `
  --data data\reparos\prepared-v1 `
  --output artifacts\reparos\tokenizer-v1 `
  --vocab-size 8000

uv run reparos train --stage base `
  --data data\reparos\prepared-v1 `
  --tokenizer artifacts\reparos\tokenizer-v1 `
  --output artifacts\reparos\base-v1

uv run reparos train --stage c1 `
  --initialize-from artifacts\reparos\base-v1\best.ckpt `
  --data data\reparos\prepared-v1 `
  --tokenizer artifacts\reparos\tokenizer-v1 `
  --output artifacts\reparos\c1-v1

uv run reparos train --stage c2 `
  --initialize-from artifacts\reparos\c1-v1\best.ckpt `
  --data data\reparos\prepared-v1 `
  --tokenizer artifacts\reparos\tokenizer-v1 `
  --output artifacts\reparos\c2-v1

uv run reparos predict `
  --model artifacts\reparos\c2-v1 `
  --text "san bay noi bai" `
  --beam-size 10 `
  --backend reference

uv run reparos export-ctranslate2 `
  --model artifacts\reparos\c2-v1 `
  --output artifacts\reparos\c2-v1-ct2 `
  --compute-type int8_float32

uv run reparos predict `
  --model artifacts\reparos\c2-v1-ct2 `
  --text "san bay noi bai" `
  --beam-size 10 `
  --backend ctranslate2

uv run reparos benchmark `
  --model artifacts\reparos\c2-v1-ct2 `
  --queries data\reparos\benchmark-queries.txt `
  --backend ctranslate2 `
  --concurrency 1

uv run reparos evaluate `
  --model artifacts\reparos\c2-v1 `
  --improvement data\reparos\eval-v1\improvement.csv `
  --regression data\reparos\eval-v1\regression.csv `
  --output artifacts\reparos\c2-v1\evaluation.json
```

Missing optional sources appear in manifests; no implicit fallback unless `--allow-fallback` is passed.

## 16. Artifact layout

```text
artifacts/reparos/<run>/
├── manifest.json
├── resolved-config.json
├── dataset-manifest.json
├── tokenizer.model
├── tokenizer.vocab
├── best.ckpt
├── last.ckpt
├── training-history.jsonl
├── validation-metrics.json
├── evaluation.json
├── latency.json
└── ctranslate2/
    ├── model.bin
    ├── config.json
    ├── shared_vocabulary.txt
    ├── tokenizer.model
    └── conversion-manifest.json
```

Artifacts must load without training data.

## 17. Tests

Unit tests cover deterministic noise, keyboard neighbours, weighted sampling, SentencePiece round trips, masks/shapes, beam termination, checkpoint equality, curriculum compatibility, 90/10 composition and metrics.

Integration tests cover tiny-corpus overfit, CPU Base→C1→C2, resume, artifact prediction, sealed evaluation and absence of cross-imports between `reparos` and `webspell`.

CTranslate2 integration tests cover checkpoint conversion, load in a fresh process, reference/runtime output parity, beam-10 parity, quantized-regression tolerance and a non-gating latency smoke benchmark.

Data tests reject duplicate IDs, non-adjudicated rows, train/evaluation overlap, tokenizer leakage, C2 without feedback and hash mismatches.

## 18. Acceptance criteria

Branch B is runnable only when:

1. `prepare` creates validated Base/C1/C2 data and manifests.
2. `train-tokenizer` creates a loadable 8K model.
3. Base creates resumable checkpoints.
4. C1 initializes from Base and C2 from C1.
5. `predict` returns beam-10 hypotheses from the reference backend.
6. A selected checkpoint exports to and loads from CTranslate2.
7. Reference and CTranslate2 parity tests pass.
8. `evaluate` reports Improvement/Regression separately.
9. Clean regression, top-k accuracy and CPU latency are included.
10. CPU smoke tests pass.
11. `reparos` has no runtime dependency on `webspell`.
12. Assumptions and missing private inputs appear in manifests.

Until then, describe Branch B as `data scaffold` or `partial reproduction`, not completed ReparoS.

## 19. Implementation order

1. Split `data.py`; finalize schemas/manifests.
2. Implement clean selection and production-grade generators.
3. Add SentencePiece.
4. Add Transformer, masks and checkpointing.
5. Add Base trainer and CPU smoke test.
6. Add C1/C2 validation and fine-tuning.
7. Add greedy and beam-10 reference inference.
8. Add CTranslate2 conversion, runtime, parity and benchmark.
9. Add Improvement/Regression evaluator.
10. Add shared WebSpell benchmark exporter.
11. Add Vietnamese/Xanh SM as a separate profile.
