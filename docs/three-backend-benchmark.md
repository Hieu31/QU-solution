# Three-backend production benchmark

## Systems under test

1. WebSpell production: independent statistical correction system.
2. ReparoS OpenNMT-py: training checkpoint and reference decoder.
3. ReparoS CTranslate2: converted deployment decoder.

OpenNMT-py and CTranslate2 share weights and tokenizer. They are reported as
separate operational backends for parity and serving measurements, but only one
ReparoS method when comparing model quality against WebSpell.

## Frozen inputs

- Synthetic production test: `reparos-production-v1/base/test.*`.
- Human-typed Improvement set: erroneous queries requiring correction.
- Human-typed Regression set: valid queries and hard negatives that must stay
  unchanged.
- The same normalized input string and gold target are sent to all backends.
- Split membership and `error_type` come from the frozen metadata. Never tune a
  threshold or decoding option on these test files.

## Execution stages

1. Validate artifact hashes, tokenizer identity, dataset row alignment, and
   zero query-group overlap.
2. Run a small deterministic smoke suite through all three backends.
3. Run a stratified development set to select WebSpell thresholds and ReparoS
   decoding settings.
4. Freeze settings and emit one JSONL prediction file per backend.
5. Score all prediction files with one backend-independent evaluator.
6. Run OpenNMT-to-CTranslate2 parity on top-1, ordered top-k, and top-k sets.
7. Measure serving performance after warmup on the intended production device.
8. Publish the aggregate report plus per-error-type and per-query regressions.

## Common quality metrics

- Clean preservation and false-correction rate.
- Query-level detection precision, recall, and F1.
- Exact query accuracy on noisy queries.
- Autocorrect precision, recall, and F1.
- Character and token error rate before and after correction.
- Improvement, unchanged, and regression rates.
- Micro and macro averages by `error_type`.

Report every common metric for `clean`, `keyboard_edit`, full and partial
missing diacritics, wrong diacritics, Telex, VNI, word boundary, address
abbreviation, address symbols, and combined errors.

## Ranking diagnostics

- ReparoS: top-1 accuracy, recall at 3/5/10, and MRR from query hypotheses.
- WebSpell: candidate recall and token rank separately; do not present its
  token candidates as query-level top-k.
- Final output metrics remain directly comparable for all three backends.

## OpenNMT/CTranslate2 parity gates

- Greedy top-1 parity is checked first.
- Beam top-1 parity, ordered top-k parity, and top-k set parity follow.
- Every top-1 mismatch is rescored against gold and classified as converted
  better, reference better, or both wrong.
- Production quality is always measured on CTranslate2 because it is the
  deployed backend. OpenNMT metrics are the reference diagnostic.

## Performance protocol

- Record cold start separately from warm inference.
- Warm up before timing.
- Measure batch 1 for online serving and batch 32/64 for offline throughput.
- Report p50/p95/p99 latency, queries per second, peak RAM/VRAM, artifact size,
  and process startup time.
- Run backends on the same hardware where comparisons are claimed.

## Decision rule

1. Reject a deployable backend that misses the clean-preservation or regression
   safety gate.
2. Among passing systems, compare noisy exact accuracy, autocorrect recall,
   macro error-type quality, and p95 latency.
3. Treat OpenNMT/CTranslate2 parity failure as a deployment blocker, not as a
   third model winning or losing the method comparison.

Initial suggested gates are clean preservation at least 99 percent, false
correction at most 1 percent, and regression at most 0.5 percent. Replace these
with product-approved values once human traffic and business-cost estimates are
available.

## Dataset lanes

The final benchmark has three frozen lanes:

| Lane | Purpose | Required composition |
|---|---|---|
| Regression | Measure false corrections | Human-valid queries, aliases, abbreviations, mixed scripts, numbers and hard negatives |
| Improvement | Measure correction ability | Human-typed errors with an adjudicated intended query and error type |
| Synthetic diagnostic | Broad controlled coverage | The leak-free production test split, stratified by every generator error type |

The Regression and Improvement lanes determine the production decision. The
synthetic lane diagnoses coverage and must not be presented as real-traffic
quality. Thresholds and decoding settings are selected only on validation data.

Group all aliases and variants of the same canonical query before splitting.
No test target, source variant, or canonical POI group may occur in training or
validation. Publish the overlap counts in the run manifest.

## Exact metric contract

For one record let `x` be input, `y` the adjudicated target, and `p` the final
top-1 output. Equality uses the benchmark's frozen NFC and whitespace policy.

### Query detection

- Gold positive: `x != y`.
- Predicted positive: `p != x`.
- TP: the query is erroneous and the backend changes it.
- FP: the query is clean and the backend changes it.
- FN: the query is erroneous and the backend leaves it unchanged.
- TN: the query is clean and the backend leaves it unchanged.

Detection precision, recall and F1 are computed from these counts. A wrong
change on an erroneous query is still a detection TP but will fail correction.

### Token detection

Align input-to-target and input-to-output with the same deterministic token
Levenshtein implementation. Substitution/deletion marks a source-token index;
insertion marks a boundary slot. Precision/recall/F1 compare predicted edit
slots with gold edit slots. Report separately:

- aligned token rows;
- structural rows containing insertions/deletions;
- alignment coverage;
- aggregate token counts.

Never silently drop word-boundary queries from token metrics.

### Correction

- Correct autocorrection: `x != y`, `p != x`, and `p == y`.
- Autocorrect precision: correct autocorrections / all changed outputs.
- Autocorrect recall: correct autocorrections / all erroneous inputs.
- Autocorrect F1: harmonic mean of those values.
- Exact noisy accuracy: exact outputs / erroneous inputs.
- Exact overall accuracy: exact outputs / all inputs.

### Safety

- Clean preservation: clean inputs with `p == x` / all clean inputs.
- False correction: clean inputs with `p != x` / all clean inputs.
- Regression: edit-distance-to-target after correction is greater than before.
- Improvement: edit-distance-to-target after correction is lower than before.
- Unchanged: the distance is equal, reported separately for literal unchanged
  output and changed-but-equally-distant output.

The implementation must satisfy `clean preservation + false correction = 1`
apart from explicitly reported invalid records.

### Text distance

Compute character and token Levenshtein distance before and after prediction:

- CER before/after;
- TER before/after;
- absolute and relative net reduction;
- per-query improved/equal/regressed classification.

Both corpus-level distance ratios and macro per-query averages are published;
they answer different questions and must not be mixed.

### Candidate ranking

For ReparoS hypotheses, normalize candidates with the same equality policy and
deduplicate while preserving rank. Compute Recall at 1/3/5/10 and reciprocal
rank of the first exact target. A missing target contributes zero.

WebSpell token candidates are not query hypotheses. Publish its token candidate
recall and token MRR in a separate diagnostic block. Its final top-1 query still
participates in all common output metrics.

## Per-error and aggregate reporting

Every quality block is emitted for:

`clean`, `keyboard_edit`, `missing_diacritics_full`,
`missing_diacritics_partial`, `wrong_diacritic`, `telex_leak`, `vni_leak`,
`word_boundary`, `address_abbreviation`, `address_symbol`, and `combined`.

Combined subtype strings remain available in the detailed JSONL but collapse to
`combined` in the primary scorecard. Report raw counts with every rate.

- Micro: pool all TP/FP/FN and edit counts before calculating the metric.
- Macro: calculate per error type first, then average equally over non-empty
  types. Clean is excluded from noisy macro correction metrics.
- Add 95-percent paired bootstrap confidence intervals for headline metrics.

## Prediction artifact contract

Each backend writes immutable JSONL before scoring. One row contains:

```json
{
  "query_id": "stable-id",
  "error_type": "telex_leak",
  "input": "saan bay nooji baif",
  "expected": "sân bay nội bài",
  "output": "sân bay nội bài",
  "hypotheses": ["sân bay nội bài"],
  "scores": [],
  "latency_ms": 10.5
}
```

The scorer joins strictly by `query_id`, rejects duplicates/missing rows, and
verifies that input/expected/error type match the frozen gold file. Model
runners never calculate final quality metrics themselves.

## Run directory

```text
benchmark/<run-id>/
  manifest.json
  gold.jsonl
  predictions/
    webspell.jsonl
    opennmt.jsonl
    ctranslate2.jsonl
  quality.json
  parity.json
  latency.json
  resources.json
  regressions.jsonl
  report.md
```

The manifest records dataset and artifact SHA-256 hashes, git commit, Python and
library versions, decoding config, thresholds, hardware, seed, start/end time,
and every excluded/invalid record.

## Latency and resource protocol

Quality inference is separated from latency measurement. For latency:

1. Load one model per process and record cold-start time.
2. Warm up at least 100 requests.
3. Run at least 10,000 measured requests in randomized order.
4. Measure end-to-end wall time with a monotonic high-resolution clock.
5. Report p50/p95/p99, mean, standard deviation, and QPS.
6. Run batch 1 with concurrency 1 for online latency.
7. Run batch 32 and 64 separately for offline throughput.
8. Repeat each backend on the same CPU/GPU allocation and power mode.

Record process peak RSS, GPU peak allocated/reserved memory, model bytes on
disk, and cold-start RSS. Do not compare local CPU latency with Kaggle GPU
latency in one ranking table.

OpenNMT is measured as a persistent reference service, not by launching its CLI
for each query. CTranslate2 is measured with its persistent translator. WebSpell
keeps its SQLite statistics and model bundle open for the full run.

## Training and update cost

Training cost is reported from immutable logs rather than rerun during quality
benchmarking:

- hardware and accelerator count;
- wall-clock duration;
- optimizer steps and processed-token budget;
- peak VRAM/RAM;
- dataset preparation time;
- artifact conversion time;
- time to incorporate a new POI or retrain after a data refresh.

For WebSpell, separate corpus/statistics, error model, classifier and evaluation
times. For ReparoS, separate tokenizer, OpenNMT training, conversion and parity.

## Final scorecard

The decision table contains, in this order:

1. clean preservation;
2. false correction;
3. regression rate;
4. noisy exact query accuracy;
5. autocorrect precision/recall/F1;
6. Recall at 1/5/10 and MRR where query hypotheses exist;
7. CER and TER net reduction;
8. macro quality by error type;
9. p95 batch-1 latency and QPS;
10. peak memory and artifact size;
11. training and update cost;
12. OpenNMT/CTranslate2 parity.

The primary production comparison is WebSpell versus ReparoS-CTranslate2.
OpenNMT occupies a reference column and must pass parity before CTranslate2 is
eligible for deployment.

## Implemented CLI flow

Create one immutable run directory and execute these stages in order:

```powershell
uv run benchmark-three prepare --data data\reparos\reparos-production-v1\base --output benchmark\<run-id>\gold.jsonl
uv run benchmark-three run-webspell --gold benchmark\<run-id>\gold.jsonl --model artifacts\osm-model-production-v4-leakfree --output benchmark\<run-id>\predictions\webspell.jsonl
uv run benchmark-three run-opennmt --gold benchmark\<run-id>\gold.jsonl --checkpoint artifacts\opennmt-production-kaggle-t4-v1\reparos_base_step_25000.pt --tokenizer artifacts\tokenizer-production-v1\tokenizer.model --decoding-config artifacts\opennmt-production-kaggle-t4-v1\decoding-config.json --output benchmark\<run-id>\predictions\opennmt.jsonl
uv run benchmark-three run-ctranslate2 --gold benchmark\<run-id>\gold.jsonl --model artifacts\opennmt-production-kaggle-t4-v1-ctranslate2-float32 --decoding-config artifacts\opennmt-production-kaggle-t4-v1\decoding-config.json --output benchmark\<run-id>\predictions\ctranslate2.jsonl
uv run benchmark-three parity --opennmt benchmark\<run-id>\predictions\opennmt.jsonl --ctranslate2 benchmark\<run-id>\predictions\ctranslate2.jsonl --output benchmark\<run-id>\parity.json
uv run benchmark-three score --gold benchmark\<run-id>\gold.jsonl --prediction webspell=benchmark\<run-id>\predictions\webspell.jsonl --prediction opennmt=benchmark\<run-id>\predictions\opennmt.jsonl --prediction ctranslate2=benchmark\<run-id>\predictions\ctranslate2.jsonl --output benchmark\<run-id>\quality.json
```

`resources` can summarize exploratory timings stored by quality runners and
artifact sizes, but marks those timings non-official. The production latency
gate still requires the persistent 100-warmup/10,000-request protocol above;
the amortized OpenNMT CLI batch time is never presented as batch-1 latency.
