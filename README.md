# WebSpell prototype

An offline, dependency-free Python prototype of the statistical architecture in
*Using the Web for Language Independent Spellchecking and Autocorrection*
(Whitelaw et al., 2009).

This repository currently implements the model graph, not the paper-scale data
pipeline. A deterministic synthetic fixture trains every component without
network access or external corpora.

## Implemented architecture

```text
term lexicon
  -> Damerau-Levenshtein candidate discovery
  -> substring error model P(observed | intended)
  -> forward/backward n-gram Stupid Backoff model
  -> context-dependent lambda reranking
  -> spellcheck classifiers (with/without suggestions)
  -> autocorrection classifier
  -> keep / flag / correct
```

The confidence layer uses a small deterministic logistic-regression
implementation written in standard Python. This keeps the prototype auditable
and avoids committing to a machine-learning framework before real scale and
storage requirements are known.

## Run the offline demo

```powershell
uv run webspell demo --text "teh quik brwon fox"
```

Expected corrected output:

```text
the quick brown fox
```

The demo also supports contextual real-word correction:

```powershell
uv run webspell demo --text "we learn form clean text"
uv run webspell demo --text "letters form a word"
```

The first sentence changes `form` to `from`; the second keeps `form`.

## Save and load a model bundle

```powershell
uv run webspell export-demo --output artifacts/demo-model
uv run webspell check --model artifacts/demo-model --text "teh quik brwon fox"
```

The bundle contains deterministic `model.json` state and a `manifest.json` with
the schema version, tokenizer identity, model size, and SHA-256 checksum. Loading
fails on corruption, unsupported schemas, or tokenizer mismatch.

## Prepare a million-sentence corpus

Use one document or sentence per UTF-8 line and build the disk-backed artifact:

```powershell
uv run webspell prepare-scale `
  --corpus data/vi.txt `
  --output artifacts/vi-statistics.sqlite3 `
  --order 5 `
  --batch-tokens 50000 `
  --symspell-distance 2 `
  --minimum-frequency 2 `
  --lowercase
```

The command makes two streaming passes. It stores stable token IDs,
forward/backward n-grams, left/right context counts, vocabulary frequencies,
and a SymSpell delete index in SQLite. Python memory is bounded by the current
batch, its vocabulary diversity, fixed caches, and the SQLite cache rather than
total corpus length.

Distance 2 is the recommended large-corpus default. Distance 3 can make the
delete index several times larger. Keep the artifact on SSD and reserve space
for SQLite WAL files during construction.

## Prepare the Vietnam OSM experiment

Install the optional PBF reader, download the current Vietnam extract, and
create deterministic entity-level splits:

```powershell
uv sync
curl.exe -L --fail -o data\osm\vietnam-latest.osm.pbf `
  https://download.geofabrik.de/asia/vietnam-latest.osm.pbf
uv run webspell prepare-osm `
  --input data\osm\vietnam-latest.osm.pbf `
  --output data\osm\prepared-v2 `
  --train-ratio 0.8 `
  --validation-ratio 0.1 `
  --seed 2026 `
  --noisy-variants 3 `
  --clean-variants 1
```

The extractor reads `name:vi`, name/alias fields, brand/operator, address
fields, roads, POIs, and administrative names. It normalizes strings to NFC,
case-folds them, groups every canonical identity and its aliases into the same
split, and only then generates deterministic location-query noise. The default
assumed distribution covers full/partial missing diacritics, wrong Vietnamese
diacritics, leaked Telex/VNI keystrokes, mobile keyboard edits, word-boundary
errors, address abbreviations/symbols, and combined errors. Aliases remain valid
clean surfaces instead of being forced to the canonical display name.

Every term also emits explicit clean-to-clean controls. Each CSV row records
`group_id`, `term_role`, `error_type`, and `variant_id`. Override an assumed
weight without changing code, for example:

```powershell
--noise-weights missing_diacritics_full=25,keyboard_edit=15
```

Unspecified weights retain their defaults. `manifest.json` stores the complete
effective distribution and observed row counts so production log estimates can
replace these assumptions later.

Each `train`, `validation`, and `test` directory contains:

- `entities.jsonl`: canonical entity, leakage-safe group id, type, and aliases.
- `corpus.txt`: one normalized clean term per line for LM/count training.
- `noisy_pairs.csv`: synthetic noisy/correct pairs with entity provenance.

`manifest.json` records the exact split/noise configuration and row counts.
Downloaded PBF files and generated artifacts are ignored by Git. OSM data is
licensed under ODbL; retain OpenStreetMap attribution when redistributing
derived datasets. Synthetic pairs are suitable for pipeline experiments, not
as evidence of real-user accuracy.

Run the reproducible infrastructure benchmark with:

```powershell
uv run python benchmarks/benchmark_scale_backend.py --sentences 100000
```

On the current development environment, 100,000 repetitive synthetic sentences
(833,334 tokens) were counted in about 14.8 seconds. Natural text has many more
distinct n-grams, so this is a baseline rather than a production estimate.

The same benchmark completed 1,000,000 synthetic sentences (8,333,334 tokens)
in 165.4 seconds at roughly 50,400 tokens/second. Its vocabulary contains only
25 terms, so it validates the million-sentence execution path and bounded-memory
design, not expected disk size or throughput for diverse natural text.

## Train the Vietnam OSM model

Train the complete reproduced paper model directly from the prepared OSM dataset splits:

```powershell
uv run webspell train-osm `
  --data data/osm/prepared `
  --output artifacts/osm-model `
  --order 5 `
  --batch-tokens 50000 `
  --symspell-distance 2 `
  --minimum-frequency 1
```

If you previously generated `artifacts/osm-train.sqlite3` with `prepare-scale`, you can reuse it immediately:

```powershell
uv run webspell train-osm `
  --data data/osm/prepared `
  --output artifacts/osm-model `
  --statistics artifacts/osm-train.sqlite3
```

The command automatically executes:
1. Corpus tokenization & counting (vocabulary, n-grams, context, SymSpell index).
2. Query alignment adapter across noisy/correct pairs.
3. Substring error model training from aggregated error triples.
4. Context-dependent $\lambda$ tuning on the validation split.
5. Confidence and autocorrection classifier training with feature caching.
6. Comprehensive evaluation on held-out test data (E1–E5, CER, FER, TER, NGS, query accuracy).
7. Atomic export of the model bundle (`model.json`, `statistics.sqlite3`, `manifest.json`, `metrics.json`).

Run interactive inference against the trained bundle:

```powershell
uv run webspell check --model artifacts/osm-model --text "san bay noi bai"
```

### Faster and resumable OSM training

Training caches context-independent candidates, error-channel scores, ranked
contexts, and language-model lookups. The CLI prints elapsed time for every
stage. A complete `statistics.sqlite3` already present in the output directory
is reused automatically when its n-gram order and SymSpell settings match.
Pass `--rebuild-statistics` after changing the corpus.

The model checkpoint is written before evaluation and coverage scans, so a
failure in those optional stages no longer discards the trained model. Coverage
is bounded to 200,000 compatible examples per split by default; use `0` for a
complete scan.

For the shortest train-only run:

```powershell
uv run webspell train-osm `
  --data data/osm/prepared `
  --output artifacts/osm-model `
  --skip-evaluation `
  --skip-coverage
```

Feature generation also supports process-based parallelism:

```powershell
uv run webspell train-osm `
  --data data/osm/prepared `
  --output artifacts/osm-model `
  --workers 2 `
  --worker-batch-size 750
```

Each worker owns its SQLite connection and caches, and output rows are merged in
input order for deterministic training. Use one worker on low-core or
memory-constrained machines. On the measured 2-core i3-1005G1 system, one
worker completed the 10% train-only benchmark in 32.3 seconds, while two workers
took 38.7 seconds because process startup and split caches outweighed CPU
parallelism. The two generated model files had identical SHA-256 hashes.

For full evaluation with bounded coverage, omit the two skip flags. Use an
existing database from another directory with `--statistics PATH`.

On the Vietnam OSM artifact in this repository, the optimized cold candidate
ranking benchmark improved from 14.449 seconds to 3.116 seconds for 500
examples (4.64x). A train-only run at 10% of the default sample limits took
32.3 seconds with precomputed statistics. Actual full-run time depends on token
and context diversity; the CLI stage timings are the authoritative measurement
for a particular dataset and machine.

### Production data policy

Positive limits no longer mean the first N compatible rows. Training and error
examples use deterministic BLAKE2b bottom-k sampling over the complete split;
evaluation samples query pairs in the same way. Samples are independent of CSV
row order and reproducible through `--sampling-seed`. Classifier samples enforce
the requested clean-to-error ratio and stratify positive labels by `error_type`.
By default, split/merge and address-symbol rows that a token classifier cannot
represent are retained in the dataset and evaluation but routed out of classifier
training. Use `--include-structural-training` only for an explicit ablation.
A zero limit still means the complete compatible split.

Every completed run writes `training-metadata.json` with the resolved config,
sampling algorithm/version, seed, sampling units, and stage timings. Optional
classifier learning curves use nested, deterministically shuffled subsets and
report validation precision, recall, and F1 in `learning-curve.json`. Evaluation
also writes query, token, candidate-recall, and top-1 metrics by `error_type`.

A production model-selection run can use the complete error model and test
split while measuring classifier saturation explicitly:

```powershell
uv run webspell train-osm `
  --data data/osm/prepared-v3 `
  --output artifacts/osm-model-production-v4 `
  --workers 2 `
  --worker-batch-size 1500 `
  --max-error-examples 0 `
  --error-model-source hybrid `
  --error-frequency-ratio 10 `
  --minimum-context-frequency 10 `
  --max-training-examples 500000 `
  --max-validation-examples 100000 `
  --max-test-examples 0 `
  --learning-curve-sizes 25000,50000,100000,200000,500000 `
  --sampling-seed 2026 `
  --clean-ratio 2 `
  --max-coverage-examples 0
```

The saved model uses `--max-training-examples`; learning-curve points are
diagnostics and do not silently replace it. The default positive-label weights
are `25/20/15/15/10/10/5` for keyboard, full missing diacritics, partial missing
diacritics, wrong diacritics, Telex, VNI, and compatible combined errors. The
effective weights and sampled classifier distribution are saved in metadata.
Promotion thresholds should be set
from product costs, especially the required autocorrect precision.

`--error-model-source web` implements Section 3.2 term-frequency and context
mining exactly. OSM is much smaller and cleaner than the paper's noisy Web
corpus, so `hybrid` is the practical default for OSM experiments: it combines
those naturally mined triples with sparse artificial errors and records both
sources in `training-metadata.json`. Use `audit-osm-scores` to inspect score
units rather than comparing raw lambda values across different corpora:

```powershell
uv run webspell audit-osm-scores `
  --model artifacts\osm-model-production-v2 `
  --pairs data\osm\prepared-v2\validation\noisy_pairs.csv `
  --max-examples 5000 `
  --output artifacts\osm-model-production-v2\score-audit.json
```

## Frozen WebSpell 2009 baseline (Branch A)

Branch A is isolated from the Xanh SM production taxonomy. It mines the error
model only from corpus frequency/context, creates Section 3.4.1 artificial data
only for lambda/confidence training and evaluation, and disables Vietnamese
candidate expansion.

```powershell
uv run webspell prepare-paper-baseline `
  --data data\osm\prepared-v3 `
  --output data\osm\paper-baseline

uv run webspell train-paper-baseline `
  --data data\osm\paper-baseline `
  --output artifacts\paper-baseline `
  --workers 1
```

The defaults use full data (`max-*=0`). The prepare command never modifies the
source directory, and the train command does not consume a human-typed set.
See [the reproduction audit](docs/webspell-2009-reproduction-audit.md) and
[the Xanh SM typed-test protocol](docs/xanh-sm-typed-test-protocol.md).

## ReparoS 2023 benchmark branch (Branch B)

Prepare the published Base -> C1 -> C2 curriculum contracts without launching a
second long training job:

```powershell
uv run reparos prepare `
  --data data\osm\prepared-v3 `
  --output data\osm\reparos
```

Optional private-data substitutes can be supplied with `--phonetic-pairs` and
`--weak-feedback`. Missing lanes remain explicit in the manifest. Build the
paper's 90/10 head-tail evaluation mixtures only from adjudicated human labels:

```powershell
uv run reparos prepare-eval `
  --labeled-queries data\typed\xanh-sm-labeled-v1.csv `
  --output data\typed\reparos-eval-v1
```

See [the ReparoS reproduction audit](docs/reparos-2023-reproduction-audit.md).

## Tests

```powershell
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
```

Tests cover edit operations, paper distance thresholds, deterministic candidate
search, substring transformations, bidirectional context, non-word correction,
real-word correction, blacklist behavior, and TER/CER/FER/NGS evaluation.

## Source layout

| Package | Responsibility |
|---|---|
| `webspell.text` | Unicode tokenization and bounded sentence context |
| `webspell.vocabulary` | Term frequencies and approximate lookup |
| `webspell.mining` | Frequency filtering, context counts, and automatic error triples |
| `webspell.error_model` | Dynamic-programming substring alignment and channel score |
| `webspell.language_model` | Forward/reverse n-gram Stupid Backoff scores |
| `webspell.candidates` | Top-20 preselection, reranking, and lambda tuning |
| `webspell.confidence` | Features, logistic classifiers, thresholds, and blacklist |
| `webspell.pipeline` | Training-row construction and end-to-end inference |
| `webspell.evaluation` | Paper-compatible E1-E5, TER, CER, FER, and NGS metrics |
| `webspell.serialization` | Atomic JSON model bundles and manifest validation |
| `webspell.scalable` | Streaming corpus, SQLite counts/LM/context, SymSpell, online training |

See [the technical specification](docs/technical-specification.md) for formulas,
module contracts, artifact requirements, and documented reproduction choices.

## Vietnamese input adapter

Candidate generation can optionally expand raw Telex and VNI input into NFC
Unicode hypotheses:

```python
from webspell.candidates import CandidateRanker
from webspell.text import VietnameseVariantGenerator

ranker = CandidateRanker(
    lexicon,
    error_model,
    language_model,
    variant_generator=VietnameseVariantGenerator(),
)
```

The adapter currently handles common transformations such as `dd -> đ`,
`aw -> ă`, `uow -> ươ`, terminal Telex tone keys, and common VNI digits. It is
an optional language adapter; the core statistical models remain independent of
Vietnamese.

## Prototype boundaries

- The reference backend remains brute force for tests; the scalable backend uses
  cached SQLite SymSpell lookup with exact edit-distance verification.
- JSON bundles are suitable for the prototype; paper-scale count tables will
  require compact binary/sharded formats rather than one monolithic JSON file.
- Error-pair and error-triple mining now expose streaming iterators; Web-scale
  training still requires distributed sharding and merge infrastructure.
- SQLite supports million-sentence single-machine experiments, but hundreds of
  millions of Web pages require a distributed count/merge job.
- The tokenizer is a generic Unicode baseline, not a production adapter for a
  particular language.
- The Vietnamese input adapter covers common Telex/VNI forms but not every
  escape convention, malformed mixed-input sequence, or syllable split/merge.
- Hyperparameters that the paper did not disclose remain explicit reproduction
  choices rather than claims about the original Google implementation.
