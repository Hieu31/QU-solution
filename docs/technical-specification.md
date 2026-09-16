# WebSpell Technical Specification

## 1. Purpose and scope

This document specifies a reproducible implementation of the statistical spellchecking and autocorrection architecture described in *Using the Web for Language Independent Spellchecking and Autocorrection* (Whitelaw et al., 2009).

The system is language-independent at its statistical core. Language-specific behavior is restricted to text normalization, tokenization, sentence segmentation, and optional character-policy configuration.

This specification covers:

- term vocabulary construction;
- approximate candidate discovery;
- automatic error-pair mining;
- substring error-model training;
- bidirectional n-gram language modeling;
- context-dependent language-model weighting;
- spellcheck and autocorrection confidence classifiers;
- inference, evaluation, serialization, and reproducibility.

It deliberately does not prescribe a particular corpus, storage service, distributed execution engine, or deployment API.

## 2. Normative language

The terms **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** describe implementation requirements.

- **Paper-defined** behavior is directly specified by the paper.
- **Reproduction choice** is behavior required for an executable system but not fully specified by the paper.
- All reproduction choices MUST be configurable and recorded in the model manifest.

## 3. System overview

The training graph is:

```text
term counts ──► term lexicon ──► close-word mining ──► contextual triples
     │                                                    │
     └────────► forward/reverse n-gram LM                 ▼
                                                  substring error model
                                                           │
                                                           ▼
clean/corrupted paired text ──► candidate ranking ──► lambda tuning
                                          │
                                          ▼
                               spellcheck classifiers
                                          │
                                          ▼
                               autocorrection classifier
```

Inference is:

```text
input text
  └─► normalization and tokenization
       └─► candidate generation
            └─► error-model preselection
                 └─► bidirectional LM reranking
                      └─► spellcheck decision
                           └─► keep, flag, or autocorrect
```

## 4. Package topology

```text
src/webspell/
├── types.py
├── config.py
├── text/
├── vocabulary/
├── mining/
├── error_model/
├── language_model/
├── candidates/
├── confidence/
├── pipeline/
├── evaluation/
└── serialization/
```

Modules MUST communicate through the contracts in this document rather than by importing another module's private state.

## 5. Canonical data types

### 5.1 Token

```python
@dataclass(frozen=True)
class Token:
    surface: str
    normalized: str
    sentence_id: int
    position: int
    character_start: int
    character_end: int
```

Invariants:

- `surface` MUST exactly reproduce the substring of the input text.
- `[character_start:character_end]` MUST select `surface` from the input.
- `position` MUST be zero-based within the sentence.
- `normalized` MUST be deterministic for a fixed normalizer configuration.

### 5.2 Context

```python
@dataclass(frozen=True)
class Context:
    left: tuple[str, ...]
    right: tuple[str, ...]
```

`left` is ordered from oldest to nearest token. `right` is ordered from nearest to furthest token. Context MUST NOT cross sentence boundaries.

### 5.3 TermRecord

```python
@dataclass(frozen=True)
class TermRecord:
    term: str
    frequency: int
    rank: int
```

### 5.4 ErrorTriple

```python
@dataclass(frozen=True)
class ErrorTriple:
    intended: str
    observed: str
    count: int
```

`count` MUST be a positive integer. Duplicate `(intended, observed)` records MUST be aggregated before error-model training.

### 5.5 Candidate

```python
@dataclass(frozen=True)
class Candidate:
    term: str
    edit_distance: int
    term_frequency: int
    error_log_score: float
    lm_log_score: float | None = None
    combined_log_score: float | None = None
```

### 5.6 Prediction

```python
Action = Literal["keep", "flag", "correct"]

@dataclass(frozen=True)
class Prediction:
    token: Token
    is_misspelled: bool
    action: Action
    correction: str | None
    candidates: tuple[Candidate, ...]
    spellcheck_confidence: float
    autocorrect_confidence: float | None
```

Rules:

- `keep` and `flag` MUST have `correction=None`.
- `correct` MUST have a non-empty correction equal to the top-ranked non-original candidate.
- A token MUST NOT be corrected unless it was first classified as misspelled.

## 6. Configuration contract

Every training or inference run MUST use a fully resolved, immutable configuration. At minimum it contains:

```yaml
schema_version: 1
text:
  unicode_normalization: NFC
  case_policy: preserve
vocabulary:
  max_terms: 10000000
  minimum_frequency: 1
mining:
  intended_observed_frequency_ratio: 10.0
  minimum_context_frequency: 10
edit_distance:
  thresholds:
    - {max_length: 4, max_distance: 1}
    - {max_length: 12, max_distance: 2}
    - {max_length: null, max_distance: 3}
error_model:
  max_source_substring_length: 2
  max_target_substring_length: 2
  smoothing: additive
  smoothing_alpha: 0.1
  alignment_iterations: 1
language_model:
  order: 5
  backoff_alpha: 0.4
  bidirectional_combination: sum
candidates:
  preselection_limit: 20
lambda_tuning:
  objective: mean_reciprocal_rank
  minimum: 0.0
  maximum: 10.0
  step: 0.05
confidence:
  classifier: logistic_regression
  threshold_objective: f1
```

The LM order, backoff factor, smoothing method, bidirectional combination, alignment iteration count, and optimizer settings are reproduction choices.

## 7. Text processing

### 7.1 Normalizer

Interface:

```python
class TextNormalizer(Protocol):
    @property
    def identity(self) -> str: ...

    def normalize_token(self, token: str) -> str: ...
```

Requirements:

- Unicode normalization MUST be explicit; NFC is the default.
- Case folding MUST NOT be applied unless configured.
- Normalization MUST NOT alter input offsets; it applies to token lookup values, not input text.
- Training and inference MUST use the same normalizer identity.

### 7.2 Tokenizer and sentence segmenter

```python
class Tokenizer(Protocol):
    @property
    def identity(self) -> str: ...

    def tokenize(self, text: str) -> list[Token]: ...
```

The implementation MUST define treatment of punctuation, combining marks, apostrophes, hyphens, numbers, URLs, and script-specific boundaries. The identity and version MUST be written to the manifest.

### 7.3 Context extraction

```python
def context_for(
    tokens: Sequence[Token],
    token_index: int,
    max_left: int,
    max_right: int,
) -> Context: ...
```

The amount of available context is:

\[
i=\min(|C_L|, n-1),\qquad j=\min(|C_R|, n-1)
\]

where `n` is the configured LM order.

### 7.4 Optional Vietnamese input adapter

The language-independent ranker MAY receive a variant generator. The Vietnamese
adapter emits NFC Unicode hypotheses for raw Unicode, Telex, and VNI tokens
before approximate lexicon search. Variant expansion changes candidate recall
only; the substring error model MUST continue scoring the original observed
token against each intended candidate.

Common mappings include `dd -> đ`, `aw -> ă`, `uow -> ươ`, Telex terminal tone
keys, and VNI diacritic/tone digits. The adapter MUST be optional because these
rules are not language-independent and can produce false hypotheses for foreign
terms. Production use MUST version the accepted input-method grammar and define
escape behavior, malformed input behavior, and mixed Unicode/input-method text.

## 8. Vocabulary subsystem

### 8.1 Term counting

Input: an iterator of normalized tokens.

Output: aggregated `(term, frequency)` records.

```python
class TermCounter:
    def update(self, tokens: Iterable[str]) -> None: ...
    def records(self) -> Iterable[TermRecord]: ...
```

Counts MUST use unsigned 64-bit integers or a representation with a greater safe range. Sorting MUST be deterministic: descending frequency, then ascending Unicode code-point order.

### 8.2 Term-list construction

The vocabulary builder MUST:

1. aggregate normalized token counts;
2. apply configured non-word filters;
3. sort deterministically;
4. retain at most `max_terms` terms;
5. assign one-based ranks.

The paper uses ten million terms. Filters such as minimum/maximum length and punctuation ratio are reproduction choices.

### 8.3 Approximate-search index

```python
class TermLexicon(Protocol):
    def contains(self, term: str) -> bool: ...
    def frequency(self, term: str) -> int: ...
    def close_terms(self, term: str, max_distance: int) -> Iterable[TermMatch]: ...
```

`close_terms` MUST return the exact set of terms whose Damerau-Levenshtein distance does not exceed the bound. Results MUST be independent of traversal or worker order.

The paper uses a trie. A SymSpell or finite-state implementation MAY be substituted if the returned set is equivalent.

## 9. Edit distance and close-word mining

### 9.1 Distance definition

The required operations are:

- character insertion;
- character deletion;
- character substitution;
- adjacent-character transposition.

Each operation has unit cost. The default bound is:

\[
d_{max}(l)=
\begin{cases}
1 & l\le4\\
2 & 5\le l\le12\\
3 & l>12
\end{cases}
\]

The length used for selecting the bound MUST be the number of Unicode code points in the normalized observed term.

### 9.2 Directional filtering

For a close pair `(a, b)`, `a` is an intended-word candidate for observed word `b` only if:

\[
f(a) \ge \rho f(b)
\]

with paper-compatible default \(\rho=10\).

Input: `TermLexicon`.

Output:

```python
@dataclass(frozen=True)
class ClosePair:
    intended: str
    observed: str
    distance: int
    intended_frequency: int
    observed_frequency: int
```

## 10. Contextual error-triple mining

For each observed term \(w\), let \(S_w\) contain \(w\) and all directionally valid intended candidates. A context is the immediately preceding and following token:

\[
c=(x_{k-1},x_{k+1})
\]

Boundary sentinels MAY be used but MUST be configured and versioned.

Contexts whose total corpus frequency is below 10 MUST be discarded by default.

For every occurrence group `(w, c)`, select:

\[
s^*(w,c)=\arg\max_{s\in S_w}\operatorname{count}(s,c)
\]

Tie-breaking MUST prefer, in order:

1. the candidate with higher global term frequency;
2. smaller edit distance from `w`;
3. ascending Unicode code-point order.

The resulting training count is:

\[
N(s,w)=\sum_c \operatorname{count}(w,c)\,\mathbf{1}[s^*(w,c)=s]
\]

The module emits `ErrorTriple(intended=s, observed=w, count=N(s,w))` for positive counts.

Tie-breaking and boundary behavior are reproduction choices.

## 11. Substring error model

### 11.1 Alignment space

For intended word \(s\) and observed word \(w\), define aligned partitions:

\[
R=(R_1,\ldots,R_m),\qquad T=(T_1,\ldots,T_m)
\]

such that:

- concatenating `R` produces `s`;
- concatenating `T` produces `w`;
- \(|R_i|\le2\) and \(|T_i|\le2\);
- either side MAY be empty;
- \((R_i,T_i)=(\epsilon,\epsilon)\) is forbidden.

### 11.2 Model

The model score is:

\[
P(w\mid s)\approx
\max_{R,T}\prod_{i=1}^{m}P(T_i\mid R_i)
\]

Implementation MUST compute in log space:

\[
\log P(w\mid s)=
\max_{R,T}\sum_i\log P(T_i\mid R_i)
\]

### 11.3 Aligner interface

```python
class SubstringAligner:
    def best_alignment(
        self,
        intended: str,
        observed: str,
        transition_scorer: TransitionScorer,
    ) -> Alignment: ...
```

Dynamic-programming state `(p, q)` represents consumed source and target prefixes. From a state, enumerate source lengths `0..2` and target lengths `0..2`, excluding `(0, 0)`. Complexity is \(O(|s||w|K_sK_t)\), with both `K` values equal to 2 by default.

Ties MUST be resolved deterministically by fewer segments, then lexicographic sequence of segment pairs.

### 11.4 Training counts

Given triples \((s,w,n_{sw})\), find the selected alignment and accumulate:

\[
C(R,T)=\sum_{s,w}n_{sw}\sum_i\mathbf{1}[(R_i,T_i)=(R,T)]
\]

Maximum-likelihood estimates are:

\[
P(T\mid R)=\frac{C(R,T)}{\sum_{T'}C(R,T')}
\]

Because unseen transitions require a finite score, the executable implementation MUST support smoothing. With additive smoothing:

\[
P(T\mid R)=
\frac{C(R,T)+\alpha}
{\sum_{T'}C(R,T')+\alpha|V_R|}
\]

where \(V_R\) is the configured target-substring support. The smoothing method and support definition are reproduction choices.

### 11.5 Initialization and iteration

The first alignment pass has no trained probabilities. It MUST use a deterministic initializer based on unit Damerau-Levenshtein costs or configured initial transition probabilities.

Optional iterative realignment is:

1. initialize alignments;
2. estimate transition probabilities;
3. realign all triples with the current model;
4. repeat for the configured iteration count.

The paper does not state an EM schedule. Therefore one estimation pass is the paper-compatible default, and additional passes are experimental.

### 11.6 Interface

```python
class SubstringErrorModel:
    def transition_logprob(self, source: str, target: str) -> float: ...
    def word_logprob(self, observed: str, intended: str) -> float: ...
```

All returned scores MUST be finite for candidates admitted by the candidate generator.

## 12. N-gram language model

### 12.1 Count model

Train word n-gram counts for orders 1 through \(n\), including explicit sentence-boundary symbols. Forward and reverse models MUST be independently counted from forward and reversed sentences.

### 12.2 Stupid Backoff

For history \(h\) and token \(x\):

\[
S(x\mid h)=
\begin{cases}
\frac{C(h,x)}{C(h)} & C(h,x)>0\\
\beta S(x\mid h') & C(h,x)=0
\end{cases}
\]

where \(h'\) drops the oldest history token and \(\beta\) defaults to 0.4. For unigrams:

\[
S(x)=\frac{C(x)}{\sum_y C(y)}
\]

An explicit unknown-token score MUST be defined. Stupid Backoff scores are ranking scores and need not form a normalized probability distribution.

### 12.3 Bidirectional score

```python
class BidirectionalLanguageModel:
    def token_logscore(self, candidate: str, context: Context) -> float: ...
```

Default reproduction choice:

\[
L(s,C)=
\log S_f(s\mid C_L)+
\log S_b(s\mid C_R)
\]

If a side has no context, its unigram score is used. Alternative averaging or length normalization MUST be configuration-controlled.

## 13. Candidate generation and ranking

### 13.1 Candidate generation

```python
class CandidateGenerator:
    def generate(self, observed: str) -> list[CandidateSeed]: ...
```

The generator MUST:

1. derive the edit-distance bound from observed-token length;
2. retrieve all close terms;
3. score each intended candidate using the error model and term frequency;
4. include the unchanged observed token as a candidate;
5. retain at most `preselection_limit` non-original suggestions.

The preselection score is:

\[
Q_{pre}(s;w)=\log P(w\mid s)+\log f(s)
\]

This follows the paper's Section 4.2.2 description. Frequency handling MUST record whether raw count, relative frequency, or a transformed count is used.

### 13.2 Contextual reranking

Given context bucket `(i, j)`, the final score is:

\[
Q(s;w,C)=\log P(w\mid s)+\lambda_{i,j}L(s,C)
\]

Candidates MUST be sorted by descending final score. Ties are resolved by descending preselection score, smaller edit distance, higher frequency, then Unicode term order.

The paper omits vocabulary frequency in its final noisy-channel equation but includes it in suggestion preselection. The implementation MUST NOT silently include frequency again during final reranking unless configured as an experiment.

## 14. Context-dependent lambda training

### 14.1 Training examples

Each example contains:

```python
@dataclass(frozen=True)
class RankingExample:
    observed: str
    intended: str
    context: Context
    candidates: tuple[CandidateSeed, ...]
```

Examples for which the intended term is absent from the candidate list MUST be reported as no-good-suggestion cases and excluded from lambda optimization.

### 14.2 Objective

Partition examples by available context `(i, j)`. For bucket \(D_{i,j}\), optimize:

\[
\lambda^*_{i,j}=
\arg\max_{\lambda\ge0}
\frac{1}{|D_{i,j}|}
\sum_{e\in D_{i,j}}
\frac{1}{\operatorname{rank}_e(\lambda)}
\]

Rank is one-based after contextual reranking.

### 14.3 Optimizer

Default optimizer: deterministic grid search over `[0, 10]` with step `0.05`, followed optionally by a finer search around the best point. Ties MUST select the smaller lambda.

For empty or undersized buckets, use the configured fallback:

1. nearest sufficiently populated bucket;
2. global lambda;
3. fixed configured default.

The selected policy MUST be stored in `lambda.json`.

## 15. Confidence feature extraction

### 15.1 Feature schema

For original candidate \(w\), top candidate \(s_1\), and second candidate \(s_2\):

\[
\Delta E_k = \log P(w\mid s_k)-\log P(w\mid w)
\]

\[
\Delta L_k = L(s_k,C)-L(w,C)
\]

Required feature groups:

1. `top1_error_delta`, `top1_lm_delta`;
2. `top2_error_delta`, `top2_lm_delta`;
3. case-signature features;
4. suggestion count;
5. token length;
6. available left and right context.

The paper calls these five sets because the score values form one set and the remaining metadata form four sets.

### 15.2 Missing candidates

Missing top-1 or top-2 values MUST use explicit indicator features and neutral numeric values, not infinities or NaNs:

```text
has_top1
has_top2
top1_error_delta = 0 when absent
top1_lm_delta    = 0 when absent
top2_error_delta = 0 when absent
top2_lm_delta    = 0 when absent
```

### 15.3 Case signature

At minimum encode:

- all lowercase;
- all uppercase;
- title case;
- mixed case;
- uncased/no cased characters;
- whether top-1 matches the observed case pattern.

The exact encoding is a reproduction choice and MUST be serialized as a named feature schema.

## 16. Confidence classifier training

### 16.1 Labels

For paired clean/corrupted text:

- spellcheck label is 1 iff `observed != intended`;
- autocorrection label is 1 iff the top-ranked non-original suggestion equals `intended`.

Alignment between clean and corrupted tokens MUST be known from the corruption process or supplied explicitly. Silent heuristic realignment is prohibited during training.

### 16.2 Spellcheck classifiers

Train two independent binary classifiers:

- `spellcheck_with_suggestions`;
- `spellcheck_without_suggestions`.

The second classifier receives only features valid when no non-original suggestion exists.

### 16.3 Autocorrection classifier

Training order is normative:

1. train and threshold-tune spellcheck classifiers;
2. run them on the autocorrection training examples;
3. retain examples predicted as misspelled and having a suggestion;
4. train the autocorrection classifier on this retained subset.

This preserves the distribution encountered by the autocorrection classifier during inference.

### 16.4 Logistic regression

For feature vector \(x\):

\[
p(y=1\mid x)=\sigma(b+\theta^Tx)
\]

Train by minimizing regularized binary cross-entropy:

\[
J(\theta,b)=
-\sum_k[y_k\log p_k+(1-y_k)\log(1-p_k)]
+\eta\lVert\theta\rVert_2^2
\]

Regularization type, strength, class weights, solver, random seed, and stopping criteria are reproduction choices and MUST be recorded.

### 16.5 Threshold tuning

For classifier probabilities \(p_k\), choose threshold \(t\) on held-out development data:

\[
t^*=\arg\max_t F_1(t)
\]

Ties MUST prefer the larger threshold to favor precision. Spellcheck and autocorrection thresholds MUST be tuned independently.

Training examples, development examples, and evaluation examples MUST be disjoint.

## 17. Blacklisting

Before confidence classification, the system MUST retain without flagging or correction:

- numbers;
- punctuation-only tokens;
- symbol-only tokens;
- single-character tokens.

Blacklist rules MUST operate on the surface and normalized forms according to documented predicates. Optional URL, email, or identifier rules are extensions and MUST be disabled in the paper-compatible configuration.

## 18. Inference algorithm

For each non-blacklisted token \(w\):

1. extract bounded left/right context;
2. generate close terms;
3. compute preselection scores and retain top suggestions;
4. add the unchanged candidate;
5. compute bidirectional LM scores;
6. select \(\lambda_{i,j}\) and compute final scores;
7. extract confidence features;
8. select the appropriate spellcheck classifier;
9. if spellcheck probability is below threshold, return `keep`;
10. if misspelled but there is no suggestion, return `flag`;
11. run the autocorrection classifier;
12. if its probability reaches its threshold, return `correct` with top-1;
13. otherwise return `flag`.

Classifier inputs MUST be computed from the same ranked candidate list returned for diagnostics.

## 19. Evaluation

For every token, define:

- `E1`: misspelled token corrected to the wrong token;
- `E2`: misspelled token flagged but not corrected;
- `E3`: misspelled token neither corrected nor flagged;
- `E4`: correct token wrongly corrected;
- `E5`: correct token wrongly flagged.

For total token count \(T\):

\[
CER=\frac{E1+E2+E3+E4}{T}
\]

\[
FER=\frac{E3+E5}{T}
\]

\[
TER=\frac{E1+E2+E3+E4+E5}{T}
\]

No Good Suggestion rate is:

\[
NGS=\frac{\#\text{misspellings without intended candidate}}
{\#\text{misspelled tokens}}
\]

The evaluator MUST additionally report raw counts, denominators, spellcheck precision/recall/F1, autocorrection accuracy, and candidate recall at `k`.

## 20. Serialization and artifact manifest

The prototype implements a versioned JSON bundle containing the complete
in-memory lexicon, substring counts, forward/backward n-gram counts, lambda
grid, confidence model weights, feature normalization state, thresholds, and
registered candidate-variant identity. Writes are atomic at the individual-file
level. The manifest validates model size, SHA-256, schema version, and tokenizer
identity before reconstruction.

At paper scale, the JSON representation MUST be replaced or supplemented by
sharded binary artifacts without changing the logical state contract below.

Each completed run has an immutable directory:

```text
artifacts/runs/<run-id>/
├── resolved-config.yaml
├── manifest.json
├── vocabulary/
├── error-mining/
├── error-model/
├── language-model/
├── ranking/
├── confidence/
└── evaluation/
```

The manifest MUST include:

- manifest and artifact schema versions;
- run ID and creation timestamp;
- source-control commit and dirty-worktree flag;
- complete resolved configuration hash;
- tokenizer and normalizer identities;
- source-input identifiers and checksums;
- dependency versions;
- random seeds;
- artifact paths, sizes, and SHA-256 checksums;
- all reproduction choices;
- parent run IDs when a stage reuses artifacts.

Loading MUST fail fast on schema incompatibility, checksum failure, tokenizer mismatch, normalizer mismatch, or missing required artifacts.

## 21. Stage interfaces

Every stage MUST support an independent API and CLI operation.

| Stage | Required inputs | Primary outputs |
|---|---|---|
| `build-vocabulary` | token stream | term records, search index |
| `count-ngrams` | sentence stream | forward and reverse counts |
| `mine-close-words` | term lexicon | directional close pairs |
| `mine-error-triples` | close pairs, contextual counts | aggregated error triples |
| `train-error-model` | error triples | substring transitions |
| `train-language-model` | n-gram counts | forward/reverse LM |
| `tune-lambda` | paired ranking examples, error model, LM | context lambda grid |
| `train-spellcheck` | labeled examples and candidates | two classifiers and thresholds |
| `train-autocorrect` | spellcheck-selected examples | classifier and threshold |
| `evaluate` | immutable model bundle, labeled examples | metrics and predictions |

Stages MUST write to temporary paths and atomically publish completed artifacts. A failed stage MUST NOT leave an artifact that can be mistaken for complete.

## 22. Determinism and parallel execution

### 22.1 Single-machine scalable backend

The scalable backend uses a replayable line corpus and two passes. Pass one
aggregates vocabulary counts to SQLite and assigns deterministic integer IDs by
descending frequency and Unicode term order. Pass two counts forward/reverse
n-grams and `(term, left, right)` contexts in bounded token batches, encoding
n-gram keys as packed unsigned integer IDs.

Approximate lookup uses a disk-backed SymSpell delete index followed by exact
Damerau-Levenshtein verification. Query and count caches have fixed maximum
sizes. Error-pair/triple mining exposes streaming iterators, and one-pass error
model training does not materialize its input. Large confidence models use
multi-pass streaming normalization/SGD and histogram-based F1 threshold tuning.

SQLite is the normative single-machine backend, not the distributed Web-scale
backend. Corpus sources MUST be replayable because vocabulary assignment must
precede integer-ID n-gram counting.

- Every stochastic component MUST accept an explicit seed.
- Distributed counts MUST be combined using associative integer addition.
- Ranking and ties MUST follow the rules in this document.
- Floating-point reduction order SHOULD be stable where practical.
- Worker count MUST NOT alter candidate sets, labels, or discrete predictions.
- Repeating a run with identical inputs, configuration, code, and dependency versions SHOULD produce identical checksums, except for explicitly excluded metadata such as timestamps.

## 23. Error handling and observability

Each stage MUST report:

- processed record count;
- rejected/malformed record count;
- elapsed time and throughput;
- peak or sampled memory usage where available;
- output cardinality;
- stage-specific diagnostics.

Required diagnostics include:

- vocabulary coverage and frequency distribution;
- close-pair count by edit distance;
- triple count and mass retained after context filtering;
- transition count and unseen-transition rate;
- candidate recall at 1, 5, 10, and 20;
- ranking MRR by context bucket;
- classifier class balance and confusion matrices;
- keep/flag/correct action distribution.

Invalid probabilities, NaNs, empty required artifacts, schema mismatches, and checksum errors MUST terminate the stage with a non-zero exit status.

## 24. Testing requirements

### 24.1 Unit tests

Tests MUST cover:

- Unicode normalization and offset preservation;
- Damerau transposition at unit cost;
- edit-distance thresholds at lengths 4, 5, 12, and 13;
- exact approximate-search recall against brute force;
- directional frequency filtering;
- context-frequency filtering and deterministic ties;
- substring insertion, deletion, substitution, and 2-to-2 transformations;
- transition normalization and smoothing;
- forward and reverse Stupid Backoff paths;
- candidate preselection and final reranking;
- lambda bucket selection;
- missing top-1/top-2 feature encoding;
- blacklist predicates;
- all inference action branches;
- exact E1-E5 metric classification.

### 24.2 Integration fixture

A tiny synthetic fixture MUST contain enough terms and sentences to produce:

- at least one valid directional close pair;
- at least one deletion, insertion, substitution, and transposition;
- one real-word error requiring context;
- a token with no suggestion;
- all three final actions.

The complete pipeline MUST train and evaluate on this fixture in continuous integration without external data or network access.

### 24.3 Regression tests

Approved small artifacts SHOULD have golden predictions and metrics. Changes to a reproduction choice MUST require an intentional golden update and a manifest-version note.

## 25. Acceptance criteria

The implementation is architecturally complete when:

1. every stage in Section 21 runs independently and through `train all`;
2. a model bundle can be loaded without access to training data;
3. inference deterministically emits keep, flag, or correction per token;
4. the error model supports all allowed 0-to-2 and 1-to-2 substring transformations;
5. forward and backward LM evidence affects ranking;
6. lambda values are optimized separately by context bucket;
7. all three confidence classifiers are trained and thresholded in the specified order;
8. TER, CER, FER, NGS, and raw error counts are reproducible;
9. manifests identify all non-paper implementation choices;
10. unit and synthetic end-to-end tests pass offline.

Matching the paper's reported accuracy is explicitly not an architectural acceptance criterion because it additionally depends on corpora, preprocessing, scale, and undocumented implementation details.

## 26. Paper ambiguities tracked as reproduction decisions

The first implementation MUST explicitly document and test these decisions:

| Decision | Recommended default |
|---|---|
| LM order | 5 |
| Stupid Backoff factor | 0.4 |
| Forward/backward score combination | sum of log-scores |
| Unknown n-gram handling | finite configured floor |
| Error-model smoothing | additive, alpha 0.1 |
| Initial substring alignment | unit edit-cost alignment |
| Alignment re-estimation | one estimation pass |
| Context tie-breaking | frequency, distance, Unicode order |
| Lambda optimizer | deterministic grid search |
| Sparse lambda buckets | global-lambda fallback |
| Logistic regression regularization | L2 |
| Threshold ties | choose higher threshold |
| Missing candidate features | indicator plus numeric zero |

No value in this table should be described as author-specified unless independently confirmed from the paper or released original implementation.
