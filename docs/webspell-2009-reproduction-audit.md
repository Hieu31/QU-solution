# WebSpell 2009 reproduction audit

Status: **Exact** means the disclosed behavior is implemented; **Approximate** means source data or undisclosed details differ; **Missing** is unavailable; **Diverged** is an intentional Xanh SM extension.

| Paper component | Status | Current code and limitation |
|---|---:|---|
| 3.1 noisy term list | Approximate | Frequencies come from OSM, not >1B Web pages/top 10M tokens. Paper filters are not disclosed. |
| 3.2 substring model, max substring 2 | Exact | Structure matches; smoothing/prior/alignment details are undisclosed and therefore approximate numerically. |
| Close terms: distance 1/2/3 for lengths <=4/<=12/>12 | Exact | Branch A builds the index to distance 3 for mining. |
| Intended frequency >=10x observed | Exact | Forced by Branch A. |
| One-left/one-right context, total >=10 | Exact | Unique contextual winner is used; ties are discarded. |
| Web-mined error triples | Approximate | Algorithm matches, but OSM scale/domain/context differ. Artificial pairs are excluded. |
| Top 20 by `P(w|s) * f(s)` | Exact | Implemented as error log-score plus log frequency. |
| Runtime suggestion edit distance 2 | Exact | Separate from the distance-3 mining index, following Section 5.2. |
| Vietnamese candidate variants | Diverged | Disabled in Branch A; retained only in the production adaptation. |
| Forward/backward Stupid Backoff LM | Approximate | Formula matches; corpus, pruning, order and backoff details are not fully disclosed. Order 5 is a local assumption. |
| Lambda per context bucket optimized for MRR | Exact | Objective/bucketing match; the local 0..10/0.25 grid is undisclosed by the paper. |
| Two spell classifiers plus gated autocorrect classifier | Exact | A fallback exists only for degenerate tiny test fixtures. |
| Five LR feature groups | Exact | Score deltas, case, suggestion count, length and context match; optimizer/scaling are undisclosed. |
| Blacklist numbers, symbols and one-character tokens | Exact | Required by System 7. |
| Independent F1 threshold tuning on dev | Exact | Histogram resolution is local. |
| 2 errors/100 chars; equal deletion/transposition/insertion | Approximate | Rate/operations match. Each OSM line is a document proxy; whitespace is preserved and each character gets at most one trial, both undisclosed choices. |
| Non-overlapping artificial train/dev/test | Approximate | Existing POI-group split is retained, but fragments replace news documents. |
| Human-typed evaluation | Missing | Validator/evaluator exist; no collected and adjudicated Xanh SM typed corpus exists yet. |
| E1-E5, CER, FER, TER, NGS | Exact | Query accuracy and per-error-type metrics are additional. |
| Xanh SM taxonomy and synthetic/hybrid triples | Diverged | Excluded from Branch A; valid only for adaptation/ablation. |

## Defensible claim

This is a **WebSpell-2009-inspired reproduction using Vietnamese OSM substitute data**, not an exact numerical reproduction. Table 2/3 numbers cannot be reproduced without the private Web counts, news data, typed sets and undisclosed training details.

## Frozen Branch A boundary

1. `corpus.txt` supplies vocabulary, frequency, LM and Section 3.2 web-context triples.
2. `noisy_pairs.csv` contains only Section 3.4.1 artificial data for lambda/classifier train-dev and artificial test evaluation.
3. Human-typed CSV is external, adjudicated, immutable and evaluation-only.
4. Branch A forbids Vietnamese variants, synthetic/hybrid error triples and Xanh SM taxonomy sampling.

