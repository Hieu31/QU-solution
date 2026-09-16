# ReparoS 2023 reproduction audit

Source: Kakkar et al., *Search Query Spell Correction with Weak Supervision in E-commerce* (ACL Industry 2023).

| Component | Status | Local reproduction |
|---|---:|---|
| Error taxonomy: edit, compounding, phonetic, combined | Exact | Dataset metadata preserves each class. |
| Edit operations: deletion, adjacent swap, keyboard replacement, insertion | Approximate | Operations exist; keyboard replacement currently uses a generic alphabet because keyboard layout/probabilities are undisclosed. |
| Brill-Moore-proportional edit sampling | Missing | Requires `(intended, observed, frequency)` query-log triples. Random edit fallback is labeled approximate. |
| Max two misspelled words in ~90% queries | Exact | Generator caps corruption at two words. The source statistic cannot be verified on OSM. |
| Correct-correct pairs | Exact | Configurable clean ratio is included to reduce overcorrection. Paper does not disclose its ratio. |
| Phonetic transliteration pipeline | Missing by default | Paper uses proprietary English-Hindi catalogue pairs, two transliteration Transformers and Hindi reformulation errors. External `correct_word,phonetic_variant` data can populate this lane but is not numerically equivalent. |
| Soundex/Metaphone candidates | Missing | Their mixing rate and implementation are undisclosed. |
| Query-chain reformulations using LLR/PMI/CTR/edit guardrails | Missing | Requires session logs and thresholds not published. |
| Weak click feedback | Approximate | CTR-filtered `(user_query, corrected_query)` ingestion exists; production intervention logs and CTR threshold are proprietary/undisclosed. |
| Base synthetic curriculum | Exact structure | Diverse synthetic+clean pairs. Scale is local rather than 270M. |
| C1 complex-only fine-tuning | Exact structure | Only edit/phonetic+compounding data. |
| C2 weak-feedback fine-tuning | Exact structure | Only available when a feedback CSV is supplied. |
| Transformer NMT | Config captured | One encoder and decoder layer, 8 heads, hidden 128. Training runtime is not bundled because OpenNMT/SentencePiece are optional external tools. |
| SentencePiece vocabulary | Config captured | Shared/independent choice and model type are not disclosed; vocabulary size is 8K. |
| Adam | Exact disclosed values | LR 1.0, beta1 .8, beta2 .998, epsilon 1e-8; fine-tuning LR .0001 and all parameters updated. |
| Beam decoding | Exact disclosed value | Beam width 10. Other decoding penalties are undisclosed. |
| Training schedule | Missing | Steps/epochs, batch, dropout, FFN size, warmup/schedule, checkpoint selection and random seeds are not published. |
| 300K seed queries from top 20% per ~50 categories by frequency/CTR | Missing | OSM corpus is a substitute and lacks category/query CTR. |
| 72K human-labelled evaluation | Missing | Must be collected for Xanh SM. |
| Regression/Improvement mixtures | Exact | Builder makes 90/10 head-tail and 10/90 mixtures from unique adjudicated queries. |
| Query-level exact accuracy | Ready | Requires model predictions; no model is trained automatically. |
| Production two-stage ML ranker | Missing | Ranker features/training data are not disclosed. Paper results explicitly omit it offline. |
| A/B experiment | Not reproducible offline | Needs product traffic, metrics and experiment infrastructure. |

## Defensible claim

Branch B reproduces the **published ReparoS experimental structure and data contracts**, not Flipkart's numerical result. Exact reproduction is impossible from the paper alone because the core phonetic resources, query/click logs, evaluation labels, ranker and several training parameters are private or undisclosed.

## Fair comparison with Branch A

- Use the same clean OSM source split and the same sealed human-typed query IDs.
- Fit/tune each method only on its own training/development lanes.
- Compare query exact accuracy, E1-E5, TER/CER/FER, clean preservation, top-k and latency.
- Report ReparoS without C2 separately when real click feedback is unavailable.
- Never compare Branch A artificial test against Branch B Improvement/Regression as if they were the same population.

