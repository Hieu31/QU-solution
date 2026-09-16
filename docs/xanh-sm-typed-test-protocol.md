# Xanh SM human-typed spellchecking test protocol

## Purpose and collection

Measure genuine user input independently of artificial training data. Recruit Vietnamese speakers across devices, input methods and regions. Ask them to type a shown location/address naturally. Keep a primary `retype_no_backspace` slice and, when privacy approval exists, a separate consented `free_search` slice. Do not ask participants to deliberately make errors; naturally clean queries must remain in the set.

## Required CSV

```text
noisy_query,correct_query,query_id,participant_id,source_text_id,collection_protocol,review_status
```

Recommended metadata: `device_class`, `keyboard_layout`, `input_method`, `region`, pseudonymous `session_id`, `annotator_ids`, and `adjudication_notes`. Never store phone numbers, precise trip coordinates or unrelated query history. `review_status` must equal `adjudicated`.

## Gold annotation

Two Vietnamese annotators independently assign intended text; a third adjudicates disagreement. Preserve acceptable aliases instead of forcing stylistic normalization. Mark ambiguous intent separately and exclude it from strict exact match while retaining it for candidate-recall diagnostics.

## Leakage controls

- Freeze the final set before viewing final metrics.
- Group by participant, source text and canonical POI/address identity.
- Keep development and sealed test sets disjoint by participant and location.
- Never use typed-test queries, corrections or derived pairs for vocabulary, counts, error mining, lambda, classifiers or thresholds.
- Hash/version the CSV; any change creates a new version.

## Reporting

Report natural prevalence, not an error-balanced headline. Cover POIs, streets, wards/districts, addresses, house numbers, alleys and transport hubs. Slice by protocol, clean/error, length, device/input method and seen/unseen location.

Primary metrics: TER, CER, FER and clean-query preservation. Secondary metrics: NGS/candidate recall, top-1, query exact match, E1-E5, latency and coverage. Publish counts and confidence intervals. Report E4 separately because false correction of a clean query has high production cost.

```powershell
uv run webspell evaluate-typed-test `
  --model artifacts\paper-baseline `
  --pairs data\typed\xanh-sm-typed-test-v1.csv `
  --output artifacts\paper-baseline\typed-test-v1-metrics.json
```
