# ReparoS Base Architecture Lock and Ablation Protocol

## Baseline claim

The locked artifact is a paper-aligned reproduction of the published ReparoS-Base
architecture. It is not a full ReparoS-C2 or Flipkart production reproduction.
Every run contains:

- `opennmt-base.json`: executable OpenNMT-py configuration.
- `resolved-architecture.json`: published, inferred, and local-default fields.
- `decoding-config.json`: one decoding contract for OpenNMT and CTranslate2.
- `opennmt-input-manifest.json`: hashes of data and tokenizer inputs.

Generation fails when a published field such as layer count, head count, hidden
size, optimizer, learning rate, or Adam parameters does not match the paper.

## Model-only ablation

The primary comparison is FFN 512 versus FFN 2048. FFN size is the largest
undisclosed architectural choice: 512 follows the common `4 * hidden_size`
ratio, while 2048 is the OpenNMT-py default. Neither value is claimed by the
paper.

Secondary one-factor-at-a-time checks are:

| Run | FFN | Dropout | Warmup |
|---|---:|---:|---:|
| `baseline-ff512` | 512 | 0.1 | 4,000 |
| `ff2048` | 2,048 | 0.1 | 4,000 |
| `dropout-0` | 512 | 0.0 | 4,000 |
| `dropout-02` | 512 | 0.2 | 4,000 |
| `warmup-8000` | 512 | 0.1 | 8,000 |

All runs must use identical data hashes, tokenizer, seed, training steps,
validation/checkpoint intervals, published optimizer fields, and decoding.

## Evaluation and selection

Select on development/validation data before opening the sealed test set.

1. Reject a variant if its clean false-correction rate exceeds the predeclared
   tolerance relative to the locked baseline.
2. Select remaining variants by Improvement exact-query accuracy.
3. Break ties by Regression exact-query accuracy, then CTranslate2 CPU p95
   latency at concurrency one.
4. Report top-10 oracle accuracy, parameter count, artifact size, and p50/p95/p99
   latency even when they are not the selection metric.
5. Compare the selected variant once on the sealed test set.

The clean false-correction tolerance must be written before training; the paper
does not supply one. Do not select it after observing test results.

## Separate tokenizer study

SentencePiece model type and normalization are not mixed into the model-only
study because they require a new tokenizer and change the input representation.
Run them as a separate experiment using:

- Unigram + `nmt_nfkc_cf` (locked tokenizer baseline).
- BPE + `nmt_nfkc_cf`.
- Unigram + a case-preserving normalization suitable for Vietnamese location
  search.

Each tokenizer lane must regenerate all five model configs or, for a smaller
budget, only `baseline-ff512` and the FFN winner selected previously.

## Commands

Generate the plan and configs without training:

```powershell
uv run reparos plan-opennmt-ablation `
  --data data\reparos\prepared-v1 `
  --tokenizer artifacts\reparos\tokenizer-v1 `
  --output artifacts\reparos\base-ablation-v1 `
  --num-workers 2
```

The generated `ablation-plan.json` contains one explicit training command per
run. Execute only the desired commands manually.
