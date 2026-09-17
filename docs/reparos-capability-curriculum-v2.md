# ReparoS capability curriculum v2

This dataset is a production-oriented experiment. It does not modify or replace
the frozen ReparoS paper baseline, and generation never starts model training.

## Frozen evaluation

`benchmark/reparos-user-centric-v2` contains 88 manually specified cases in 11
buckets. It includes bare location acronyms (`ltk`, `nct`, `hbt`, `dbp`, `pvh`,
`ntmk`, `nvl`) and abbreviation plus word-boundary errors. The current Base
scores 0% Top-1 and 0% Recall@5 on both new buckets. The manifest is
assistant-reviewed and still requires domain-owner sign-off before it becomes a
formal production acceptance set.

The 4K paired compositional benchmark and 10K stratified diagnostic remain
external tests. No benchmark row is copied into curriculum data.

## Training stages

1. `stage1-primitives`: clean/protected replay, missing diacritics, boundary,
   keyboard edits, Telex and VNI.
2. `stage2-composition`: one-operation replay plus two- and three-operation
   corruptions. Every noisy variant points directly to the fully clean target.
3. `stage3-lexical`: curated/contextual acronym, address abbreviation, lexical
   typo, address symbol, mixed replay and clean safety examples.

Automatically mined acronyms are contextual only. A bare acronym can only come
from `CURATED_ACRONYMS`; automatic OSM mining can never produce a bare mapping.
One percent of Stage 3 train rows is reserved for reviewed acronym templates so
reservoir sampling cannot drop this dictionary knowledge.

## Materialized profiles

| Profile | Stage 1 train | Stage 2 train | Stage 3 train | Validation per stage |
|---|---:|---:|---:|---:|
| pilot | 120,000 | 160,000 | 120,000 | 10,000 |
| full | 1,000,000 | 1,200,000 | 600,000 | 40,000 |

Locations:

```text
data/reparos/curriculum-v2/pilot
data/reparos/curriculum-v2/full
```

Each stage contains aligned `train.src`, `train.tgt`, `train.meta.jsonl`,
`validation.src`, `validation.tgt`, and `validation.meta.jsonl`. The metadata
records operation traces and both query/source group identifiers.

Recreate either profile with:

```powershell
uv run python scripts\prepare_curriculum_v2.py `
  --source data\osm\prepared-v4-leakfree `
  --output data\reparos\curriculum-v2\pilot `
  --profile pilot `
  --seed 2026
```

Use `--profile full` and the `full` output folder for the full materialization.

## Audit result

- All src/tgt/meta line counts match.
- Train/validation normalized target-group overlap is zero in every stage.
- Every curated bare acronym has exactly one canonical target.
- Full row counts match the resolved manifest.
- Test suite: 26 passed, 1 optional test skipped.

The `lexical_typo` lane is synthetic until the reviewed company dictionary is
available. Do not describe it as coverage of real brand/user typos. Add company
entries as a versioned reviewed resource rather than silently mixing them into
the generator.

## GPU notebooks

- `notebook/train_reparos.ipynb`: Google Colab + Drive.
- `notebook/train_reparos_kaggle.ipynb`: Kaggle attached datasets.

Both notebooks execute the same locked workflow:

1. load the trusted Base checkpoint, vocabulary and tokenizer;
2. train Pilot Stage 1, Stage 2 and Stage 3 sequentially;
3. convert every pilot checkpoint to CTranslate2 and run the frozen user,
   composition and stratified benchmarks;
4. write `pilot-gates.json` and stop if any stage gate fails;
5. require the operator to explicitly set `ALLOW_FULL_TRAIN=True`;
6. train Full Stage 1, Stage 2 and Stage 3 as a fresh branch from Base;
7. export and evaluate the final full checkpoint.

Full training cannot start merely because training code runs successfully. It
requires both passing quality gates and explicit operator acknowledgement.

For Colab, archive `data/reparos/curriculum-v2` as
`reparos-curriculum-v2.zip` in `MyDrive/reparos`. For Kaggle, upload that folder
as a dataset together with the trusted Base artifact directory and
`tokenizer.model`.
