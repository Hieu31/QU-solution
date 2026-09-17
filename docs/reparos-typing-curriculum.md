# ReparoS production typing curriculum

This is a production experiment separate from the frozen ReparoS-Base paper
baseline. It prepares data only; it never launches a training job.

## Stages

1. `stage1-ime`: 70% Telex/VNI transformations and 30% clean examples.
2. `stage2-abbreviation`: 60% contextual abbreviation, 20% IME replay and
   20% clean replay.
3. `stage3-mixed`: 50% combined errors, 15% IME replay, 15% abbreviation
   replay and 20% clean replay.
4. `stage4-domain`: 60% canonical domain data, 15% mixed replay, 10% IME
   replay, 5% abbreviation replay and 10% clean replay.

Stage 4 uses canonical OSM/location queries as the domain lane until reviewed
real user logs are available. Test rows remain diagnostic and are never used as
replay data. Every row has `group_id`, `split`, `stage`, `error_type`,
`source_family`, and `is_clean` in its adjacent `meta.jsonl` file.

## Generate

```powershell
uv run python scripts\prepare_typing_curriculum.py `
  --source data\osm\prepared-v4-leakfree `
  --output data\reparos\typing-curriculum-v1 `
  --seed 2026 `
  --ime-variants 2 `
  --abbreviation-variants 2 `
  --mixed-variants 2
```

The source dataset and all existing model artifacts are read-only inputs. The
output contains parallel `*.src`, `*.tgt`, and `*.meta.jsonl` files plus a
resolved `curriculum-manifest.json`.

Train sequentially from the best validation checkpoint of the preceding stage.
Do not freeze encoder, decoder, or embeddings between stages. Freeze the Stage
3 artifact only as an evaluation artifact; copy it before Stage 4 fine-tuning.

## Required gates

- Stage 1: Telex/VNI exact accuracy and clean preservation.
- Stage 2: abbreviation exact accuracy and abbreviation hard-negative safety.
- Stage 3: exact query accuracy per error type and regression rate.
- Stage 4: production-domain macro score, false-correction rate, and regression.

Before production use, replace assumed mixture weights with reviewed user-log
statistics and extend the clean lane with business names, acronyms, route codes,
foreign text, and other abbreviation hard negatives.
