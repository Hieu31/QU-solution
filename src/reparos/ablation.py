from __future__ import annotations

from pathlib import Path

from reparos.manifests import fingerprint_files, sha256_file, write_json


MODEL_VARIANTS: tuple[dict[str, object], ...] = (
    {
        'id': 'baseline-ff512',
        'purpose': 'locked Base reference',
        'overrides': {'transformer_ff': 512, 'dropout': 0.1, 'warmup_steps': 4000},
        'comparison_group': 'ffn-primary',
    },
    {
        'id': 'ff2048',
        'purpose': 'OpenNMT default-sized FFN alternative',
        'overrides': {'transformer_ff': 2048, 'dropout': 0.1, 'warmup_steps': 4000},
        'comparison_group': 'ffn-primary',
    },
    {
        'id': 'dropout-0',
        'purpose': 'dropout sensitivity check',
        'overrides': {'transformer_ff': 512, 'dropout': 0.0, 'warmup_steps': 4000},
        'comparison_group': 'local-default-sensitivity',
    },
    {
        'id': 'dropout-02',
        'purpose': 'dropout sensitivity check',
        'overrides': {'transformer_ff': 512, 'dropout': 0.2, 'warmup_steps': 4000},
        'comparison_group': 'local-default-sensitivity',
    },
    {
        'id': 'warmup-8000',
        'purpose': 'Noam warmup sensitivity check',
        'overrides': {'transformer_ff': 512, 'dropout': 0.1, 'warmup_steps': 8000},
        'comparison_group': 'local-default-sensitivity',
    },
)


def build_base_ablation_plan(
    data: str | Path,
    tokenizer_model: str | Path,
    output: str | Path,
    *,
    train_steps: int = 100_000,
    valid_steps: int = 5_000,
    save_checkpoint_steps: int = 5_000,
    batch_size: int = 4096,
    bucket_size: int = 8192,
    num_workers: int = 2,
    gpu_rank: int | None = None,
) -> Path:
    from reparos.training.opennmt import build_opennmt_config

    data_root, tokenizer, output_root = Path(data), Path(tokenizer_model), Path(output)
    inputs = [
        data_root / 'base' / 'train.src', data_root / 'base' / 'train.tgt',
        data_root / 'base' / 'validation.src', data_root / 'base' / 'validation.tgt',
    ]
    missing = [str(path) for path in [*inputs, tokenizer] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing ablation inputs: {missing}')
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for variant in MODEL_VARIANTS:
        run_root = output_root / str(variant['id'])
        overrides = dict(variant['overrides'])
        config = build_opennmt_config(
            data_root, tokenizer, run_root,
            train_steps=train_steps,
            valid_steps=valid_steps,
            save_checkpoint_steps=save_checkpoint_steps,
            batch_size=batch_size,
            bucket_size=bucket_size,
            num_workers=num_workers,
            transformer_ff=int(overrides['transformer_ff']),
            dropout=float(overrides['dropout']),
            warmup_steps=int(overrides['warmup_steps']),
            gpu_rank=gpu_rank,
        )
        runs.append({
            **variant,
            'config': str(config.resolve()),
            'command': f'uv run reparos train-opennmt --config "{config.resolve()}"',
        })
    plan = {
        'schema_version': 1,
        'profile': 'reparos-base-local-default-ablation',
        'execution': 'configs-only; no training was launched',
        'hypotheses': {
            'primary': 'FFN 512 vs 2048 tests the largest undisclosed architectural choice',
            'sensitivity': 'dropout and warmup are varied one factor at a time around FFN 512',
        },
        'controlled': {
            'published_architecture': '1 encoder / 1 decoder / 8 heads / hidden 128',
            'optimizer': 'Adam lr=1.0 beta1=0.8 beta2=0.998 epsilon=1e-8',
            'data_sha256': fingerprint_files(inputs),
            'tokenizer_sha256': sha256_file(tokenizer),
            'seed': 2026,
            'train_steps': train_steps,
            'validation_interval': valid_steps,
            'checkpoint_interval': save_checkpoint_steps,
            'selection_rule': 'lowest validation loss; report same-step checkpoint as sensitivity',
            'decoding': 'shared decoding-config.json from every run',
        },
        'runs': runs,
        'evaluation': {
            'primary_metrics': [
                'Improvement exact query accuracy',
                'Regression exact query accuracy',
                'clean false-correction rate',
            ],
            'secondary_metrics': [
                'top-10 oracle accuracy', 'parameter count', 'artifact bytes',
                'CTranslate2 CPU p50/p95/p99 latency at concurrency 1',
            ],
            'selection': [
                'Reject any variant with worse clean false-correction rate beyond the predeclared tolerance.',
                'Among remaining variants, select by Improvement accuracy; break ties with Regression accuracy then p95 latency.',
                'Do not inspect the sealed test set until the variant is selected on validation/development data.',
            ],
        },
        'separate_tokenizer_study': {
            'reason': 'SentencePiece model type and normalization require retraining the tokenizer and are not mixed into the model-only study.',
            'lanes': ['unigram+nmt_nfkc_cf (locked)', 'bpe+nmt_nfkc_cf', 'unigram+identity/case-preserving'],
        },
    }
    destination = output_root / 'ablation-plan.json'
    write_json(destination, plan)
    return destination
