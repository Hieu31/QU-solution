from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from reparos.data import (
    prepare_improvement_regression_sets,
    prepare_production_data,
    prepare_query_group_split,
    prepare_reparos_data,
)


def _tokenizer_model(value: str) -> Path:
    path = Path(value)
    return path / 'tokenizer.model' if path.is_dir() else path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='reparos',
        description='Independent ReparoS 2023 reproduction pipeline',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare = subparsers.add_parser(
        'prepare',
        help='prepare Base/C1/C2 curriculum data without starting training',
    )
    prepare.add_argument('--data', required=True, help='group-split clean corpus root')
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--variants', type=int, default=4)
    prepare.add_argument('--clean-ratio', type=float, default=1.0)
    prepare.add_argument('--seed', type=int, default=2026)
    prepare.add_argument('--phonetic-pairs', default=None, help='CSV: correct_word,phonetic_variant')
    prepare.add_argument('--weak-feedback', default=None, help='CSV: user_query,corrected_query,corrected_query_ctr')
    prepare.add_argument('--minimum-feedback-ctr', type=float, default=0.1)

    production = subparsers.add_parser(
        'prepare-production',
        help='convert leak-free WebSpell pairs to ReparoS data without resampling',
    )
    production.add_argument('--data', required=True, help='prepared WebSpell root')
    production.add_argument('--output', required=True)

    resplit = subparsers.add_parser(
        'resplit-query-groups',
        help='re-split clean corpora by normalized query text to prevent leakage',
    )
    resplit.add_argument('--data', required=True)
    resplit.add_argument('--output', required=True)
    resplit.add_argument('--train-ratio', type=float, default=0.8)
    resplit.add_argument('--validation-ratio', type=float, default=0.1)
    resplit.add_argument('--seed', type=int, default=2026)

    evaluation = subparsers.add_parser(
        'prepare-eval',
        help='build 90/10 head-tail Improvement and Regression sets',
    )
    evaluation.add_argument('--labeled-queries', required=True)
    evaluation.add_argument('--output', required=True)
    evaluation.add_argument('--examples-per-set', type=int, default=0)
    evaluation.add_argument('--seed', type=int, default=2026)

    pilot = subparsers.add_parser(
        'prepare-vietnamese-pilot',
        help='build a stratified Vietnamese search pilot separate from the paper baseline',
    )
    pilot.add_argument('--data', required=True, help='group-split clean corpus root')
    pilot.add_argument('--output', required=True)
    pilot.add_argument('--train-per-class', type=int, default=4096)
    pilot.add_argument('--validation-per-class', type=int, default=512)
    pilot.add_argument('--train-clean-multiplier', type=int, default=1)
    pilot.add_argument('--seed', type=int, default=2026)

    tokenizer = subparsers.add_parser('train-tokenizer', help='train an 8K SentencePiece model from Base train data')
    tokenizer.add_argument('--data', required=True)
    tokenizer.add_argument('--output', required=True)
    tokenizer.add_argument('--vocab-size', type=int, default=8000)

    train = subparsers.add_parser('train', help='train the ReparoS-Base reference Transformer')
    train.add_argument('--stage', choices=('base',), default='base')
    train.add_argument('--data', required=True)
    train.add_argument('--tokenizer', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--epochs', type=int, default=10)
    train.add_argument('--batch-size', type=int, default=64)
    train.add_argument('--device', default='auto')

    predict = subparsers.add_parser('predict', help='predict with the reference Base checkpoint')
    predict.add_argument('--model', required=True)
    predict.add_argument('--tokenizer', required=True)
    predict.add_argument('--text', required=True)
    predict.add_argument('--beam-size', type=int, default=10)
    predict.add_argument('--device', default='cpu')

    score = subparsers.add_parser('evaluate-predictions', help='score precomputed top-k predictions without model dependencies')
    score.add_argument('--input', required=True, help='CSV with noisy_query,correct_query,hypotheses_json')
    score.add_argument('--output', default=None)

    export = subparsers.add_parser('export-ctranslate2', help='convert an OpenNMT-py Base checkpoint for CPU serving')
    export.add_argument('--model', required=True)
    export.add_argument('--tokenizer', required=True)
    export.add_argument('--output', required=True)
    export.add_argument('--compute-type', default='float32')
    export.add_argument(
        '--trust-checkpoint', action='store_true',
        help='allow pickle deserialization only for a checkpoint you trust',
    )

    onmt_config = subparsers.add_parser('build-opennmt-config', help='create paper-aligned OpenNMT-py Base config')
    onmt_config.add_argument('--data', required=True)
    onmt_config.add_argument('--tokenizer', required=True)
    onmt_config.add_argument('--output', required=True)
    onmt_config.add_argument('--train-steps', type=int, default=100000)
    onmt_config.add_argument('--valid-steps', type=int, default=5000)
    onmt_config.add_argument('--save-checkpoint-steps', type=int, default=5000)
    onmt_config.add_argument('--batch-size', type=int, default=4096)
    onmt_config.add_argument('--bucket-size', type=int, default=8192)
    onmt_config.add_argument('--num-workers', type=int, default=2)
    onmt_config.add_argument('--transformer-ff', type=int, default=512)
    onmt_config.add_argument('--dropout', type=float, default=0.1)
    onmt_config.add_argument('--warmup-steps', type=int, default=4000)
    onmt_config.add_argument('--gpu-rank', type=int, default=None)

    ablation = subparsers.add_parser('plan-opennmt-ablation', help='write controlled Base ablation configs without training')
    ablation.add_argument('--data', required=True)
    ablation.add_argument('--tokenizer', required=True)
    ablation.add_argument('--output', required=True)
    ablation.add_argument('--train-steps', type=int, default=100000)
    ablation.add_argument('--valid-steps', type=int, default=5000)
    ablation.add_argument('--save-checkpoint-steps', type=int, default=5000)
    ablation.add_argument('--batch-size', type=int, default=4096)
    ablation.add_argument('--bucket-size', type=int, default=8192)
    ablation.add_argument('--num-workers', type=int, default=2)
    ablation.add_argument('--gpu-rank', type=int, default=None)

    onmt_train = subparsers.add_parser('train-opennmt', help='build vocabulary and train Base using OpenNMT-py')
    onmt_train.add_argument('--config', required=True)

    onmt_vocab = subparsers.add_parser('build-opennmt-vocab', help='validate config and build OpenNMT-py vocabularies')
    onmt_vocab.add_argument('--config', required=True)

    parity = subparsers.add_parser('check-ctranslate2-parity', help='compare OpenNMT and CTranslate2 beam outputs')
    parity.add_argument('--checkpoint', required=True)
    parity.add_argument('--model', required=True, help='converted CTranslate2 directory')
    parity.add_argument('--tokenizer', required=True)
    parity.add_argument('--queries', required=True)
    parity.add_argument('--beam-size', type=int, default=10)
    parity.add_argument('--n-best', type=int, default=10)
    parity.add_argument('--decoding-config', default=None, help='resolved decoding-config.json; overrides beam/n-best flags')
    parity.add_argument('--output', default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == 'prepare':
        manifest = prepare_reparos_data(
            args.data,
            args.output,
            variants_per_query=args.variants,
            clean_ratio=args.clean_ratio,
            seed=args.seed,
            phonetic_pairs=args.phonetic_pairs,
            weak_feedback=args.weak_feedback,
            minimum_feedback_ctr=args.minimum_feedback_ctr,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.command == 'prepare-production':
        manifest = prepare_production_data(args.data, args.output)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.command == 'resplit-query-groups':
        manifest = prepare_query_group_split(
            args.data, args.output,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.command == 'prepare-eval':
        counts = prepare_improvement_regression_sets(
            args.labeled_queries,
            args.output,
            examples_per_set=args.examples_per_set,
            seed=args.seed,
        )
        print(json.dumps({'output': args.output, **counts}, indent=2, ensure_ascii=False))
        return 0
    if args.command == 'prepare-vietnamese-pilot':
        from reparos.pilot import prepare_vietnamese_search_pilot
        manifest = prepare_vietnamese_search_pilot(
            args.data, args.output,
            train_per_class=args.train_per_class,
            validation_per_class=args.validation_per_class,
            train_clean_multiplier=args.train_clean_multiplier,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.command == 'train-tokenizer':
        from reparos.config import TokenizerConfig
        from reparos.tokenization import train_sentencepiece
        data = Path(args.data)
        manifest = train_sentencepiece(
            (data / 'base' / 'train.src', data / 'base' / 'train.tgt'),
            args.output,
            TokenizerConfig(vocab_size=args.vocab_size),
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.command == 'train':
        from reparos.config import TrainingConfig
        from reparos.training import train_base
        manifest = train_base(
            args.data, _tokenizer_model(args.tokenizer), args.output,
            training_config=TrainingConfig.for_stage(
                'base', epochs=args.epochs, batch_size=args.batch_size, device=args.device,
            ),
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.command == 'predict':
        from reparos.inference import ReferencePredictor
        result = ReferencePredictor(args.model, _tokenizer_model(args.tokenizer), args.device).predict(args.text, args.beam_size)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == 'evaluate-predictions':
        from reparos.evaluation import evaluate_queries
        with Path(args.input).open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        metrics = evaluate_queries(
            [row['noisy_query'] for row in rows],
            [row['correct_query'] for row in rows],
            [json.loads(row['hypotheses_json']) for row in rows],
        )
        if args.output:
            Path(args.output).write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == 'export-ctranslate2':
        from reparos.serving import export_opennmt_checkpoint
        manifest = export_opennmt_checkpoint(
            args.model, _tokenizer_model(args.tokenizer), args.output,
            quantization=args.compute_type, trust_checkpoint=args.trust_checkpoint,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    if args.command == 'build-opennmt-config':
        from reparos.training.opennmt import build_opennmt_config
        path = build_opennmt_config(
            args.data, _tokenizer_model(args.tokenizer), args.output,
            train_steps=args.train_steps, valid_steps=args.valid_steps,
            save_checkpoint_steps=args.save_checkpoint_steps,
            batch_size=args.batch_size, bucket_size=args.bucket_size,
            num_workers=args.num_workers, transformer_ff=args.transformer_ff,
            dropout=args.dropout, warmup_steps=args.warmup_steps,
            gpu_rank=args.gpu_rank,
        )
        print(json.dumps({'config': str(path)}, indent=2))
        return 0
    if args.command == 'plan-opennmt-ablation':
        from reparos.ablation import build_base_ablation_plan
        path = build_base_ablation_plan(
            args.data, _tokenizer_model(args.tokenizer), args.output,
            train_steps=args.train_steps, valid_steps=args.valid_steps,
            save_checkpoint_steps=args.save_checkpoint_steps,
            batch_size=args.batch_size, bucket_size=args.bucket_size,
            num_workers=args.num_workers, gpu_rank=args.gpu_rank,
        )
        print(json.dumps({'ablation_plan': str(path), 'training_started': False}, indent=2))
        return 0
    if args.command == 'train-opennmt':
        from reparos.training.opennmt import train_opennmt
        print(json.dumps(train_opennmt(args.config), indent=2, ensure_ascii=False))
        return 0
    if args.command == 'build-opennmt-vocab':
        from reparos.training.opennmt import build_opennmt_vocabulary
        build_opennmt_vocabulary(args.config)
        print(json.dumps({'config': args.config, 'vocabulary_built': True}, indent=2))
        return 0
    if args.command == 'check-ctranslate2-parity':
        from reparos.architecture import DecodingConfig
        from reparos.serving.parity import compare_opennmt_ctranslate2
        decoding = (
            DecodingConfig(**json.loads(Path(args.decoding_config).read_text(encoding='utf-8')))
            if args.decoding_config else
            DecodingConfig(beam_size=args.beam_size, num_hypotheses=args.n_best)
        )
        report = compare_opennmt_ctranslate2(
            args.checkpoint, args.model, _tokenizer_model(args.tokenizer), args.queries,
            output=args.output, decoding=decoding,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    raise AssertionError(f'unknown command: {args.command}')


if __name__ == '__main__':
    raise SystemExit(main())
