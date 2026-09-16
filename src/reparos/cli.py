from __future__ import annotations

import argparse
import json

from reparos.data import prepare_improvement_regression_sets, prepare_reparos_data


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

    evaluation = subparsers.add_parser(
        'prepare-eval',
        help='build 90/10 head-tail Improvement and Regression sets',
    )
    evaluation.add_argument('--labeled-queries', required=True)
    evaluation.add_argument('--output', required=True)
    evaluation.add_argument('--examples-per-set', type=int, default=0)
    evaluation.add_argument('--seed', type=int, default=2026)
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
    if args.command == 'prepare-eval':
        counts = prepare_improvement_regression_sets(
            args.labeled_queries,
            args.output,
            examples_per_set=args.examples_per_set,
            seed=args.seed,
        )
        print(json.dumps({'output': args.output, **counts}, indent=2, ensure_ascii=False))
        return 0
    raise AssertionError(f'unknown command: {args.command}')


if __name__ == '__main__':
    raise SystemExit(main())
