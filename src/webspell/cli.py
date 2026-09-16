from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from webspell.demo import build_demo_model
from webspell.osm import OSMPreparationConfig, OSMTrainingConfig, QueryAlignmentAdapter, prepare_osm, train_osm, write_token_level_pairs
from webspell.osm.noise import DEFAULT_NOISE_WEIGHTS
from webspell.osm.alignment import TRAINING_ERROR_TYPE_WEIGHTS
from webspell.osm.training import audit_score_scales, evaluate_osm_model
from webspell.osm.paper import prepare_artificial_data, validate_typed_test, PROFILE
from webspell.pipeline import WebSpellModel
from webspell.scalable import LineCorpus, SQLiteCorpusStatistics, SQLiteSymSpellIndex


def _parse_positive_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError('sizes must be comma-separated integers') from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError('sizes must contain positive integers')
    return sizes


def _parse_noise_weights(value: str) -> tuple[tuple[str, int], ...]:
    weights = dict(DEFAULT_NOISE_WEIGHTS)
    try:
        for item in value.split(','):
            name, raw_weight = item.strip().split('=', 1)
            if name not in weights:
                raise argparse.ArgumentTypeError(f'unknown error type: {name}')
            weights[name] = int(raw_weight)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError('weights must be error_type=integer pairs') from error
    if any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
        raise argparse.ArgumentTypeError('weights must be non-negative with a positive total')
    return tuple(weights.items())


def _parse_training_weights(value: str) -> tuple[tuple[str, int], ...]:
    weights = dict(TRAINING_ERROR_TYPE_WEIGHTS)
    try:
        for item in value.split(','):
            name, raw_weight = item.strip().split('=', 1)
            if name not in weights:
                raise argparse.ArgumentTypeError(f'unknown training error type: {name}')
            weights[name] = int(raw_weight)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError('weights must be error_type=integer pairs') from error
    if any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
        raise argparse.ArgumentTypeError('weights must be non-negative with a positive total')
    return tuple(weights.items())


def _demo(text: str) -> int:
    model = build_demo_model()
    return _print_predictions(model, text)


def _print_predictions(model: WebSpellModel, text: str) -> int:
    predictions = model.predict(text)
    payload = {
        "input": text,
        "output": model.corrected_text(text, predictions),
        "tokens": [
            {
                "token": prediction.token.surface,
                "action": prediction.action,
                "correction": prediction.correction,
                "spellcheck_confidence": round(prediction.spellcheck_confidence, 6),
                "autocorrect_confidence": (
                    round(prediction.autocorrect_confidence, 6)
                    if prediction.autocorrect_confidence is not None
                    else None
                ),
                "top_candidates": [candidate.term for candidate in prediction.candidates[:3]],
            }
            for prediction in predictions
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _export_demo(output: str) -> int:
    model = build_demo_model()
    model.save(output)
    print(json.dumps({"saved_model": output}, ensure_ascii=False))
    return 0


def _prepare_scale(args: argparse.Namespace) -> int:
    corpus = LineCorpus(args.corpus, lowercase=args.lowercase)
    statistics = SQLiteCorpusStatistics.build(
        corpus,
        args.output,
        order=args.order,
        batch_tokens=args.batch_tokens,
    )
    index = SQLiteSymSpellIndex.build(
        args.output,
        max_distance=args.symspell_distance,
        minimum_frequency=args.minimum_frequency,
    )
    index.close()
    payload = {
        "artifact": args.output,
        "terms": statistics.term_count(),
        **statistics.metadata(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _prepare_osm(args: argparse.Namespace) -> int:
    config = OSMPreparationConfig(
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        noisy_variants_per_term=args.noisy_variants,
        minimum_length=args.minimum_length,
        character_error_rate=args.character_error_rate,
        clean_variants_per_term=args.clean_variants,
        noise_weights=args.noise_weights,
    )
    counts = prepare_osm(args.input, args.output, config)
    print(json.dumps({"output": args.output, **counts}, indent=2, ensure_ascii=False))
    return 0


def _train_osm(args: argparse.Namespace) -> int:
    config = OSMTrainingConfig(
        order=args.order,
        batch_tokens=args.batch_tokens,
        symspell_distance=args.symspell_distance,
        minimum_frequency=args.minimum_frequency,
        max_error_examples=args.max_error_examples,
        max_training_examples=args.max_training_examples,
        max_validation_examples=args.max_validation_examples,
        max_test_examples=args.max_test_examples,
        classifier_epochs=args.classifier_epochs,
        clean_ratio=args.clean_ratio,
        token_compatible_only=args.token_compatible_only,
        error_type_weights=args.training_error_weights,
        reuse_statistics=args.statistics,
        max_coverage_examples=args.max_coverage_examples,
        skip_evaluation=args.skip_evaluation,
        skip_coverage=args.skip_coverage,
        rebuild_statistics=args.rebuild_statistics,
        workers=args.workers,
        worker_batch_size=args.worker_batch_size,
        sampling_seed=args.sampling_seed,
        learning_curve_sizes=args.learning_curve_sizes,
        error_model_source=args.error_model_source,
        error_frequency_ratio=args.error_frequency_ratio,
        minimum_context_frequency=args.minimum_context_frequency,
    )
    state = train_osm(
        args.data,
        args.output,
        config,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(json.dumps({
        "output": args.output,
        "pair_coverage": state["pair_coverage"],
        "test_metrics": state["test_metrics"],
        "stage_times_seconds": state["stage_times_seconds"],
        "learning_curve": state["learning_curve"],
    }, indent=2, ensure_ascii=False))
    return 0


def _evaluate_osm_token(args: argparse.Namespace) -> int:
    counts = write_token_level_pairs(args.pairs, args.filtered_pairs)
    with WebSpellModel.load(args.model) as model:
        metrics = evaluate_osm_model(model, args.filtered_pairs, QueryAlignmentAdapter(order=model.ranker.language_model.config.order), max_examples=args.max_test_examples, sampling_seed=args.sampling_seed)
    payload = {'scope': 'token-level/same-token-count', 'source_pairs': args.pairs, 'filtered_pairs': args.filtered_pairs, 'filter_counts': counts, 'metrics': metrics}
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _audit_osm_scores(args: argparse.Namespace) -> int:
    with WebSpellModel.load(args.model) as model:
        audit = audit_score_scales(
            model, args.pairs,
            QueryAlignmentAdapter(order=model.ranker.language_model.config.order),
            max_examples=args.max_examples,
            sampling_seed=args.sampling_seed,
        )
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as stream:
            json.dump(audit, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


def _prepare_paper_baseline(args: argparse.Namespace) -> int:
    profile = prepare_artificial_data(
        args.data,
        args.output,
        variants_per_document=args.variants,
        seed=args.seed,
        character_error_rate=0.02,
    )
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    return 0


def _train_paper_baseline(args: argparse.Namespace) -> int:
    profile_path = Path(args.data) / 'reproduction-profile.json'
    if not profile_path.is_file():
        raise FileNotFoundError(
            f'{profile_path} is required; run prepare-paper-baseline first'
        )
    profile_state = json.loads(profile_path.read_text(encoding='utf-8'))
    if profile_state.get('profile') != PROFILE:
        raise ValueError('data is not the frozen WebSpell 2009 baseline profile')
    config = OSMTrainingConfig(
        order=args.order,
        batch_tokens=args.batch_tokens,
        # Mining needs the paper's length-dependent 1/2/3 neighborhood.
        symspell_distance=3,
        minimum_frequency=args.minimum_frequency,
        max_error_examples=args.max_error_examples,
        max_training_examples=args.max_training_examples,
        max_validation_examples=args.max_validation_examples,
        max_test_examples=args.max_test_examples,
        classifier_epochs=args.classifier_epochs,
        clean_ratio=(1.0 - 0.092) / 0.092,
        token_compatible_only=True,
        max_coverage_examples=args.max_coverage_examples,
        skip_coverage=args.skip_coverage,
        rebuild_statistics=args.rebuild_statistics,
        workers=args.workers,
        worker_batch_size=args.worker_batch_size,
        sampling_seed=args.sampling_seed,
        error_model_source='web',
        error_frequency_ratio=10.0,
        minimum_context_frequency=10,
        # Paper Section 5.2 says runtime suggestions were limited to distance 2.
        candidate_max_edit_distance=2,
        candidate_variant_generator='none',
        experiment_profile=PROFILE,
    )
    state = train_osm(
        args.data, args.output, config,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    target_profile = Path(args.output) / 'reproduction-profile.json'
    target_profile.write_text(
        json.dumps({**profile_state, 'resolved_training_config': state['config']}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps({
        'output': args.output,
        'profile': PROFILE,
        'test_metrics': state['test_metrics'],
        'stage_times_seconds': state['stage_times_seconds'],
    }, indent=2, ensure_ascii=False))
    return 0


def _evaluate_typed_test(args: argparse.Namespace) -> int:
    validation = validate_typed_test(args.pairs)
    with WebSpellModel.load(args.model) as model:
        metrics = evaluate_osm_model(
            model, args.pairs,
            QueryAlignmentAdapter(order=model.ranker.language_model.config.order),
            max_examples=0,
            sampling_seed=args.sampling_seed,
        )
    payload = {
        'scope': 'held-out-human-typed/evaluation-only',
        'validation': validation,
        'metrics': metrics,
    }
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WebSpell statistical prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="train and run the offline synthetic demo")
    demo.add_argument("--text", default="teh quik brwon fox")
    export = subparsers.add_parser("export-demo", help="train and save the synthetic model")
    export.add_argument("--output", required=True)
    check = subparsers.add_parser("check", help="load a saved model and check text")
    check.add_argument("--model", required=True)
    check.add_argument("--text", required=True)
    prepare = subparsers.add_parser(
        "prepare-scale",
        help="stream a line corpus into disk-backed vocabulary, LM, context, and index",
    )
    prepare.add_argument("--corpus", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--order", type=int, default=5)
    prepare.add_argument("--batch-tokens", type=int, default=50_000)
    prepare.add_argument("--symspell-distance", type=int, default=2, choices=(1, 2, 3))
    prepare.add_argument("--minimum-frequency", type=int, default=2)
    prepare.add_argument("--lowercase", action="store_true")
    osm = subparsers.add_parser("prepare-osm", help="extract and split search terms from an OSM PBF")
    osm.add_argument("--input", required=True, help="path to an .osm.pbf extract")
    osm.add_argument("--output", required=True)
    osm.add_argument("--train-ratio", type=float, default=0.8)
    osm.add_argument("--validation-ratio", type=float, default=0.1)
    osm.add_argument("--seed", type=int, default=2026)
    osm.add_argument("--noisy-variants", type=int, default=3)
    osm.add_argument("--minimum-length", type=int, default=2)
    osm.add_argument("--character-error-rate", type=float, default=0.02, help="deprecated compatibility option; taxonomy weights now control noise")
    osm.add_argument("--clean-variants", type=int, default=1, help="explicit clean-to-clean controls per term")
    osm.add_argument("--noise-weights", type=_parse_noise_weights, default=tuple(DEFAULT_NOISE_WEIGHTS.items()), help="comma-separated overrides such as missing_diacritics_full=25,keyboard_edit=15")
    train = subparsers.add_parser("train-osm", help="train the reproduced paper model from prepared OSM splits")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--order", type=int, default=5)
    train.add_argument("--batch-tokens", type=int, default=50_000)
    train.add_argument("--symspell-distance", type=int, default=2, choices=(1, 2, 3))
    train.add_argument("--minimum-frequency", type=int, default=1)
    train.add_argument("--max-error-examples", type=int, default=200_000, help="0 means all compatible examples")
    train.add_argument("--max-training-examples", type=int, default=50_000, help="0 means all compatible examples")
    train.add_argument("--max-validation-examples", type=int, default=10_000, help="0 means all compatible examples")
    train.add_argument("--max-test-examples", type=int, default=10_000, help="0 means all compatible examples")
    train.add_argument("--classifier-epochs", type=int, default=3)
    train.add_argument("--clean-ratio", type=float, default=2.0, help="clean-to-misspelled token ratio; 2 matches prepared-v3 token prevalence")
    train.add_argument("--training-error-weights", type=_parse_training_weights, default=tuple(TRAINING_ERROR_TYPE_WEIGHTS.items()), help="stratified positive-label weights by error type")
    train.add_argument("--include-structural-training", dest="token_compatible_only", action="store_false", help="also train token classifier on split/merge and address-symbol rows")
    train.set_defaults(token_compatible_only=True)
    train.add_argument("--statistics", default=None, help="optional path to existing precomputed SQLite statistics to reuse")
    train.add_argument("--max-coverage-examples", type=int, default=200_000, help="coverage rows per split; 0 means all")
    train.add_argument("--skip-evaluation", action="store_true", help="save the trained model without test evaluation")
    train.add_argument("--skip-coverage", action="store_true", help="skip the optional pair-coverage scan")
    train.add_argument("--rebuild-statistics", action="store_true", help="rebuild SQLite statistics after the corpus or index settings change")
    train.add_argument("--workers", type=int, default=1, help="feature-generation processes; use 2 on a dual-core machine when RAM permits")
    train.add_argument("--worker-batch-size", type=int, default=750, help="examples sent to each worker task")
    train.add_argument("--sampling-seed", type=int, default=2026, help="deterministic representative-sampling seed")
    train.add_argument("--learning-curve-sizes", type=_parse_positive_sizes, default=(), help="comma-separated classifier sample sizes, e.g. 25000,50000,100000")
    train.add_argument('--error-model-source', choices=('synthetic', 'web', 'hybrid'), default='synthetic', help='triple source; web implements paper Section 3.2, hybrid combines both')
    train.add_argument('--error-frequency-ratio', type=float, default=10.0, help='minimum intended/observed frequency ratio for web mining')
    train.add_argument('--minimum-context-frequency', type=int, default=10, help='minimum total count for a one-left/one-right context')
    evaluate_token = subparsers.add_parser('evaluate-osm-token', help='create and evaluate a separate same-token-count OSM test split')
    evaluate_token.add_argument('--model', required=True)
    evaluate_token.add_argument('--pairs', required=True, help='original noisy_pairs.csv (read-only)')
    evaluate_token.add_argument('--filtered-pairs', required=True, help='new filtered CSV to create')
    evaluate_token.add_argument('--output', default=None, help='optional JSON metrics output')
    evaluate_token.add_argument('--max-test-examples', type=int, default=0, help='0 evaluates every filtered pair')
    evaluate_token.add_argument('--sampling-seed', type=int, default=2026)
    audit = subparsers.add_parser('audit-osm-scores', help='report error/LM score scales and lambda by context bucket')
    audit.add_argument('--model', required=True)
    audit.add_argument('--pairs', required=True)
    audit.add_argument('--output', default=None)
    audit.add_argument('--max-examples', type=int, default=10000)
    audit.add_argument('--sampling-seed', type=int, default=2026)
    paper_prepare = subparsers.add_parser(
        'prepare-paper-baseline',
        help='create Section 3.4.1 artificial data from already split clean corpus',
    )
    paper_prepare.add_argument('--data', required=True, help='prepared OSM root; read-only')
    paper_prepare.add_argument('--output', required=True, help='new independent baseline data root')
    paper_prepare.add_argument('--variants', type=int, default=1)
    paper_prepare.add_argument('--seed', type=int, default=2026)
    paper_train = subparsers.add_parser(
        'train-paper-baseline',
        help='train frozen WebSpell 2009 System-7-style baseline',
    )
    paper_train.add_argument('--data', required=True)
    paper_train.add_argument('--output', required=True)
    paper_train.add_argument('--order', type=int, default=5)
    paper_train.add_argument('--batch-tokens', type=int, default=50_000)
    paper_train.add_argument('--minimum-frequency', type=int, default=1)
    paper_train.add_argument('--max-error-examples', type=int, default=0)
    paper_train.add_argument('--max-training-examples', type=int, default=0)
    paper_train.add_argument('--max-validation-examples', type=int, default=0)
    paper_train.add_argument('--max-test-examples', type=int, default=0)
    paper_train.add_argument('--classifier-epochs', type=int, default=3)
    paper_train.add_argument('--max-coverage-examples', type=int, default=0)
    paper_train.add_argument('--skip-coverage', action='store_true')
    paper_train.add_argument('--rebuild-statistics', action='store_true')
    paper_train.add_argument('--workers', type=int, default=1)
    paper_train.add_argument('--worker-batch-size', type=int, default=750)
    paper_train.add_argument('--sampling-seed', type=int, default=2026)
    typed = subparsers.add_parser(
        'evaluate-typed-test',
        help='validate and evaluate an adjudicated human-typed test set',
    )
    typed.add_argument('--model', required=True)
    typed.add_argument('--pairs', required=True)
    typed.add_argument('--output', default=None)
    typed.add_argument('--sampling-seed', type=int, default=2026)
    return parser


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = build_parser().parse_args()
    if args.command == "demo":
        return _demo(args.text)
    if args.command == "export-demo":
        return _export_demo(args.output)
    if args.command == "check":
        return _print_predictions(WebSpellModel.load(args.model), args.text)
    if args.command == "prepare-scale":
        return _prepare_scale(args)
    if args.command == "prepare-osm":
        return _prepare_osm(args)
    if args.command == "train-osm":
        return _train_osm(args)
    if args.command == 'evaluate-osm-token':
        return _evaluate_osm_token(args)
    if args.command == 'audit-osm-scores':
        return _audit_osm_scores(args)
    if args.command == 'prepare-paper-baseline':
        return _prepare_paper_baseline(args)
    if args.command == 'train-paper-baseline':
        return _train_paper_baseline(args)
    if args.command == 'evaluate-typed-test':
        return _evaluate_typed_test(args)
    raise AssertionError(f"unknown command: {args.command}")

if __name__ == "__main__":
    raise SystemExit(main())
