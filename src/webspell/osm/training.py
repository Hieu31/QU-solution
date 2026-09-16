from __future__ import annotations

import json
import random
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

from webspell.candidates import CandidateRanker, LambdaGrid
from webspell.config import CandidateConfig
from webspell.confidence import ConfidenceModel, TrainingRow
from webspell.confidence.model import is_blacklisted
from webspell.error_model import SubstringErrorModel
from webspell.evaluation import EvaluationMetrics, evaluate_predictions
from webspell.osm.alignment import (
    AlignmentDiagnostics,
    QueryAlignmentAdapter,
    TRAINING_ERROR_TYPE_WEIGHTS,
    iter_sampled_query_rows,
)
from webspell.mining import ClosePair, iter_close_pairs, iter_error_triples
from webspell.pipeline import WebSpellModel, iter_training_rows
from webspell.scalable import (
    LineCorpus, SQLiteBidirectionalLanguageModel, SQLiteCorpusStatistics,
    SQLiteSymSpellIndex, StreamingClassifierConfig,
    StreamingConfidenceTrainer, tune_lambdas_streaming,
)
from webspell.scalable.context import SQLiteContextStatistics
from webspell.serialization import save_model_bundle
from webspell.text import SimpleTokenizer, VietnameseVariantGenerator
from webspell.types import Context, ErrorTriple, LabeledToken


@dataclass(frozen=True)
class OSMTrainingConfig:
    order: int = 5
    batch_tokens: int = 50_000
    symspell_distance: int = 2
    minimum_frequency: int = 1
    max_error_examples: int = 200_000
    max_training_examples: int = 50_000
    max_validation_examples: int = 10_000
    max_test_examples: int = 10_000
    classifier_epochs: int = 3
    clean_ratio: float = 2.0
    token_compatible_only: bool = True
    error_type_weights: tuple[tuple[str, int], ...] = tuple(TRAINING_ERROR_TYPE_WEIGHTS.items())
    reuse_statistics: str | None = None
    max_coverage_examples: int = 200_000
    skip_evaluation: bool = False
    skip_coverage: bool = False
    rebuild_statistics: bool = False
    workers: int = 1
    worker_batch_size: int = 750
    sampling_seed: int = 2026
    learning_curve_sizes: tuple[int, ...] = ()
    error_model_source: str = 'synthetic'
    error_frequency_ratio: float = 10.0
    minimum_context_frequency: int = 10
    candidate_max_edit_distance: int = 3
    candidate_variant_generator: str = 'vietnamese'
    experiment_profile: str = 'xanh-sm-adaptation'


def iter_pair_examples(
    path: str | Path,
    order: int = 5,
    limit: int = 0,
    include_clean: bool = False,
    clean_ratio: float = 1.0,
) -> Iterator[LabeledToken]:
    """Compatibility wrapper yielding LabeledToken instances via QueryAlignmentAdapter."""
    adapter = QueryAlignmentAdapter(order=order)
    yield from adapter.iter_examples(
        path, limit=limit, include_clean=include_clean, clean_ratio=clean_ratio
    )


def count_pair_coverage(
    path: str | Path,
    order: int = 5,
    max_examples: int = 0,
    sampling_seed: int = 2026,
) -> dict[str, int]:
    """Analyze query pair alignment coverage and diagnostics."""
    adapter = QueryAlignmentAdapter(order=order)
    _, diagnostics = adapter.mine_error_triples(
        path, max_examples=max_examples, sampling_seed=sampling_seed
    )
    return diagnostics.to_dict()


def _confidence_state(model: ConfidenceModel) -> dict[str, object]:
    return {
        "with_suggestions": model.with_suggestions.to_state(),
        "without_suggestions": model.without_suggestions.to_state(),
        "autocorrect": model.autocorrect.to_state(),
        "thresholds": {
            "with_suggestions": model.with_threshold,
            "without_suggestions": model.without_threshold,
            "autocorrect": model.autocorrect_threshold,
        },
    }


def _balanced_training_subset(
    rows: list[TrainingRow],
    size: int,
    clean_ratio: float,
    seed: int = 2026,
    error_type_weights: dict[str, int] | None = None,
) -> list[TrainingRow]:
    if size >= len(rows):
        return rows
    clean_target = int(size * clean_ratio / (1.0 + clean_ratio))
    error_target = size - clean_target
    clean = [row for row in rows if not row.misspelled]
    random.Random(f'{seed}:clean').shuffle(clean)
    weights = dict(error_type_weights or TRAINING_ERROR_TYPE_WEIGHTS)
    buckets: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        if row.misspelled:
            kind = 'combined' if row.error_type.startswith('combined:') else row.error_type
            buckets[kind].append(row)
    active = [(name, weight) for name, weight in weights.items() if weight > 0]
    total_weight = sum(weight for _, weight in active)
    targets = {name: int(error_target * weight / total_weight) for name, weight in active}
    for name, _ in active[:error_target - sum(targets.values())]:
        targets[name] += 1
    errors: list[TrainingRow] = []
    for name, target in targets.items():
        bucket = buckets.get(name, [])
        random.Random(f'{seed}:error:{name}').shuffle(bucket)
        errors.extend(bucket[:target])
    selected = errors + clean[:clean_target]
    if len(selected) < size:
        selected_ids = {id(row) for row in selected}
        selected.extend(row for row in rows if id(row) not in selected_ids)
    selected = selected[:size]
    random.Random(f'{seed}:mix:{size}').shuffle(selected)
    return selected


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _classifier_metrics(
    model: ConfidenceModel,
    rows: list[TrainingRow],
) -> dict[str, float | int]:
    spell_tp = spell_fp = spell_fn = 0
    auto_tp = auto_fp = auto_fn = 0
    for row in rows:
        spell_predicted = (
            model.spellcheck_probability(row.features, row.has_suggestions)
            >= model.spellcheck_threshold(row.has_suggestions)
        )
        spell_tp += int(spell_predicted and row.misspelled)
        spell_fp += int(spell_predicted and not row.misspelled)
        spell_fn += int(not spell_predicted and row.misspelled)
        auto_predicted = (
            spell_predicted
            and row.has_suggestions
            and model.autocorrect_probability(row.features)
            >= model.autocorrect_threshold
        )
        auto_tp += int(auto_predicted and row.top_is_correct)
        auto_fp += int(auto_predicted and not row.top_is_correct)
        auto_fn += int(not auto_predicted and row.top_is_correct)
    spell_precision, spell_recall, spell_f1 = _prf(spell_tp, spell_fp, spell_fn)
    auto_precision, auto_recall, auto_f1 = _prf(auto_tp, auto_fp, auto_fn)
    return {
        'examples': len(rows),
        'spellcheck_precision': round(spell_precision, 6),
        'spellcheck_recall': round(spell_recall, 6),
        'spellcheck_f1': round(spell_f1, 6),
        'autocorrect_precision': round(auto_precision, 6),
        'autocorrect_recall': round(auto_recall, 6),
        'autocorrect_f1': round(auto_f1, 6),
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {'p10': 0.0, 'p50': 0.0, 'p90': 0.0}
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]
    return {key: round(at(fraction), 6) for key, fraction in (('p10', .1), ('p50', .5), ('p90', .9))}


def audit_score_scales(
    model: WebSpellModel,
    pairs_path: str | Path,
    adapter: QueryAlignmentAdapter,
    max_examples: int = 10_000,
    sampling_seed: int = 2026,
) -> dict[str, object]:
    """Measure error/LM score deltas in the same units used by lambda ranking."""
    buckets: dict[tuple[int, int], dict[str, object]] = defaultdict(
        lambda: {'examples': 0, 'candidate_hits': 0, 'error': [], 'lm': [], 'weighted_lm': []}
    )
    examples = adapter.iter_examples(
        pairs_path, limit=max_examples, include_clean=False,
        sampling_seed=sampling_seed,
    )
    max_context = model.ranker.language_model.config.order - 1
    for example in examples:
        bucket = (min(len(example.context.left), max_context), min(len(example.context.right), max_context))
        record = buckets[bucket]
        record['examples'] = int(record['examples']) + 1
        candidates = model.ranker.rank(example.observed, example.context, 0.0)
        original = next(item for item in candidates if item.term == example.observed)
        intended = next((item for item in candidates if item.term == example.intended), None)
        if intended is None:
            continue
        record['candidate_hits'] = int(record['candidate_hits']) + 1
        error_delta = intended.error_log_score - original.error_log_score
        lm_delta = float(intended.lm_log_score) - float(original.lm_log_score)
        lambda_value = model.lambdas.for_context(example.context, max_context)
        record['error'].append(error_delta)  # type: ignore[union-attr]
        record['lm'].append(lm_delta)  # type: ignore[union-attr]
        record['weighted_lm'].append(lambda_value * lm_delta)  # type: ignore[union-attr]
    output = []
    for bucket, record in sorted(buckets.items()):
        examples_count = int(record['examples'])
        hits = int(record['candidate_hits'])
        output.append({
            'left': bucket[0], 'right': bucket[1],
            'lambda': model.lambdas.values.get(bucket, model.lambdas.default),
            'examples': examples_count, 'candidate_hits': hits,
            'candidate_recall': round(hits / examples_count, 6) if examples_count else 0.0,
            'error_score_delta': _quantiles(record['error']),  # type: ignore[arg-type]
            'lm_score_delta': _quantiles(record['lm']),  # type: ignore[arg-type]
            'weighted_lm_delta': _quantiles(record['weighted_lm']),  # type: ignore[arg-type]
        })
    return {'examples': sum(int(item['examples']) for item in output), 'buckets': output}


def mine_web_error_triples(
    index: SQLiteSymSpellIndex,
    database: str | Path,
    frequency_ratio: float = 10.0,
    minimum_context_frequency: int = 10,
    max_triples: int = 0,
) -> tuple[list[ErrorTriple], dict[str, int | float | str]]:
    """Mine Section 3.2 triples from term frequency and one-word contexts."""
    contexts = SQLiteContextStatistics(database)
    close_pair_count = 0
    blacklisted_pair_count = 0

    def word_like(term: str) -> bool:
        return (
            2 <= len(term) <= 64
            and not any(char.isdigit() for char in term)
            and sum(char.isalpha() for char in term) / len(term) >= 0.8
        )

    def counted_pairs() -> Iterator[ClosePair]:
        nonlocal close_pair_count, blacklisted_pair_count
        for pair in iter_close_pairs(index, frequency_ratio=frequency_ratio):
            if (
                is_blacklisted(pair.observed)
                or is_blacklisted(pair.intended)
                or not word_like(pair.observed)
                or not word_like(pair.intended)
            ):
                blacklisted_pair_count += 1
                continue
            close_pair_count += 1
            yield pair

    try:
        stream = iter_error_triples(
            counted_pairs(), contexts,
            minimum_context_frequency=minimum_context_frequency,
        )
        triples = list(islice(stream, max_triples)) if max_triples else list(stream)
    finally:
        contexts.close()
    return triples, {
        'source': 'web-context-mining',
        'frequency_ratio': frequency_ratio,
        'minimum_context_frequency': minimum_context_frequency,
        'close_pairs_scanned': close_pair_count,
        'blacklisted_pairs_excluded': blacklisted_pair_count,
        'triples_emitted': len(triples),
        'triple_occurrences': sum(item.count for item in triples),
    }


def _merge_error_triples(*groups: list[ErrorTriple]) -> list[ErrorTriple]:
    counts: Counter[tuple[str, str]] = Counter()
    for group in groups:
        counts.update({(item.intended, item.observed): item.count for item in group})
    return [
        ErrorTriple(intended, observed, count)
        for (intended, observed), count in sorted(counts.items())
    ]


def evaluate_osm_model(
    model: WebSpellModel,
    test_pairs_path: str | Path,
    adapter: QueryAlignmentAdapter,
    max_examples: int = 10_000,
    sampling_seed: int = 2026,
) -> dict[str, object]:
    """Run full evaluation on test split according to Whitelaw et al. 2009.

    Computes:
    - Token-level paper metrics: E1-E5, CER, FER, TER, NGS
    - Candidate ranker metrics: candidate recall, top-1 accuracy
    - Query-level metrics: full query exact match accuracy, preserved clean queries
    """
    total_queries = 0
    exact_match_queries = 0
    clean_queries = 0
    clean_queries_preserved = 0
    by_error_type: dict[str, Counter[str]] = defaultdict(Counter)

    tokens_total = 0
    e1_total = e2_total = e3_total = e4_total = e5_total = 0
    no_good_total = 0
    misspelled_total = 0

    ranker_total = 0
    candidate_hits = 0
    ranker_top1 = 0

    max_context = model.ranker.language_model.config.order - 1

    sampled_rows = iter_sampled_query_rows(
        test_pairs_path,
        limit=max_examples,
        seed=sampling_seed,
        namespace='evaluation-query',
    )
    with nullcontext(sampled_rows) as rows:
        for row in rows:
            noisy_text = row["noisy_query"].strip()
            correct_text = row["correct_query"].strip()
            if not noisy_text or not correct_text:
                continue

            raw_error_type = row.get('error_type') or row.get('noise_source', 'unknown')
            metric_type = 'combined' if raw_error_type.startswith('combined:') else raw_error_type
            type_counts = by_error_type[metric_type]
            total_queries += 1
            type_counts['queries'] += 1
            is_clean_query = noisy_text == correct_text
            if is_clean_query:
                clean_queries += 1

            predictions = model.predict(noisy_text)
            corrected = model.corrected_text(noisy_text, predictions)
            if corrected == correct_text:
                exact_match_queries += 1
                type_counts['exact_queries'] += 1
                if is_clean_query:
                    clean_queries_preserved += 1

            intended_tokens = correct_text.split()
            if len(predictions) == len(intended_tokens):
                try:
                    eval_res = evaluate_predictions(predictions, intended_tokens)
                    tokens_total += eval_res.tokens
                    e1_total += eval_res.e1
                    e2_total += eval_res.e2
                    e3_total += eval_res.e3
                    e4_total += eval_res.e4
                    e5_total += eval_res.e5
                    no_good_total += eval_res.no_good_suggestion
                    misspelled_total += eval_res.misspelled_tokens
                    type_counts.update({
                        'tokens': eval_res.tokens,
                        'misspelled_tokens': eval_res.misspelled_tokens,
                        'e1': eval_res.e1, 'e2': eval_res.e2, 'e3': eval_res.e3,
                        'e4': eval_res.e4, 'e5': eval_res.e5,
                        'no_good_suggestion': eval_res.no_good_suggestion,
                    })
                except ValueError:
                    pass

            aligned_tokens, _, _ = adapter.align_pair(noisy_text, correct_text, include_clean=False)
            for ex in aligned_tokens:
                if ex.observed != ex.intended:
                    ranker_total += 1
                    lambda_val = model.lambdas.for_context(ex.context, max_context)
                    candidates = model.ranker.rank(ex.observed, ex.context, lambda_val)
                    alternatives = [item for item in candidates if item.term != ex.observed]
                    hit = any(item.term == ex.intended for item in alternatives)
                    candidate_hits += int(hit)
                    top1 = bool(alternatives) and alternatives[0].term == ex.intended
                    ranker_top1 += int(top1)
                    type_counts['ranker_examples'] += 1
                    type_counts['candidate_hits'] += int(hit)
                    type_counts['top1_hits'] += int(top1)

    cer = (e1_total + e2_total + e3_total + e4_total) / tokens_total if tokens_total else 0.0
    fer = (e3_total + e5_total) / tokens_total if tokens_total else 0.0
    ter = (e1_total + e2_total + e3_total + e4_total + e5_total) / tokens_total if tokens_total else 0.0
    ngs = no_good_total / misspelled_total if misspelled_total else 0.0
    query_acc = exact_match_queries / total_queries if total_queries else 0.0

    def error_type_payload(counts: Counter[str]) -> dict[str, object]:
        tokens = counts['tokens']
        misspelled = counts['misspelled_tokens']
        ranker_examples = counts['ranker_examples']
        return {
            'queries': counts['queries'],
            'query_accuracy': round(counts['exact_queries'] / counts['queries'], 6) if counts['queries'] else 0.0,
            'tokens': tokens,
            'misspelled_tokens': misspelled,
            'e1': counts['e1'], 'e2': counts['e2'], 'e3': counts['e3'],
            'e4': counts['e4'], 'e5': counts['e5'],
            'ngs': round(counts['no_good_suggestion'] / misspelled, 6) if misspelled else 0.0,
            'candidate_recall': round(counts['candidate_hits'] / ranker_examples, 6) if ranker_examples else 0.0,
            'top1_accuracy': round(counts['top1_hits'] / ranker_examples, 6) if ranker_examples else 0.0,
        }

    return {
        "cer": round(cer, 6),
        "fer": round(fer, 6),
        "ter": round(ter, 6),
        "ngs": round(ngs, 6),
        "query_accuracy": round(query_acc, 6),
        "token_metrics": {
            "tokens": tokens_total,
            "misspelled_tokens": misspelled_total,
            "e1": e1_total,
            "e2": e2_total,
            "e3": e3_total,
            "e4": e4_total,
            "e5": e5_total,
            "no_good_suggestion": no_good_total,
        },
        "ranker_metrics": {
            "examples": ranker_total,
            "candidate_recall": round(candidate_hits / ranker_total, 6) if ranker_total else 0.0,
            "top1_accuracy": round(ranker_top1 / ranker_total, 6) if ranker_total else 0.0,
        },
        "query_metrics": {
            "total_queries": total_queries,
            "exact_match_queries": exact_match_queries,
            "clean_queries": clean_queries,
            "clean_queries_preserved": clean_queries_preserved,
        },
        "error_type_metrics": {
            name: error_type_payload(counts)
            for name, counts in sorted(by_error_type.items())
        },
    }


def _validate_data(root: Path) -> None:
    for split in ("train", "validation", "test"):
        for filename in ("corpus.txt", "noisy_pairs.csv"):
            path = root / split / filename
            if not path.is_file():
                raise FileNotFoundError(path)


def train_osm(
    data: str | Path,
    output: str | Path,
    config: OSMTrainingConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    settings = config or OSMTrainingConfig()
    if settings.workers < 1:
        raise ValueError('workers must be positive')
    if settings.worker_batch_size < 1:
        raise ValueError('worker batch size must be positive')
    if settings.clean_ratio < 0:
        raise ValueError('clean ratio must be non-negative')
    error_weights = dict(settings.error_type_weights)
    if set(error_weights).difference(TRAINING_ERROR_TYPE_WEIGHTS):
        raise ValueError('unknown classifier error type weight')
    if not error_weights or any(weight < 0 for weight in error_weights.values()) or sum(error_weights.values()) <= 0:
        raise ValueError('classifier error type weights must have a positive total')
    if any(size <= 0 for size in settings.learning_curve_sizes):
        raise ValueError('learning curve sizes must be positive')
    if settings.error_model_source not in {'synthetic', 'web', 'hybrid'}:
        raise ValueError('error model source must be synthetic, web, or hybrid')
    if settings.error_frequency_ratio <= 1:
        raise ValueError('error frequency ratio must be greater than 1')
    if settings.minimum_context_frequency < 1:
        raise ValueError('minimum context frequency must be positive')
    if settings.candidate_max_edit_distance not in {1, 2, 3}:
        raise ValueError('candidate max edit distance must be 1, 2, or 3')
    if settings.candidate_variant_generator not in {'none', 'vietnamese'}:
        raise ValueError('candidate variant generator must be none or vietnamese')
    if (
        settings.max_training_examples
        and settings.learning_curve_sizes
        and max(settings.learning_curve_sizes) > settings.max_training_examples
    ):
        raise ValueError(
            'learning curve sizes cannot exceed max training examples'
        )
    report = progress or (lambda message: None)
    stage_times: dict[str, float] = {}
    total_started = time.perf_counter()

    def begin_stage(name: str) -> float:
        report(f'[{name}] starting')
        return time.perf_counter()

    def finish_stage(name: str, started: float) -> None:
        elapsed = time.perf_counter() - started
        stage_times[name] = round(elapsed, 3)
        report(f'[{name}] {elapsed:.2f}s')

    data_root, output_root = Path(data), Path(output)
    _validate_data(data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    database = output_root / "statistics.sqlite3"

    adapter = QueryAlignmentAdapter(order=settings.order)
    started = begin_stage('statistics')

    reuse_output = False
    if database.is_file() and not settings.rebuild_statistics:
        try:
            metadata = SQLiteCorpusStatistics(database).metadata()
            reuse_output = (
                metadata.get('counting_complete') is True
                and metadata.get('symspell_complete') is True
                and int(metadata.get('order', -1)) == settings.order
                and int(metadata.get('symspell_max_distance', -1)) == settings.symspell_distance
                and int(metadata.get('symspell_minimum_frequency', -1)) == settings.minimum_frequency
            )
        except (OSError, TypeError, ValueError, sqlite3.DatabaseError):
            reuse_output = False

    if settings.reuse_statistics and Path(settings.reuse_statistics).is_file():
        if database.resolve() != Path(settings.reuse_statistics).resolve():
            shutil.copy2(settings.reuse_statistics, database)
        statistics = SQLiteCorpusStatistics(database)
        index = SQLiteSymSpellIndex(database)
    elif reuse_output:
        statistics = SQLiteCorpusStatistics(database)
        index = SQLiteSymSpellIndex(database)
    else:
        if database.exists():
            database.unlink()
        corpus = LineCorpus(data_root / "train" / "corpus.txt", lowercase=False)
        statistics = SQLiteCorpusStatistics.build(
            corpus, database, order=settings.order, batch_tokens=settings.batch_tokens
        )
        index = SQLiteSymSpellIndex.build(
            database,
            max_distance=settings.symspell_distance,
            minimum_frequency=settings.minimum_frequency,
        )
    finish_stage('statistics', started)

    language_model = SQLiteBidirectionalLanguageModel(database)
    tokenizer = SimpleTokenizer("NFC")

    try:
        train_pairs = data_root / "train" / "noisy_pairs.csv"
        validation_pairs = data_root / "validation" / "noisy_pairs.csv"
        test_pairs = data_root / "test" / "noisy_pairs.csv"

        started = begin_stage('error_model')
        synthetic_triples: list[ErrorTriple] = []
        train_diagnostics = AlignmentDiagnostics()
        error_mining: dict[str, object] = {'source': settings.error_model_source}
        if settings.error_model_source in {'synthetic', 'hybrid'}:
            synthetic_triples, train_diagnostics = adapter.mine_error_triples(
                train_pairs,
                max_examples=settings.max_error_examples,
                sampling_seed=settings.sampling_seed,
            )
            error_mining['synthetic_triples'] = len(synthetic_triples)
            error_mining['synthetic_occurrences'] = sum(item.count for item in synthetic_triples)
        web_triples: list[ErrorTriple] = []
        if settings.error_model_source in {'web', 'hybrid'}:
            web_triples, web_diagnostics = mine_web_error_triples(
                index, database,
                frequency_ratio=settings.error_frequency_ratio,
                minimum_context_frequency=settings.minimum_context_frequency,
                max_triples=settings.max_error_examples,
            )
            error_mining['web'] = web_diagnostics
        triples = _merge_error_triples(synthetic_triples, web_triples)
        if not triples:
            raise ValueError(
                'error-model mining produced no triples; use hybrid/synthetic '
                'or lower the frequency/context thresholds'
            )
        error_model = SubstringErrorModel.fit(triples)
        finish_stage('error_model', started)
        ranker = CandidateRanker(
            index,
            error_model,
            language_model,
            CandidateConfig(maximum_edit_distance=settings.candidate_max_edit_distance),
            variant_generator=(
                VietnameseVariantGenerator()
                if settings.candidate_variant_generator == 'vietnamese'
                else None
            ),
        )

        started = begin_stage('lambda_tuning')
        validation_examples = list(
            adapter.iter_examples(
                validation_pairs,
                limit=settings.max_validation_examples,
                include_clean=False,
                sampling_seed=settings.sampling_seed,
            )
        )
        lambdas = tune_lambdas_streaming(lambda: iter(validation_examples), ranker)
        finish_stage('lambda_tuning', started)

        started = begin_stage('feature_generation')
        report(
            f'[feature_generation] workers={settings.workers} '
            f'batch_size={settings.worker_batch_size}'
        )
        train_examples = list(
            adapter.iter_examples(
                train_pairs,
                limit=settings.max_training_examples,
                include_clean=True,
                clean_ratio=settings.clean_ratio,
                sampling_seed=settings.sampling_seed,
                compatible_only=settings.token_compatible_only,
                error_type_weights=dict(settings.error_type_weights),
            )
        )
        val_conf_examples = list(
            adapter.iter_examples(
                validation_pairs,
                limit=settings.max_validation_examples,
                include_clean=True,
                clean_ratio=settings.clean_ratio,
                sampling_seed=settings.sampling_seed,
                compatible_only=settings.token_compatible_only,
                error_type_weights=dict(settings.error_type_weights),
            )
        )

        if settings.workers > 1:
            from webspell.scalable.parallel import build_training_rows_parallel

            training_rows_cache, dev_rows_cache = build_training_rows_parallel(
                train_examples,
                val_conf_examples,
                ranker,
                lambdas,
                str(database),
                settings.workers,
                settings.worker_batch_size,
                settings.candidate_variant_generator,
            )
        else:
            training_rows_cache = list(
                iter_training_rows(train_examples, ranker, lambdas)
            )
            dev_rows_cache = list(
                iter_training_rows(val_conf_examples, ranker, lambdas)
            )
        finish_stage('feature_generation', started)

        classifier_distribution: Counter[str] = Counter()
        for row in training_rows_cache:
            kind = 'clean' if not row.misspelled else ('combined' if row.error_type.startswith('combined:') else row.error_type)
            classifier_distribution[kind] += 1

        started = begin_stage('classifier')
        confidence = StreamingConfidenceTrainer(
            StreamingClassifierConfig(epochs=settings.classifier_epochs)
        ).fit(lambda: iter(training_rows_cache), lambda: iter(dev_rows_cache))
        finish_stage('classifier', started)

        learning_curve: list[dict[str, object]] = []
        curve_sizes = sorted(set(settings.learning_curve_sizes))
        if curve_sizes and curve_sizes[-1] > len(training_rows_cache):
            raise ValueError(
                'learning curve size exceeds available training examples'
            )
        if curve_sizes:
            started = begin_stage('learning_curve')
            for size in curve_sizes:
                subset = _balanced_training_subset(
                    training_rows_cache, size, settings.clean_ratio,
                    settings.sampling_seed, dict(settings.error_type_weights),
                )
                if size == len(training_rows_cache):
                    curve_model = confidence
                else:
                    curve_model = StreamingConfidenceTrainer(
                        StreamingClassifierConfig(epochs=settings.classifier_epochs)
                    ).fit(lambda rows=subset: iter(rows), lambda: iter(dev_rows_cache))
                learning_curve.append(
                    {
                        'training_examples': len(subset),
                        **_classifier_metrics(curve_model, dev_rows_cache),
                    }
                )
            finish_stage('learning_curve', started)

        model = WebSpellModel(tokenizer, ranker, lambdas, confidence)
        started = begin_stage('checkpoint')
        save_model_bundle(model, output_root)
        finish_stage('checkpoint', started)

        metrics: dict[str, object] = {}
        if not settings.skip_evaluation:
            started = begin_stage('evaluation')
            metrics = evaluate_osm_model(
                model,
                test_pairs,
                adapter,
                max_examples=settings.max_test_examples,
                sampling_seed=settings.sampling_seed,
            )
            finish_stage('evaluation', started)

        coverage: dict[str, dict[str, int]] = {}
        if not settings.skip_coverage:
            started = begin_stage('coverage')
            coverage = {
                split: count_pair_coverage(
                    data_root / split / 'noisy_pairs.csv',
                    order=settings.order,
                    max_examples=settings.max_coverage_examples,
                    sampling_seed=settings.sampling_seed,
                )
                for split in ('validation', 'test')
            }
            if settings.max_coverage_examples == settings.max_error_examples:
                coverage['train'] = train_diagnostics.to_dict()
            else:
                coverage['train'] = count_pair_coverage(
                    train_pairs,
                    order=settings.order,
                    max_examples=settings.max_coverage_examples,
                    sampling_seed=settings.sampling_seed,
                )
            finish_stage('coverage', started)
        (output_root / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        source_manifest = data_root / "manifest.json"
        if source_manifest.is_file():
            shutil.copy2(source_manifest, output_root / "data-manifest.json")

        stage_times['total'] = round(time.perf_counter() - total_started, 3)
        training_metadata = {
            'schema_version': 1,
            'config': asdict(settings),
            'sampling': {
                'algorithm': 'blake2b-128-bottom-k/v1',
                'seed': settings.sampling_seed,
                'training_unit': 'labeled-token-occurrence',
                'error_model_unit': 'error-token-occurrence',
                'evaluation_unit': 'query-pair',
                'clean_ratio': settings.clean_ratio,
                'token_compatible_only': settings.token_compatible_only,
                'error_type_weights': dict(settings.error_type_weights),
                'classifier_distribution': dict(sorted(classifier_distribution.items())),
            },
            'stage_times_seconds': stage_times,
            'error_model_mining': error_mining,
        }
        (output_root / 'training-metadata.json').write_text(
            json.dumps(training_metadata, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        if learning_curve:
            (output_root / 'learning-curve.json').write_text(
                json.dumps(learning_curve, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8',
            )

        return {
            "schema_version": 1,
            "config": asdict(settings),
            "statistics": statistics.metadata(),
            "error_model": {
                "config": asdict(error_model.config),
                "counts": error_model.transitions(),
                "targets": sorted(error_model._targets),
            },
            "lambdas": {
                "default": lambdas.default,
                "values": [
                    {"left": key[0], "right": key[1], "value": value}
                    for key, value in sorted(lambdas.values.items())
                ],
            },
            "confidence": _confidence_state(confidence),
            "pair_coverage": coverage,
            "test_metrics": metrics,
            "stage_times_seconds": stage_times,
            "learning_curve": learning_curve,
        }
    finally:
        index.close()
        language_model.close()
