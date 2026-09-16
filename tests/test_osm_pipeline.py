from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from webspell.osm.alignment import QueryAlignmentAdapter, iter_sampled_query_rows, write_token_level_pairs
from webspell.osm.prepare import OSMEntity, OSMPreparationConfig, prepare_entities, resplit_prepared_queries, synthetic_query_variants
from webspell.osm.noise import generate_noise
from webspell.osm.training import OSMTrainingConfig, _balanced_training_subset, audit_score_scales, mine_web_error_triples, train_osm
from webspell.confidence import TrainingRow
from webspell.pipeline import WebSpellModel
from webspell.scalable import LineCorpus, SQLiteCorpusStatistics, SQLiteSymSpellIndex


class OSMAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = QueryAlignmentAdapter(order=3)

    def test_resplit_prepared_queries_groups_shared_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            header = (
                'noisy_query', 'correct_query', 'entity_id', 'group_id',
                'term_role', 'noise_source', 'error_type', 'variant_id',
            )
            for index, split in enumerate(('train', 'validation', 'test')):
                folder = source / split
                folder.mkdir(parents=True)
                with (folder / 'noisy_pairs.csv').open(
                    'w', encoding='utf-8', newline=''
                ) as stream:
                    writer = csv.writer(stream)
                    writer.writerow(header)
                    writer.writerow((
                        'ho guom', 'hồ gươm', str(index), f'entity-{index}',
                        'alias', 'synthetic', 'edit', '0',
                    ))
                    writer.writerow((
                        'hồ gươm', 'hồ gươm', str(index), f'entity-{index}',
                        'alias', 'clean', 'clean', 'clean-0',
                    ))
            output = root / 'output'
            manifest = resplit_prepared_queries(source, output)
            self.assertEqual(manifest['overlaps'], {
                'train_validation': 0, 'train_test': 0, 'validation_test': 0,
            })
            locations = []
            for split in ('train', 'validation', 'test'):
                with (output / split / 'noisy_pairs.csv').open(
                    encoding='utf-8', newline=''
                ) as stream:
                    rows = list(csv.DictReader(stream))
                if rows:
                    locations.append(split)
                    self.assertEqual(len(rows), 6)
            self.assertEqual(len(locations), 1)

    def test_align_equal_length_query_pairs(self) -> None:
        noisy = "san bay noi bai"
        clean = "sân bay nội bài"
        tokens, error_pairs, is_sm = self.adapter.align_pair(noisy, clean, include_clean=True)
        self.assertFalse(is_sm)
        self.assertEqual(len(error_pairs), 3)
        self.assertIn(("sân", "san"), error_pairs)
        self.assertIn(("bài", "bai"), error_pairs)
        noi_item = next(t for t in tokens if t.observed == "noi")
        self.assertEqual(noi_item.intended, "nội")
        self.assertEqual(noi_item.context.left, ("san", "bay"))
        self.assertEqual(noi_item.context.right, ("bai",))

    def test_training_context_matches_noisy_inference_context(self) -> None:
        tokens, _, _ = self.adapter.align_pair(
            'wrong neighbor typo', 'right neighbor type', include_clean=False
        )
        typo = next(item for item in tokens if item.observed == 'typo')
        self.assertEqual(typo.context.left, ('wrong', 'neighbor'))

    def test_align_differing_length_query_pairs(self) -> None:
        noisy = "san noi bai"
        clean = "sân bay nội bài"
        tokens, error_pairs, is_sm = self.adapter.align_pair(noisy, clean, include_clean=False)
        self.assertFalse(is_sm)
        self.assertEqual(len(error_pairs), 3)
        self.assertIn(("sân", "san"), error_pairs)
        self.assertIn(("nội", "noi"), error_pairs)
        self.assertIn(("bài", "bai"), error_pairs)

    def test_detect_merged_query(self) -> None:
        noisy = "châuđốc"
        clean = "châu đốc"
        tokens, error_pairs, is_sm = self.adapter.align_pair(noisy, clean, include_clean=False)
        self.assertTrue(is_sm)

    def test_mine_error_triples_aggregates_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "pairs.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("noisy_query", "correct_query", "entity_id", "noise_source"))
                writer.writerow(("ha noi", "hà nội", "1", "synthetic"))
                writer.writerow(("ha noi", "hà nội", "2", "synthetic"))
                writer.writerow(("ha noi dep", "hà nội đẹp", "3", "synthetic"))

            triples, diag = self.adapter.mine_error_triples(csv_path)
            triple_map = {(t.intended, t.observed): t.count for t in triples}
            self.assertEqual(triple_map[("hà", "ha")], 3)
            self.assertEqual(triple_map[("nội", "noi")], 3)
            self.assertEqual(triple_map[("đẹp", "dep")], 1)
            self.assertEqual(diag.total_query_pairs, 3)
            self.assertEqual(diag.unique_error_triples, 3)

    def test_hash_sample_is_stable_when_rows_are_reordered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / 'first.csv'
            second = Path(directory) / 'second.csv'
            rows = [
                (f'noisy-{index}', f'correct-{index}', f'entity-{index}', 'synthetic')
                for index in range(20)
            ]
            for path, ordered_rows in ((first, rows), (second, list(reversed(rows)))):
                with path.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(
                        ('noisy_query', 'correct_query', 'entity_id', 'noise_source')
                    )
                    writer.writerows(ordered_rows)
            selected_first = {
                row['entity_id'] for row in iter_sampled_query_rows(first, limit=7)
            }
            selected_second = {
                row['entity_id'] for row in iter_sampled_query_rows(second, limit=7)
            }
            self.assertEqual(selected_first, selected_second)

    def test_limited_training_sample_balances_clean_and_error_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pairs.csv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ('noisy_query', 'correct_query', 'entity_id', 'noise_source')
                )
                for index in range(20):
                    writer.writerow(('helo', 'hello', f'entity-{index}', 'synthetic'))
                    writer.writerow(('hello', 'hello', f'clean-{index}', 'clean'))
            examples = list(
                self.adapter.iter_examples(
                    path, limit=10, include_clean=True, clean_ratio=1.0
                )
            )
            clean = sum(item.observed == item.intended for item in examples)
            self.assertEqual(len(examples), 10)
            self.assertEqual(clean, 5)

    def test_classifier_sampler_stratifies_and_excludes_structural_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pairs.csv'
            kinds = ('keyboard_edit', 'missing_diacritics_full', 'missing_diacritics_partial', 'wrong_diacritic', 'telex_leak', 'vni_leak', 'combined:keyboard_edit+wrong_diacritic')
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(('noisy_query', 'correct_query', 'entity_id', 'noise_source', 'error_type', 'variant_id'))
                for index, kind in enumerate(kinds):
                    for repeat in range(3):
                        writer.writerow((f'bad{index}', f'good{index}', f'e{index}', 'synthetic', kind, repeat))
                for index in range(20):
                    writer.writerow((f'clean{index}', f'clean{index}', f'c{index}', 'clean', 'clean', 0))
                writer.writerow(('sanbay', 'san bay', 'structural', 'synthetic', 'word_boundary', 0))
            examples = list(self.adapter.iter_examples(path, limit=21, include_clean=True, clean_ratio=2, compatible_only=True, error_type_weights={name: 1 for name in ('keyboard_edit', 'missing_diacritics_full', 'missing_diacritics_partial', 'wrong_diacritic', 'telex_leak', 'vni_leak', 'combined')}))
            errors = Counter(item.error_type if not item.error_type.startswith('combined:') else 'combined' for item in examples if item.observed != item.intended)
            self.assertEqual(errors, Counter({name: 1 for name in ('keyboard_edit', 'missing_diacritics_full', 'missing_diacritics_partial', 'wrong_diacritic', 'telex_leak', 'vni_leak', 'combined')}))
            self.assertNotIn('sanbay', [item.observed for item in examples])

    def test_learning_curve_subsets_are_nested_and_shuffled(self) -> None:
        rows = []
        kinds = ('keyboard_edit', 'missing_diacritics_full', 'missing_diacritics_partial', 'wrong_diacritic', 'telex_leak', 'vni_leak', 'combined')
        for index in range(70):
            rows.append(TrainingRow((float(index),), 1, True, 1, kinds[index % len(kinds)]))
        rows.extend(TrainingRow((float(index),), 0, True, 0, 'clean') for index in range(70, 210))
        small = _balanced_training_subset(rows, 30, 2, seed=2026)
        large = _balanced_training_subset(rows, 60, 2, seed=2026)
        self.assertTrue({id(row) for row in small}.issubset({id(row) for row in large}))
        labels = [row.misspelled for row in small]
        self.assertGreater(sum(a != b for a, b in zip(labels, labels[1:])), 1)

    def test_write_token_level_pairs_creates_separate_filtered_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'noisy_pairs.csv'
            filtered = Path(directory) / 'noisy_pairs_token_level.csv'
            original = 'noisy_query,correct_query,entity_id,noise_source\nha noi,hà nội,1,synthetic\nhanoi,hà nội,2,synthetic\nda nangg,đà nẵng,3,synthetic\n'
            source.write_text(original, encoding='utf-8')
            counts = write_token_level_pairs(source, filtered)
            self.assertEqual(counts, {'source_rows': 3, 'kept_rows': 2, 'excluded_rows': 1})
            self.assertEqual(source.read_text(encoding='utf-8'), original)
            with filtered.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['entity_id'] for row in rows], ['1', '3'])

    def test_location_generator_is_deterministic_and_emits_only_noisy_rows(self) -> None:
        first = synthetic_query_variants('thanh pho ho chi minh', 100, 2026, 0.02)
        second = synthetic_query_variants('thanh pho ho chi minh', 100, 2026, 0.02)
        self.assertEqual(first, second)
        self.assertTrue(all(noisy != 'thanh pho ho chi minh' for noisy, _, _ in first))
        self.assertTrue(all(source.startswith('synthetic_') for _, source, _ in first))

    def test_each_location_error_type_can_be_generated(self) -> None:
        import random
        fixtures = {
            'missing_diacritics_full': 'sân bay nội bài',
            'missing_diacritics_partial': 'sân bay nội bài',
            'wrong_diacritic': 'sân bay nội bài',
            'telex_leak': 'đường nguyễn huệ',
            'vni_leak': 'đường nguyễn huệ',
            'keyboard_edit': 'sân bay nội bài',
            'word_boundary': 'sân bay nội bài',
            'address_abbreviation': 'thành phố hà nội',
            'address_symbol': '12/3 đường huệ',
        }
        for error_type, term in fixtures.items():
            with self.subTest(error_type=error_type):
                result = generate_noise(term, error_type, random.Random(7))
                self.assertIsNotNone(result)
                self.assertNotEqual(result[0], term)
        self.assertEqual(generate_noise('thành phố', 'telex_leak', random.Random(7))[0], 'thanhf phoos')
        self.assertEqual(generate_noise('thành phố', 'vni_leak', random.Random(7))[0], 'thanh2 pho61')

    def test_prepare_entities_writes_clean_controls_and_noise_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = OSMPreparationConfig(
                noisy_variants_per_term=100, character_error_rate=0.02
            )
            counts = prepare_entities(
                [OSMEntity('1', 'place:city', 'thanh pho ha noi', ('thanh pho ha noi',))],
                directory,
                config,
            )
            self.assertEqual(counts['noisy_pairs'], 101)
            self.assertGreater(counts['clean_pairs'], 0)
            self.assertGreater(counts['misspelled_pairs'], 0)
            rows = []
            for name in ('train', 'validation', 'test'):
                with (Path(directory) / name / 'noisy_pairs.csv').open(encoding='utf-8', newline='') as stream:
                    rows.extend(csv.DictReader(stream))
            self.assertIn('variant_id', rows[0])
            self.assertIn('group_id', rows[0])
            self.assertIn('term_role', rows[0])
            self.assertIn('error_type', rows[0])
            self.assertTrue(any(row['noise_source'] == 'clean' for row in rows))
            self.assertTrue(any(row['noise_source'].startswith('synthetic_') for row in rows))

    def test_aliases_of_same_canonical_entity_never_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entities = [
                OSMEntity('1', 'place:city', 'ha noi', ('ha noi', 'thang long')),
                OSMEntity('2', 'place:city', 'ha noi', ('ha noi', 'thu do')),
            ]
            prepare_entities(entities, directory, OSMPreparationConfig(noisy_variants_per_term=1))
            populated = []
            for split in ('train', 'validation', 'test'):
                with (Path(directory) / split / 'noisy_pairs.csv').open(encoding='utf-8', newline='') as stream:
                    if list(csv.DictReader(stream)):
                        populated.append(split)
            self.assertEqual(len(populated), 1)


class OSMTrainingPipelineTests(unittest.TestCase):
    def _create_mini_osm_dataset(self, root: Path) -> None:
        train_dir = root / "train"
        val_dir = root / "validation"
        test_dir = root / "test"
        for d in (train_dir, val_dir, test_dir):
            d.mkdir(parents=True, exist_ok=True)

        corpus_lines = [
            "sân bay nội bài",
            "khách sạn hà nội",
            "thành phố hồ chí minh",
            "nhà hát lớn hà nội",
            "chợ bến thành",
            "quận hoàn kiếm hà nội",
            "đường nguyễn huệ",
        ] * 10

        train_dir.joinpath("corpus.txt").write_text("\n".join(corpus_lines) + "\n", encoding="utf-8")
        val_dir.joinpath("corpus.txt").write_text("\n".join(corpus_lines[:5]) + "\n", encoding="utf-8")
        test_dir.joinpath("corpus.txt").write_text("\n".join(corpus_lines[:5]) + "\n", encoding="utf-8")

        train_pairs = [
            ("san bay noi bai", "sân bay nội bài"),
            ("khach san ha noi", "khách sạn hà nội"),
            ("thanh pho ho chi minh", "thành phố hồ chí minh"),
            ("cho ben thanh", "chợ bến thành"),
            ("duong nguyen hue", "đường nguyễn huệ"),
            ("nha hat lon ha noi", "nhà hát lớn hà nội"),
            ("sân bay nội bài", "sân bay nội bài"),
            ("chợ bến thành", "chợ bến thành"),
        ] * 5

        for split, pairs in (("train", train_pairs), ("validation", train_pairs[:6]), ("test", train_pairs[:6])):
            csv_path = root / split / "noisy_pairs.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("noisy_query", "correct_query", "entity_id", "noise_source"))
                for idx, (noisy, clean) in enumerate(pairs):
                    writer.writerow((noisy, clean, f"entity_{idx}", "synthetic"))

        manifest = {
            "config": {"seed": 2026},
            "counts": {"entities": 10, "terms": 20, "noisy_pairs": len(train_pairs)},
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_train_osm_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            output_dir = Path(temp_dir) / "model"
            self._create_mini_osm_dataset(data_dir)

            config = OSMTrainingConfig(
                order=3,
                batch_tokens=1000,
                symspell_distance=2,
                minimum_frequency=1,
                max_error_examples=100,
                max_training_examples=100,
                max_validation_examples=50,
                max_test_examples=50,
                classifier_epochs=3,
                learning_curve_sizes=(10, 20),
            )

            result = train_osm(data_dir, output_dir, config)

            # 1. Verify returned result structure
            self.assertIn("test_metrics", result)
            self.assertIn("pair_coverage", result)
            self.assertIn("error_model", result)
            self.assertIn("lambdas", result)
            self.assertIn("confidence", result)
            self.assertIn("stage_times_seconds", result)
            self.assertGreater(result["stage_times_seconds"]["total"], 0)
            test_metrics = result["test_metrics"]
            self.assertIn("ter", test_metrics)
            self.assertIn("cer", test_metrics)
            self.assertIn("query_accuracy", test_metrics)
            self.assertIn("ranker_metrics", test_metrics)
            self.assertIn("error_type_metrics", test_metrics)

            # 2. Verify files created in output bundle
            self.assertTrue((output_dir / "statistics.sqlite3").is_file())
            self.assertTrue((output_dir / "model.json").is_file())
            self.assertTrue((output_dir / "manifest.json").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "data-manifest.json").is_file())
            self.assertTrue((output_dir / 'training-metadata.json').is_file())
            self.assertTrue((output_dir / 'learning-curve.json').is_file())
            self.assertEqual(
                [point['training_examples'] for point in result['learning_curve']],
                [10, 20],
            )

            # 3. Verify loading model bundle into WebSpellModel
            with WebSpellModel.load(output_dir) as loaded_model:
                self.assertIsNotNone(loaded_model)

                # 4. Run prediction with loaded model
                predictions = loaded_model.predict("san bay noi bai")
                corrected = loaded_model.corrected_text("san bay noi bai", predictions)
                self.assertEqual(corrected, "sân bay nội bài")
                audit = audit_score_scales(
                    loaded_model, data_dir / 'test' / 'noisy_pairs.csv',
                    QueryAlignmentAdapter(order=3), max_examples=10,
                )
                self.assertIn('buckets', audit)

    def test_web_error_mining_uses_frequency_ratio_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / 'corpus.txt'
            database = Path(directory) / 'statistics.sqlite3'
            corpus_path.write_text(
                ('left hello right\n' * 20) + ('left helo right\n' * 2),
                encoding='utf-8',
            )
            corpus = LineCorpus(corpus_path)
            SQLiteCorpusStatistics.build(corpus, database, order=3, batch_tokens=100)
            index = SQLiteSymSpellIndex.build(database, max_distance=2, minimum_frequency=1)
            try:
                triples, diagnostics = mine_web_error_triples(
                    index, database, frequency_ratio=10,
                    minimum_context_frequency=10,
                )
            finally:
                index.close()
            counts = {(item.intended, item.observed): item.count for item in triples}
            self.assertEqual(counts[('hello', 'helo')], 2)
            self.assertGreaterEqual(diagnostics['close_pairs_scanned'], 1)
            self.assertEqual(diagnostics['source'], 'web-context-mining')


if __name__ == "__main__":
    unittest.main()
