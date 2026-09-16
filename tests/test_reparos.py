from __future__ import annotations

import csv
import json
import tempfile
import unittest
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

from reparos.data import (
    PROFILE,
    noisy_query,
    prepare_improvement_regression_sets,
    prepare_production_data,
    prepare_reparos_data,
)
from reparos.config import ModelConfig, TrainingConfig
from reparos.evaluation import evaluate_queries
from reparos.manifests import fingerprint_files, require_matching_checksum, sha256_file


class ReparoSTests(unittest.TestCase):
    def test_production_conversion_preserves_pairs_taxonomy_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'webspell'
            header = (
                'noisy_query', 'correct_query', 'entity_id', 'group_id',
                'term_role', 'noise_source', 'error_type', 'variant_id',
            )
            rows = {
                'train': [
                    ('ha noi', 'ha noi', 'n1', 'ha noi', 'canonical', 'clean', 'clean', 'clean-0'),
                    ('haf nooij', 'ha noi', 'n1', 'ha noi', 'canonical', 'synthetic_telex', 'telex_leak', '0'),
                ],
                'validation': [
                    ('san bay', 'san bay', 'n2', 'san bay', 'canonical', 'clean', 'clean', 'clean-0'),
                    ('sna bay', 'san bay', 'n2', 'san bay', 'canonical', 'synthetic_edit', 'keyboard_edit', '0'),
                ],
                'test': [
                    ('cho ben', 'cho ben', 'n3', 'cho ben', 'canonical', 'clean', 'clean', 'clean-0'),
                    ('chowj beens', 'cho ben', 'n3', 'cho ben', 'canonical', 'synthetic_telex', 'telex_leak', '0'),
                ],
            }
            for split, values in rows.items():
                folder = source / split
                folder.mkdir(parents=True)
                with (folder / 'noisy_pairs.csv').open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(header)
                    writer.writerows(values)
            output = root / 'production'
            manifest = prepare_production_data(source, output)
            self.assertEqual(manifest['overlaps'], {
                'train_validation': 0, 'train_test': 0, 'validation_test': 0,
            })
            self.assertEqual(manifest['counts']['train']['clean_fraction'], 0.5)
            self.assertEqual(
                (output / 'base' / 'train.src').read_text(encoding='utf-8').splitlines(),
                ['ha noi', 'haf nooij'],
            )
            metadata = [
                json.loads(line) for line in
                (output / 'base' / 'train.meta.jsonl').read_text(encoding='utf-8').splitlines()
            ]
            self.assertEqual(metadata[1]['error_type'], 'telex_leak')
            self.assertEqual(metadata[1]['group_id'], 'ha noi')

    def test_reparos_import_does_not_load_webspell(self) -> None:
        output = subprocess.check_output(
            [sys.executable, '-c', "import reparos,sys; print(any(x == 'webspell' or x.startswith('webspell.') for x in sys.modules))"],
            text=True,
        ).strip()
        self.assertEqual(output, 'False')
    def test_model_config_requires_head_divisibility(self) -> None:
        ModelConfig(hidden_size=128, attention_heads=8).validate()
        with self.assertRaises(ValueError):
            ModelConfig(hidden_size=127, attention_heads=8).validate()

    def test_stage_learning_rates_are_explicit(self) -> None:
        self.assertEqual(TrainingConfig.for_stage('base').learning_rate, 1.0)
        self.assertEqual(TrainingConfig.for_stage('c1').learning_rate, 0.0001)

    def test_manifest_hash_detects_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.txt'
            path.write_text('first', encoding='utf-8')
            first = sha256_file(path)
            self.assertEqual(fingerprint_files([path])[path.as_posix()], first)
            path.write_text('second', encoding='utf-8')
            with self.assertRaises(ValueError):
                require_matching_checksum(path, first)

    def test_query_group_resplit_is_disjoint_and_deterministic(self) -> None:
        from reparos.data import prepare_query_group_split
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            rows = {
                'train': ['Hồ Gươm', 'Sân bay Nội Bài', 'Trùng Tên'],
                'validation': ['hồ   gươm', 'Bệnh viện Bạch Mai', 'trùng tên'],
                'test': ['HỒ GƯƠM', 'Ga Hà Nội', 'TRÙNG TÊN'],
            }
            for split, values in rows.items():
                folder = source / split
                folder.mkdir(parents=True)
                (folder / 'corpus.txt').write_text('\n'.join(values) + '\n', encoding='utf-8')
            first = prepare_query_group_split(source, root / 'first', seed=7)
            second = prepare_query_group_split(source, root / 'second', seed=7)
            self.assertEqual(first['counts'], second['counts'])
            self.assertEqual(first['overlaps'], {
                'train_validation': 0, 'train_test': 0, 'validation_test': 0,
            })
            locations = {}
            for split in ('train', 'validation', 'test'):
                values = (root / 'first' / split / 'corpus.txt').read_text(
                    encoding='utf-8'
                ).splitlines()
                for value in values:
                    locations.setdefault(' '.join(value.casefold().split()), set()).add(split)
            self.assertTrue(all(len(splits) == 1 for splits in locations.values()))

    def test_query_metrics_include_clean_regression_and_topk(self) -> None:
        metrics = evaluate_queries(
            ['clean', 'eror', 'bad'],
            ['clean', 'error', 'good'],
            [['changed', 'clean'], ['wrong', 'error'], ['good']],
        )
        self.assertEqual(metrics['query_exact_accuracy'], 1 / 3)
        self.assertEqual(metrics['topk_oracle_accuracy'], 1.0)
        self.assertEqual(metrics['clean_false_correction_rate'], 1.0)

    def test_opennmt_config_matches_published_base_shape(self) -> None:
        from reparos.training.opennmt import build_opennmt_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'data' / 'base'
            base.mkdir(parents=True)
            for split in ('train', 'validation'):
                (base / f'{split}.src').write_text('helo\n', encoding='utf-8')
                (base / f'{split}.tgt').write_text('hello\n', encoding='utf-8')
            tokenizer = root / 'tokenizer.model'
            tokenizer.write_bytes(b'fixture')
            path = build_opennmt_config(
                root / 'data', tokenizer, root / 'run', train_steps=10,
                model_dtype='fp16',
            )
            state = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(state['enc_layers'], 1)
            self.assertEqual(state['dec_layers'], 1)
            self.assertEqual(state['heads'], 8)
            self.assertEqual(state['hidden_size'], 128)
            self.assertEqual(state['self_attn_type'], 'scaled-dot')
            self.assertEqual(state['model_dtype'], 'fp16')
            self.assertEqual(state['train_steps'], 10)
            self.assertEqual(state['data']['corpus_1']['transforms'], ['sentencepiece'])
            resolved = json.loads((root / 'run' / 'resolved-architecture.json').read_text(encoding='utf-8'))
            self.assertEqual(resolved['published']['hidden_size'], 128)
            self.assertEqual(resolved['local_defaults']['transformer_ff'], 512)
            self.assertEqual(resolved['inferred']['attention_type'], 'scaled-dot')
            decoding = json.loads((root / 'run' / 'decoding-config.json').read_text(encoding='utf-8'))
            self.assertEqual(decoding['beam_size'], 10)
            self.assertEqual(decoding['max_decoding_length'], 100)

    def test_opennmt_translation_uses_resolved_decoding(self) -> None:
        from reparos.architecture import DecodingConfig
        from reparos.training.opennmt import translate_opennmt
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            'reparos.training.opennmt._require_opennmt'
        ), mock.patch('reparos.training.opennmt.subprocess.run') as run:
            root = Path(directory)
            translate_opennmt(
                root / 'model.pt', root / 'tokenizer.model', root / 'queries.txt',
                root / 'predictions.txt',
                decoding=DecodingConfig(
                    beam_size=4, num_hypotheses=2, min_decoding_length=1,
                    max_decoding_length=37, no_repeat_ngram_size=3,
                ),
            )
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('-beam_size') + 1], '4')
            self.assertEqual(command[command.index('-n_best') + 1], '2')
            self.assertEqual(command[command.index('-max_length') + 1], '37')
            self.assertEqual(command[command.index('-block_ngram_repeat') + 1], '3')

    def test_opennmt_vocabulary_uses_all_training_examples(self) -> None:
        from reparos.training.opennmt import build_opennmt_vocabulary
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            'reparos.training.opennmt._require_opennmt'
        ), mock.patch('reparos.training.opennmt.subprocess.run') as run:
            root = Path(directory)
            src_vocab = root / 'vocab.src'
            tgt_vocab = root / 'vocab.tgt'
            src_vocab.write_text('token\t1\n', encoding='utf-8')
            tgt_vocab.write_text('token\t1\n', encoding='utf-8')
            config = root / 'opennmt-base.json'
            config.write_text(json.dumps({
                'src_vocab': str(src_vocab),
                'tgt_vocab': str(tgt_vocab),
            }), encoding='utf-8')

            build_opennmt_vocabulary(config)

            command = run.call_args.args[0]
            self.assertEqual(command[command.index('-n_sample') + 1], '-1')

    def test_ablation_plan_is_controlled_and_does_not_train(self) -> None:
        from reparos.ablation import build_base_ablation_plan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'data' / 'base'
            base.mkdir(parents=True)
            for split in ('train', 'validation'):
                (base / f'{split}.src').write_text('helo\n', encoding='utf-8')
                (base / f'{split}.tgt').write_text('hello\n', encoding='utf-8')
            tokenizer = root / 'tokenizer.model'
            tokenizer.write_bytes(b'fixture')
            plan_path = build_base_ablation_plan(
                root / 'data', tokenizer, root / 'ablation', train_steps=10,
                valid_steps=5, save_checkpoint_steps=5, num_workers=0,
            )
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
            self.assertEqual(plan['execution'], 'configs-only; no training was launched')
            self.assertEqual(len(plan['runs']), 5)
            by_id = {run['id']: run for run in plan['runs']}
            self.assertEqual(by_id['baseline-ff512']['overrides']['transformer_ff'], 512)
            self.assertEqual(by_id['ff2048']['overrides']['transformer_ff'], 2048)
            for run in plan['runs']:
                self.assertTrue(Path(run['config']).is_file())
                self.assertFalse(list(Path(run['config']).parent.glob('*.pt')))

    @unittest.skipUnless(
        os.environ.get('REPAROS_RUN_OPENNMT_SMOKE') == '1'
        and importlib.util.find_spec('onmt')
        and importlib.util.find_spec('ctranslate2'),
        'set REPAROS_RUN_OPENNMT_SMOKE=1 with reparos-opennmt installed',
    )
    def test_opennmt_to_ctranslate2_smoke(self) -> None:
        from reparos.config import TokenizerConfig
        from reparos.serving.ctranslate2 import export_opennmt_checkpoint
        from reparos.serving.parity import compare_opennmt_ctranslate2
        from reparos.tokenization import train_sentencepiece
        from reparos.training.opennmt import build_opennmt_config, train_opennmt

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'data' / 'base'
            base.mkdir(parents=True)
            rows = [('ha noi', 'hà nội'), ('noi bai', 'nội bài')] * 16
            for split in ('train', 'validation'):
                (base / f'{split}.src').write_text(
                    '\n'.join(source for source, _ in rows) + '\n', encoding='utf-8'
                )
                (base / f'{split}.tgt').write_text(
                    '\n'.join(target for _, target in rows) + '\n', encoding='utf-8'
                )
            tokenizer_root = root / 'tokenizer'
            train_sentencepiece(
                (base / 'train.src', base / 'train.tgt'), tokenizer_root,
                TokenizerConfig(vocab_size=32),
            )
            tokenizer = tokenizer_root / 'tokenizer.model'
            run_root = root / 'onmt'
            config = build_opennmt_config(
                root / 'data', tokenizer, run_root,
                train_steps=1, valid_steps=1, save_checkpoint_steps=1, batch_size=32,
                bucket_size=64, num_workers=0,
            )
            training = train_opennmt(config)
            checkpoint = Path(str(training['latest_checkpoint']))
            converted = root / 'ctranslate2'
            manifest = export_opennmt_checkpoint(
                checkpoint, tokenizer, converted, trust_checkpoint=True,
            )
            self.assertEqual(manifest['source_format'], 'OpenNMT-py')
            queries = root / 'queries.txt'
            queries.write_text('ha noi\n', encoding='utf-8')
            report = compare_opennmt_ctranslate2(
                checkpoint, converted, tokenizer, queries,
                beam_size=1, n_best=1,
            )
            self.assertEqual(report['queries'], 1)
            self.assertTrue(report['records'][0]['opennmt'][0])
            self.assertTrue(report['records'][0]['ctranslate2'][0])

    def test_error_classes_are_deterministic(self) -> None:
        import random
        clean = 'san bay noi bai'
        first = noisy_query(clean, 'edit_compounding', random.Random(7), {})
        second = noisy_query(clean, 'edit_compounding', random.Random(7), {})
        self.assertEqual(first, second)
        self.assertNotEqual(first, clean)

    def test_vietnamese_pilot_is_stratified_and_group_disjoint(self) -> None:
        from reparos.pilot import ERROR_CLASSES, prepare_vietnamese_search_pilot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, offset in (('train', 0), ('validation', 100)):
                folder = root / 'clean' / split
                folder.mkdir(parents=True)
                rows = [f'đường nguyễn trãi số {offset + index}' for index in range(20)]
                (folder / 'corpus.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
            output = root / 'pilot'
            manifest = prepare_vietnamese_search_pilot(
                root / 'clean', output, train_per_class=5, validation_per_class=4,
            )
            self.assertTrue(manifest['group_split_verified'])
            self.assertFalse(manifest['test_split_used'])
            for split, expected in (('train', 5), ('validation', 4)):
                metadata = [
                    json.loads(line) for line in
                    (output / 'base' / f'{split}.meta.jsonl').read_text(encoding='utf-8').splitlines()
                ]
                counts = {kind: 0 for kind in ERROR_CLASSES}
                for row in metadata:
                    counts[row['error_class']] += 1
                self.assertEqual(counts, {kind: expected for kind in ERROR_CLASSES})
            train_groups = {
                json.loads(line)['group_id'] for line in
                (output / 'base' / 'train.meta.jsonl').read_text(encoding='utf-8').splitlines()
            }
            validation_groups = {
                json.loads(line)['group_id'] for line in
                (output / 'base' / 'validation.meta.jsonl').read_text(encoding='utf-8').splitlines()
            }
            self.assertFalse(train_groups & validation_groups)

    def test_vietnamese_pilot_drops_validation_groups_seen_in_train(self) -> None:
        from reparos.pilot import prepare_vietnamese_search_pilot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / 'clean' / 'train'
            validation = root / 'clean' / 'validation'
            train.mkdir(parents=True)
            validation.mkdir(parents=True)
            shared = 'đường nguyễn trãi số 1'
            train_rows = [shared] + [f'đường lê lợi số {index}' for index in range(20)]
            validation_rows = [shared] + [f'đường hai bà trưng số {index}' for index in range(20)]
            (train / 'corpus.txt').write_text('\n'.join(train_rows) + '\n', encoding='utf-8')
            (validation / 'corpus.txt').write_text(
                '\n'.join(validation_rows) + '\n', encoding='utf-8'
            )
            output = root / 'pilot'
            prepare_vietnamese_search_pilot(
                root / 'clean', output, train_per_class=10, validation_per_class=10,
            )
            validation_targets = (
                output / 'base' / 'validation.tgt'
            ).read_text(encoding='utf-8').splitlines()
            self.assertNotIn(shared, validation_targets)

    def test_vietnamese_pilot_clean_multiplier_only_changes_train(self) -> None:
        from reparos.pilot import prepare_vietnamese_search_pilot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, offset in (('train', 0), ('validation', 100)):
                folder = root / 'clean' / split
                folder.mkdir(parents=True)
                rows = [f'đường trần phú số {offset + index}' for index in range(30)]
                (folder / 'corpus.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
            manifest = prepare_vietnamese_search_pilot(
                root / 'clean', root / 'pilot',
                train_per_class=4, validation_per_class=3, train_clean_multiplier=3,
            )
            self.assertEqual(manifest['counts']['train']['clean'], 12)
            self.assertEqual(manifest['counts']['train']['edit'], 4)
            self.assertEqual(manifest['counts']['validation']['clean'], 3)

    def test_prepare_writes_curriculum_and_marks_missing_private_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'source', root / 'reparos'
            for split in ('train', 'validation', 'test'):
                target = source / split
                target.mkdir(parents=True)
                (target / 'corpus.txt').write_text('san bay noi bai\ncho ben thanh\n', encoding='utf-8')
            manifest = prepare_reparos_data(source, output, variants_per_query=2)
            self.assertEqual(manifest['profile'], PROFILE)
            self.assertFalse(manifest['available']['phonetic_pairs'])
            self.assertTrue((output / 'base' / 'train.src').is_file())
            self.assertTrue((output / 'c1' / 'train.tgt').is_file())
            config = json.loads((output / 'opennmt-disclosed-config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['common']['attention_heads'], 8)
            self.assertEqual(config['common']['beam_width'], 10)

    def test_improvement_regression_head_tail_mixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'labels.csv', root / 'eval'
            with source.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(('noisy_query', 'correct_query', 'query_id', 'monthly_frequency', 'review_status'))
                for index in range(20):
                    writer.writerow((f'q{index}', f'q{index}', f'id{index}', 100-index, 'adjudicated'))
            counts = prepare_improvement_regression_sets(source, output, examples_per_set=10)
            self.assertEqual(counts['regression_queries'], 10)
            with (output / 'regression.csv').open(encoding='utf-8', newline='') as stream:
                regression = list(csv.DictReader(stream))
            with (output / 'improvement.csv').open(encoding='utf-8', newline='') as stream:
                improvement = list(csv.DictReader(stream))
            self.assertEqual(sum(row['frequency_bucket'] == 'head' for row in regression), 9)
            self.assertEqual(sum(row['frequency_bucket'] == 'tail' for row in improvement), 9)

    @unittest.skipUnless(
        importlib.util.find_spec('torch') and importlib.util.find_spec('sentencepiece'),
        'reparos-train optional dependencies are not installed',
    )
    def test_tiny_base_training_checkpoint_and_prediction(self) -> None:
        from reparos.config import ModelConfig, TokenizerConfig, TrainingConfig
        from reparos.inference import ReferencePredictor
        from reparos.tokenization import train_sentencepiece
        from reparos.training import train_base

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data' / 'base'
            data.mkdir(parents=True)
            rows = [
                ('ha noi', 'hà nội'), ('hanoi', 'hà nội'),
                ('noi bai', 'nội bài'), ('noibai', 'nội bài'),
            ] * 3
            for split in ('train', 'validation'):
                (data / f'{split}.src').write_text('\n'.join(a for a, _ in rows) + '\n', encoding='utf-8')
                (data / f'{split}.tgt').write_text('\n'.join(b for _, b in rows) + '\n', encoding='utf-8')
            tokenizer_root = root / 'tokenizer'
            train_sentencepiece(
                (data / 'train.src', data / 'train.tgt'), tokenizer_root,
                TokenizerConfig(vocab_size=32),
            )
            from reparos.tokenization import SentencePieceTokenizer
            tokenizer_path = tokenizer_root / 'tokenizer.model'
            tokenizer = SentencePieceTokenizer(tokenizer_path)
            self.assertEqual(tokenizer.decode(tokenizer.encode('hà nội')), 'hà nội')
            model_root = root / 'model'
            train_base(
                root / 'data', tokenizer_path, model_root,
                model_config=ModelConfig(
                    vocab_size=tokenizer.vocab_size, hidden_size=16,
                    attention_heads=4, feedforward_size=32, dropout=0.0,
                    max_length=16,
                ),
                training_config=TrainingConfig.for_stage(
                    'base', epochs=1, batch_size=4, warmup_steps=1, device='cpu',
                ),
            )
            self.assertTrue((model_root / 'best.ckpt').is_file())
            prediction = ReferencePredictor(model_root / 'best.ckpt', tokenizer_path).predict('ha noi', beam_size=2)
            self.assertEqual(prediction['backend'], 'reference')
            self.assertEqual(len(prediction['hypotheses']), 2)


if __name__ == '__main__':
    unittest.main()
