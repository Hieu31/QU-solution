from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from reparos.data import (
    PROFILE,
    noisy_query,
    prepare_improvement_regression_sets,
    prepare_reparos_data,
)


class ReparoSTests(unittest.TestCase):
    def test_error_classes_are_deterministic(self) -> None:
        import random
        clean = 'san bay noi bai'
        first = noisy_query(clean, 'edit_compounding', random.Random(7), {})
        second = noisy_query(clean, 'edit_compounding', random.Random(7), {})
        self.assertEqual(first, second)
        self.assertNotEqual(first, clean)

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


if __name__ == '__main__':
    unittest.main()
