from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from webspell.config import CandidateConfig
from webspell.osm.paper import (
    CHARACTER_ERROR_RATE,
    PROFILE,
    corrupt_document,
    prepare_artificial_data,
    validate_typed_test,
)


class PaperBaselineTests(unittest.TestCase):
    def test_corruption_is_deterministic_and_uses_only_paper_operations(self) -> None:
        first = corrupt_document('a sufficiently long clean document fragment ' * 20, 7, 0.2)
        second = corrupt_document('a sufficiently long clean document fragment ' * 20, 7, 0.2)
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], 'a sufficiently long clean document fragment ' * 20)
        self.assertTrue(set(first[1]).issubset({'deletion', 'transposition', 'insertion'}))

    def test_prepare_artificial_data_keeps_source_untouched_and_writes_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'source', root / 'paper'
            original = 'sân bay nội bài\nđường nguyễn huệ\n'
            for split in ('train', 'validation', 'test'):
                split_root = source / split
                split_root.mkdir(parents=True)
                (split_root / 'corpus.txt').write_text(original, encoding='utf-8')
            profile = prepare_artificial_data(source, output, variants_per_document=2)
            self.assertEqual(profile['profile'], PROFILE)
            self.assertEqual(profile['artificial_generation']['character_error_rate'], CHARACTER_ERROR_RATE)
            self.assertEqual((source / 'train' / 'corpus.txt').read_text(encoding='utf-8'), original)
            self.assertTrue((output / 'test' / 'noisy_pairs.csv').is_file())
            saved = json.loads((output / 'reproduction-profile.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['lanes']['typed'].split(';')[0], 'external held-out human-typed CSV')

    def test_typed_test_requires_adjudication_and_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'typed.csv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(('noisy_query', 'correct_query', 'query_id', 'participant_id', 'source_text_id', 'collection_protocol', 'review_status'))
                writer.writerow(('san bay noi bai', 'sân bay nội bài', 'q1', 'p1', 'poi1', 'retype_no_backspace', 'adjudicated'))
                writer.writerow(('hà nội', 'hà nội', 'q2', 'p2', 'poi2', 'free_search', 'adjudicated'))
            counts = validate_typed_test(path)
            self.assertEqual(counts['queries'], 2)
            self.assertEqual(counts['changed_queries'], 1)
            self.assertEqual(counts['clean_queries'], 1)

    def test_candidate_distance_cap_separates_mining_from_serving(self) -> None:
        paper_runtime = CandidateConfig(maximum_edit_distance=2)
        mining = CandidateConfig(maximum_edit_distance=3)
        self.assertEqual(paper_runtime.max_edit_distance(20), 2)
        self.assertEqual(mining.max_edit_distance(20), 3)


if __name__ == '__main__':
    unittest.main()
