from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from webspell.candidates import tune_lambdas
from webspell.demo import _examples, build_demo_model
from webspell.error_model import SubstringErrorModel
from webspell.language_model import BidirectionalLanguageModel
from webspell.mining import iter_close_pairs, iter_error_triples
from webspell.confidence import TrainingRow
from webspell.scalable import (
    SQLiteBidirectionalLanguageModel,
    SQLiteContextStatistics,
    SQLiteCorpusStatistics,
    SQLiteSymSpellIndex,
    StreamingClassifierConfig,
    StreamingConfidenceTrainer,
    tune_lambdas_streaming,
)
from webspell.types import Context
from webspell.vocabulary import TermLexicon


class ScalableBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences = [
            ("the", "quick", "brown", "fox"),
            ("the", "quick", "red", "fox"),
            ("we", "learn", "from", "clean", "text"),
        ] * 4

    def _build(self, directory: str) -> Path:
        path = Path(directory) / "statistics.sqlite3"
        SQLiteCorpusStatistics.build(self.sentences, path, order=3, batch_tokens=10)
        return path

    def test_streamed_counts_and_disk_lm_match_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._build(directory)
            metadata = SQLiteCorpusStatistics(path).metadata()
            self.assertEqual(metadata["sentence_count"], 12)
            self.assertEqual(metadata["token_count"], 52)
            disk = SQLiteBidirectionalLanguageModel(path)
            memory = BidirectionalLanguageModel.fit(self.sentences, disk.config)
            context = Context(("we", "learn"), ("clean", "text"))
            self.assertAlmostEqual(
                disk.token_logscore("from", context),
                memory.token_logscore("from", context),
            )
            disk.close()

    def test_disk_context_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._build(directory)
            contexts = SQLiteContextStatistics(path)
            self.assertEqual(contexts.count("from", ("learn", "clean")), 4)
            self.assertEqual(contexts.total(("learn", "clean")), 4)
            contexts.close()

    def test_symspell_matches_brute_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._build(directory)
            index = SQLiteSymSpellIndex.build(
                path, max_distance=2, minimum_frequency=1, batch_rows=20
            )
            reference = TermLexicon.from_sentences(self.sentences)
            expected = [item.term for item in reference.close_terms("teh", 2)]
            actual = [item.term for item in index.close_terms("teh", 2)]
            self.assertEqual(actual, expected)
            index.close()

    def test_streaming_confidence_training_is_replayable(self) -> None:
        rows = [
            TrainingRow((3.0, 1.0), 1, True, 1),
            TrainingRow((2.0, 1.0), 1, True, 1),
            TrainingRow((-2.0, 1.0), 0, True, 0),
            TrainingRow((-3.0, 1.0), 0, True, 0),
            TrainingRow((2.0, 0.0), 1, False, 0),
            TrainingRow((-2.0, 0.0), 0, False, 0),
        ]
        factory = lambda: iter(rows)
        trainer = StreamingConfidenceTrainer(
            StreamingClassifierConfig(epochs=20, learning_rate=0.05)
        )
        model = trainer.fit(factory, factory)
        self.assertGreater(
            model.spellcheck_probability((3.0, 1.0), True),
            model.spellcheck_probability((-3.0, 1.0), True),
        )

    def test_streaming_lambda_tuner_matches_reference(self) -> None:
        model = build_demo_model()
        examples = _examples()
        reference = tune_lambdas(examples, model.ranker)
        streamed = tune_lambdas_streaming(lambda: iter(examples), model.ranker)
        self.assertEqual(streamed, reference)

    def test_disk_backends_feed_streaming_error_model(self) -> None:
        sentences = [("please", "receive", "this")] * 20
        sentences += [("please", "recieve", "this")] * 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statistics.sqlite3"
            SQLiteCorpusStatistics.build(sentences, path, order=3, batch_tokens=20)
            index = SQLiteSymSpellIndex.build(
                path, max_distance=2, minimum_frequency=1
            )
            contexts = SQLiteContextStatistics(path)
            triples = iter_error_triples(
                iter_close_pairs(index, frequency_ratio=10.0),
                contexts,
                minimum_context_frequency=10,
            )
            error_model = SubstringErrorModel.fit(triples)
            self.assertGreater(
                error_model.word_logprob("recieve", "receive"),
                error_model.word_logprob("zzzzzzz", "receive"),
            )
            contexts.close()
            index.close()


if __name__ == "__main__":
    unittest.main()
