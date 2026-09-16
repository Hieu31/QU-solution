from __future__ import annotations

import unittest

from webspell.demo import build_demo_model
from webspell.evaluation import evaluate_predictions


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_demo_model()

    def test_nonword_corrections(self) -> None:
        text = "teh quik brwon fox"
        predictions = self.model.predict(text)
        self.assertEqual(
            self.model.corrected_text(text, predictions),
            "the quick brown fox",
        )
        self.assertEqual([item.action for item in predictions], [
            "correct", "correct", "correct", "keep"
        ])

    def test_real_word_correction_depends_on_context(self) -> None:
        wrong = "we learn form clean text"
        correct = "letters form a word"
        wrong_predictions = self.model.predict(wrong)
        correct_predictions = self.model.predict(correct)
        self.assertEqual(
            self.model.corrected_text(wrong, wrong_predictions),
            "we learn from clean text",
        )
        self.assertEqual(
            self.model.corrected_text(correct, correct_predictions),
            correct,
        )

    def test_blacklist_and_unknown_flagging(self) -> None:
        predictions = self.model.predict("a ! qqqqqq")
        self.assertEqual(predictions[0].action, "keep")
        self.assertEqual(predictions[1].action, "keep")
        self.assertEqual(predictions[2].action, "flag")

    def test_paper_metrics(self) -> None:
        predictions = self.model.predict("teh quik brwon fox")
        metrics = evaluate_predictions(predictions, ["the", "quick", "brown", "fox"])
        self.assertEqual((metrics.e1, metrics.e2, metrics.e3, metrics.e4, metrics.e5), (0, 0, 0, 0, 0))
        self.assertEqual(metrics.ter, 0.0)
        self.assertEqual(metrics.ngs, 0.0)


if __name__ == "__main__":
    unittest.main()
