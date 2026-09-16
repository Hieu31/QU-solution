from __future__ import annotations

import unittest

from webspell.config import CandidateConfig
from webspell.demo import DEMO_SENTENCES, DEMO_TRIPLES
from webspell.error_model import SubstringErrorModel
from webspell.error_model.model import SubstringAligner, _initial_score
from webspell.language_model import BidirectionalLanguageModel
from webspell.types import Context
from webspell.vocabulary import TermLexicon, damerau_levenshtein


class EditDistanceTests(unittest.TestCase):
    def test_damerau_operations(self) -> None:
        self.assertEqual(damerau_levenshtein("the", "teh"), 1)
        self.assertEqual(damerau_levenshtein("spell", "spel"), 1)
        self.assertEqual(damerau_levenshtein("cat", "cart"), 1)
        self.assertEqual(damerau_levenshtein("cat", "cut"), 1)

    def test_bounded_damerau_matches_exact_within_cutoff(self) -> None:
        pairs = (("kitten", "sitting"), ("the", "teh"), ("abc", "xyz"), ("", "abc"))
        for source, target in pairs:
            exact = damerau_levenshtein(source, target)
            for cutoff in range(4):
                expected = exact if exact <= cutoff else cutoff + 1
                self.assertEqual(
                    damerau_levenshtein(source, target, cutoff), expected
                )

    def test_paper_distance_thresholds(self) -> None:
        config = CandidateConfig()
        self.assertEqual(config.max_edit_distance(4), 1)
        self.assertEqual(config.max_edit_distance(5), 2)
        self.assertEqual(config.max_edit_distance(12), 2)
        self.assertEqual(config.max_edit_distance(13), 3)

    def test_lexicon_search_is_deterministic(self) -> None:
        lexicon = TermLexicon({"the": 20, "then": 10, "tea": 5})
        terms = [match.term for match in lexicon.close_terms("teh", 2)]
        self.assertEqual(terms, ["the", "tea", "then"])


class StatisticalModelTests(unittest.TestCase):
    def test_score_only_alignment_matches_full_alignment(self) -> None:
        aligner = SubstringAligner()
        for intended, observed in (("the", "teh"), ("abc", "ac"), ("", "x")):
            self.assertEqual(
                aligner.best_score(intended, observed, _initial_score),
                aligner.best_alignment(intended, observed, _initial_score).score,
            )

    def test_error_model_learns_transposition(self) -> None:
        model = SubstringErrorModel.fit(DEMO_TRIPLES)
        known = model.word_logprob("teh", "the")
        unknown = model.word_logprob("xyz", "the")
        self.assertGreater(known, unknown)
        self.assertIn("eh", model.transitions().get("he", {}))

    def test_bidirectional_lm_uses_context(self) -> None:
        model = BidirectionalLanguageModel.fit(list(DEMO_SENTENCES) * 10)
        context = Context(("we", "learn"), ("clean", "text"))
        self.assertGreater(
            model.token_logscore("from", context),
            model.token_logscore("form", context),
        )


if __name__ == "__main__":
    unittest.main()
