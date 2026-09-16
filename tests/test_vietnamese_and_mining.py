from __future__ import annotations

import unittest

from webspell.candidates import CandidateRanker
from webspell.config import LanguageModelConfig
from webspell.error_model import SubstringErrorModel
from webspell.language_model import BidirectionalLanguageModel
from webspell.mining import ContextStatistics, mine_close_pairs, mine_error_triples
from webspell.text import VietnameseVariantGenerator, decode_telex, decode_vni
from webspell.types import ErrorTriple
from webspell.vocabulary import TermLexicon


class ErrorTripleMiningTests(unittest.TestCase):
    def test_context_selects_frequent_intended_term(self) -> None:
        sentences = [("please", "receive", "this")] * 20
        sentences += [("please", "recieve", "this")] * 2
        lexicon = TermLexicon.from_sentences(sentences)
        pairs = mine_close_pairs(lexicon, frequency_ratio=10.0)
        triples = mine_error_triples(
            pairs,
            ContextStatistics.from_sentences(sentences),
            minimum_context_frequency=10,
        )
        self.assertIn(ErrorTriple("receive", "recieve", 2), triples)

    def test_context_tie_does_not_infer_an_error(self) -> None:
        sentences = [('common', 'receive', 'context')]
        sentences += [('common', 'recieve', 'context')]
        sentences += [('other', 'receive', 'place')] * 19
        sentences += [('else', 'recieve', 'where')]
        lexicon = TermLexicon.from_sentences(sentences)
        pairs = mine_close_pairs(lexicon, frequency_ratio=10.0)
        triples = mine_error_triples(
            pairs, ContextStatistics.from_sentences(sentences),
            minimum_context_frequency=2,
        )
        self.assertNotIn(ErrorTriple('receive', 'recieve', 1), triples)


class VietnameseAdapterTests(unittest.TestCase):
    def test_telex_decoding(self) -> None:
        self.assertEqual(decode_telex("tieengs"), "tiếng")
        self.assertEqual(decode_telex("nguowif"), "người")
        self.assertEqual(decode_telex("ddawng"), "đăng")

    def test_vni_decoding(self) -> None:
        self.assertEqual(decode_vni("tie6ng1"), "tiếng")

    def test_variant_bypasses_raw_edit_distance_cutoff(self) -> None:
        sentences = [("tôi", "nói", "tiếng", "việt")] * 10
        lexicon = TermLexicon.from_sentences(sentences)
        error_model = SubstringErrorModel.fit(
            [ErrorTriple("tiếng", "tieengs", 20)]
        )
        language_model = BidirectionalLanguageModel.fit(
            sentences, LanguageModelConfig(order=3)
        )
        ranker = CandidateRanker(
            lexicon,
            error_model,
            language_model,
            variant_generator=VietnameseVariantGenerator(),
        )
        self.assertIn("tiếng", [candidate.term for candidate in ranker.generate("tieengs")])


if __name__ == "__main__":
    unittest.main()
