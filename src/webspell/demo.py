from __future__ import annotations

from collections import Counter

from webspell.candidates import CandidateRanker, tune_lambdas
from webspell.confidence import ConfidenceModel
from webspell.config import WebSpellConfig
from webspell.error_model import SubstringErrorModel
from webspell.language_model import BidirectionalLanguageModel
from webspell.pipeline import WebSpellModel, build_training_rows
from webspell.text import SimpleTokenizer
from webspell.types import Context, ErrorTriple, LabeledToken
from webspell.vocabulary import TermLexicon


DEMO_SENTENCES: tuple[tuple[str, ...], ...] = (
    ("the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"),
    ("the", "quick", "red", "fox", "runs", "through", "the", "field"),
    ("we", "spell", "words", "using", "their", "context"),
    ("context", "helps", "choose", "the", "correct", "word"),
    ("letters", "form", "a", "word"),
    ("we", "learn", "from", "clean", "text"),
    ("a", "language", "model", "uses", "word", "context"),
    ("the", "model", "can", "correct", "spelling", "errors"),
)

DEMO_TRIPLES: tuple[ErrorTriple, ...] = (
    ErrorTriple("the", "teh", 80),
    ErrorTriple("quick", "quik", 50),
    ErrorTriple("brown", "brwon", 40),
    ErrorTriple("spell", "spel", 50),
    ErrorTriple("words", "wrods", 50),
    ErrorTriple("using", "usign", 30),
    ErrorTriple("context", "contex", 30),
    ErrorTriple("from", "form", 15),
)


def _examples() -> list[LabeledToken]:
    return [
        LabeledToken("teh", "the", Context((), ("quick", "brown"))),
        LabeledToken("quik", "quick", Context(("the",), ("brown", "fox"))),
        LabeledToken("brwon", "brown", Context(("the", "quick"), ("fox",))),
        LabeledToken("spel", "spell", Context(("we",), ("words", "using"))),
        LabeledToken("wrods", "words", Context(("we", "spell"), ("using",))),
        LabeledToken("usign", "using", Context(("spell", "words"), ("their", "context"))),
        LabeledToken("contex", "context", Context(("their",), ())),
        LabeledToken("form", "from", Context(("we", "learn"), ("clean", "text"))),
        LabeledToken("form", "form", Context(("letters",), ("a", "word"))),
        LabeledToken("from", "from", Context(("we", "learn"), ("clean", "text"))),
        LabeledToken("word", "word", Context(("correct",), ())),
        LabeledToken("fox", "fox", Context(("quick", "brown"), ())),
        LabeledToken("dog", "dog", Context(("the", "lazy"), ())),
        LabeledToken("clean", "clean", Context(("learn", "from"), ("text",))),
        LabeledToken("text", "text", Context(("from", "clean"), ())),
        LabeledToken("model", "model", Context(("a", "language"), ("uses",))),
        LabeledToken("xylophone", "xylophone", Context((), ())),
        LabeledToken("qqqqqq", "missing", Context((), ())),
    ]


def build_demo_model(config: WebSpellConfig | None = None) -> WebSpellModel:
    """Train the complete architecture on a tiny deterministic offline fixture."""
    settings = config or WebSpellConfig()
    expanded_sentences = list(DEMO_SENTENCES) * 20
    frequencies: Counter[str] = Counter()
    for sentence in expanded_sentences:
        frequencies.update(sentence)
    frequencies["xylophone"] = 5

    lexicon = TermLexicon(frequencies)
    error_model = SubstringErrorModel.fit(DEMO_TRIPLES, settings.error_model)
    language_model = BidirectionalLanguageModel.fit(
        expanded_sentences, settings.language_model
    )
    ranker = CandidateRanker(lexicon, error_model, language_model, settings.candidates)
    examples = _examples()
    lambdas = tune_lambdas(examples, ranker, settings.lambda_tuning)
    confidence = ConfidenceModel(settings.classifier)
    confidence.fit(build_training_rows(examples, ranker, lambdas))
    tokenizer = SimpleTokenizer(settings.unicode_normalization)
    return WebSpellModel(tokenizer, ranker, lambdas, confidence)
