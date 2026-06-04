"""Surface text-feature extraction.

These features are the *prior* signal only (section "Length/development
proxy"). They anchor a criterion prediction but never decide it on their own;
the semantic/LLM layer is expected to supply the real per-criterion evidence.
Pure standard library so the engine runs anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_RE = re.compile(r"[.!?]+")
_ENDMARKS = ".!?"
_PUNCT_CHARS = ".,!?;:\"'()-—"


@dataclass
class TextFeatures:
    char_count: int
    word_count: int
    line_count: int
    sentence_count: int
    avg_sentence_length: float
    avg_word_length: float
    long_word_rate: float        # fraction of words with > 6 letters
    type_token_ratio: float
    comma_density: float         # commas per word
    endmark_density: float       # . ! ? per word
    punctuation_variety: int     # distinct punctuation characters used

    def to_dict(self) -> dict:
        return asdict(self)


def extract_text_features(text: str | None) -> TextFeatures:
    """Compute surface features from raw script text."""
    text = text or ""
    char_count = len(text)
    line_count = text.count("\n") + 1 if text.strip() else 0

    words = _WORD_RE.findall(text)
    word_count = len(words)

    if word_count == 0:
        return TextFeatures(
            char_count=char_count, word_count=0, line_count=line_count,
            sentence_count=0, avg_sentence_length=0.0, avg_word_length=0.0,
            long_word_rate=0.0, type_token_ratio=0.0, comma_density=0.0,
            endmark_density=0.0, punctuation_variety=0,
        )

    # Sentences: at least one if there is any text but no terminal punctuation.
    sentence_count = max(1, len(_SENTENCE_RE.findall(text)))

    long_words = sum(1 for w in words if len(w) > 6)
    total_word_chars = sum(len(w) for w in words)
    distinct = len({w.lower() for w in words})
    commas = text.count(",")
    endmarks = sum(text.count(ch) for ch in _ENDMARKS)
    variety = len({ch for ch in text if ch in _PUNCT_CHARS})

    return TextFeatures(
        char_count=char_count,
        word_count=word_count,
        line_count=line_count,
        sentence_count=sentence_count,
        avg_sentence_length=round(word_count / sentence_count, 2),
        avg_word_length=round(total_word_chars / word_count, 2),
        long_word_rate=round(long_words / word_count, 3),
        type_token_ratio=round(distinct / word_count, 3),
        comma_density=round(commas / word_count, 3),
        endmark_density=round(endmarks / word_count, 3),
        punctuation_variety=variety,
    )
