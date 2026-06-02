"""
scribe_detector.py
==================
Deterministic pre-filter for scribe detection (Phase 6 TB2).
Runs before the LLM text-level classifier to flag strong structural signals.
"""

import re


# Scribe note patterns at start of script
_SCRIBE_NOTE_START = re.compile(
    r"^\s*(scribed?\s*(by|for)?|scribe\s*[:\-]|dictated\s*(by|to)?|"
    r"written\s*(by|for)\s*scribe)",
    re.IGNORECASE,
)

# Word-list patterns at end of script (10+ lines of single-word or short entries)
_WORD_LIST_LINE = re.compile(r"^\s*\d+[\.)\]\s*\w+\s*$", re.MULTILINE)


def detect_scribe_signals(raw_text):
    """
    Scan raw text for deterministic scribe signals.

    Returns a dict:
      {
        "has_top_note": bool,       # "scribed by..." at start
        "has_word_list": bool,      # numbered word list at end
        "top_note_text": str,       # the note text if found
        "word_list_count": int,     # number of word-list items found
      }
    """
    text = str(raw_text or "").strip()
    signals = {
        "has_top_note": False,
        "has_word_list": False,
        "top_note_text": "",
        "word_list_count": 0,
    }

    if not text:
        return signals

    # Check first 100 chars for scribe note
    top_match = _SCRIBE_NOTE_START.match(text[:200])
    if top_match:
        signals["has_top_note"] = True
        first_line = text.split("\n")[0][:100]
        signals["top_note_text"] = first_line

    # Check last 400 chars for a word list
    tail = text[-400:]
    word_list_matches = _WORD_LIST_LINE.findall(tail)
    if len(word_list_matches) >= 8:
        signals["has_word_list"] = True
        signals["word_list_count"] = len(word_list_matches)

    return signals
