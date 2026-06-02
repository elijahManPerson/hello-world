"""
vocab_classifier.py
===================
Phase VP -- Vocabulary precision.

Populates `VocabCategory` and `VocabCategoryConfidence` on the word_map
from the LLM-classified Spelling_with_VocabClass_LLM.csv lexicon.

Source priority:
  1. VocabCategory_LLM (Column Z) -- primary, LLM-labelled
  2. Fallback to 'simple' when Column Z is empty (most gap words are
     high-frequency function/common words mis-labelled by the rule-based pass)

VP1 must run AFTER Phase SP so that WordKind=proper_noun_abbreviation
and junk are already populated (used to skip proper nouns).
"""

import os
import pandas as pd

_VOCAB_LEXICON = None
_DEFAULT_VOCAB_PATH = "Spelling_with_VocabClass_LLM.csv"


def load_vocab_lexicon(path=None):
    """Load the vocab lexicon once and cache it.

    Returns a dict: word_lower -> (category, confidence)
    where category is 'precise', 'specific', or 'simple'.
    """
    global _VOCAB_LEXICON
    if _VOCAB_LEXICON is not None:
        return _VOCAB_LEXICON

    csv_path = path or _DEFAULT_VOCAB_PATH
    if not os.path.exists(csv_path):
        # Try relative to this file's directory
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(here, csv_path)
    if not os.path.exists(csv_path):
        print(f"  [VP1] vocab lexicon not found: {csv_path}")
        _VOCAB_LEXICON = {}
        return _VOCAB_LEXICON

    df = pd.read_csv(csv_path, keep_default_na=False, low_memory=False)

    # Validate required columns
    if "VocabCategory_LLM" not in df.columns:
        print(f"  [VP1] VocabCategory_LLM column missing from {csv_path}")
        _VOCAB_LEXICON = {}
        return _VOCAB_LEXICON

    conf_col = "VocabConfidence" if "VocabConfidence" in df.columns else None

    _VOCAB_LEXICON = {}
    for _, row in df.iterrows():
        key = str(row.get("Word", "")).lower().strip()
        if not key or key in _VOCAB_LEXICON:
            continue   # first occurrence wins (lexicon is roughly freq-ordered)

        z_val  = str(row.get("VocabCategory_LLM", "")).strip()
        z_conf = str(row.get(conf_col, "high")).strip() if conf_col else "high"

        if z_val:
            _VOCAB_LEXICON[key] = (z_val, z_conf)
        else:
            # Column Z empty -- fallback to NA (WM1: changed from 'simple')
            _VOCAB_LEXICON[key] = ("NA", "fallback_NA")

    return _VOCAB_LEXICON


def lookup_vocab_category(word, lexicon=None):
    """Return (category, confidence) for a word.

    Returns ('NA', 'NA') if word not in lexicon.
    """
    if not word:
        return ("NA", "NA")
    lex = lexicon if lexicon is not None else load_vocab_lexicon()
    entry = lex.get(str(word).lower().strip())
    if entry is None:
        return ("NA", "NA")
    return entry


def assign_vocab_columns(wm, lexicon_path=None):
    """
    Populate VocabCategory and VocabCategoryConfidence on the word_map.

    Only word tokens get classified. Non-word tokens (punctuation, etc.)
    get empty strings. Proper nouns (WordKind=proper_noun_abbreviation)
    and junk tokens get 'NA' -- they are not vocabulary signals.
    """
    lex = load_vocab_lexicon(lexicon_path)
    if not lex:
        print("  [VP1] empty lexicon -- VocabCategory not populated")
        return wm

    vocab_cats  = []
    vocab_confs = []

    for _, row in wm.iterrows():
        tok_cat  = str(row.get("TokenCategory", ""))
        word_kind = str(row.get("WordKind", ""))

        if tok_cat != "word":
            vocab_cats.append("")
            vocab_confs.append("")
            continue

        # Proper nouns and junk are not vocabulary signals
        if word_kind in ("proper_noun_abbreviation", "junk", "name_foreign"):
            vocab_cats.append("NA")
            vocab_confs.append("NA")
            continue

        cat, conf = lookup_vocab_category(row.get("corr_token"), lex)
        vocab_cats.append(cat)
        vocab_confs.append(conf)

    wm = wm.copy()
    wm["VocabCategory"] = vocab_cats
    wm["VocabCategoryConfidence"] = vocab_confs
    return wm
