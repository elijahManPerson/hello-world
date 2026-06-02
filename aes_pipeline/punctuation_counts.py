"""
Phase PF -- Punctuation counts.

Adds per-script Punctuation count columns to the texts dataframe, mirroring
spelling_final.add_spelling_counts, break_classifications.add_break_counts and
ss_counts.add_ss_counts. Call once, after the sentence layer and the word-class
/ capitalization fill have run:

    import punctuation_counts as pc
    df = pc.add_punctuation_counts(df, sent, wm, id_col=cfg["id_col"])

Three parallel blocks, each with Total / Correct / Incorrect / CorrectPct /
IncorrectPct, plus a standalone stray-capital count.

Sentence punctuation (per sentence, from the sentence-level "Punc boundary"
verdict -- the opening capital and end mark of each sentence):
    Pun_Sentence_Total  Pun_Sentence_Correct  Pun_Sentence_Incorrect
    Pun_Sentence_CorrectPct  Pun_Sentence_IncorrectPct

Other punctuation (per sentence, from "Punc internal" -- commas, apostrophes
and other within-sentence marks):
    Pun_Other_Total  Pun_Other_Correct  Pun_Other_Incorrect
    Pun_Other_CorrectPct  Pun_Other_IncorrectPct

Noun capitalization (per token, from the word-level "Cap class" in
{Noun capital, Abbrev./Acronym} -- proper nouns, abbreviations and acronyms;
NOT the sentence-opening "Boundary capital", which is already covered by the
Sentence block):
    Pun_NounCaps_Total  Pun_NounCaps_Correct  Pun_NounCaps_Incorrect
    Pun_NounCaps_CorrectPct  Pun_NounCaps_IncorrectPct

Stray capitals (per token, "Cap class" == "Stray capital" -- a capital where
none belongs; these are over-capitalization errors by definition):
    Pun_StrayCaps

Counting unit
-------------
The Sentence and Other blocks count per sentence (each sentence is correct or
incorrect on that punctuation dimension). The totals are the judged sentences
only (a verdict of Correct or Incorrect); sentences with no verdict, e.g.
excluded artifacts or sentences with no internal punctuation to assess, are
NaN in the source and so fall out of the denominator. Percentages are over the
judged total, so unjudged sentences never drag a rate down. The capitalization
blocks count per token. No artifact exclusion is applied to the token counts;
if proper nouns inside titles or headings should be excluded, that is a small
add-on.
"""
from __future__ import annotations

import pandas as pd

NOUN_CAP_CLASSES = ["Noun capital", "Abbrev./Acronym"]
STRAY_CAP_CLASS = "Stray capital"


def add_punctuation_counts(texts_df: pd.DataFrame, sent_df: pd.DataFrame,
                           word_map_df: pd.DataFrame,
                           id_col: str = "Research ID") -> pd.DataFrame:
    df = texts_df.copy()
    # Drop any Pun_ count columns from a previous run (keeps the marker "Pun").
    df = df.drop(columns=[c for c in df.columns if c.startswith("Pun_")],
                 errors="ignore")
    ids = df[id_col].astype(str)

    def _block(prefix, correct_mask, incorrect_mask, gid):
        def per_script(mask):
            return mask.astype(int).groupby(gid).sum()
        cor = ids.map(per_script(correct_mask)).fillna(0).astype(int)
        inc = ids.map(per_script(incorrect_mask)).fillna(0).astype(int)
        tot = cor + inc
        df[f"{prefix}_Total"]        = tot
        df[f"{prefix}_Correct"]      = cor
        df[f"{prefix}_Incorrect"]    = inc
        df[f"{prefix}_CorrectPct"]   = (100.0 * cor / tot.where(tot > 0)).round(1).fillna(0.0)
        df[f"{prefix}_IncorrectPct"] = (100.0 * inc / tot.where(tot > 0)).round(1).fillna(0.0)

    # ---- sentence-level blocks ----
    def scol(name):
        if name in sent_df.columns:
            return sent_df[name].astype(str)
        return pd.Series([""] * len(sent_df), index=sent_df.index)

    sgid = scol("Identifier")
    pb = scol("Punc boundary")
    pi = scol("Punc internal")
    _block("Pun_Sentence", pb == "Correct", pb == "Incorrect", sgid)
    _block("Pun_Other",    pi == "Correct", pi == "Incorrect", sgid)

    # ---- word-level capitalization blocks ----
    def wcol(name):
        if name in word_map_df.columns:
            return word_map_df[name].astype(str)
        return pd.Series([""] * len(word_map_df), index=word_map_df.index)

    wgid = wcol("Identifier")
    cap_class = wcol("Cap class")
    cap_err = wcol("Cap error").str.lower()
    is_noun = cap_class.isin(NOUN_CAP_CLASSES)
    _block("Pun_NounCaps",
           is_noun & (cap_err == "false"),
           is_noun & (cap_err == "true"),
           wgid)

    stray = (cap_class == STRAY_CAP_CLASS)
    df["Pun_StrayCaps"] = ids.map(
        stray.astype(int).groupby(wgid).sum()).fillna(0).astype(int)

    return df
