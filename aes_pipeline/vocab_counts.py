"""
Phase VC -- Vocabulary counts and band.

Adds per-script Vocabulary columns to the texts dataframe, mirroring
ss_counts.add_ss_counts and punctuation_counts.add_punctuation_counts. Call
once, after the word layer and the vocabulary-precision fill (vocab_classifier
/ word_map_chunker_v3) have run:

    import vocab_counts as vc
    df = vc.add_vocab_columns(df, wm, id_col=cfg["id_col"])

It aggregates the word-level LexicalPrecision tags (simple / precise /
specific) into per-script counts, then applies the Vocabulary band ladder.

Columns added:
    Voc_ContentWords         lexical/content words: simple + precise + specific
    Voc_PreciseWords         words tagged 'precise'
    Voc_SpecificWords        words tagged 'specific'
    Voc_PreciseOrSpecific    precise + specific (specific is the more-precise tier)
    Voc_PrecisePct           100 * PreciseOrSpecific / ContentWords (the density)
    Voc_InappropriateChoices STUB. Filled later by the appropriateness pass (a
                             Sonnet check: is each content word used
                             appropriately and accurately, e.g. "onerous" used
                             for "ominous"?). NaN until that pass runs.
    Voc_Band                 0-4 from the ladder below.
    Voc_Band5Candidate       True when a script meets the band-4 bar and is a
                             candidate for band 5 pending the appropriateness
                             and effectiveness check.
    Voc_BandReason           which rung produced the band.

The ladder (rubric, 0-5 scale):
    0  under 2 words, or title only
    1  2+ words, fewer than MIN_PRECISE precise words
    2  MIN_PRECISE+ precise words, but not yet sustained (< SUSTAINED_WORDS)
    3  sustained, precise words present, below PRECISE_DENSITY_THRESHOLD% density
    4  sustained AND >= PRECISE_DENSITY_THRESHOLD% precise-or-specific density
    5  (LLM gate) an effective range of precise words AND no inappropriate or
       inaccurate word choices. Reserved: the deterministic ladder tops out at
       4 and flags band-4 scripts as Voc_Band5Candidate. When the
       appropriateness pass runs, it promotes clean candidates to 5 and can
       demote scripts whose "precise" words are actually misused.

Thresholds were set on a 69-script expert sample (bands 2-3 sat near 6%
precise density, bands 4-5 near 13%, so ~10% is the 3/4 divider) and should be
re-confirmed on the full 364-script expert run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUSTAINED_WORDS = 200
MIN_PRECISE = 4
PRECISE_DENSITY_THRESHOLD = 10.0  # percent of content words that are precise/specific
_PREC = ("precise", "specific")
_LEX = ("simple", "precise", "specific")


def add_vocab_columns(texts_df: pd.DataFrame, word_map_df: pd.DataFrame,
                      id_col: str = "Research ID") -> pd.DataFrame:
    df = texts_df.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Voc_")],
                 errors="ignore")
    ids = df[id_col].astype(str)

    def wcol(name):
        if name in word_map_df.columns:
            return word_map_df[name].astype(str)
        return pd.Series([""] * len(word_map_df), index=word_map_df.index)

    lp = wcol("LexicalPrecision")
    gid = wcol("Identifier")

    def per_script(mask):
        return mask.astype(int).groupby(gid).sum()

    content = ids.map(per_script(lp.isin(_LEX))).fillna(0).astype(int)
    precise = ids.map(per_script(lp == "precise")).fillna(0).astype(int)
    specific = ids.map(per_script(lp == "specific")).fillna(0).astype(int)
    prec_spec = precise + specific
    pct = (100.0 * prec_spec / content.where(content > 0)).round(1).fillna(0.0)

    df["Voc_ContentWords"] = content
    df["Voc_PreciseWords"] = precise
    df["Voc_SpecificWords"] = specific
    df["Voc_PreciseOrSpecific"] = prec_spec
    df["Voc_PrecisePct"] = pct
    df["Voc_InappropriateChoices"] = np.nan  # filled by the appropriateness pass

    wc = (pd.to_numeric(df["WordCount"], errors="coerce").fillna(0)
          if "WordCount" in df.columns else pd.Series(0, index=df.index))
    if "TitleOnly" in df.columns:
        title_only = df["TitleOnly"].astype(str).str.lower().isin(["true", "1"])
    else:
        title_only = pd.Series(False, index=df.index)

    c = content.values
    ps = prec_spec.values
    w = wc.values
    p = pct.values
    to = title_only.values

    bands, reasons = [], []
    for i in range(len(df)):
        if c[i] < 2 or to[i]:
            bands.append(0); reasons.append("under 2 words or title only")
        elif ps[i] < MIN_PRECISE:
            bands.append(1); reasons.append(f"2+ words, fewer than {MIN_PRECISE} precise")
        elif w[i] < SUSTAINED_WORDS:
            bands.append(2); reasons.append(f"{MIN_PRECISE}+ precise words, not yet sustained (<{SUSTAINED_WORDS} words)")
        elif p[i] < PRECISE_DENSITY_THRESHOLD:
            bands.append(3); reasons.append(f"sustained, precise present, below {PRECISE_DENSITY_THRESHOLD:.0f}% density")
        else:
            bands.append(4); reasons.append(f"sustained with {PRECISE_DENSITY_THRESHOLD:.0f}%+ precise density (band-5 candidate pending appropriateness check)")

    df["Voc_Band"] = bands
    df["Voc_BandReason"] = reasons
    df["Voc_Band5Candidate"] = df["Voc_Band"] == 4
    return df
