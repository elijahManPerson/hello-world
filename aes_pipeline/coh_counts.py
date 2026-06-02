"""
Phase CC -- Cohesion counts and band (two-tier error model).

Adds per-script Cohesion columns to the texts dataframe, mirroring
vocab_counts.add_vocab_columns. Call once, after the sentence and word layers:

    import coh_counts as cc
    df = cc.add_coh_columns(df, sent, wm, id_col=cfg["id_col"])

Construct-validity design (the two error tiers):
    A cohesion mark conflates two questions. "Does the text hold together at
    all?" is broken by DEFINITIVE errors -- a reference with no antecedent, a
    wrong referent, a tense that derails the reader, an article/determiner
    slip. "Does it hold together flawlessly?" is disturbed even by a momentary
    AMBIGUITY (a 'light' error). So the two tiers gate different jumps:
      - DEFINITIVE errors (with connective variety) gate the 2 -> 3 jump, at a
        low rate rather than zero, since experts tolerate the occasional slip
        in an otherwise cohesive text.
      - LIGHT errors (ambiguity) are held back to gate the 3 -> 4 jump: a band-4
        candidate must additionally be free of ambiguity (and sustained), before
        the 'flows well' pass confirms the top band.
    Strictness therefore rises up the ladder.

    Validated against expert marks (n=69): the band-3 gate below reaches ~0.72
    Spearman; zero-tolerance on definitive errors falls to ~0.66, confirming the
    low-rate (not zero) choice at band 3.

Columns added:
    Coh_Sentences            sentence count (all sentences count)
    Coh_CohesiveWords        references + conjunctions
    Coh_References           CohesionRole == 'reference'
    Coh_Connectives          CohesionRole startswith 'conj_'
    Coh_ConnectiveVariety    distinct conjunction types used (0-5)
    Coh_TenseErrors          sentences with Tense drift == Incorrect
    Coh_PronounErrors        sentences with Pronoun ref == Incorrect (all)
    Coh_HardPronounErrors    pronoun breaks: No antecedent / Wrong referent
    Coh_LightErrors          ambiguity: Ambiguous referent (the light tier)
    Coh_ArticleDetErrors     determiner grammar errors (thin signal)
    Coh_DefinitiveErrors     tense + hard pronoun + article/determiner
    Coh_DefinitiveRate       definitive errors per sentence (the band-3 measure)
    Coh_FlowsWell            STUB. Filled by the band-4 'flows well' LLM pass.
    Coh_Band                 0-3 from the ladder below.
    Coh_Band4Candidate       band 3, free of light errors, and FLOW_WORDS+ words.
    Coh_BandReason           which rung produced the band.

The ladder (rubric, 0-4 scale):
    0  title only
    1  short: < MIN_SENTENCES sentences, or < MIN_TIES cohesive words
    2  MIN_SENTENCES+ sentences and MIN_TIES+ ties, but not yet band 3
    3  sustained (SUSTAINED_WORDS+) AND connective variety >= BAND3_MIN_VARIETY
       AND definitive-error rate < BAND3_MAX_DEFINITIVE_RATE per sentence
    4  (LLM gate) FLOW_WORDS+ words, free of light errors, and reads as flowing
       well. Reserved: the deterministic ladder tops at 3 and flags qualifying
       scripts as Coh_Band4Candidate.

To revert to a single-tier rule, fold Coh_LightErrors back into the band-3 rate
and drop the light-error condition from Coh_Band4Candidate. All thresholds are
provisional on 69 scripts; re-confirm on the full 364-script expert run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SENTENCES = 3
MIN_TIES = 2
SUSTAINED_WORDS = 200
FLOW_WORDS = 300
BAND3_MIN_VARIETY = 3            # distinct connective types required for band 3
BAND3_MAX_DEFINITIVE_RATE = 0.15  # definitive errors per sentence allowed at band 3
BAND4_MAX_LIGHT_ERRORS = 0      # light (ambiguity) errors allowed for a band-4 candidate

_SENT_ID = "Identifier"
_HARD_PRONOUN = ("No antecedent", "Wrong referent")
_LIGHT_PRONOUN = ("Ambiguous referent",)
_CONJ_TYPES = ("conj_additive", "conj_temporal", "conj_adversative",
               "conj_causal", "conj_other")


def add_coh_columns(texts_df: pd.DataFrame, sent_df: pd.DataFrame,
                    word_map_df: pd.DataFrame,
                    id_col: str = "Research ID") -> pd.DataFrame:
    df = texts_df.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Coh_")],
                 errors="ignore")
    ids = df[id_col].astype(str)

    sid = sent_df[_SENT_ID].astype(str) if _SENT_ID in sent_df.columns else pd.Series([""] * len(sent_df))

    def sent_agg(mask):
        return mask.astype(int).groupby(sid).sum()

    n_sent = ids.map(sid.groupby(sid).size()).fillna(0).astype(int)

    def col_eq(col, val):
        s = sent_df[col].astype(str) if col in sent_df.columns else pd.Series([""] * len(sent_df))
        return ids.map(sent_agg(s == val)).fillna(0).astype(int)

    def col_isin(col, vals):
        s = sent_df[col].astype(str) if col in sent_df.columns else pd.Series([""] * len(sent_df))
        return ids.map(sent_agg(s.isin(vals))).fillna(0).astype(int)

    tense_err = col_eq("Tense drift", "Incorrect")
    pron_err = col_eq("Pronoun ref", "Incorrect")
    hard_pron = col_isin("Pronoun ref subtype", _HARD_PRONOUN)
    light_pron = col_isin("Pronoun ref subtype", _LIGHT_PRONOUN)
    artdet_err = col_eq("Grammar subtype (most serious)", "determiner")

    def wcol(name):
        if name in word_map_df.columns:
            return word_map_df[name].astype(str)
        return pd.Series([""] * len(word_map_df), index=word_map_df.index)

    cr = wcol("CohesionRole")
    wid = wcol("Identifier")

    def w_count(mask):
        return mask.astype(int).groupby(wid).sum()

    n_ref = ids.map(w_count(cr == "reference")).fillna(0).astype(int)
    n_conj = ids.map(w_count(cr.str.startswith("conj_"))).fillna(0).astype(int)
    n_ties = n_ref + n_conj
    variety = ids.map(
        cr.where(cr.isin(_CONJ_TYPES)).groupby(wid).nunique()
    ).fillna(0).astype(int)

    definitive = tense_err + hard_pron + artdet_err
    def_rate = (definitive / n_sent.where(n_sent > 0)).fillna(0.0)

    df["Coh_Sentences"] = n_sent
    df["Coh_CohesiveWords"] = n_ties
    df["Coh_References"] = n_ref
    df["Coh_Connectives"] = n_conj
    df["Coh_ConnectiveVariety"] = variety
    df["Coh_TenseErrors"] = tense_err
    df["Coh_PronounErrors"] = pron_err
    df["Coh_HardPronounErrors"] = hard_pron
    df["Coh_LightErrors"] = light_pron
    df["Coh_ArticleDetErrors"] = artdet_err
    df["Coh_DefinitiveErrors"] = definitive
    df["Coh_DefinitiveRate"] = def_rate.round(3)
    df["Coh_FlowsWell"] = np.nan  # filled by the band-4 'flows well' pass

    wc = (pd.to_numeric(df["WordCount"], errors="coerce").fillna(0)
          if "WordCount" in df.columns else pd.Series(0, index=df.index))
    if "TitleOnly" in df.columns:
        title_only = df["TitleOnly"].astype(str).str.lower().isin(["true", "1"])
    else:
        title_only = pd.Series(False, index=df.index)

    ns = n_sent.values
    nt = n_ties.values
    w = wc.values
    var = variety.values
    drate = def_rate.values
    light = light_pron.values
    to = title_only.values

    bands, reasons = [], []
    for i in range(len(df)):
        if to[i] or ns[i] == 0:
            bands.append(0); reasons.append("title only")
        elif ns[i] < MIN_SENTENCES or nt[i] < MIN_TIES:
            bands.append(1); reasons.append(f"short: <{MIN_SENTENCES} sentences or <{MIN_TIES} cohesive words")
        elif w[i] < SUSTAINED_WORDS:
            bands.append(2); reasons.append(f"{MIN_SENTENCES}+ sentences and {MIN_TIES}+ cohesive words, not yet sustained (<{SUSTAINED_WORDS} words)")
        elif var[i] >= BAND3_MIN_VARIETY and drate[i] < BAND3_MAX_DEFINITIVE_RATE:
            bands.append(3); reasons.append(f"sustained, range of cohesive devices (variety {int(var[i])}), definitive-error rate {drate[i]:.2f} ok")
        elif var[i] < BAND3_MIN_VARIETY:
            bands.append(2); reasons.append(f"sustained but limited range of cohesive devices (variety {int(var[i])} < {BAND3_MIN_VARIETY}); held at band 2")
        else:
            bands.append(2); reasons.append(f"sustained and varied but definitive-error rate {drate[i]:.2f} >= {BAND3_MAX_DEFINITIVE_RATE}; held at band 2")

    df["Coh_Band"] = bands
    df["Coh_BandReason"] = reasons
    df["Coh_Band4Candidate"] = (
        (df["Coh_Band"] == 3)
        & (light_pron <= BAND4_MAX_LIGHT_ERRORS)
        & (wc >= FLOW_WORDS)
    )
    return df
