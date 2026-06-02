"""
Phase SS -- Sentence Structure counts.

Adds per-script Sentence Structure count columns to the texts dataframe,
mirroring spelling_final.add_spelling_counts and
break_classifications.add_break_counts. Call it once, after the sentence
layer (and the engine sentence-typing stage) have run:

    import ss_counts as ssc
    df = ssc.add_ss_counts(df, sent, id_col=cfg["id_col"])

Columns produced
----------------
Totals (per script):
    SS_AssessableCount   sentences in the assessable set
    SS_ExcludedCount     sentences lifted out (see "Assessable set" below)
    SS_CorrectCount      assessable sentences with SS internal == Correct
    SS_IncorrectCount    assessable sentences with SS internal == Incorrect
    SS_CorrectPct        100 * Correct / (Correct + Incorrect), 1 dp.
                         Computed over JUDGED sentences only, so untyped /
                         un-judged sentences do not drag the rate down.
                         0.0 when no judged sentences.

By type (assessable only), correct / incorrect split for the climb-to-fall
ladder, one pair per type. Complex - Projected sentences count under Complex;
there is no separate Projected pair, since projection is qualitative:
    SS_Fragment_Correct          SS_Fragment_Incorrect
    SS_Interjection_Correct      SS_Interjection_Incorrect
    SS_Simple_Correct            SS_Simple_Incorrect
    SS_Compound_Correct          SS_Compound_Incorrect
    SS_Complex_Correct           SS_Complex_Incorrect

Qualitative informer (NOT part of any structural tally or the variety index):
    SS_ProjectedCount      assessable sentences using projection (reported or
                           direct speech / thought). A signal about the kind
                           of writing move, not a structural-variety count.

Variety:
    SS_DistinctTypesUsed   how many of the three STRUCTURAL types
                           (Simple, Compound, Complex) appear at least once.
                           0..3. Fragment, Interjection and Projected are
                           deliberately NOT counted as variety: the first two
                           are not positive structural range, and projection
                           is qualitative and already folded into Complex.

Note: until the engine sentence-typing stage runs, "Sentence type" is the
TBD(engine) placeholder, so assessable sentences are counted in
SS_AssessableCount but land in no per-type column. The per-type pairs will
therefore read mostly zero on a deterministic-only run; that is expected,
not a bug. Inline time-slips are the exception -- they are forced to
Fragment here regardless of the placeholder (see invariant below).

Assessable set
--------------
A sentence is EXCLUDED (not typed, not scored) when it is:
    - a sound effect          (Sentence type == "SoundEffect")
    - a cutoff                (TextualArtifact == "CUTOFF")
    - a title                 (TextualArtifact == "TITLE")
    - an ending               (TextualArtifact == "ENDING")
    - an isolated time-slip   (TextualArtifact in {"TIMESKIP", "TIMESLIP"})
Everything else, including OPENING, is assessable.

KEY INVARIANT (locked): an inline time-slip
(TimeSlipStatus == "inline_missing_break") is ALWAYS assessable and is
forced to type Fragment, regardless of its Sentence type or any artifact
tag. In the current data every TIMESKIP-tagged row is in fact an inline
time-slip, so this override is load-bearing, not theoretical.
"""
from __future__ import annotations

import pandas as pd

# Type labels -- MUST match engine_prototype._SENT_TYPES exactly.
# "Complex - Projected" is a QUALITATIVE informer, not a structural variety.
# For every quantitative count it folds into Complex; it is reported only as
# the separate SS_ProjectedCount, which is NOT part of the variety tally.
STRUCTURAL_TYPES = ["Simple", "Compound", "Complex"]
ALL_TYPES = ["Fragment", "Interjection"] + STRUCTURAL_TYPES
PROJECTED = "Complex - Projected"

EXCLUDE_ARTIFACTS = {"TITLE", "ENDING", "CUTOFF", "TIMESKIP", "TIMESLIP"}
INLINE_TIMESLIP = "inline_missing_break"


def _safe(label: str) -> str:
    """Column-safe form of a type label: 'Complex - Projected' -> 'ComplexProjected'."""
    return label.replace(" - ", "").replace(" ", "")


def add_ss_counts(texts_df: pd.DataFrame, sent_df: pd.DataFrame,
                  id_col: str = "Research ID") -> pd.DataFrame:
    df = texts_df.copy()
    # Drop any SS_ count columns from a previous run so re-running is clean
    # and stale columns (e.g. an old SS_ComplexProjected pair) cannot linger.
    df = df.drop(columns=[c for c in df.columns if c.startswith("SS_")],
                 errors="ignore")
    ids = df[id_col].astype(str)
    s = sent_df

    def col(name):
        if name in s.columns:
            return s[name].astype(str)
        return pd.Series([""] * len(s), index=s.index)

    stype = col("Sentence type")
    ta = col("TextualArtifact")
    tss = col("TimeSlipStatus")
    ss = col("SS internal")
    gid = col("Identifier")

    inline = tss == INLINE_TIMESLIP
    excluded = (~inline) & ((stype == "SoundEffect") | ta.isin(EXCLUDE_ARTIFACTS))
    assessable = ~excluded

    # Effective type: inline time-slips are forced to Fragment (the invariant).
    eff = stype.where(~inline, "Fragment")
    # Quantitative type: Complex - Projected folds into Complex. The projected
    # signal is kept separately below as a qualitative informer only.
    eff_q = eff.where(eff != PROJECTED, "Complex")

    correct = ss == "Correct"
    incorrect = ss == "Incorrect"

    def per_script(mask: pd.Series) -> pd.Series:
        # sum of booleans grouped by script id -> per-script count
        return mask.astype(int).groupby(gid).sum()

    df["SS_AssessableCount"] = ids.map(per_script(assessable)).fillna(0).astype(int)
    df["SS_ExcludedCount"]   = ids.map(per_script(excluded)).fillna(0).astype(int)
    df["SS_CorrectCount"]    = ids.map(per_script(assessable & correct)).fillna(0).astype(int)
    df["SS_IncorrectCount"]  = ids.map(per_script(assessable & incorrect)).fillna(0).astype(int)

    den = df["SS_CorrectCount"] + df["SS_IncorrectCount"]
    df["SS_CorrectPct"] = (100.0 * df["SS_CorrectCount"]
                           / den.where(den > 0)).round(1).fillna(0.0)

    for t in ALL_TYPES:
        name = _safe(t)
        is_t = eff_q == t
        df[f"SS_{name}_Correct"]   = ids.map(per_script(assessable & is_t & correct)).fillna(0).astype(int)
        df[f"SS_{name}_Incorrect"] = ids.map(per_script(assessable & is_t & incorrect)).fillna(0).astype(int)

    # Qualitative informer ONLY -- NOT part of the structural variety tally.
    # Counts assessable sentences the type pass flagged as projection.
    df["SS_ProjectedCount"] = ids.map(
        per_script(assessable & (eff == PROJECTED))).fillna(0).astype(int)

    # Distinct structural types present per script (variety index, 0..3).
    struct_mask = assessable & eff_q.isin(STRUCTURAL_TYPES)
    distinct = eff_q[struct_mask].groupby(gid[struct_mask]).nunique()
    df["SS_DistinctTypesUsed"] = ids.map(distinct).fillna(0).astype(int)

    return df
