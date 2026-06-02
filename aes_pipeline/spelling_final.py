"""
spelling_final.py
=================
Phase SF -- consolidated SpellingFinal column on word_map and 11 per-script
count columns on texts.

SpellingFinal has exactly one of 11 values per word token. The eight
tier-correctness combinations (Simple/Common/Difficult/Challenging x
Correct/Incorrect) plus NA for non-assessable tokens, plus two mitigation
categories that hold obvious slips and non-spelling substitutions aside
from the catch-to-fall:

  - Typo:        Simple/Common-Incorrect token that is unmistakably a keying
                 slip (the writer plainly knows the word)
  - NotSpelling: a correctly-spelled real word that is not an attempt at the
                 intended word (a different word, or a proper noun)

Two gates produce the mitigation categories on top of the base label:

  Gate 1 (deterministic): a Simple/Common-Incorrect token whose corrected
    form appears spelled correctly somewhere else in the SAME script is
    unquestionably a slip -> Typo.

  Gate 2 (light LLM):     remaining Simple/Common-Incorrect tokens get a
    constrained three-way classification (Typo / NotSpelling / Error). The
    default on any uncertainty -- or any missing LLM response -- is Error
    (i.e. leave the token as a genuine Incorrect). Gate 2 is gated on an
    api_key; with no key it simply does not run, which keeps local-only
    runs deterministic and free.

Difficult-Incorrect and Challenging-Incorrect are NEVER reclassified -- the
rubric is meant to credit / debit hard-word attempts.

An optional audit column `SpellingFinalGate` records which gate (if any)
reclassified each token: empty / gate1_deterministic / gate2_llm.
"""
from __future__ import annotations

import json
import re

import pandas as pd

# Tier source: WordKind (after SP1 reconciliation with SpellDifficulty).
_TIER_MAP = {
    "simple":      "Simple",
    "common":      "Common",
    "difficult":   "Difficult",
    "challenging": "Challenging",
}

SPELLING_FINAL_VALUES = [
    "NA",
    "Simple-Correct", "Simple-Incorrect",
    "Common-Correct", "Common-Incorrect",
    "Difficult-Correct", "Difficult-Incorrect",
    "Challenging-Correct", "Challenging-Incorrect",
    "Typo", "NotSpelling",
]

_COUNT_COL_NAMES = {
    "NA":                    "Spell_NA",
    "Simple-Correct":        "Spell_Simple_Correct",
    "Simple-Incorrect":      "Spell_Simple_Incorrect",
    "Common-Correct":        "Spell_Common_Correct",
    "Common-Incorrect":      "Spell_Common_Incorrect",
    "Difficult-Correct":     "Spell_Difficult_Correct",
    "Difficult-Incorrect":   "Spell_Difficult_Incorrect",
    "Challenging-Correct":   "Spell_Challenging_Correct",
    "Challenging-Incorrect": "Spell_Challenging_Incorrect",
    "Typo":                  "Spell_Typo",
    "NotSpelling":           "Spell_NotSpelling",
}


def _is_spelling_assessable_row(row) -> bool:
    """SP3 mirror -- excludes proper nouns, junk, insert/delete-ops, non-word
    tokens and tokens with no tier. Also excludes Spell class == "NA" (the
    spelling engine signal for "not assessed") and corrector insertions
    (op == "insert"; raw_token is empty). Kept local to avoid an import
    cycle with run_all.py."""
    if str(row.get("TokenCategory", "")) != "word":
        return False
    if str(row.get("Spell class", "")) in ("ProperNoun-ignored", "NA"):
        return False
    if str(row.get("WordKind", "")) in (
            "proper_noun_abbreviation", "junk", "name_foreign", "NA"):
        return False
    if str(row.get("op", "")) in ("delete", "insert"):
        return False
    return True


def _base_label(row) -> str:
    """SF1 Step 1 -- combine WordKind tier with Spell class correctness.

    Returns "" for non-word tokens (punctuation etc.) so SpellingFinal is
    only populated on word tokens. Returns "NA" for word tokens that are
    not spelling-assessable (proper nouns, junk, numerals, delete-ops)."""
    if str(row.get("TokenCategory", "")) != "word":
        return ""
    if not _is_spelling_assessable_row(row):
        return "NA"
    tier = _TIER_MAP.get(str(row.get("WordKind", "")).lower())
    if tier is None:
        return "NA"
    correctness = ("Correct"
                   if str(row.get("Spell class", "")) == "Correct"
                   else "Incorrect")
    return f"{tier}-{correctness}"


def apply_base_and_gate1(wm: pd.DataFrame) -> pd.DataFrame:
    """SF1 Steps 1 + 2 -- add SpellingFinal + SpellingFinalGate columns.

    Step 1: base label from WordKind x Spell class.
    Step 2 (Gate 1, deterministic): a Simple/Common-Incorrect whose corrected
    form appears correctly spelled elsewhere in the same script is relabelled
    Typo.
    """
    df = wm.copy()
    # Step 1 -- base label, vectorised via apply
    df["SpellingFinal"] = df.apply(_base_label, axis=1)
    df["SpellingFinalGate"] = ""

    # Step 2 -- Gate 1
    idcol = "Identifier" if "Identifier" in df.columns else "ID"
    candidate_mask = df["SpellingFinal"].isin(
        ["Simple-Incorrect", "Common-Incorrect"])
    if not candidate_mask.any():
        return df

    n_gate1 = 0
    for ident, g in df.groupby(idcol, sort=False):
        # Correctly-spelled corrected tokens in this script (case-insensitive)
        correct_forms = set(
            g.loc[g["Spell class"] == "Correct", "corr_token"]
            .astype(str).str.lower().str.strip()
        )
        if not correct_forms:
            continue
        # Gate 1 only acts on Spell class == "Incorrect" (genuine unknown).
        # Homophone-flagged tokens are kept as Incorrect so Gate 2 can
        # correctly classify them as Error per the spec's Gate-2 guidance.
        cands = g[
            g["SpellingFinal"].isin(["Simple-Incorrect", "Common-Incorrect"])
            & (g["Spell class"] == "Incorrect")
        ]
        for idx, row in cands.iterrows():
            corrected = str(row.get("corr_token", "")).lower().strip()
            if corrected and corrected in correct_forms:
                df.at[idx, "SpellingFinal"] = "Typo"
                df.at[idx, "SpellingFinalGate"] = "gate1_deterministic"
                n_gate1 += 1

    df.attrs["sf_gate1_count"] = n_gate1
    return df


# ---------------------------------------------------------------------------
# SF1 Step 3 -- Gate 2 (light LLM classification)
# ---------------------------------------------------------------------------

_GATE2_SYSTEM = """You classify flagged spelling tokens for a Year-5 writing\
 marking pipeline.

For each flagged token return exactly one of: Typo, NotSpelling, Error.

- Typo: the misspelling is UNQUESTIONABLY a keying slip. The writer plainly\
 knows the word (e.g. "th" for "the"). Choose this ONLY when there is no\
 plausible way a student could genuinely not know the spelling.

- NotSpelling: the student wrote a real, correctly-spelled word that is NOT\
 an attempt at the intended word. This means a proper noun/name (e.g. "Daisy"\
 the character), OR a different word the student chose or omitted (e.g. wrote\
 "to" where "into" was intended). The student's own word is correctly spelled\
 and is simply a different word.

- Error: a genuine spelling error. This INCLUDES:
  - homophone confusions (where/were, their/there, to/too) -- these are\
 spelling errors even though the result is a real word, because the student\
 was reaching for the intended word and produced the wrong form
  - phonetic spellings (bak/back, agen/again, sed/said)
  - any case where the student was trying to produce the intended word but\
 got its form wrong

CRITICAL: When in doubt between any two labels, choose Error. NotSpelling and\
 Typo are narrow exceptions -- only use them when the case is unmistakable.\
 If there is ANY chance the student was reaching for the intended word and\
 fumbled it, the answer is Error.

Return ONLY a JSON object mapping each student token to its label. No prose."""


def _gate2_user_message(token_pairs) -> str:
    lines = [f"- {raw} -> {corr}" for raw, corr in token_pairs]
    return ("Flagged tokens (student wrote -> intended):\n"
            + "\n".join(lines)
            + "\n\nReturn JSON: {\"<student_token>\": "
              "\"Typo|NotSpelling|Error\", ...}")


def _gate2_classify_script(token_pairs, model, api_key) -> dict:
    """One Gate 2 call for a single script. Returns a dict mapping the
    student (raw) token to one of Typo / NotSpelling / Error. On any failure
    returns {} -- callers default missing entries to Error (conservative)."""
    if not token_pairs:
        return {}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            system=[{"type": "text", "text": _GATE2_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user",
                       "content": _gate2_user_message(token_pairs)}],
        )
        try:
            from cost_tracker import tracker
            tracker.record("spelling_llm", msg.usage, model=model)
        except Exception:
            pass
        out = next((b.text for b in msg.content if b.type == "text"), "{}")
        out = re.sub(r"```json|```", "", out).strip()
        parsed = json.loads(out)
        if not isinstance(parsed, dict):
            return {}
        # Keep only valid labels
        return {str(k): v for k, v in parsed.items()
                if v in ("Typo", "NotSpelling", "Error")}
    except Exception:
        return {}   # conservative: caller defaults missing entries to Error


def apply_gate2(wm: pd.DataFrame, model: str, api_key: str,
                mock: bool = False) -> pd.DataFrame:
    """SF1 Step 3 -- Gate 2 LLM classification for the remaining
    Simple/Common-Incorrect tokens.

    Tokens not present in (or returned as Error by) the response stay as
    their base Incorrect label. Difficult/Challenging tokens are NEVER
    touched. When mock=True or api_key is empty, this is a no-op and every
    candidate remains its base Incorrect label (conservative default)."""
    df = wm.copy()
    if "SpellingFinal" not in df.columns:
        return df
    if mock or not api_key:
        df.attrs["sf_gate2_count"] = 0
        return df

    idcol = "Identifier" if "Identifier" in df.columns else "ID"
    n_typo = n_notspelling = 0
    for ident, g in df.groupby(idcol, sort=False):
        cands = g[g["SpellingFinal"].isin(
            ["Simple-Incorrect", "Common-Incorrect"])]
        if cands.empty:
            continue
        token_pairs = [
            (str(r.get("raw_token", "")).strip(),
             str(r.get("corr_token", "")).strip())
            for _, r in cands.iterrows()
        ]
        # Drop empty pairs
        token_pairs = [(r, c) for r, c in token_pairs if r and c]
        if not token_pairs:
            continue
        labels = _gate2_classify_script(token_pairs, model=model,
                                        api_key=api_key)
        for idx, row in cands.iterrows():
            raw = str(row.get("raw_token", "")).strip()
            lab = labels.get(raw, "Error")
            if lab == "Typo":
                df.at[idx, "SpellingFinal"] = "Typo"
                df.at[idx, "SpellingFinalGate"] = "gate2_llm"
                n_typo += 1
            elif lab == "NotSpelling":
                df.at[idx, "SpellingFinal"] = "NotSpelling"
                df.at[idx, "SpellingFinalGate"] = "gate2_llm"
                n_notspelling += 1
            # Error -> leave base Incorrect label in place

    df.attrs["sf_gate2_typo"] = n_typo
    df.attrs["sf_gate2_notspelling"] = n_notspelling
    return df


# ---------------------------------------------------------------------------
# SF2 -- per-script counts on the texts table
# ---------------------------------------------------------------------------

def add_spelling_counts(texts_df: pd.DataFrame, wm: pd.DataFrame,
                        id_col: str = "Research ID") -> pd.DataFrame:
    """SF2 -- add 11 count columns (one per SpellingFinal value) to texts.
    Counts are integer per-script frequencies; missing scripts get 0."""
    df = texts_df.copy()
    if "SpellingFinal" not in wm.columns:
        for value in SPELLING_FINAL_VALUES:
            df[_COUNT_COL_NAMES[value]] = 0
        return df

    wm_idcol = "Identifier" if "Identifier" in wm.columns else "ID"
    ids = df[id_col].astype(str)
    for value in SPELLING_FINAL_VALUES:
        col = _COUNT_COL_NAMES[value]
        counts = (wm[wm["SpellingFinal"] == value]
                  .groupby(wm_idcol).size())
        df[col] = ids.map(counts).fillna(0).astype(int)
    return df
