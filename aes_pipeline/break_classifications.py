"""
break_classifications.py
========================
Phase BK2 -- break-correctness classifications layered on the BK1 foundation.

Four patches:
  BK2-1  ParagraphBreakClass  (true / missing / superfluous / na)
  BK2-2  SpeechBreakClass     (true / missing / na)
  BK2-3  StructuralBreak      (after_title / before_ending / none)
  BK2-4  Time-slip tagging    (TIMESLIP TextualArtifact + TimeSlipStatus)

Plus per-script break counts on the texts table.

Design decisions reflected here
-------------------------------
- True / Superfluous paragraph breaks are deterministic (this module).
  Missing paragraph breaks are produced by:
    (a) inline temporal fragments (option ii) -- always, no LLM,
    (b) the corrector pipeline -- optional, gated on api_key.
- Speech breaks are two-state (true / missing). Extra breaks inside
  continuous speech are NOT flagged superfluous.
- TIMESLIP is a NEW TextualArtifact value distinct from the existing
  TIMESKIP. A sentence is upgraded to TIMESLIP only when it is BOTH a
  temporal fragment AND structurally isolated (paragraph/line break, or
  colon/dash separator). Inline temporal fragments do NOT get TIMESLIP;
  they remain Fragments for SS/Punctuation and record a missing paragraph
  break (option ii).
- Sentence type may be unset in local-only runs (the C4 LLM stage hasn't
  written 'Fragment'). When that happens we fall back to a deterministic
  fragment heuristic: <=4 content words AND matches a temporal phrase /
  connective.
"""
from __future__ import annotations

import json
import re

import pandas as pd


# ---------------------------------------------------------------------------
# Temporal lexicon (shared by BK2-4)
# ---------------------------------------------------------------------------

_TEMPORAL_CONNECTIVES = {
    "later", "meanwhile", "afterwards", "eventually", "soon",
    "then", "next", "finally", "suddenly", "now",
    "today", "tomorrow", "yesterday",
}

_TEMPORAL_PHRASE_RES = [
    re.compile(r'\b\d+\s+(second|minute|hour|day|week|month|year)s?\s+later\b',
               re.IGNORECASE),
    re.compile(r'\blater\s+that\s+(day|night|morning|evening|afternoon|week)\b',
               re.IGNORECASE),
    re.compile(r'\bthe\s+next\s+(day|morning|week|year|night)\b',
               re.IGNORECASE),
    re.compile(r'\ba?\s*(while|moment|few\s+\w+)\s+later\b', re.IGNORECASE),
    re.compile(r'\b(soon|not\s+long)\s+after\b', re.IGNORECASE),
    re.compile(r'\bthat\s+(night|morning|evening|day)\b', re.IGNORECASE),
    re.compile(r'\bthe\s+following\s+(day|morning|week)\b', re.IGNORECASE),
    re.compile(r'\bmeanwhile\b', re.IGNORECASE),
    re.compile(r'\beventually\b', re.IGNORECASE),
]


def _matches_temporal_phrase(text: str) -> bool:
    return any(p.search(text) for p in _TEMPORAL_PHRASE_RES)


def _content_words(text: str) -> list:
    """Words minus terminal punctuation."""
    t = re.sub(r'[.!?,;:"\']+',' ', str(text or '')).split()
    return [w for w in t if re.search(r'[A-Za-z0-9]', w)]


# ---------------------------------------------------------------------------
# BK2-1  Paragraph break classification (deterministic part)
# ---------------------------------------------------------------------------

def classify_paragraph_breaks(sent_df: pd.DataFrame) -> pd.DataFrame:
    """True / Superfluous deterministic from PrecededByBreak.

    Superfluous-script heuristic: on a multi-sentence script, if more than
    70% of sentences are paragraph-firsts, the writer is breaking randomly
    and every break in that script is labelled superfluous instead of true.

    'na' is the default for any sentence with no preceding break -- the
    BK2-4 inline time-slip rule may later override these to 'missing', and
    the corrector pipeline (optional) may flag additional missing breaks."""
    df = sent_df.copy()
    df["ParagraphBreakClass"] = "na"

    for ident, g in df.groupby("Identifier", sort=False):
        g_sorted = g.sort_values("SentenceRef")
        n_sent = len(g_sorted)
        # +1 because the first paragraph isn't preceded by a break but it
        # still represents an "implicit" paragraph in the count.
        n_breaks = int((g_sorted["PrecededByBreak"] == "paragraph").sum()) + 1
        superfluous = (n_sent >= 4) and (n_breaks / n_sent > 0.7)

        for idx in g_sorted.index:
            if df.at[idx, "PrecededByBreak"] == "paragraph":
                df.at[idx, "ParagraphBreakClass"] = (
                    "superfluous" if superfluous else "true")
    return df


# ---------------------------------------------------------------------------
# BK2-2  Speech break classification
# ---------------------------------------------------------------------------

def _sentence_has_dialogue(sentence_ref: str, dialogue_map: dict) -> bool:
    """True if the sentence has any DIALOGUE=True token in the word map."""
    return bool(dialogue_map.get(sentence_ref, False))


def _starts_new_speaker_turn(text: str, has_dialogue: bool) -> bool:
    """Heuristic for the START of a new speaker turn: the sentence carries
    DIALOGUE tokens AND its leading non-whitespace character is a quote
    mark (\", curly variants, or single quote). Conservative -- doesn't
    catch attribution-led turns (\"\"X,\" said Y. Then Y said \"Z\"\")"""
    if not has_dialogue:
        return False
    t = str(text or "").lstrip()
    if not t:
        return False
    return t[0] in {'"', '“', '”', "'", '‘', '’'}


def classify_speech_breaks(sent_df: pd.DataFrame,
                           wm: pd.DataFrame) -> pd.DataFrame:
    """For each sentence that starts a new speaker turn AND follows another
    speech sentence (i.e. there was a previous turn to break from), check
    whether a break precedes it.
      true:    new speaker AND prev was speech AND PrecededByBreak in
               {paragraph, line_break}
      missing: new speaker AND prev was speech AND no such break
      na:      anything else (no speaker change detected)
    """
    df = sent_df.copy()
    df["SpeechBreakClass"] = "na"

    # Per-sentence dialogue flag built from word_map's DIALOGUE column
    dialogue_map: dict = {}
    if "DIALOGUE" in wm.columns and "SentenceRef" in wm.columns:
        # Treat anything truthy (True / "TRUE" / "True" / 1) as dialogue
        flag = wm["DIALOGUE"].astype(str).str.lower().isin(
            ["true", "1", "yes"])
        if flag.any():
            present = wm.loc[flag, "SentenceRef"].dropna().unique()
            dialogue_map = {str(r): True for r in present}

    for ident, g in df.groupby("Identifier", sort=False):
        g_sorted = g.sort_values("SentenceRef")
        prev_was_speech = False
        for idx in g_sorted.index:
            ref = str(df.at[idx, "SentenceRef"])
            text = str(df.at[idx, "CorrectedSentence"] or "")
            has_dlg = _sentence_has_dialogue(ref, dialogue_map)
            is_new_turn = _starts_new_speaker_turn(text, has_dlg)
            if is_new_turn and prev_was_speech:
                if df.at[idx, "PrecededByBreak"] in ("paragraph", "line_break"):
                    df.at[idx, "SpeechBreakClass"] = "true"
                else:
                    df.at[idx, "SpeechBreakClass"] = "missing"
            prev_was_speech = has_dlg
    return df


# ---------------------------------------------------------------------------
# BK2-3  Structural breaks
# ---------------------------------------------------------------------------

def classify_structural_breaks(sent_df: pd.DataFrame) -> pd.DataFrame:
    """after_title: the first non-title sentence in a script that has a
                    title (title_styled block precedes it).
    before_ending: the sentence tagged ENDING (the break before it is
                    structural by definition).
    none:           everything else."""
    df = sent_df.copy()
    df["StructuralBreak"] = "none"

    for ident, g in df.groupby("Identifier", sort=False):
        g_sorted = g.sort_values("SentenceRef")
        has_title = (g_sorted["BlockStyle"] == "title_styled").any()
        # First sentence whose BlockStyle is not title_styled
        if has_title:
            non_title = g_sorted[g_sorted["BlockStyle"] != "title_styled"]
            if not non_title.empty:
                first_body_idx = non_title.index[0]
                df.at[first_body_idx, "StructuralBreak"] = "after_title"
        for idx in g_sorted.index:
            if str(df.at[idx, "TextualArtifact"] or "") == "ENDING":
                df.at[idx, "StructuralBreak"] = "before_ending"
    return df


# ---------------------------------------------------------------------------
# BK2-4  Time-slip tagging
# ---------------------------------------------------------------------------

def _is_temporal_fragment(row) -> bool:
    """Fragment-like AND essentially a temporal connective/phrase.

    Primary signal: Sentence type == 'Fragment' AND the text matches a
    temporal phrase or ends in a temporal connective.

    Fallback (when Sentence type isn't filled by the C4 LLM stage, e.g.
    local-only runs): use the existing CB5 TextualArtifact == 'TIMESKIP'
    tag. _is_bare_timeskip in mech_v2.py already restricts that tag to
    bare-pattern time skips ("2 weeks later.", "Later that night.") --
    a much tighter filter than a generic short-sentence heuristic, which
    over-fires on complete sentences containing "now"/"then"/"finally".
    """
    text = str(row.get("CorrectedSentence", "") or "").strip()
    if not text:
        return False

    stype = str(row.get("Sentence type", "") or "")
    if stype == "Fragment":
        words = _content_words(text)
        matches_phrase = _matches_temporal_phrase(text)
        has_connective = any(w.lower() in _TEMPORAL_CONNECTIVES for w in words)
        return matches_phrase or has_connective

    # Fallback: rely on the existing TIMESKIP tag (set by mech_v2's
    # _is_bare_timeskip, which catches "N units later." style fragments).
    return str(row.get("TextualArtifact", "") or "") == "TIMESKIP"


def _is_isolated(row) -> bool:
    """Structurally set off: a paragraph/line break precedes, OR the raw
    sentence starts with a colon/dash separator (separator-style time
    slip)."""
    if row.get("PrecededByBreak") in ("paragraph", "line_break"):
        return True
    raw = str(row.get("RawSentence", "") or "").lstrip()
    return raw.startswith((":", "-", "—"))


def tag_time_slips(sent_df: pd.DataFrame) -> pd.DataFrame:
    """BK2-4. Adds TimeSlipStatus to every sentence and, for confirmed
    (isolated) temporal fragments, sets TextualArtifact = TIMESLIP. Inline
    temporal fragments stay Fragments and pick up
    ParagraphBreakClass = 'missing' per option (ii)."""
    df = sent_df.copy()
    if "TimeSlipStatus" not in df.columns:
        df["TimeSlipStatus"] = "none"
    if "ParagraphBreakClass" not in df.columns:
        df["ParagraphBreakClass"] = "na"
    if "TextualArtifact" not in df.columns:
        df["TextualArtifact"] = ""

    n_confirmed = n_inline = 0
    for idx, row in df.iterrows():
        if not _is_temporal_fragment(row):
            continue
        if _is_isolated(row):
            # Don't overwrite a higher-priority artifact (TITLE/CUTOFF win)
            ta = str(df.at[idx, "TextualArtifact"] or "")
            if ta not in ("TITLE", "CUTOFF"):
                df.at[idx, "TextualArtifact"] = "TIMESLIP"
            df.at[idx, "TimeSlipStatus"] = "confirmed"
            n_confirmed += 1
        else:
            df.at[idx, "TimeSlipStatus"] = "inline_missing_break"
            df.at[idx, "ParagraphBreakClass"] = "missing"
            n_inline += 1

    df.attrs["bk2_confirmed_timeslips"] = n_confirmed
    df.attrs["bk2_inline_timeslips"] = n_inline
    return df


# ---------------------------------------------------------------------------
# BK2-final Part 2 — Missing-break LLM stage
# ---------------------------------------------------------------------------

_MISSING_BREAK_PROMPT = """\
PARAGRAPH BREAK CHECK

Below is the student's narrative, with each sentence numbered. A new
paragraph is expected at MAJOR narrative shifts only:
  - a new scene or location
  - a significant time jump
  - a clear change of topic or focus
  - a shift between narration and an extended new speaker's turn

Identify any numbered sentence that clearly SHOULD start a new paragraph
but does not (no paragraph break precedes it in the student's original).

Be conservative. Only flag a CLEAR, major shift. Do NOT flag:
  - minor topic movement within a scene
  - stylistic choices
  - every new sentence
When in doubt, do NOT flag.

Sentences (number: text; [BREAK] marks where the student already broke):
{numbered_sentences}

Return JSON only: {{"missing_break_before": [list of sentence numbers]}}"""


def _call_missing_break_api(numbered_sentences: str, model: str,
                             api_key: str) -> dict:
    """One LLM call per script; returns {"missing_break_before": [...]}."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    prompt = _MISSING_BREAK_PROMPT.format(numbered_sentences=numbered_sentences)
    msg = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    # Strip markdown code fences if the model wraps the JSON
    if text.startswith("```"):
        text = re.sub(r'^```[a-z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text.strip())
    return json.loads(text)


def apply_missing_break_llm(sent_df: pd.DataFrame, model: str,
                             api_key: str) -> pd.DataFrame:
    """Optional corrector-pipeline stage. Flags sentences that should start a
    new paragraph but don't, at major narrative shifts. Gated on api_key.
    Sets ParagraphBreakClass = 'missing' on flagged sentences; only upgrades
    'na' -- never overrides 'true', 'superfluous', or existing 'missing'."""
    if not api_key:
        return sent_df  # local-only: skip cleanly

    df = sent_df.copy()

    for ident, g in df.groupby('Identifier', sort=False):
        g_sorted = g.sort_values('SentenceRef')

        lines = []
        ref_by_num: dict = {}
        for n, (idx, row) in enumerate(g_sorted.iterrows(), 1):
            brk = '[BREAK] ' if row['PrecededByBreak'] in (
                'paragraph', 'line_break') else ''
            lines.append(f"{n}: {brk}{row['CorrectedSentence']}")
            ref_by_num[n] = idx
        numbered = "\n".join(lines)

        try:
            result = _call_missing_break_api(numbered, model, api_key)
        except Exception as e:
            print(f"  [BK2-missing] LLM error on {ident}: {e}")
            continue

        for num in result.get('missing_break_before', []):
            idx = ref_by_num.get(int(num))
            if idx is None:
                continue
            # Only upgrade na -> missing; deterministic classifications win
            if df.at[idx, 'ParagraphBreakClass'] == 'na':
                df.at[idx, 'ParagraphBreakClass'] = 'missing'

    return df


# ---------------------------------------------------------------------------
# BK2-4  Per-script counts on the texts table
# ---------------------------------------------------------------------------

_BREAK_COUNT_COLS = {
    "Para_Breaks_True":          ("ParagraphBreakClass", "true"),
    "Para_Breaks_Missing":       ("ParagraphBreakClass", "missing"),
    "Para_Breaks_Superfluous":   ("ParagraphBreakClass", "superfluous"),
    "Speech_Breaks_True":        ("SpeechBreakClass",    "true"),
    "Speech_Breaks_Missing":     ("SpeechBreakClass",    "missing"),
    "TimeSlips_Confirmed":       ("TimeSlipStatus",      "confirmed"),
    "TimeSlips_InlineMissing":   ("TimeSlipStatus",      "inline_missing_break"),
}


def add_break_counts(texts_df: pd.DataFrame, sent_df: pd.DataFrame,
                     id_col: str = "Research ID") -> pd.DataFrame:
    df = texts_df.copy()
    ids = df[id_col].astype(str)
    for col, (src_col, value) in _BREAK_COUNT_COLS.items():
        if src_col not in sent_df.columns:
            df[col] = 0
            continue
        counts = (sent_df[sent_df[src_col] == value]
                  .groupby("Identifier").size())
        df[col] = ids.map(counts).fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Convenience: run BK2 end-to-end on a (sent_df, wm) pair.
# ---------------------------------------------------------------------------

def apply_all(sent_df: pd.DataFrame, wm: pd.DataFrame) -> pd.DataFrame:
    """Run BK2-1 .. BK2-4 in the required order. Returns the augmented
    sentences dataframe (texts counts are added separately by
    add_break_counts before save)."""
    sent_df = classify_paragraph_breaks(sent_df)
    sent_df = classify_speech_breaks(sent_df, wm)
    sent_df = classify_structural_breaks(sent_df)
    sent_df = tag_time_slips(sent_df)
    return sent_df
