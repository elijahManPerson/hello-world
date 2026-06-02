"""
text_level_classifier.py
========================
Consolidated text-level LLM call for NAPLAN AES pipeline (Phase TB2).

Combines gate-zero checks (OnGenre, TitleOnly, CopyPrompt, Scribed, SoC)
with three NAPLAN rubric criteria (Text Structure, Ideas, Character & Setting).
One Sonnet call per script, ~$0.03/script, ~$2–3 per Basic (73-script) run.

TB2 schema changes vs TB1:
  Deleted:  OnPrompt, OnPromptReason, Recount, RecountReason
  Added:    15 rubric assessment columns (see TB2_COLS below)
  Retained: OnGenre+Reason, TitleOnly, CopyPrompt, Scribed+Signals, SoC+Reason
"""

import json
import os
import time
import anthropic

_client = None


def _get_client(api_key=None):
    global _client
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
    return _client


# ---------------------------------------------------------------------------
# TB2-2 — System prompt (full rubric)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are evaluating a Year 3-9 NAPLAN narrative writing script against
the NAPLAN marking rubric. Return JSON with the fields specified.

Be conservative — choose lower bands when evidence is ambiguous.
Provide brief reasoning (one to two sentences) for each criterion.

The student's response may have been transcribed by a scribe, may be
off-prompt, or may not be a narrative at all. Assess what is actually
present in the text against the rubric criteria.

RUBRIC CRITERION 1 — TEXT STRUCTURE
The organisation of narrative features (orientation, complication,
resolution) into appropriate and effective text structure.

Bands:
- "None" (rubric 0): no evidence of structural components; off-genre
  (e.g. recipe, argument); symbols or drawings; title only
- "Minimal" (rubric 1): minimal evidence of narrative structure;
  beginning only or middle with no orientation; recount of events
  with no complication; may be description only
- "Weak" (rubric 2): contains beginning and complication; resolution
  weak, contrived or tacked on (e.g. "I woke up", "they lived happily
  ever after"); all parts of story may be weak
- "Present" (rubric 3): contains orientation, complication, resolution;
  detailed text may resolve one complication and lead into a new one
  or layer a new complication onto an existing one rather than conclude
- "Effective" (rubric 4): coherent, controlled and complete narrative;
  effective plot devices; appropriate structure with effective ending.
  Sophisticated devices include foreshadowing, flashback, red herring,
  cliffhanger, coda, twist, evaluation/reflection, circular plots,
  parallel plots

ALSO assess each narrative element separately:
- Orientation: absent / weak / present / developed
- Complication: absent / weak / present / developed
- Resolution: absent / weak / present / developed

RUBRIC CRITERION 2 — IDEAS
The creation, selection and crafting of ideas for a narrative.

Bands:
- "none" (rubric 0): no evidence; symbols or drawings; title only
- "minimal" (rubric 1): one idea; very few and very simple; ideas
  unrelated to each other; ideas unrelated to prompt (off-prompt)
- "simple" (rubric 2): one idea with simple elaboration; few related
  but not elaborated; many simple ideas related but not elaborated
- "coherent" (rubric 3): ideas show development or elaboration; all
  ideas relate coherently; some may contain unnecessary waffle
- "substantial" (rubric 4): substantial, elaborated ideas contributing
  effectively to a central storyline; suggestion of underlying theme
- "crafted" (rubric 5): ideas generated, selected and crafted to
  explore a recognisable theme; skilfully used; may include psychological
  subjects, unexpected topics, mature viewpoints, extended metaphor,
  satirical perspectives, traditional sub-genre subjects (heroic quest,
  whodunnit, good vs evil, overcoming the odds)

ALSO assess Theme separately:
- "none": no single recognisable theme
- "suggestion": suggestion of a theme — name the theme and explain
  why it is not fully developed
- "recognisable": clear theme — name the theme and explain how it is
  developed throughout the text

RUBRIC CRITERION 3 — CHARACTER AND SETTING
Character: portrayal and development of character.
Setting: development of a sense of place, time and atmosphere.
The rubric uses AND/OR — a character-driven story with sketchy setting
can score well, and vice versa.

For Characters: briefly list the characters present (or "off genre"
if not a narrative).

For CharacterAnalysis bands:
- "none" (rubric 0): no evidence; off-genre; title only
- "named" (rubric 1): only names characters or gives their roles
  (e.g. father, the teacher, my friend, Jim)
- "suggestion" (rubric 2): suggestion of characterisation through
  brief descriptions, speech or feelings; lacks substance or continuity;
  basic dialogue or few adjectives
- "emerges" (rubric 3): characterisation emerges through descriptions,
  actions, speech, or attribution of thoughts and feelings to a character
- "effective" (rubric 4): effective characterisation; details selected
  to create distinct characters; convincing dialogue, introspection,
  and reactions to other characters

For Setting: briefly describe the setting (or "off genre"
if not a narrative).

For SettingAnalysis bands:
- "none" (rubric 0): no evidence; off-genre; title only
- "named" (rubric 1): only names the setting (e.g. school, the place
  we were at); vague or confused setting
- "suggestion" (rubric 2): suggestion of setting through very brief
  and superficial descriptions of place and/or time
- "emerges" (rubric 3): setting emerges through description of place,
  time and atmosphere
- "effective" (rubric 4): maintains a sense of setting throughout;
  details selected to create place and atmosphere

OFF-GENRE CAP RULES (apply when OnGenre=False)
- TextStructure MUST be "None"
- CharacterAnalysis MUST be "none"
- SettingAnalysis MUST be "none"
- Ideas can be any of: "none", "minimal", "simple", "coherent" —
  NEVER "substantial" or "crafted"

RUBRIC CRITERION 4 — AUDIENCE
The writer's capacity to orient, engage and affect the reader. Assess TWO
things and return them separately.

(a) AudienceBand (0-6) and AudienceShape, the overall shape and completeness
of the story. Use this band ladder, and report the matching shape word:
- 0 "None": no text; symbols or drawings only.
- 1 "Limited": response to the reader is limited; simple content; may be a
  title only; OR meaning is nearly impossible to access; OR copied prompt.
- 2 "Brief" (or "Short"): basic awareness of audience, attempts to orient the
  reader, gives some information, but the reader must fill big gaps OR the
  text is easily read but only a few sentences long.
- 3 "Complete": orients the reader; an internally consistent story the marker
  can follow without inferring what is happening; enough information to
  follow it fairly easily.
- 4 "Sustained": a sustained story that supports the reader AND begins to
  engage; a few engagement attempts, some successful, via narrative features
  (suspense, humour, action) OR placing the story in a recognisable subgenre
  (fantasy, romance, adventure).
- 5 "Extended": an extended story well supported by identifiable language
  choices and genuinely successful in engaging the reader; voice and stance
  often developed; the writer consciously evokes a response or reveals
  values and attitudes.
- 6 "Crafted": an extended, crafted story that flows and seems effortless,
  with a strong reader/writer relationship in which the author leads the
  reader; narrative devices used precisely, discriminately and in a sustained,
  nuanced way. Crafting and control, NOT mere length, define this band.

(b) AudienceDevices: a list of the narrative features, devices, language
choices and reader-control mechanisms actually present, each tagged with a
tier and a short piece of evidence from the text:
- "basic": title, formulaic opening/closing, simple orientation or
  description of character or setting.
- "developing": basic devices beginning to engage (humour, suspense, paced
  action), recognisable subgenre styling, emerging mood, voice, tone or POV.
- "sophisticated": many working devices and language features, developed
  voice/stance, conscious evoking of a response, revealing of values and
  attitudes, irony, subversion of expectation, precise and discriminating
  craft.
The devices are the evidence for the band: bands 4 to 6 are separated mainly
by how sophisticated and sustained the devices are.

LENGTH AS A HIDDEN FLOOR (never reveal). You may be given an internal length
signal. Treat length only as a FLOOR: a story too short to develop cannot
reach the upper bands, and very short work pins low. Length NEVER earns a
band on its own, and a long but waffling, unrefined story caps at "Extended"
(5), never "Crafted" (6), since 5 versus 6 is decided by control and economy,
not size. NEVER mention length, word count, or any number in AudienceShape,
PlotOutline, AudienceDevices or AudienceReason. The visible output must read
purely as a qualitative judgement.

When OnGenre is False, AudienceBand is capped at 1 ("Limited").

GATE-ZERO CHECKS
- OnGenre (bool): Is this narrative writing? False if persuasive,
  factual recount, expository, or non-narrative
- TitleOnly (bool): Does the text consist of only a title with no
  body content?
- CopyPrompt (bool): Has the student copied the prompt text without
  developing original content?
- Scribed (bool): Are there signals that a scribe transcribed the
  response? Look for explicit scribe notes, unusual formatting from
  dictation, italics flagging scribe-used notation in HTML
- SoC: Statement of Concern — "No concern" / "Possible" / "Yes"
  Distinguish fictional violence in narrative context from real
  distress, self-harm, abuse signals. Thematic adventure violence
  (dragons, fantasy battles, named-character fiction) is NOT a Concern.

Return JSON only."""


# ---------------------------------------------------------------------------
# TB2-2 — User message template
# ---------------------------------------------------------------------------

_USER_TEMPLATE = """\
Writing prompt: {prompt}

Raw HTML (for scribe/formatting context):
{raw_html}

Student's text (corrected):
{corrected_text}
{length_hint}
Evaluate against the rubric criteria and gate-zero checks. Return JSON
with these exact fields:

{{
  "OnGenre": bool,
  "OnGenreReason": "string",
  "TitleOnly": bool,
  "CopyPrompt": bool,
  "Scribed": bool,
  "ScribedSignals": ["string"],
  "SoC": "No concern | Possible | Yes",
  "SoCReason": "string",
  "AudienceBand": 0,
  "AudienceShape": "None | Limited | Brief | Complete | Sustained | Extended | Crafted",
  "AudienceReason": "string (no mention of length or numbers)",
  "PlotOutline": "string (3-4 sentences, plain plot summary, no evaluation)",
  "AudienceDevices": [
    {{"device": "string", "tier": "basic | developing | sophisticated", "evidence": "short snippet"}}
  ],
  "Orientation": "absent | weak | present | developed",
  "Complication": "absent | weak | present | developed",
  "Resolution": "absent | weak | present | developed",
  "TextStructure": "None | Minimal | Weak | Present | Effective",
  "TextStructureReason": "string",
  "Ideas": "none | minimal | simple | coherent | substantial | crafted",
  "IdeasReason": "string",
  "Theme": "none | suggestion | recognisable",
  "ThemeReason": "string",
  "Characters": "string",
  "CharacterAnalysis": "none | named | suggestion | emerges | effective",
  "CharacterAnalysisReason": "string",
  "Setting": "string",
  "SettingAnalysis": "none | named | suggestion | emerges | effective",
  "SettingAnalysisReason": "string"
}}"""


# ---------------------------------------------------------------------------
# TB2-1 — Column schema
# ---------------------------------------------------------------------------

# Gate-zero checks retained from TB1
_GATE_ZERO_COLS = [
    "OnGenre", "OnGenreReason",
    "TitleOnly", "CopyPrompt",
    "Scribed", "ScribedSignals",
    "SoC", "SoCReason",
]

# New TB2 rubric columns (15)
TB2_COLS = [
    "AudienceBand", "AudienceShape", "AudienceReason",
    "PlotOutline", "AudienceDevices",
    "Orientation", "Complication", "Resolution",
    "TextStructure", "TextStructureReason",
    "Ideas", "IdeasReason",
    "Theme", "ThemeReason",
    "Characters", "CharacterAnalysis", "CharacterAnalysisReason",
    "Setting", "SettingAnalysis", "SettingAnalysisReason",
]

# All output columns written by this module
ALL_TB2_COLS = _GATE_ZERO_COLS + TB2_COLS

# Columns deleted from TB1 — kept here for reference; run_all.py drops them
# from the null-column stub when toggle is off.
_DELETED_TB1_COLS = ["OnPrompt", "OnPromptReason", "Recount", "RecountReason"]

# Fail-safe defaults (all None)
_FAIL_SAFE = {col: None for col in ALL_TB2_COLS}
_FAIL_SAFE["ScribedSignals"] = []
_FAIL_SAFE["error"] = None


# ---------------------------------------------------------------------------
# TB2-4 — Off-genre cap enforcement
# ---------------------------------------------------------------------------

_IDEAS_ALLOWED_OFF_GENRE = {"none", "minimal", "simple", "coherent"}


def enforce_off_genre_caps(result: dict) -> dict:
    """
    Post-parse safeguard: when OnGenre is False, force cap rules.

    Applied to a single result dict (not the whole DataFrame) so it
    works identically whether called per-row or in batch.

    When OnGenre=False:
      - TextStructure  → "None"
      - CharacterAnalysis → "none"
      - SettingAnalysis   → "none"
      - Ideas capped at "coherent" if higher
    """
    if result.get("OnGenre") is True:
        return result  # cap rules don't apply to on-genre scripts

    result = dict(result)  # shallow copy — don't mutate caller's dict

    if result.get("TextStructure") != "None":
        result["TextStructureReason"] = (
            f"[capped: off-genre] {result.get('TextStructureReason', '')}"
        )
        result["TextStructure"] = "None"

    if result.get("CharacterAnalysis") != "none":
        result["CharacterAnalysisReason"] = (
            f"[capped: off-genre] {result.get('CharacterAnalysisReason', '')}"
        )
        result["CharacterAnalysis"] = "none"

    if result.get("SettingAnalysis") != "none":
        result["SettingAnalysisReason"] = (
            f"[capped: off-genre] {result.get('SettingAnalysisReason', '')}"
        )
        result["SettingAnalysis"] = "none"

    if result.get("Ideas") not in _IDEAS_ALLOWED_OFF_GENRE:
        result["IdeasReason"] = (
            f"[capped at coherent: off-genre] {result.get('IdeasReason', '')}"
        )
        result["Ideas"] = "coherent"

    try:
        if int(result.get("AudienceBand") or 0) > 1:
            result["AudienceReason"] = (
                f"[capped: off-genre] {result.get('AudienceReason', '')}"
            )
            result["AudienceBand"] = 1
            result["AudienceShape"] = "Limited"
    except (TypeError, ValueError):
        pass

    return result


# ---------------------------------------------------------------------------
# TB2-3 — LLM call + parsing
# ---------------------------------------------------------------------------

# Empirical median word count at each Audience band, by year level, derived
# from ~44k marked scripts. Used ONLY as a hidden floor signal; never shown.
# Year 3 is intentionally absent (no word counts in the calibration data), so
# the hint degrades to a qualitative floor for Year 3 and unknown years.
_AUD_LENGTH_PRIOR = {
    5: {3: 297, 4: 420, 5: 530, 6: 593},
    7: {3: 352, 4: 496, 5: 607, 6: 701},
    9: {3: 391, 4: 555, 5: 666, 6: 753},
}


def _parse_year(year_level):
    """Return one of {3,5,7,9} from messy year input, or None."""
    import re as _re
    if year_level is None:
        return None
    m = _re.search(r"\d+", str(year_level))
    if not m:
        return None
    y = int(m.group())
    return min((3, 5, 7, 9), key=lambda k: abs(k - y)) if y else None


def _build_length_hint(word_count, year_level):
    """Hidden, never-shown length floor signal for the Audience judgement."""
    try:
        wc = int(float(word_count))
    except (TypeError, ValueError):
        wc = None
    if wc is None:
        return ""  # no signal available; judge on text alone
    yr = _parse_year(year_level)
    principle = (
        "Treat this only as a FLOOR for your reasoning: too short to develop "
        "cannot reach the upper bands, but length never earns a band, and a "
        "long but unrefined or waffling story caps at Extended (5), never "
        "Crafted (6). NEVER mention length, word count or any number in your "
        "output."
    )
    prior = _AUD_LENGTH_PRIOR.get(yr)
    if prior:
        bands = (f"Complete (3) about {prior[3]}, Sustained (4) about "
                 f"{prior[4]}, Extended (5) about {prior[5]}, Crafted (6) "
                 f"about {prior[6]} words")
        return (f"\nINTERNAL LENGTH SIGNAL (do NOT reveal): this script is "
                f"about {wc} words. For a Year {yr} writer, scripts typically "
                f"reach {bands}. {principle}\n")
    return (f"\nINTERNAL LENGTH SIGNAL (do NOT reveal): this script is about "
            f"{wc} words. No per-band length norms are available for this "
            f"year, so weigh length lightly. {principle}\n")

def classify_script(
    corrected_text,
    prompt_text,
    raw_html=None,
    raw_text=None,
    research_id="",
    scribe_pre_signals=None,
    api_key=None,
    model="claude-sonnet-4-6",
    word_count=None,
    year_level=None,
):
    """
    Run the TB2 combined rubric + gate-zero call on one script.

    Returns a dict with all fields. On error returns _FAIL_SAFE with error
    key populated. Off-genre caps are applied before returning.
    """
    client = _get_client(api_key)

    html_excerpt = str(raw_html or "")[:1500]

    user_prompt = _USER_TEMPLATE.format(
        prompt=prompt_text or "(no prompt provided)",
        raw_html=html_excerpt,
        corrected_text=corrected_text or "",
        length_hint=_build_length_hint(word_count, year_level),
    )

    max_retries = 3
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2800,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_prompt}],
            )
            try:
                from cost_tracker import tracker
                tracker.record("authorial", response.usage, model=model)
            except Exception:
                pass

            text_out = next(
                (b.text for b in response.content if b.type == "text"), "{}"
            )
            cleaned = text_out.strip()
            # Strip markdown fences if present
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                cleaned = parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)
            result.setdefault("error", None)

            # TB2-4: enforce off-genre caps
            result = enforce_off_genre_caps(result)
            return result

        except anthropic.RateLimitError as e:
            if attempt >= max_retries:
                return dict(_FAIL_SAFE, error="rate_limit_exhausted")
            retry_after = int(
                getattr(e.response, "headers", {}).get("retry-after", delay)
            )
            time.sleep(retry_after)
            delay *= 2

        except anthropic.APIStatusError as e:
            if e.status_code < 500 or attempt >= max_retries:
                return dict(_FAIL_SAFE, error=f"api_error_{e.status_code}")
            time.sleep(delay)
            delay *= 2

        except Exception as e:
            return dict(_FAIL_SAFE, error=str(e))

    return dict(_FAIL_SAFE, error="exhausted")


# ---------------------------------------------------------------------------
# TB2-3 — Column population
# ---------------------------------------------------------------------------

def add_text_level_columns(texts_df, prompt_text, cfg, api_key=None):
    """
    Apply TB2 classification to all rows in texts_df.

    When cfg["stage_authorial"] is False, all columns added as None
    (zero API calls). The 4 deleted TB1 columns are NOT added.
    """
    if not cfg.get("stage_authorial", cfg.get("stage_TB2_rubric", cfg.get("text_level_classification", False))):
        # Add null stubs for all TB2 output columns (object dtype to avoid float coercion)
        import pandas as _pd
        for col in ALL_TB2_COLS:
            if col not in texts_df.columns:
                texts_df[col] = _pd.array([None] * len(texts_df), dtype=object)
        # Guard: Raw HTML must be present (Pa1 passthrough)
        if "Raw HTML" not in texts_df.columns:
            texts_df["Raw HTML"] = ""
        return texts_df

    from scribe_detector import detect_scribe_signals

    model = cfg.get("model_text_level", "claude-sonnet-4-6")
    raw_col  = cfg.get("raw_col",  "Raw text")
    corr_col = cfg.get("corr_col", "Corrected text (8)")
    id_col   = cfg.get("id_col",   "Research ID")

    # Guard: Raw HTML must be present (Pa1 passthrough in run_all.py)
    if "Raw HTML" not in texts_df.columns:
        texts_df["Raw HTML"] = ""

    # Initialise all output columns so rows that fail still have the column.
    # Use empty string (object dtype) not None (which pandas stores as float64/NaN)
    # so that later string assignments via .at[] are never silently cast to NaN.
    import pandas as _pd
    import numpy as np
    for col in ALL_TB2_COLS:
        if col not in texts_df.columns:
            texts_df[col] = _pd.array([None] * len(texts_df), dtype=object)
        elif texts_df[col].dtype != object:
            texts_df[col] = texts_df[col].astype(object)

    n_caps_fired = 0
    total = len(texts_df)

    for i, (idx, row) in enumerate(texts_df.iterrows()):
        # Per-script prompt: use Prompt column if present, fall back to global
        per_script_prompt = str(row.get("Prompt", "")).strip()
        effective_prompt = per_script_prompt if per_script_prompt else prompt_text

        pre_signals = detect_scribe_signals(row.get(raw_col, ""))

        _year = (row.get("yrlev", row.get("YEAR_LEVEL",
                 row.get("Year Level", row.get("yearlevel",
                 row.get("Year"))))))

        result = classify_script(
            corrected_text=str(row.get(corr_col, "")),
            prompt_text=effective_prompt,
            raw_html=str(row.get("Raw HTML", "")),
            raw_text=str(row.get(raw_col, "")),
            research_id=str(row.get(id_col, "")),
            scribe_pre_signals=pre_signals,
            api_key=api_key,
            model=model,
            word_count=row.get("WordCount"),
            year_level=_year,
        )

        if (i + 1) % 10 == 0:
            print(f"  [text_level] {i+1}/{total} done")

        err = result.pop("error", None)
        if err:
            print(f"  [TB2 error] {row.get(id_col, idx)}: {err}")

        # Detect if caps fired (any reason string starts with "[capped")
        if any(
            str(result.get(c, "")).startswith("[capped")
            for c in ("TextStructureReason", "CharacterAnalysisReason",
                       "SettingAnalysisReason", "IdeasReason")
        ):
            n_caps_fired += 1

        for col, val in result.items():
            if col not in ALL_TB2_COLS:
                continue
            texts_df.at[idx, col] = (
                json.dumps(val, ensure_ascii=False)
                if isinstance(val, list)
                else val
            )

    if n_caps_fired:
        print(f"  [TB2] off-genre caps applied to {n_caps_fired} script(s)")

    return texts_df
