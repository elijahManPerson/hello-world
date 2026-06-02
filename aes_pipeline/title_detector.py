"""
title_detector.py
==================
Per-script Claude call that finds the parts of a script that are NOT the
narrative: a leading title, a trailing ending sign-off, and clearly
non-story fragments (bylines, word counts, notes to the marker, prompt
echoes). It restores what the old upstream NLP did, self-contained.

WHY A MODEL, NOT A RULE
  The raw text has no line break between a title and the story
  ("The Failed Submarine I had always wanted..."), so the boundary is a
  meaning judgement a rule cannot make.

TWO DELIBERATE BIASES (different directions, on purpose)
  - TITLE leans slightly toward flagging. A missed title bleeds its words
    into the first real sentence and corrupts that sentence's analysis,
    which is worse than a slightly odd first sentence.
  - "OTHER non-story" is the opposite: cautious. Over-flagging here would
    delete real narrative. Only clearly non-story fragments count.

FIRM CARVE-OUT
  Students open stories with onomatopoeia and dramatic exclamations:
  "BANG!", "CRASH!", "Whoosh...", "AAARGH!". These are the story starting,
  not a title or an artifact, and are often the strongest openings. They
  are always narrative.

OUTPUT IS WORD COUNTS / TEXT, NOT CHARACTER OFFSETS
  Models count characters badly. The marking step in mech_v2 turns these
  into token spans robustly, and a failed or empty call simply falls back
  to the existing heuristic. aes_canonical is never modified.

  mock=True prints the exact prompt and a sample shape at zero cost.
"""

import json
import re
import time

ENGINE = "TBD(engine)"

_SYSTEM = """You separate a student's narrative from the parts that are \
NOT the story. You are given one script. Identify three things.

1) TITLE: a short heading that names the story and stands outside the \
action, for example "The Failed Submarine", "NOSE THE FISH", "Make it to \
the Moon". Lean slightly toward identifying a title: if there is a \
plausible heading-like opener before the narrative starts, treat it as a \
title, because missing it corrupts the first sentence.

CRUCIAL CARVE-OUT: a sound word or dramatic exclamation that begins the \
action is NOT a title and NOT an artifact. "BANG!", "CRASH!", "Whoosh", \
"AAARGH!", "Splash!" are the story starting. Onomatopoeia and \
interjections are always narrative, never a title, even in capitals.

2) ENDING: a trailing sign-off that is not story, for example "The End", \
"THE END", "the end!".

3) OTHER non-story: ONLY clearly non-story fragments -- a byline like \
"by Sam", a word count like "WC 250", a note to the marker like "I hope \
you liked my story", or an echoed writing prompt. Be cautious here: if it \
could be narrative, it is narrative. Headings inside the piece, dialogue, \
unusual openings and onomatopoeia are NOT "other".

Count words as whitespace-separated words in the order they appear.

Return STRICT JSON only, no prose:
{"title": {"present": bool, "text": "", "word_count": 0},
 "ending": {"present": bool, "text": "", "word_count": 0},
 "other": [{"text": "", "kind": "byline|wordcount|note|prompt-echo"}]}
If absent: present false, text "", word_count 0, other []."""


# TC1 -- Title/ending detector helpers used to tighten the model's output
# -----------------------------------------------------------------------

_CONNECTIVE_WORDS_TC1 = {
    "that", "which", "when", "because", "after", "before", "while",
    "since", "although", "though", "if", "unless", "until",
}

# Patterns that strongly suggest a narrative ending (TC1b)
_ENDING_PATTERNS = [
    re.compile(r'\bthe\s+end\b', re.IGNORECASE),
    re.compile(r'\bhappily\s+ever\s+after\b', re.IGNORECASE),
    re.compile(r'\band\s+that(\'s|\s+is)\s+how\b', re.IGNORECASE),
    re.compile(r'\bto\s+this\s+day\b', re.IGNORECASE),
    re.compile(r'\bfrom\s+(then|that\s+day)\s+on\b', re.IGNORECASE),
    re.compile(r'\bthe\s+moral\s+of\b', re.IGNORECASE),
    re.compile(r'\band\s+they\s+all\s+lived\b', re.IGNORECASE),
    re.compile(r'\bin\s+the\s+end\b', re.IGNORECASE),
    # BR14 -- extended ending patterns
    re.compile(r'\bto\s+be\s+continued\b', re.IGNORECASE),
    re.compile(r'\bend\s+of\s+(?:part|chapter|episode|scene)\b', re.IGNORECASE),
    re.compile(r'[~*\-]\s*(?:the\s+)?end\s*[~*\-]', re.IGNORECASE),
    re.compile(r'[~*\-]\s*fin\s*[~*\-]', re.IGNORECASE),
    re.compile(r'[~*\-]\s*to\s+be\s+continued\s*[~*\-]', re.IGNORECASE),
    # BR14b -- "finally" only when it closes the tail (not mid-narrative)
    re.compile(r'\bfinally[.,]?\s*$', re.IGNORECASE),
]


def _is_likely_title_text(text):
    """
    TC1a: Return False for text that looks like an opening sentence rather
    than a title.  Used to post-filter model title detections.
    Returns True only when the text passes all rejection checks.
    """
    t = str(text or "").strip()
    if not t:
        return False
    # Rejection 1: lowercase opener with significant length
    if not t[0].isupper() and len(t.split()) >= 6:
        return False
    # Rejection 2: multi-clause structure (connective + long)
    words = [w.lower().strip(",.!?\"'") for w in t.split()]
    conn_count = sum(1 for w in words if w in _CONNECTIVE_WORDS_TC1)
    if conn_count >= 1 and len(words) >= 8:
        return False
    # Rejection 3: very long (> 8 words)
    if len(words) > 8:
        return False
    # BR11 -- Rejection 4: unclosed opening quote (indicates dialogue start, not a title)
    if (t.startswith('"') or t.startswith('“') or t.startswith("''")):
        if not (t.endswith('"') or t.endswith('”') or t.endswith("''")):
            return False
    # BR11 -- Rejection 5: looks like a sound effect (handled by BR13)
    # Short + mostly caps + ends with ! -> not a title
    letters = [c for c in t if c.isalpha()]
    if letters:
        cap_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if len(words) <= 2 and cap_ratio >= 0.8 and t.rstrip().endswith('!'):
            return False
    return True


def _detect_ending_in_tail(corrected_text, tail_chars=200):
    """
    TC1b: Search the last tail_chars of corrected_text for narrative ending
    patterns.  Returns word count of the matched span, or 0 if none found.
    """
    if not corrected_text:
        return 0
    tail = str(corrected_text)[-tail_chars:]
    for pattern in _ENDING_PATTERNS:
        m = pattern.search(tail)
        if m:
            return max(1, len(m.group(0).split()))
    return 0


def detect_artifacts(text, model="claude-opus-4-7", api_key=None,
                     mock=False):
    text = str(text or "").strip()
    if mock:
        return {
            "_mock": True,
            "system_prompt": _SYSTEM,
            "user_payload": text[:300] + (" ..." if len(text) > 300 else ""),
            "example_output_shape": {
                "title": {"present": True, "text": "The Failed Submarine",
                          "word_count": 3},
                "ending": {"present": True, "text": "The End",
                           "word_count": 2},
                "other": [{"text": "by Sam", "kind": "byline"}],
            },
        }
    if not text:
        return {"title": {"present": False, "text": "", "word_count": 0},
                "ending": {"present": False, "text": "", "word_count": 0},
                "other": []}

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    _FAIL_SAFE = {"title": {"present": False, "text": "", "word_count": 0},
                  "ending": {"present": False, "text": "", "word_count": 0},
                  "other": []}

    max_retries = 3
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            msg = client.messages.create(
                model=model, max_tokens=500,
                system=[{"type": "text", "text": _SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": text}],
            )
            try:
                from cost_tracker import tracker
                tracker.record("artifacts", msg.usage, model=model)
            except Exception:
                pass
            out = next((b.text for b in msg.content if b.type == "text"), "{}")
            out = re.sub(r"```json|```", "", out).strip()
            try:
                r = json.loads(out)
                # minimal shape guard; on anything odd, fail to "nothing flagged"
                for k in ("title", "ending"):
                    r.setdefault(k, {})
                    r[k].setdefault("present", False)
                    r[k].setdefault("text", "")
                    r[k].setdefault("word_count", 0)
                r.setdefault("other", [])
                return r
            except Exception:
                # fail safe: detect nothing -> heuristic fallback still applies,
                # and no real narrative is deleted.
                return dict(_FAIL_SAFE, _parse_failed=True)

        except anthropic.RateLimitError as e:
            if attempt >= max_retries:
                print(f"  [artifacts] rate limit, giving up after {max_retries} retries")
                return dict(_FAIL_SAFE, _rate_limited=True)
            retry_after = int(
                getattr(e.response, "headers", {}).get("retry-after", delay))
            print(f"  [artifacts] rate limited, waiting {retry_after}s "
                  f"(attempt {attempt+1}/{max_retries})")
            time.sleep(retry_after)
            delay *= 2

        except anthropic.APIStatusError as e:
            if e.status_code < 500 or attempt >= max_retries:
                print(f"  [artifacts] API error {e.status_code}, giving up")
                return dict(_FAIL_SAFE, _api_error=e.status_code)
            print(f"  [artifacts] server error {e.status_code}, retrying in "
                  f"{delay}s (attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
            delay *= 2

    return dict(_FAIL_SAFE, _exhausted=True)  # should not reach here


def run_artifacts(texts, id_col="Research ID", raw_col="Raw text",
                  model="claude-opus-4-7", api_key=None, mock=False):
    """Add columns the mech_v2 marking step reads:
       TitleWordCount, EndingWordCount, OtherSpansJSON, ArtifactText.
    Cautious by construction: anything uncertain becomes 0 / empty, which
    leaves the existing heuristic in charge rather than deleting text."""
    df = texts.copy()
    df["Identifier"] = df[id_col].astype(str)
    tw, ew, ot, info = [], [], [], []
    for raw in df[raw_col].astype(str):
        r = detect_artifacts(raw, model=model, api_key=api_key, mock=mock)
        if r.get("_mock"):
            tw.append(ENGINE); ew.append(ENGINE); ot.append(ENGINE)
            info.append("MOCK: see detect_artifacts(text, mock=True)")
            continue
        t = r.get("title", {})
        e = r.get("ending", {})
        # TC1a: post-filter model title detection
        raw_tw = int(t.get("word_count", 0)) if t.get("present") else 0
        title_text = t.get("text", "")
        if raw_tw > 0 and not _is_likely_title_text(title_text):
            raw_tw = 0  # reject -- looks like an opening clause, not a title

        # TC1b: supplement model ending detection with pattern matching
        raw_ew = int(e.get("word_count", 0)) if e.get("present") else 0
        if raw_ew == 0:
            raw_ew = _detect_ending_in_tail(raw)

        tw.append(raw_tw)
        ew.append(raw_ew)
        ot.append(json.dumps(r.get("other", []), ensure_ascii=False))
        info.append(json.dumps({"title": t.get("text", ""),
                                "ending": e.get("text", "")},
                               ensure_ascii=False))
    df["TitleWordCount"] = tw
    df["EndingWordCount"] = ew
    df["OtherSpansJSON"] = ot
    df["ArtifactText"] = info
    return df


def detect_prompt_copy(sentence_text, prompt_text, threshold_ratio=0.20):
    """
    B2 -- Return True if sentence_text is a near-copy of the writing prompt.
    Uses character-level edit distance (Levenshtein or difflib fallback).
    threshold_ratio: fraction of prompt length that is the maximum edit distance.
    Only intended for the first 3-5 sentences of a script to avoid mid-narrative
    false positives.
    """
    if not prompt_text or not sentence_text:
        return False
    st = str(sentence_text).lower().strip()
    pt = str(prompt_text).lower().strip()
    threshold = max(10, int(len(pt) * threshold_ratio))

    try:
        import Levenshtein as _lev
        distance = _lev.distance(st, pt)
    except ImportError:
        import difflib
        # Use difflib SequenceMatcher as fallback
        ratio = difflib.SequenceMatcher(None, st, pt).ratio()
        distance = int((1 - ratio) * max(len(st), len(pt)))

    return distance <= threshold


def tag_prompt_copies_in_wm(df_map, prompt_text, max_sentence_index=5):
    """
    B2: Walk the first max_sentence_index sentences per script and tag
    any sentence that is a near-copy of the prompt with
    TextualArtifact = 'PROMPT'.
    Returns modified df_map.
    """
    # Guard: ensure prompt_text is a non-empty string (catches None, float NaN, etc.)
    try:
        prompt_text = str(prompt_text or "").strip()
    except Exception:
        prompt_text = ""
    if not prompt_text or prompt_text.lower() == "nan":
        return df_map
    df = df_map.copy()
    idcol = "Identifier" if "Identifier" in df.columns else "ID"

    for ID, g in df.groupby(idcol, sort=False):
        if "corr_index" not in g.columns:
            continue
        g_sorted = g.sort_values("corr_index", kind="mergesort")
        if "CorrSentenceID" not in g_sorted.columns:
            continue
        for csid, sg in g_sorted.groupby("CorrSentenceID", sort=True):
            try:
                sid_int = int(csid)
            except Exception:
                continue
            if sid_int > max_sentence_index:
                break
            # Reconstruct sentence text -- explicit str() guard on each token
            toks = [str(t) for t in sg["corr_token"].tolist()
                    if t is not None and str(t).strip() not in ("", "nan")]
            sent_text = " ".join(toks)
            if detect_prompt_copy(sent_text, prompt_text):
                df.loc[sg.index, "TextualArtifact"] = "PROMPT"
    return df


def is_likely_title_at_body_start(raw_text, sentence_index):
    """
    SA2 -- Catch titles the model missed because they sit at sentence positions
    _s001-_s003 rather than _s000.  Triggers when ALL of these hold:
      - sentence_index is 1, 2, or 3 (first three body positions)
      - raw text is ALL CAPS
      - <= 20 characters (after stripping)
      - <= 4 whitespace-separated words

    Deliberately conservative (all four conditions required) so that normal
    ALL-CAPS shouted dialogue later in the script is never mistakenly tagged.
    """
    if sentence_index < 1 or sentence_index > 3:
        return False
    raw = str(raw_text or "").strip()
    if not raw:
        return False
    if len(raw) > 20:
        return False
    if raw != raw.upper():
        return False
    if len(raw.split()) > 4:
        return False
    return True


if __name__ == "__main__":
    print("Title/artifact detector. Inspect the prompt with:")
    print("  from title_detector import detect_artifacts")
    print("  print(detect_artifacts('x', mock=True)['system_prompt'])")
