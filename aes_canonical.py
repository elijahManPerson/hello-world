# -*- coding: utf-8 -*-
"""
aes_canonical.py
================
ONE canonical module. Steps 1-9, single definition of every function,
plus an input-quality gate that runs BEFORE sentence scoring.
"""

import re
import json
import math
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

CANONICAL_VERSION = "aes_canonical_2026-05-17_r4"

# ----------------------------------------------------------------------
# Step 8A. Correction
# ----------------------------------------------------------------------

_MOJIBAKE_FIXES = [
    (r"\u00e2\u20ac\u201d", "\u2014"), (r"\u00e2\u20ac\u201c", "\u2013"),
    (r"\u00e2\u20ac\u02dc", "\u2018"), (r"\u00e2\u20ac\u2122", "\u2019"),
    (r"\u00e2\u20ac\u0153", "\u201c"), (r"\u00e2\u20ac\x9d",   "\u201d"),
    (r"\u00e2\u20ac\u00a6", "\u2026"), ("\u00c2 ",               " "),
]


def normalize_mojibake(s):
    if s is None:
        return ""
    out = str(s)
    for pat, repl in _MOJIBAKE_FIXES:
        out = re.sub(pat, repl, out)
    return out


def mock_corrector(raw):
    s = normalize_mojibake(str(raw or ""))
    t = s.strip()
    m = re.search(r"[A-Za-z]", t)
    if m:
        i = m.start()
        t = t[:i] + t[i].upper() + t[i + 1:]
    if t and not re.search(r"[.!?…]\s*$", t):
        t += "."
    return t


def run_correct_only(df_in, corrector, text_col="Raw text", id_col="ID",
                     out_col="Corrected text (8)"):
    if text_col not in df_in.columns:
        raise KeyError(f"Missing required column: {text_col}")
    df = df_in.copy()
    if id_col not in df.columns:
        df[id_col] = pd.RangeIndex(len(df)).astype(str)
    df[id_col] = df[id_col].astype(str).str.replace(r"\.0$", "", regex=True)

    cache = {}
    corrected = []
    src = []
    for raw in df[text_col].astype(str).tolist():
        if raw in cache:
            c = cache[raw]
        else:
            c = corrector(raw)
            cache[raw] = c
        corrected.append(c)
        src.append(getattr(corrector, "__name__", "corrector"))
    df[out_col] = corrected
    df["CorrectedBy"] = src
    df["NarrativeTagsJSON"] = "[]"
    df["DialogueSpansJSON"] = "[]"
    return df


# ----------------------------------------------------------------------
# Step 8B. Token map
# ----------------------------------------------------------------------

_WORD_RX = re.compile(r"\w", flags=re.UNICODE)
_VOWELS = frozenset("aeiouAEIOUyY")  # y/Y acts as vowel in "types", "gym", etc.
_ORDINAL_RX = re.compile(r"^\d+(?:st|nd|rd|th)$", re.I)
_DECADE_RX  = re.compile(r"^\d{4}s$", re.I)


def _is_gibberish(tok):
    """Return True for non-comprehensible tokens where no reasonable word
    can be inferred:
      - mixed alphanumeric that is not an ordinal (19th) or decade (1980s)
      - pure-alphabetic string longer than 3 chars with no vowels
    jiopjikop-style tokens (vowels present) are NOT caught — no dictionary."""
    if not isinstance(tok, str) or not tok:
        return False
    has_alpha = bool(re.search(r"[a-zA-Z]", tok))
    has_digit = bool(re.search(r"\d", tok))
    if has_alpha and has_digit:
        return not (_ORDINAL_RX.match(tok) or _DECADE_RX.match(tok))
    if has_alpha and not has_digit:
        return len(tok) > 3 and not any(c in _VOWELS for c in tok)
    return False


def _simple_tokenize(s):
    # Numerals (including decimal/thousands separators) are parsed as a single
    # token so that number-internal "." and "," never become potential sentence
    # boundaries.  The numeral pattern must precede \w+ to take priority.
    return re.findall(r"\d+(?:[.,]\d+)*(?!\w)|\w+|\.\.\.|[^\w\s]", s or "", flags=re.UNICODE)


def _rebuild_offsets(text, tokens):
    spans = []
    i = 0
    text = text or ""
    for tok in tokens:
        start = text.find(tok, i)
        if start < 0:
            start = i
        end = start + len(tok)
        spans.append((start, end))
        i = end
    return spans


def _is_word(tok):
    if not isinstance(tok, str):
        return False
    return bool(tok) and bool(_WORD_RX.search(tok))


_UPPER_LOWER_SEAM_RX = re.compile(r'([A-Z]{2,})([a-z])')
_WORD_PUNCT_PENALTY = 4  # substitution cost for word<->punct mistype


def _normalise_raw_spacing(text):
    """Split at uppercase->lowercase seams: 'YAAAYwe' -> 'YAAAY we'."""
    return _UPPER_LOWER_SEAM_RX.sub(r'\1 \2', text)


def _align(raw_tokens, corr_tokens):
    """Type-aware DP aligner that heavily penalises word<->punct substitutions.

    Returns a list of (op, raw_idx_or_None, corr_idx_or_None) individual
    token-level operations: 'equal', 'replace', 'delete', 'insert'.
    """
    n, m = len(raw_tokens), len(corr_tokens)

    def _sub_cost(i, j):
        rt, ct = raw_tokens[i], corr_tokens[j]
        if _is_word(rt) != _is_word(ct):
            return _WORD_PUNCT_PENALTY
        return 0 if rt.lower() == ct.lower() else 1

    # Build DP cost table.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]   # 0=sub, 1=del, 2=ins
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c_s = dp[i - 1][j - 1] + _sub_cost(i - 1, j - 1)
            c_d = dp[i - 1][j] + 1
            c_i = dp[i][j - 1] + 1
            best = min(c_s, c_d, c_i)
            dp[i][j] = best
            bt[i][j] = 0 if best == c_s else (1 if best == c_d else 2)

    # Traceback.
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            ops.append(("insert", None, j - 1))
            j -= 1
        elif j == 0:
            ops.append(("delete", i - 1, None))
            i -= 1
        elif bt[i][j] == 0:
            rt, ct = raw_tokens[i - 1], corr_tokens[j - 1]
            op = ("equal"
                  if rt.lower() == ct.lower() and _is_word(rt) == _is_word(ct)
                  else "replace")
            ops.append((op, i - 1, j - 1))
            i -= 1
            j -= 1
        elif bt[i][j] == 1:
            ops.append(("delete", i - 1, None))
            i -= 1
        else:
            ops.append(("insert", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def build_word_map(raw_text, corr_text):
    raw_text = _normalise_raw_spacing(str(raw_text or ""))
    corr_text = str(corr_text or "")
    raw_tokens = _simple_tokenize(raw_text)
    corr_tokens = _simple_tokenize(corr_text)
    raw_spans = _rebuild_offsets(raw_text, raw_tokens)
    corr_spans = _rebuild_offsets(corr_text, corr_tokens)

    rows = []
    for op, ri, ci in _align(raw_tokens, corr_tokens):
        rt = raw_tokens[ri] if ri is not None else None
        ct = corr_tokens[ci] if ci is not None else None
        rs, re_ = raw_spans[ri] if ri is not None else (None, None)
        cs, ce  = corr_spans[ci] if ci is not None else (None, None)

        if op == "equal":
            rows.append(dict(raw_index=ri, raw_token=rt, raw_start=rs,
                             raw_end=re_, corr_index=ci, corr_token=ct,
                             corr_start=cs, corr_end=ce, op="equal",
                             equal_ci=(rt == ct), error_type="Equal"))
        elif op == "replace":
            err = ("Spelling" if rt.isalpha() and ct.isalpha()
                               and rt.lower() != ct.lower() else "Replacement")
            rows.append(dict(raw_index=ri, raw_token=rt, raw_start=rs,
                             raw_end=re_, corr_index=ci, corr_token=ct,
                             corr_start=cs, corr_end=ce, op="replace",
                             equal_ci=(rt.lower() == ct.lower()),
                             error_type=err))
        elif op == "delete":
            rows.append(dict(raw_index=ri, raw_token=rt, raw_start=rs,
                             raw_end=re_, corr_index=None, corr_token=None,
                             corr_start=None, corr_end=None, op="delete",
                             equal_ci=False,
                             error_type=("PunctuationDeletion"
                                         if not _is_word(rt) else "Deletion")))
        else:  # insert
            rows.append(dict(raw_index=None, raw_token=None, raw_start=None,
                             raw_end=None, corr_index=ci, corr_token=ct,
                             corr_start=cs, corr_end=ce, op="insert",
                             equal_ci=False,
                             error_type=("PunctuationInsertion"
                                         if not _is_word(ct) else "Insertion")))
    return rows


def _taxonomy_row(row):
    """Assign broad_category, error_type, error_subtype to one word-map row.

    Taxonomy (Entry 003):
      broad_category : word | punctuation | capitalisation
      error_type     : Spelling | SentencePunctuation | OtherPunctuation
                       | NounCapitalisation | SentenceStructure | Equal
      error_subtype  : correct | insertion | deletion | changed

    This function operates on the raw values; sentence-punctuation context
    (position within sentence) is enriched later by run_step9 once the
    boundary layer exists.
    """
    op = row.get("op", "equal")
    rt = row.get("raw_token")
    ct = row.get("corr_token")
    err = row.get("error_type", "Equal")

    rt_str = str(rt) if rt is not None else ""
    ct_str = str(ct) if ct is not None else ""

    # Fix 4: gibberish raw token — comprehensible word absent, loss = SS/deletion.
    if _is_gibberish(rt_str):
        return "word", "SentenceStructure", "deletion"

    is_raw_word = _is_word(rt_str)
    is_corr_word = _is_word(ct_str)

    if op == "equal":
        # equal_ci=False means SequenceMatcher matched on lowercase but the
        # actual tokens differ in case — a capitalisation-only change.
        if not row.get("equal_ci", True) and rt_str and ct_str:
            return "capitalisation", "NounCapitalisation", "changed"
        return "word" if is_corr_word else "punctuation", "Equal", "correct"

    if op == "insert":
        if is_corr_word:
            return "word", "SentenceStructure", "insertion"
        return "punctuation", "OtherPunctuation", "insertion"

    if op == "delete":
        if is_raw_word:
            return "word", "SentenceStructure", "deletion"
        return "punctuation", "OtherPunctuation", "deletion"

    # op == "replace"
    if is_raw_word and is_corr_word:
        # Capitalisation-only change?
        if rt_str.lower() == ct_str.lower():
            return "capitalisation", "NounCapitalisation", "changed"
        # Spelling: same intended word, different spelling.
        if err == "Spelling":
            return "word", "Spelling", "changed"
        # Otherwise structural change.
        return "word", "SentenceStructure", "changed"

    if not is_raw_word and not is_corr_word:
        return "punctuation", "OtherPunctuation", "changed"

    # word <-> punct mismatch (should be rare after _typed_realign).
    return "word", "SentenceStructure", "changed"


def run_mapping_only(df_corr, id_col="ID", raw_col="Raw text",
                     corr_col="Corrected text (8)"):
    need = {raw_col, corr_col}
    missing = need - set(df_corr.columns)
    if missing:
        raise KeyError(f"run_mapping_only missing: {missing}")
    df = df_corr.copy()
    if id_col not in df.columns:
        df[id_col] = pd.RangeIndex(len(df)).astype(str)
    df[id_col] = df[id_col].astype(str)

    rows = []
    for rid, raw, cor in zip(df[id_col].tolist(),
                             df[raw_col].astype(str).tolist(),
                             df[corr_col].astype(str).tolist()):
        mapped = build_word_map(raw, cor)
        if not mapped:
            mapped = [dict(raw_index=np.nan, raw_token=None, raw_start=np.nan,
                           raw_end=np.nan, corr_index=np.nan, corr_token=None,
                           corr_start=np.nan, corr_end=np.nan, op="empty",
                           equal_ci=False, error_type="EmptyText")]
        for r in mapped:
            rows.append({"ID": rid, **r})
    df_map = pd.DataFrame(rows)
    for c in ["corr_index", "corr_start", "corr_end",
              "raw_index", "raw_start", "raw_end"]:
        if c in df_map.columns:
            df_map[c] = pd.to_numeric(df_map[c], errors="coerce")

    # Entry 003: marking taxonomy columns.
    taxonomy = df_map.apply(_taxonomy_row, axis=1, result_type="expand")
    taxonomy.columns = ["broad_category", "error_type", "error_subtype"]
    df_map[["broad_category", "error_type", "error_subtype"]] = taxonomy

    texts = df.copy()
    return df_map, texts


# ----------------------------------------------------------------------
# Step 8C. Sentence IDs
# ----------------------------------------------------------------------

ABBREV = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
          "etc.", "e.g.", "i.e.", "cf.", "fig.", "ex.", "no.", "jan.", "feb.",
          "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.",
          "nov.", "dec."}
TERMINALS = {".", "!", "?", "…", "...", "?!", "!?"}
CLOSERS = {")", "]", "}", "\u201d", "'", "\u00bb"}
OPENERS = {"(", "[", "{", "\u201c", "'", "\u00ab"}
_RE_INITIAL = re.compile(r"^[A-Z]\.$")
_RE_NUM_DOT = re.compile(r"^\d+\.$")
_RE_ELLIPSIS = re.compile(r"^\.\.\.$")


def _tok(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)


def _is_terminal_token(tok, prev_tok, next_tok):
    t = tok.strip()
    if not t:
        return False
    if _RE_ELLIPSIS.fullmatch(t) or t in {"...", "?!", "!?", "…"}:
        return True
    if t in {".", "!", "?"}:
        p = (prev_tok or "").strip()
        n = (next_tok or "").strip()
        if p.lower() in ABBREV:
            return False
        if _RE_INITIAL.fullmatch(p) or _RE_NUM_DOT.fullmatch(p):
            return False
        if len(p) == 1 and p.isalpha():
            return False
        if n.isdigit():
            return False
        return True
    return False


def assign_corr_sentence_ids(df_map):
    df = df_map.copy()
    df["ID"] = df.get("ID", df.index.astype(str)).astype(str)
    if "corr_index" not in df.columns:
        df["corr_index"] = np.nan
    df["_rowpos"] = np.arange(len(df))
    df["_sort"] = (pd.to_numeric(df["corr_index"], errors="coerce")
                   .fillna(1e12) + df["_rowpos"] * 1e-9)

    out_ids = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"),
                        index=df.index)
    for ID, g in df.sort_values(["ID", "_sort"], kind="mergesort").groupby(
            "ID", sort=False):
        toks = (g["corr_token"] if "corr_token" in g.columns
                else g["raw_token"]).map(_tok).tolist()
        sids = []
        sid = 1  # 0 is reserved for titles
        pending = False
        for i, raw_tok in enumerate(toks):
            t = raw_tok.strip()
            prev_tok = toks[i - 1].strip() if i > 0 else ""
            next_tok = toks[i + 1].strip() if i + 1 < len(toks) else ""
            if pending:
                if t in CLOSERS:
                    sids.append(sid)
                    continue
                sid += 1
                pending = False
                sids.append(sid)
            else:
                sids.append(sid)
            if _is_terminal_token(t, prev_tok, next_tok):
                pending = True
        out_ids.loc[g.index] = pd.array(sids, dtype="Int64")
    df["CorrSentenceID"] = out_ids
    df.drop(columns=["_rowpos", "_sort"], inplace=True, errors="ignore")
    return df


# ----------------------------------------------------------------------
# Step 8D / 8E. Title+dialogue marking and boundary flags
# ----------------------------------------------------------------------

OPENING_PUNCT = {'"', "\u201c", "'", "\u00ab", "(", "[", "{"}

# Tunable threshold for TextualArtifact detection (marker calibrates this).
ARTIFACT_THRESHOLD = 0.45

_FORMULAIC_ENDING_RX = re.compile(
    r"^(the\s+end|to\s+be\s+continued|fin\.?|the\s+end\.)$", re.I)


def _is_cutoff(tokens):
    """True if this final fragment is a cut-off: 0 word tokens, or exactly
    1 word token with no terminal punctuation.  Single-word/pure-punctuation
    trailing fragments are not sentence attempts and must not be scored."""
    str_toks = [t for t in tokens if isinstance(t, str) and t]
    words = [t for t in str_toks if _is_word(t)]
    has_terminal = any(t in TERMINALS for t in str_toks)
    if len(words) == 0:
        return True
    if len(words) == 1 and not has_terminal:
        return True
    return False


def _score_artifact(tokens, position):
    """Score a sentence's tokens for title/ending likelihood.

    position: "first" (potential TITLE) or "last" (potential ENDING).
    Returns (artifact_type, score) where artifact_type is "TITLE", "ENDING",
    or "" and score is in [0, 1].
    """
    str_toks = [t for t in tokens if isinstance(t, str) and t]
    words = [t for t in str_toks if _is_word(t)]
    n_words = len(words)
    sentence_text = " ".join(str_toks)

    # Formulaic ending is near-decisive on its own.
    if position == "last" and _FORMULAIC_ENDING_RX.fullmatch(sentence_text.strip()):
        return "ENDING", 1.0

    score = 0.0

    # Fragment length clue.
    if n_words == 1:
        score += 0.35
    elif n_words <= 4:
        score += 0.20
    elif n_words <= 6:
        score += 0.05
    else:
        score -= 0.10

    # ALL CAPS (multi-char words only, to ignore "I").
    long_words = [w for w in words if len(w) > 1]
    if long_words and all(w.isupper() for w in long_words):
        score += 0.30

    # Title Case (every alpha word starts with upper).
    alpha_words = [w for w in words if w.isalpha()]
    if alpha_words and all(w[0].isupper() for w in alpha_words):
        score += 0.15

    # No terminal punctuation.
    if not any(t in TERMINALS for t in str_toks):
        score += 0.10

    score = max(0.0, min(score, 1.0))
    if score >= ARTIFACT_THRESHOLD:
        return ("ENDING" if position == "last" else "TITLE"), score
    return "", score


def _loads_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            return json.loads(x)
        except Exception:
            return []
    return []


def mark_title_and_dialogue(df_map, df_texts):
    df = df_map.copy()
    if "Sentence Boundaries" not in df.columns:
        df["Sentence Boundaries"] = ""
    df["TITLE"] = False
    df["DIALOGUE"] = False
    df["TextualArtifact"] = ""
    if "ID" not in df.columns:
        df["ID"] = df.index.astype(str)

    tags_by_id = {str(i): _loads_list(t) for i, t in zip(
        df_texts["ID"].astype(str), df_texts.get("NarrativeTagsJSON", []))}
    dlg_by_id = {str(i): _loads_list(t) for i, t in zip(
        df_texts["ID"].astype(str), df_texts.get("DialogueSpansJSON", []))}

    parts = []
    for ID, g in df.groupby("ID", sort=False):
        g = g.sort_values(["CorrSentenceID", "corr_index"],
                          kind="mergesort").copy()
        ID = str(ID)

        # --- explicit JSON tags (from upstream NLP) ---
        for tag in tags_by_id.get(ID, []):
            if isinstance(tag, dict) and tag.get("type") == "title":
                st, en = int(tag.get("start", 0)), int(tag.get("end", 0))
                mask = (g["corr_start"] >= st) & (g["corr_end"] <= en)
                if mask.any():
                    g.loc[mask, "TITLE"] = True
                    g.loc[mask, "TextualArtifact"] = "TITLE"
        for sp in dlg_by_id.get(ID, []):
            if isinstance(sp, dict):
                st, en = int(sp.get("start", 0)), int(sp.get("end", 0))
                mask = (g["corr_start"] < en) & (g["corr_end"] > st)
                if mask.any():
                    g.loc[mask, "DIALOGUE"] = True

        # --- weighted-clue detection for first and last sentence groups ---
        sent_ids = sorted(g["CorrSentenceID"].dropna().unique())
        for position, sid in [("first", sent_ids[0] if sent_ids else None),
                               ("last",  sent_ids[-1] if sent_ids else None)]:
            if sid is None:
                continue
            # Skip if already flagged by explicit tags.
            grp = g[g["CorrSentenceID"] == sid]
            if grp["TextualArtifact"].any():
                continue
            toks = grp["corr_token"].fillna("").astype(str).tolist()
            # CUTOFF check (last position only): 0-word or 1-word fragment
            # with no terminal punctuation is excluded, not scored.
            if position == "last" and _is_cutoff(toks):
                g.loc[grp.index, "TextualArtifact"] = "CUTOFF"
                continue
            artifact_type, _ = _score_artifact(toks, position)
            if artifact_type:
                g.loc[grp.index, "TextualArtifact"] = artifact_type
                if artifact_type == "TITLE":
                    g.loc[grp.index, "TITLE"] = True
                    # Reserve CorrSentenceID 0 for titles (Entry 002).
                    g.loc[grp.index, "CorrSentenceID"] = 0

        g.loc[g["TITLE"], "Sentence Boundaries"] = "Title"
        parts.append(g)
    df = pd.concat(parts).reset_index(drop=True)

    def _sid3(x):
        try:
            return f"{int(x):03d}"
        except Exception:
            return "000"
    df["SentenceRef"] = (df["ID"].astype(str) + "_s"
                         + df["CorrSentenceID"].map(_sid3))
    return df


def _begins_with_upper_raw(raw_tok):
    s = str(raw_tok or "")
    m = re.search(r"[A-Za-z]", s)
    if not m:
        return None
    return s[m.start()].isupper()


def add_sentence_boundary_flags(df_map):
    df = df_map.copy()
    for c in ("Sentence Boundaries", "BoundaryCheck"):
        if c not in df.columns:
            df[c] = ""
    if "corr_index" not in df.columns:
        df["corr_index"] = np.nan
    df["_rowpos"] = np.arange(len(df))
    df["_sort"] = (pd.to_numeric(df["corr_index"], errors="coerce")
                   .fillna(1e12) + df["_rowpos"] * 1e-9)
    df = df.sort_values(["ID", "CorrSentenceID", "_sort"], kind="mergesort")

    def first_content_row(g):
        for idx, tok in zip(g.index, g["corr_token"].fillna("").astype(str)):
            if tok in OPENING_PUNCT:
                continue
            if _is_word(tok):
                return idx
        return None

    def last_terminal_row(g):
        toks = g["corr_token"].fillna("").astype(str).tolist()
        for pos in range(len(toks) - 1, -1, -1):
            if toks[pos] in TERMINALS:
                return g.index[pos]
        return None

    for (_id, _sid), g in df.groupby(["ID", "CorrSentenceID"], sort=False):
        artifact = g["TextualArtifact"].iloc[0] if "TextualArtifact" in g.columns else ""
        if artifact or g["TITLE"].all():
            # No CorrectBeginning / CorrectEnding for titles or endings.
            continue
        g = g.sort_values("_sort", kind="mergesort")
        b = first_content_row(g)
        e = last_terminal_row(g)
        if b is not None:
            prev = df.at[b, "Sentence Boundaries"]
            df.at[b, "Sentence Boundaries"] = (
                prev + (" | " if prev else "") + "Sentence Beginning")
            cap = _begins_with_upper_raw(df.at[b, "raw_token"]
                                         if "raw_token" in df.columns else None)
            tag = ("Correct Beginning" if cap is True else
                   "Incorrect Beginning" if cap is False else
                   "Unknown Beginning")
            prevc = df.at[b, "BoundaryCheck"]
            df.at[b, "BoundaryCheck"] = prevc + (" | " if prevc else "") + tag
        if e is not None:
            prev = df.at[e, "Sentence Boundaries"]
            df.at[e, "Sentence Boundaries"] = (
                prev + (" | " if prev else "") + "Sentence Ending")
            rawe = str(df.at[e, "raw_token"] or "") if "raw_token" in df.columns else ""
            corre = str(df.at[e, "corr_token"] or "")
            tag = "Correct Ending" if rawe == corre else "Incorrect Ending"
            prevc = df.at[e, "BoundaryCheck"]
            df.at[e, "BoundaryCheck"] = prevc + (" | " if prevc else "") + tag

    df.drop(columns=["_rowpos", "_sort"], inplace=True, errors="ignore")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# Step 8.5  INPUT-QUALITY GATE
# ----------------------------------------------------------------------

_PROMPT_MARKERS = re.compile(
    r"\b(write a (narrative|story|recount|persuasive)|your story could be|"
    r"plan your (writing|story)|you (could|can|may) (write|include)|"
    r"think about (what|who|where)|use the (picture|stimulus)|"
    r"information orbs?)\b", re.I)


def assess_input_quality(raw_text):
    s = str(raw_text or "")
    stripped = s.strip()
    reasons = []

    words = re.findall(r"\w+", stripped)
    n_words = len(words)
    alpha_chars = sum(c.isalpha() for c in stripped)
    total_chars = max(len(stripped), 1)
    alpha_ratio = alpha_chars / total_chars

    if n_words < 15:
        reasons.append(f"too short for narrative scoring ({n_words} words)")
    if alpha_ratio < 0.55:
        reasons.append(f"low alphabetic ratio {alpha_ratio:.2f} "
                       "(likely OCR debris or garbled capture)")
    if _PROMPT_MARKERS.search(stripped):
        reasons.append("contains writing-prompt instruction language")

    if words:
        tiny = sum(1 for w in words if len(w) <= 2)
        if tiny / len(words) > 0.45:
            reasons.append(f"high proportion of 1-2 char tokens "
                           f"({tiny}/{len(words)}) suggests broken capture")

    if not reasons:
        return "ok", []
    hard = any("low alphabetic" in r for r in reasons)
    return ("quarantine" if hard else "review"), reasons


def run_input_quality_gate(df_texts, raw_col="Raw text"):
    df = df_texts.copy()
    verdicts, reason_json = [], []
    for raw in df[raw_col].astype(str):
        v, rs = assess_input_quality(raw)
        verdicts.append(v)
        reason_json.append(json.dumps(rs, ensure_ascii=False))
    df["InputQuality"] = verdicts
    df["InputQualityReasons"] = reason_json
    return df


# ----------------------------------------------------------------------
# Step 9. Sentence summariser
# ----------------------------------------------------------------------

NO_SPACE_BEFORE = set(list(".,;:!?)]}\"'\u00bb\u201d\u2019\u2026"))
NO_SPACE_AFTER = set(list("([{\"'\u00ab\u201c\u2018"))


def _detok(tokens):
    out = []
    for t in tokens:
        if t is None or (isinstance(t, float) and math.isnan(t)):
            continue
        t = str(t)
        if not out:
            out.append(t)
            continue
        prev = out[-1]
        if t in NO_SPACE_BEFORE or re.fullmatch(r"[.]{3}", t):
            out[-1] = prev + t
        elif prev in NO_SPACE_AFTER:
            out[-1] = prev + t
        else:
            out.append(" " + t)
    return re.sub(r"\.\s*\.\s*\.", "...", "".join(out)).strip()


def _summarize_sentence(g):
    corr_tokens = g["corr_token"].tolist()
    raw_tokens = [x for x in g.get("raw_token", pd.Series([], dtype=object)
                                   ).tolist() if not pd.isna(x)]

    artifact = ""
    if "TextualArtifact" in g.columns:
        vals = g["TextualArtifact"].dropna()
        artifact = str(vals.iloc[0]) if not vals.empty else ""

    # Boundary verdicts are blank for textual artifacts (Entry 001).
    begin_ok = np.nan
    end_ok = np.nan
    if not artifact:
        b_rows = g[g["Sentence Boundaries"].astype(str).str.contains(
            "Sentence Beginning", na=False)]
        e_rows = g[g["Sentence Boundaries"].astype(str).str.contains(
            "Sentence Ending", na=False)]
        if not b_rows.empty:
            chk = " | ".join(b_rows["BoundaryCheck"].dropna().astype(str))
            begin_ok = 1 if "Correct Beginning" in chk else (
                0 if "Incorrect Beginning" in chk else np.nan)
        if not e_rows.empty:
            chk = " | ".join(e_rows["BoundaryCheck"].dropna().astype(str))
            end_ok = 1 if "Correct Ending" in chk else (
                0 if "Incorrect Ending" in chk else np.nan)

    ops = g.get("op", pd.Series([], dtype=object))
    return pd.Series({
        "SentenceRef": g["SentenceRef"].iloc[0],
        "TextualArtifact": artifact,
        "CorrectedSentence": _detok(corr_tokens),
        "RawSentence": _detok(raw_tokens) if raw_tokens else "",
        "TokensInSentence": int(len(g)),
        "EditsInSentence": int((ops != "equal").sum()) if not ops.empty else np.nan,
        "EqualsInSentence": int((ops == "equal").sum()) if not ops.empty else np.nan,
        "Insertions": int((ops == "insert").sum()) if not ops.empty else np.nan,
        "Deletions": int((ops == "delete").sum()) if not ops.empty else np.nan,
        "Replacements": int((ops == "replace").sum()) if not ops.empty else np.nan,
        "CorrectBeginning": begin_ok,
        "CorrectEnding": end_ok,
    })


def run_step9(df_map, df_texts):
    need = {"ID", "CorrSentenceID", "corr_token", "Sentence Boundaries",
            "BoundaryCheck", "SentenceRef", "TITLE"}
    missing = need - set(df_map.columns)
    if missing:
        raise KeyError(f"df_map missing for Step 9: {missing}")
    sort_cols = ["ID", "CorrSentenceID"]
    if "corr_index" in df_map.columns:
        sort_cols.append("corr_index")
    wm = df_map.sort_values(sort_cols, kind="mergesort").copy()
    # Keep all rows including artifacts; _summarize_sentence handles them.

    q = df_texts.set_index("ID")[["InputQuality"]] if "InputQuality" \
        in df_texts.columns else None

    out = []
    for (ID, _sid), g in wm.groupby(["ID", "CorrSentenceID"], sort=False):
        rec = _summarize_sentence(g)
        if q is not None and ID in q.index:
            rec["InputQuality"] = q.loc[ID, "InputQuality"]
        else:
            rec["InputQuality"] = "ok"
        out.append(rec)
    sent_df = (pd.DataFrame(out)
               .sort_values("SentenceRef", kind="mergesort")
               .reset_index(drop=True))
    return sent_df


# ----------------------------------------------------------------------
# Pipeline entry point
# ----------------------------------------------------------------------

def _refine_capitalisation_taxonomy(df_map):
    """Re-route sentence-initial capitalisation changes to SentencePunctuation.

    Must run AFTER add_sentence_boundary_flags (Entry 007). At taxonomy
    build time the boundary layer does not exist yet, so all capitalisation
    changes were provisionally assigned NounCapitalisation. Now that
    Sentence Boundaries is populated, any NounCapitalisation token that
    sits at a Sentence Beginning is re-routed to SentencePunctuation /
    punctuation / insertion.
    """
    df = df_map.copy()
    sb = df.get("Sentence Boundaries", pd.Series("", index=df.index))
    is_sent_begin = sb.fillna("").astype(str).str.contains("Sentence Beginning",
                                                            na=False)
    mask = (df["error_type"] == "NounCapitalisation") & is_sent_begin
    df.loc[mask, "broad_category"] = "punctuation"
    df.loc[mask, "error_type"]     = "SentencePunctuation"
    df.loc[mask, "error_subtype"]  = "insertion"
    return df


def run_pipeline(df_preprocessed, corrector=mock_corrector,
                 raw_col="Raw text", id_col="ID"):
    df_corr = run_correct_only(df_preprocessed, corrector,
                               text_col=raw_col, id_col=id_col)
    df_map, df_texts = run_mapping_only(df_corr, id_col=id_col,
                                        raw_col=raw_col)
    df_map = assign_corr_sentence_ids(df_map)
    df_map = mark_title_and_dialogue(df_map, df_texts)
    df_map = add_sentence_boundary_flags(df_map)
    df_map = _refine_capitalisation_taxonomy(df_map)   # Entry 007
    df_texts = run_input_quality_gate(df_texts, raw_col=raw_col)
    sent_df = run_step9(df_map, df_texts)
    return {
        "version": CANONICAL_VERSION,
        "corrector": getattr(corrector, "__name__", "corrector"),
        "df_texts": df_texts,
        "df_map": df_map,
        "sent_df": sent_df,
    }
