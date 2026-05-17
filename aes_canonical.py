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

CANONICAL_VERSION = "aes_canonical_2026-05-17_r1"

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


def _simple_tokenize(s):
    return re.findall(r"\w+|[^\w\s]", s or "", flags=re.UNICODE)


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


def build_word_map(raw_text, corr_text):
    raw_text = str(raw_text or "")
    corr_text = str(corr_text or "")
    raw_tokens = _simple_tokenize(raw_text)
    corr_tokens = _simple_tokenize(corr_text)
    raw_spans = _rebuild_offsets(raw_text, raw_tokens)
    corr_spans = _rebuild_offsets(corr_text, corr_tokens)

    sm = SequenceMatcher(a=[t.lower() for t in raw_tokens],
                         b=[t.lower() for t in corr_tokens],
                         autojunk=False)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                rt, ct = raw_tokens[i1 + k], corr_tokens[j1 + k]
                rs, re_ = raw_spans[i1 + k]
                cs, ce = corr_spans[j1 + k]
                rows.append(dict(raw_index=i1 + k, raw_token=rt, raw_start=rs,
                                 raw_end=re_, corr_index=j1 + k, corr_token=ct,
                                 corr_start=cs, corr_end=ce, op="equal",
                                 equal_ci=(rt == ct), error_type="Equal"))
        elif tag == "replace":
            m = min(i2 - i1, j2 - j1)
            for k in range(m):
                rt, ct = raw_tokens[i1 + k], corr_tokens[j1 + k]
                rs, re_ = raw_spans[i1 + k]
                cs, ce = corr_spans[j1 + k]
                err = ("Spelling" if (rt.lower() != ct.lower()
                       and rt.isalpha() and ct.isalpha()) else "Replacement")
                rows.append(dict(raw_index=i1 + k, raw_token=rt, raw_start=rs,
                                 raw_end=re_, corr_index=j1 + k, corr_token=ct,
                                 corr_start=cs, corr_end=ce, op="replace",
                                 equal_ci=(rt.lower() == ct.lower()),
                                 error_type=err))
            for k in range(i1 + m, i2):
                rt = raw_tokens[k]
                rs, re_ = raw_spans[k]
                rows.append(dict(raw_index=k, raw_token=rt, raw_start=rs,
                                 raw_end=re_, corr_index=None, corr_token=None,
                                 corr_start=None, corr_end=None, op="delete",
                                 equal_ci=False,
                                 error_type=("PunctuationDeletion"
                                             if not _is_word(rt) else "Deletion")))
            for k in range(j1 + m, j2):
                ct = corr_tokens[k]
                cs, ce = corr_spans[k]
                rows.append(dict(raw_index=None, raw_token=None, raw_start=None,
                                 raw_end=None, corr_index=k, corr_token=ct,
                                 corr_start=cs, corr_end=ce, op="insert",
                                 equal_ci=False,
                                 error_type=("PunctuationInsertion"
                                             if not _is_word(ct) else "Insertion")))
        elif tag == "delete":
            for k in range(i1, i2):
                rt = raw_tokens[k]
                rs, re_ = raw_spans[k]
                rows.append(dict(raw_index=k, raw_token=rt, raw_start=rs,
                                 raw_end=re_, corr_index=None, corr_token=None,
                                 corr_start=None, corr_end=None, op="delete",
                                 equal_ci=False,
                                 error_type=("PunctuationDeletion"
                                             if not _is_word(rt) else "Deletion")))
        elif tag == "insert":
            for k in range(j1, j2):
                ct = corr_tokens[k]
                cs, ce = corr_spans[k]
                rows.append(dict(raw_index=None, raw_token=None, raw_start=None,
                                 raw_end=None, corr_index=k, corr_token=ct,
                                 corr_start=cs, corr_end=ce, op="insert",
                                 equal_ci=False,
                                 error_type=("PunctuationInsertion"
                                             if not _is_word(ct) else "Insertion")))
    return rows


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
        sid = 0
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
        for tag in tags_by_id.get(ID, []):
            if isinstance(tag, dict) and tag.get("type") == "title":
                st, en = int(tag.get("start", 0)), int(tag.get("end", 0))
                mask = (g["corr_start"] >= st) & (g["corr_end"] <= en)
                if mask.any():
                    g.loc[mask, "TITLE"] = True
        for sp in dlg_by_id.get(ID, []):
            if isinstance(sp, dict):
                st, en = int(sp.get("start", 0)), int(sp.get("end", 0))
                mask = (g["corr_start"] < en) & (g["corr_end"] > st)
                if mask.any():
                    g.loc[mask, "DIALOGUE"] = True
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
        if g["TITLE"].all():
            df.loc[g.index, "Sentence Boundaries"] = "Title"
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
    hard = any("prompt instruction" in r or "low alphabetic" in r
               for r in reasons)
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
    b_rows = g[g["Sentence Boundaries"].astype(str).str.contains(
        "Sentence Beginning", na=False)]
    e_rows = g[g["Sentence Boundaries"].astype(str).str.contains(
        "Sentence Ending", na=False)]

    begin_ok = np.nan
    end_ok = np.nan
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
    wm = wm[~wm["TITLE"].astype(bool)].copy()

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

def run_pipeline(df_preprocessed, corrector=mock_corrector,
                 raw_col="Raw text", id_col="ID"):
    df_corr = run_correct_only(df_preprocessed, corrector,
                               text_col=raw_col, id_col=id_col)
    df_map, df_texts = run_mapping_only(df_corr, id_col=id_col,
                                        raw_col=raw_col)
    df_map = assign_corr_sentence_ids(df_map)
    df_map = mark_title_and_dialogue(df_map, df_texts)
    df_map = add_sentence_boundary_flags(df_map)
    df_texts = run_input_quality_gate(df_texts, raw_col=raw_col)
    sent_df = run_step9(df_map, df_texts)
    return {
        "version": CANONICAL_VERSION,
        "corrector": getattr(corrector, "__name__", "corrector"),
        "df_texts": df_texts,
        "df_map": df_map,
        "sent_df": sent_df,
    }
