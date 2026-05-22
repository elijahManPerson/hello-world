"""
mech_v2.py
==========
Additive layer on top of aes_canonical.  It does NOT modify your original
file.  It does three things:

  1. Fixes the tokeniser so contractions ("don't", "Let's") stay one token.
  2. Teaches the sentence splitter that straight double quotes are quotes,
     so dialogue stops fragmenting.
  3. Adds the agreed schema: a token broad category + subcategory, plus the
     twelve columns (4 families x error / subtype / class).

Cheap, reliable columns are filled here with pandas only.  The two columns
that need a part-of-speech / grammar engine are written as the literal
string ENGINE so it is obvious they are deliberately deferred, not missing:
  - broad subcategory for word / contraction rows (the word class)
  - Sentence structure class (the grammatical subtype)

Run validate_on_r5() to see before/after on the saved r5 corrected text
WITHOUT calling any API.
"""

import re
import pandas as pd
import numpy as np
import aes_canonical as aes

ENGINE = "TBD(engine)"

# ----------------------------------------------------------------------
# 1.  Apostrophe-aware tokeniser
# ----------------------------------------------------------------------
# Original split '\w+' so "don't" became don / ' / t.  This keeps an
# apostrophe that sits between letters inside the word, and also handles
# trailing possessive  dogs'  and leading  'cause / 'em.
_TOKEN_RX = re.compile(
    r"\d+(?:[.,]\d+)*(?!\w)"          # numbers: 1,000  3.14
    r"|[A-Za-z]+(?:['’][A-Za-z]+)+['’]?"   # true contraction: letter ' letter
    r"|[A-Za-z]*s['’](?![A-Za-z])"              # plural possessive: dogs'
    r"|[A-Za-z]+"                      # plain word
    r"|\w+"                            # any other word char run
    r"|\.\.\.|[^\w\s]",               # ellipsis or single punctuation
    flags=re.UNICODE,
)
# Note: a *leading* apostrophe (e.g. 'cause, or an opening single quote glued
# to a word) is now its own punctuation token. This is the safe choice for
# dialogue-heavy school data; rare genuine elisions lose the apostrophe nuance.

def tokenize(s):
    return _TOKEN_RX.findall(s or "")

_APOS = ("'", "’")

def _has_internal_apostrophe(tok):
    if not tok or len(tok) < 3:
        return False
    # letter ' letter  (don't, Let's)  OR  letters'  (dogs')
    if re.search(r"[A-Za-z]['’][A-Za-z]", tok):
        return True
    if re.fullmatch(r"[A-Za-z]*s['’]", tok):
        return True
    return False


# ----------------------------------------------------------------------
# 2.  Build word map with the fixed tokeniser, reusing the proven aligner
# ----------------------------------------------------------------------
def build_word_map_fixed(raw_text, corr_text):
    raw_text = aes._normalise_raw_spacing(str(raw_text or ""))
    corr_text = str(corr_text or "")
    rtok, ctok = tokenize(raw_text), tokenize(corr_text)
    rsp = aes._rebuild_offsets(raw_text, rtok)
    csp = aes._rebuild_offsets(corr_text, ctok)
    rows = []
    for op, ri, ci in aes._align(rtok, ctok):
        rt = rtok[ri] if ri is not None else None
        ct = ctok[ci] if ci is not None else None
        rs, re_ = rsp[ri] if ri is not None else (None, None)
        cs, ce = csp[ci] if ci is not None else (None, None)
        rows.append(dict(raw_index=ri, raw_token=rt, raw_start=rs, raw_end=re_,
                         corr_index=ci, corr_token=ct, corr_start=cs,
                         corr_end=ce, op=op,
                         equal_ci=(str(rt).lower() == str(ct).lower()
                                   if rt is not None and ct is not None
                                   else (rt == ct))))
    return rows


# ----------------------------------------------------------------------
# 3.  Schema enrichment (the cheap columns)
# ----------------------------------------------------------------------
_MARK_NAME = {
    ".": "Full stop", "!": "Exclamation mark", "?": "Question mark",
    ",": "Comma", ";": "Semicolon", ":": "Colon",
    "'": "Apostrophe", "’": "Apostrophe",
    '"': "Quotation mark", "“": "Quotation mark",
    "”": "Quotation mark",
    "(": "Bracket", ")": "Bracket", "[": "Bracket", "]": "Bracket",
    "-": "Hyphen", "—": "Dash", "–": "Dash",
    "...": "Ellipsis", "…": "Ellipsis",
}

# Tiny starter homophone set — deliberately small; the engine refines this.
_HOMOPHONES = {
    ("their", "there"), ("there", "their"), ("their", "they're"),
    ("they're", "their"), ("to", "too"), ("too", "to"), ("to", "two"),
    ("your", "you're"), ("you're", "your"), ("its", "it's"),
    ("it's", "its"), ("then", "than"), ("than", "then"),
    ("were", "where"), ("where", "were"), ("of", "off"),
    ("no", "know"), ("new", "knew"), ("hear", "here"), ("here", "hear"),
}

_PUNCT_RX = re.compile(r"^[^\w\s]+$|^\.\.\.$")

# Very common short words where an adjacent-letter reversal is almost
# certainly a slip, not a spelling gap. Kept small on purpose.
_COMMON_SHORT = {
    "the", "and", "a", "to", "of", "in", "is", "it", "was", "for",
    "that", "with", "as", "at", "but", "his", "her", "had", "have",
    "not", "this", "be", "are", "my", "said", "she", "he", "they",
    "we", "you", "on", "so", "if", "or", "an", "from", "then", "them",
}

def _is_obvious_typo(a, b):
    """True only for an unambiguous mechanical slip the cheap layer can be
    sure of: an adjacent-letter reversal that produces a short or very
    common word (teh->the, adn->and, fro->for). Deliberately narrow;
    the engine widens 'Typo' to other not-plausible-as-spelling slips."""
    if not a or not b or len(a) != len(b) or a == b:
        return False
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if len(diff) != 2:
        return False
    i, j = diff
    if j != i + 1:
        return False
    if not (a[i] == b[j] and a[j] == b[i]):     # exactly an adjacent swap
        return False
    return len(b) <= 4 or b in _COMMON_SHORT


def _cat_of(tok):
    """word | contraction | punctuation | numeral/symbol"""
    if tok is None:
        return None
    t = str(tok)
    if _has_internal_apostrophe(t):
        return "contraction"
    if re.fullmatch(r"\d+(?:[.,]\d+)*", t):
        return "numeral/symbol"
    if re.search(r"[A-Za-z]", t) and re.fullmatch(r"[A-Za-z'’]+", t):
        return "word"
    if _PUNCT_RX.match(t):
        return "punctuation"
    return "numeral/symbol"

def _core(tok):
    """letters only, lowercased, apostrophes stripped"""
    return re.sub(r"[^a-z]", "", str(tok or "").lower())

def _apos_signature(tok):
    """(present?, char_before, char_after) for the first apostrophe."""
    t = str(tok or "")
    for i, c in enumerate(t):
        if c in _APOS:
            before = t[i-1].lower() if i > 0 else ""
            after = t[i+1].lower() if i+1 < len(t) else ""
            return (True, before, after)
    return (False, "", "")

_ACTION = {"equal": "Correct", "delete": "Taken away",
           "insert": "Added", "replace": "Changed"}

def _enrich_row(r, sentence_initial):
    rt, ct, op = r.get("raw_token"), r.get("corr_token"), r.get("op", "equal")
    tok_for_cat = ct if ct is not None else rt
    cat = _cat_of(tok_for_cat)
    is_wordish = cat in ("word", "contraction")

    out = {"TokenCategory": cat}

    # ---- broad subcategory ----
    if is_wordish:
        out["TokenSubcat"] = ENGINE          # word class -> engine
    elif cat == "punctuation":
        out["TokenSubcat"] = _MARK_NAME.get(str(tok_for_cat), "Other mark")
    elif cat == "numeral/symbol":
        out["TokenSubcat"] = ("numeral"
                              if re.fullmatch(r"\d+(?:[.,]\d+)*",
                                              str(tok_for_cat or "")) else "symbol")
    else:
        out["TokenSubcat"] = None

    NA = "NA"
    # defaults
    cap_e = cap_s = cap_c = NA
    pun_e = pun_s = pun_c = NA
    spl_e = spl_s = spl_c = NA
    ss_e = ss_s = ss_c = NA

    rt_s, ct_s = (str(rt) if rt is not None else ""), (str(ct) if ct is not None else "")
    action = _ACTION.get(op, "Changed")

    # ===== CAPITALISATION (word/contraction, need both sides) =====
    if is_wordish and op in ("equal", "replace"):
        first_corr_alpha = next((c for c in ct_s if c.isalpha()), "")
        first_raw_alpha = next((c for c in rt_s if c.isalpha()), "")
        corr_cap = first_corr_alpha.isupper()
        raw_cap = first_raw_alpha.isupper()
        is_acronym = ct_s.isupper() and len(_core(ct_s)) > 1
        # what role does a capital play here?
        if is_acronym:
            cap_c = "Abbrev./Acronym"
        elif corr_cap and sentence_initial:
            cap_c = "Boundary capital"
        elif corr_cap and not sentence_initial:
            cap_c = "Noun capital"
        elif (not corr_cap) and raw_cap:
            cap_c = "Stray capital"
        else:
            cap_c = NA                     # ordinary lowercase word, no capital concept
        same_case = (first_raw_alpha == first_corr_alpha)
        if cap_c == NA:
            cap_e, cap_s = "FALSE", "Correct"
        else:
            cap_e = "FALSE" if same_case else "TRUE"
            cap_s = "Correct" if same_case else "Changed"

    # ===== PUNCTUATION =====
    if cat == "punctuation":
        pun_c = _MARK_NAME.get(str(tok_for_cat), "Other mark")
        pun_e = "FALSE" if op == "equal" else "TRUE"
        pun_s = action
    elif is_wordish:
        # apostrophe placement rule (agreed examples 7 vs 8)
        rp, rb, ra = _apos_signature(rt_s)
        cp, cb, ca = _apos_signature(ct_s)
        if not rp and not cp:
            pun_e = pun_s = pun_c = NA          # no apostrophe in play
        else:
            pun_c = "Apostrophe"
            if rp and cp and (rb, ra) == (cb, ca):
                pun_e, pun_s = "FALSE", "Correct"
            elif rp and not cp:
                pun_e, pun_s = "TRUE", "Taken away"   # extra apostrophe removed
            elif cp and not rp:
                pun_e, pun_s = "TRUE", "Added"        # missing apostrophe added
            else:
                pun_e, pun_s = "TRUE", "Changed"      # misplaced

    # ===== SPELLING (word/contraction only) =====
    if is_wordish:
        # proper-noun guard: capitalised in corrected, not just sentence start.
        # Contractions and the pronoun "I" are never proper nouns.
        corr_cap = bool(ct_s) and ct_s[:1].isupper()
        proper = (corr_cap and not sentence_initial and not ct_s.isupper()
                  and cat == "word" and _core(ct_s) != "i")
        if proper:
            spl_e, spl_s, spl_c = NA, NA, "ProperNoun-ignored"
        elif op in ("insert", "delete"):
            spl_e = spl_s = spl_c = NA            # add/remove word = SS, not spelling
        else:
            cr, cc = _core(rt_s), _core(ct_s)
            if cr == cc:
                spl_e, spl_s, spl_c = "FALSE", "Correct", "Correct"
            else:
                # Cheap deterministic default: a genuine misspelling reads
                # "Incorrect". "Typo" is reserved for an unambiguous
                # mechanical slip the cheap layer can be sure of: an
                # adjacent-letter reversal of a short/common word
                # (adn, teh). Edit distance alone cannot tell "teh" from
                # "recieve", so the engine widens Typo to other
                # not-plausible-as-spelling slips; the cheap layer stays
                # conservative.
                spl_e, spl_s = "TRUE", "Changed"
                if (cr, cc) in _HOMOPHONES:
                    spl_c = "Homophone"
                elif _is_obvious_typo(cr, cc):
                    spl_c = "Typo"
                else:
                    spl_c = "Incorrect"

    # ===== SENTENCE STRUCTURE =====
    if is_wordish:
        if op == "insert":
            # correction added a word the student left out
            ss_e, ss_s, ss_c = "TRUE", "Added", "missing word"
        elif op == "delete":
            # correction removed a word the student put in
            ss_e, ss_s, ss_c = "TRUE", "Taken away", "extra word"
        elif op == "equal":
            ss_e, ss_s, ss_c = "FALSE", "Correct", NA
        else:
            # replace: in the cheap layer this is treated as a spelling
            # change, so SS is NA (request 3). The engine reassigns the
            # genuine word-form changes (ate->eat): for those it sets SS
            # and flips Spelling to NA, always with a class.
            ss_e, ss_s, ss_c = NA, NA, NA
    elif aes._is_gibberish(rt_s):
        ss_e, ss_s, ss_c = "TRUE", "Taken away", "extra word"

    out.update({
        "Cap error": cap_e, "Cap subtype": cap_s, "Cap class": cap_c,
        "Punc error": pun_e, "Punc subtype": pun_s, "Punc class": pun_c,
        "Spell error": spl_e, "Spell subtype": spl_s, "Spell class": spl_c,
        "SS error": ss_e, "SS subtype": ss_s, "SS class": ss_c,
    })
    return out


def enrich(df_map):
    """Add schema columns. df_map must have CorrSentenceID + corr_index so we
    know which token is sentence-initial."""
    df = df_map.copy()
    # sentence-initial = first word-ish token of each (Identifier, CorrSentenceID)
    df["_init"] = False
    key = ([c for c in ["Identifier", "CorrSentenceID"] if c in df.columns]
           or ["Identifier"])
    for _, g in df.groupby(key, sort=False):
        gg = g.sort_values("corr_index", kind="mergesort")
        for idx in gg.index:
            tok = gg.at[idx, "corr_token"]
            if tok is not None and re.search(r"[A-Za-z]", str(tok)):
                df.at[idx, "_init"] = True
                break
    recs = [_enrich_row(r, r["_init"]) for _, r in df.iterrows()]
    add = pd.DataFrame(recs, index=df.index)
    df = pd.concat([df.drop(columns=["_init"]), add], axis=1)
    return df


# ----------------------------------------------------------------------
# 4.  One clean entry point  (handles all the ID -> Identifier juggling)
# ----------------------------------------------------------------------
def run_layer(texts, id_col="Research ID",
              raw_col="Raw text", corr_col="Corrected text (8)"):
    """Run the fixed mechanical layer + schema on a texts dataframe.

    Returns df_map with the script identifier in a column called
    'Identifier'.  Your original aes_canonical is not modified, and the
    IDeas criterion column 'ID' (if present) is left untouched.
    """
    # Proper quote-aware segmentation now handles this (see section 7);
    # the old constant-patching hack is removed.

    t = texts.copy()
    if id_col not in t.columns:
        raise KeyError(f"id_col {id_col!r} not found. Columns: {list(t.columns)}")
    t["Identifier"] = t[id_col].astype(str)

    # word map — _ap (alignment position, per-script) is kept for Section 8
    rows = []
    for _, r in t.iterrows():
        for ap, d in enumerate(build_word_map_fixed(r[raw_col], r[corr_col])):
            rows.append({"Identifier": r["Identifier"], "_ap": ap, **d})
    wm = pd.DataFrame(rows)

    # aes steps need a column literally called 'ID'. Give them one as an
    # alias, and protect any pre-existing 'ID' (your IDeas criterion).
    wm["ID"] = wm["Identifier"]
    t_aes = t.copy()
    if "ID" in t_aes.columns:
        t_aes = t_aes.rename(columns={"ID": "ID__ideas_criterion"})
    t_aes["ID"] = t_aes["Identifier"]

    wm = assign_corr_sentence_ids_v2(wm)
    wm = _fix_delete_sentence_ids(wm)                # Change 7
    wm = aes.mark_title_and_dialogue(wm, t_aes)
    wm = mark_dialogue_by_quotes(wm)                 # self-contained DIALOGUE
    wm = mark_artifacts_by_model(wm, t_aes)          # title/ending/other
    wm = aes.add_sentence_boundary_flags(wm)

    wm = wm.drop(columns=["ID"], errors="ignore")   # drop the alias
    wm = enrich(wm)                                  # uses 'Identifier'
    wm = _add_boundary_retained_columns(wm)          # Section 8
    wm = inject_paragraph_breaks(wm, texts, id_col=id_col, raw_col=raw_col)
    _idcol = "Identifier" if "Identifier" in wm.columns else "ID"
    wm = (wm.assign(_ap_sort=pd.to_numeric(wm["_ap"], errors="coerce"))
            .sort_values([_idcol, "_ap_sort"], kind="mergesort")
            .drop(columns="_ap_sort")
            .reset_index(drop=True))
    return wm


# ----------------------------------------------------------------------
# 5.  Sentence layer  (six new columns on top of aes.run_step9)
# ----------------------------------------------------------------------
# EDIT THIS ONE LIST to retune column-3 severity. First band = most serious.
# Pronoun is deliberately absent: it lives in its own Pronoun column.
SS_SEVERITY = [
    ["missing word", "extra word", "word order", "verb tense", "verb agreement"],
    ["noun number", "preposition", "conjunction"],
    ["determiner", "adjective", "adverb", "other"],
]

def _most_serious(labels):
    """Pick the single most serious grammar label present (column 3)."""
    labels = [str(x) for x in labels if x and str(x) not in ("NA", "nan")]
    if not labels:
        return "NA"
    if all(x == ENGINE for x in labels):
        return ENGINE
    for band in SS_SEVERITY:
        for lbl in labels:                       # first-in-sentence within band
            if lbl in band:
                return lbl
    return next((x for x in labels if x != ENGINE), ENGINE)

_PAST = {"was","were","had","did","said","went","ran","saw","came","got",
         "took","made","found","told","gave","knew","thought","felt","began",
         "looked","walked","turned","asked","called","wanted","could","would"}
_PRES = {"is","are","am","has","have","do","does","say","says","go","goes",
         "run","runs","see","sees","get","gets","come","comes","want","wants"}

def _sentence_tense(tokens):
    """Crude past/present/unknown for one sentence. Honest first pass only:
    no parser here, so it reads obvious finite-verb cues and -ed endings."""
    p = q = 0
    for t in tokens:
        w = re.sub(r"[^a-z]", "", str(t).lower())
        if not w:
            continue
        if w in _PAST:
            p += 1
        elif w in _PRES:
            q += 1
        elif w.endswith("ed") and len(w) > 3:
            p += 1
    if p == q:
        return "unknown"
    return "past" if p > q else "present"

def enrich_sentences(wm, sent_df):
    """Attach the six sentence-level columns. wm is the enriched word map
    (must have Identifier, CorrSentenceID, SS error, SS class, op,
     corr_token, DIALOGUE). sent_df is the output of aes.run_step9."""
    s = sent_df.copy()

    # group the word map by sentence via SentenceRef
    wm = wm.copy()
    if "SentenceRef" not in wm.columns:
        raise KeyError("word map needs SentenceRef (run mark_title first)")
    by_ref = {r: g for r, g in wm.groupby("SentenceRef", sort=False)}

    # per-script prevailing tense (non-dialogue sentences only)
    script_tense = {}
    for ident, g in wm.groupby("Identifier", sort=False):
        votes = {"past": 0, "present": 0}
        for ref, gg in g.groupby("SentenceRef", sort=False):
            if bool(gg.get("DIALOGUE", pd.Series([False])).any()):
                continue
            t = _sentence_tense(gg["corr_token"].tolist())
            if t in votes:
                votes[t] += 1
        script_tense[ident] = ("past" if votes["past"] >= votes["present"]
                               else "present")

    col1, col2, col3, col4, col5 = [], [], [], [], []
    pron_v, pron_s, tense_v = [], [], []
    prev_ident = None
    for _, row in s.iterrows():
        ref = row["SentenceRef"]
        ident = ref.rsplit("_s", 1)[0]
        g = by_ref.get(ref)
        artifact = str(row.get("TextualArtifact") or "")
        sclass = str(row.get("ScriptClass") or "")
        # Off-genre / foreign / garbled / prompt-copy cannot be fairly
        # scored as narrative, so the verdict columns are NA'd. Concern 1
        # and Concern 2 are deliberately NOT here: a concerning or sweary
        # script is still a narrative and is still scored normally.
        is_art = bool(artifact) or sclass in NON_SCORABLE

        # ---- column 1: internal grammaticality ----
        if g is None or is_art:
            col1.append("NA")
        else:
            bad = (g["SS error"].astype(str) == "TRUE").any()
            col1.append("Incorrect" if bad else "Correct")

        # ---- column 4 + 2: punctuation boundary ----
        if is_art or g is None:
            cat = "NA"
        else:
            b = row.get("CorrectBeginning")
            begin_ok = (None if pd.isna(b) else bool(b))
            term = g[g["corr_token"].astype(str).isin(
                list(aes.TERMINALS))]
            if term.empty:
                end_state = "missing"
            else:
                op = str(term.iloc[-1]["op"])
                end_state = ("correct" if op == "equal"
                             else "missing" if op == "insert"
                             else "wrong")
            if begin_ok is None:
                cat = "NA"
            elif begin_ok and end_state == "correct":
                cat = "Correct"
            elif (not begin_ok) and end_state == "correct":
                cat = "missing cap"
            elif begin_ok and end_state == "missing":
                cat = "missing end boundary"
            elif begin_ok and end_state == "wrong":
                cat = "wrong end boundary"
            else:                                  # begin wrong + end not ok
                cat = "missing all"
        col4.append(cat)
        col2.append("NA" if cat == "NA"
                    else "Correct" if cat == "Correct" else "Incorrect")

        # ---- column 3: most serious grammar subtype ----
        if g is None or is_art:
            col3.append("NA")
        else:
            errs = g[g["SS error"].astype(str) == "TRUE"]
            col3.append(_most_serious(errs["SS class"].tolist())
                        if not errs.empty else "NA")

        # ---- column 5: sentence type (engine) ----
        col5.append("NA" if is_art else ENGINE)

        # ---- Pronoun column (engine) ----
        if is_art or g is None:
            pron_v.append("NA"); pron_s.append("NA")
        else:
            pron_v.append(ENGINE); pron_s.append(ENGINE)

        # ---- Tense drift column (cheap heuristic now) ----
        if is_art or g is None:
            tense_v.append("NA")
        else:
            in_dlg = bool(g.get("DIALOGUE", pd.Series([False])).any())
            t = _sentence_tense(g["corr_token"].tolist())
            if in_dlg or t == "unknown" or ident != prev_ident:
                tense_v.append("NA")               # no fair comparison
            else:
                tense_v.append("Incorrect"
                               if t != script_tense.get(ident) else "Correct")
        prev_ident = ident

    s["SS internal"] = col1
    s["Punc boundary"] = col2
    s["Grammar subtype (most serious)"] = col3
    s["Punc boundary subtype"] = col4
    s["Sentence type"] = col5
    s["Pronoun ref"] = pron_v
    s["Pronoun ref subtype"] = pron_s
    s["Tense drift"] = tense_v
    return s

def run_sentence_layer(texts, id_col="Research ID",
                       raw_col="Raw text", corr_col="Corrected text (8)",
                       precomputed_wm=None):
    """Full path: fixed word layer -> aes.run_step9 -> six sentence columns.
    Returns sent_df with the script id in 'Identifier'.

    Pass precomputed_wm = an already engine-enriched word map (word class
    and SS class filled) so the sentence grammar-subtype rollup resolves
    instead of staying TBD(engine)."""
    wm = (precomputed_wm if precomputed_wm is not None
          else run_layer(texts, id_col=id_col, raw_col=raw_col,
                          corr_col=corr_col))

    # aes.run_step9 needs 'ID'; alias then restore.
    wm_a = wm.copy()
    wm_a["ID"] = wm_a["Identifier"]
    t = texts.copy()
    t["Identifier"] = t[id_col].astype(str)
    if "ID" in t.columns:
        t = t.rename(columns={"ID": "ID__ideas_criterion"})
    t["ID"] = t["Identifier"]
    # NEW: script classification replaces the old quality gate.
    t = classify_scripts(t, raw_col=raw_col)
    t["InputQuality"] = t["ScriptClass"]          # keep run_step9 happy
    sent = aes.run_step9(wm_a, t)
    sent["Identifier"] = sent["SentenceRef"].str.rsplit("_s", n=1).str[0]
    # replace aes._detok output with the quote-aware, state-carrying one
    corr_txt = restate_sentences(wm_a, token_col="corr_token")
    raw_txt = restate_sentences(wm_a, token_col="raw_token")
    sent["CorrectedSentence"] = sent["SentenceRef"].map(corr_txt).fillna(
        sent["CorrectedSentence"])
    sent["RawSentence"] = sent["SentenceRef"].map(raw_txt).fillna(
        sent.get("RawSentence", ""))
    # Change 10 — merge segmentation-artefact sentences (content is
    # punctuation only, no alphabetic characters) into the previous
    # valid sentence in the same script. Mutates wm in place so the
    # word_map and sentences stay aligned on SentenceRef.
    sent = _merge_punctuation_only_sentences(wm, sent)
    # carry the script-level flags onto every sentence row
    cmap = t.set_index("Identifier")
    for col in ["ScriptClass", "ScriptClassReason", "Concern1",
                "Concern1Reason"]:
        sent[col] = sent["Identifier"].map(cmap[col])
    return enrich_sentences(wm, sent)


# ----------------------------------------------------------------------
# 6.  Script classification  (replaces the old quality gate)
# ----------------------------------------------------------------------
# Priority order. Welfare first, then offence, then "can't be scored as
# narrative" reasons, then ok. 'ok' is ONLY ever set by the engine, never
# by the cheap pass, so a missed welfare disclosure is never auto-cleared.
SCRIPT_PRIORITY = [
    "Script of Concern 1",     # welfare  (engine only, by design)
    "Script of Concern 2",     # offence  (cheap profanity + engine nuance)
    "Unscorable/garbled",      # OCR debris / empty   (cheap)
    "Foreign language",        # (cheap heuristic)
    "Off genre",               # not narrative        (engine)
    "Prompt copy",             # echoes the prompt    (cheap markers)
    "ok",                      # engine only
]

# Small, deliberately mild starter list for the *swear word* part of
# Concern 2 only. Hate speech / slurs are NOT a keyword list here: that is
# engine work, because a slur list is both incomplete and harmful to ship.
_PROFANITY = {"damn", "hell", "crap", "bloody", "bastard", "shit",
              "piss", "bugger", "arse"}

_COMMON_EN = {"the","and","a","to","of","in","is","it","was","i","he","she",
              "they","we","you","that","for","on","with","as","at","but",
              "his","her","had","have","not","this","be","are","my","said"}

def _looks_foreign(text):
    s = str(text or "")
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 20:
        return False
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    if non_ascii / len(letters) > 0.30:
        return True
    words = re.findall(r"[A-Za-z']+", s.lower())
    if len(words) >= 12:
        cover = sum(1 for w in words if w in _COMMON_EN) / len(words)
        if cover < 0.08:               # almost no English function words
            return True
    return False

def _looks_garbled(text):
    s = str(text or "").strip()
    if len(s) < 3:
        return True
    alpha = sum(c.isalpha() for c in s)
    if alpha / max(len(s), 1) < 0.45:
        return True
    words = re.findall(r"\w+", s)
    if words and sum(1 for w in words if len(w) <= 2) / len(words) > 0.55:
        return True
    return False

def _has_profanity(text):
    toks = set(re.findall(r"[a-z]+", str(text or "").lower()))
    hit = toks & _PROFANITY
    return sorted(hit) if hit else None

def classify_script_cheap(text):
    """Returns (label, reason). Only assigns labels the cheap pass can be
    sure of. Otherwise label is ENGINE meaning 'engine must decide
    Concern1 / Off-genre / ok'. Never returns 'ok'."""
    prof = _has_profanity(text)
    if prof:
        return "Script of Concern 2", f"profanity: {', '.join(prof)}"
    if _looks_garbled(text):
        return "Unscorable/garbled", "OCR debris / empty / broken capture"
    if _looks_foreign(text):
        return "Foreign language", "low English coverage / non-Latin script"
    if aes._PROMPT_MARKERS.search(str(text or "")):
        return "Prompt copy", "contains writing-prompt instruction language"
    return ENGINE, "engine must check Concern1 / off-genre / ok"

def classify_scripts(texts, raw_col="Raw text"):
    """Adds two columns to a texts-style frame:
       ScriptClass         - the single priority-ordered label
       ScriptClassReason   - why
       Concern1            - dedicated advisory column (ENGINE until engine
                             runs); NEVER read by the scoring layer.
       Concern1Reason
    """
    df = texts.copy()
    labels, reasons = [], []
    for t in df[raw_col].astype(str):
        lbl, why = classify_script_cheap(t)
        labels.append(lbl)
        reasons.append(why)
    df["ScriptClass"] = labels
    df["ScriptClassReason"] = reasons
    # Concern 1 is engine-only by design. Cheap pass never asserts it and
    # never clears it. It stays ENGINE until a real model judgement exists.
    df["Concern1"] = ENGINE
    df["Concern1Reason"] = ENGINE
    return df

# Which script classes mean "cannot be fairly scored as a narrative".
# Concern 1 and 2 are NOT here: a concerning or sweary script is still a
# narrative and is still scored. Welfare never suppresses the score.
NON_SCORABLE = {"Unscorable/garbled", "Foreign language",
                "Off genre", "Prompt copy"}


# ----------------------------------------------------------------------
# 7.  Proper quote-aware sentence segmentation + detok
# ----------------------------------------------------------------------
# Replaces the constant-patching hack. A straight " is an opener or a
# closer depending on whether we are currently inside a quote, so we track
# that with a toggle instead of guessing from the character.
_HARD_CLOSERS = {")", "]", "}", "”", "’", "»"}  # unambiguous

def assign_corr_sentence_ids_v2(df_map):
    df = df_map.copy()
    df["ID"] = df.get("ID", df.index.astype(str)).astype(str)
    if "corr_index" not in df.columns:
        df["corr_index"] = np.nan
    df["_rp"] = np.arange(len(df))
    df["_s"] = (pd.to_numeric(df["corr_index"], errors="coerce")
                .fillna(1e12) + df["_rp"] * 1e-9)
    out = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)

    for ID, g in df.sort_values(["ID", "_s"], kind="mergesort").groupby(
            "ID", sort=False):
        toks = (g["corr_token"] if "corr_token" in g.columns
                else g["raw_token"]).map(_tok_str).tolist()
        sids = []
        sid = 1
        in_dq = False          # inside straight double quote
        in_sq = False          # inside straight single quote
        pending = False
        for i, raw in enumerate(toks):
            t = raw.strip()
            prev = toks[i - 1].strip() if i > 0 else ""
            nxt = toks[i + 1].strip() if i + 1 < len(toks) else ""

            is_dq = (t == '"')
            is_sq = (t == "'")
            dq_closer = is_dq and in_dq
            sq_closer = is_sq and in_sq
            is_closer = (t in _HARD_CLOSERS) or dq_closer or sq_closer

            if pending:
                if is_closer:
                    # a closing mark belongs to the sentence that just ended
                    sids.append(sid)
                    if is_dq:
                        in_dq = not in_dq
                    elif is_sq:
                        in_sq = not in_sq
                    continue
                # anything else (including an OPENING quote) starts the next
                sid += 1
                pending = False
                sids.append(sid)
            else:
                sids.append(sid)

            if is_dq:
                in_dq = not in_dq
            elif is_sq:
                in_sq = not in_sq

            if aes._is_terminal_token(t, prev, nxt):
                pending = True

        out.loc[g.index] = pd.array(sids, dtype="Int64")

    df["CorrSentenceID"] = out
    df.drop(columns=["_rp", "_s"], inplace=True, errors="ignore")
    return df


def _tok_str(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)


def detok_v2(tokens, in_dq_start=False, in_sq_start=False):
    """Quote-aware detokeniser using an explicit state machine.
    Pass in_dq_start=True when this sentence opens already inside a
    double quote that was opened in an earlier sentence."""
    nb = aes.NO_SPACE_BEFORE - {'"', "'"}
    na = aes.NO_SPACE_AFTER - {'"', "'"}
    parts = []
    in_dq, in_sq = in_dq_start, in_sq_start
    glue_next = False
    first = True
    for t in tokens:
        if t is None or (isinstance(t, float) and pd.isna(t)):
            continue
        t = str(t)
        is_dq = (t == '"')
        is_sq = (t == "'")
        opener = (is_dq and not in_dq) or (is_sq and not in_sq)
        closer = (is_dq and in_dq) or (is_sq and in_sq)

        if first:
            space = False
            first = False
        elif glue_next:
            space = False
        elif closer or t in nb or re.fullmatch(r"[.]{3}", t):
            space = False
        else:
            space = True

        parts.append((" " if space else "") + t)

        # set glue for the *next* token
        if opener:
            glue_next = True
        elif closer:
            glue_next = False
        elif t in na:
            glue_next = True
        else:
            glue_next = False

        if is_dq:
            in_dq = not in_dq
        elif is_sq:
            in_sq = not in_sq

    return re.sub(r"\.\ s*\.\s*\.", "...", "".join(parts)).strip()


def restate_sentences(wm, token_col="corr_token"):
    """Return {SentenceRef: text} detokenised with quote state carried
    correctly across sentence boundaries within each script."""
    out = {}
    w = wm.sort_values(["Identifier", "CorrSentenceID", "corr_index"],
                        kind="mergesort")
    for ident, g in w.groupby("Identifier", sort=False):
        in_dq = in_sq = False
        for ref, gg in g.groupby("SentenceRef", sort=False):
            toks = [t for t in gg[token_col].tolist() if t is not None]
            out[ref] = detok_v2(toks, in_dq, in_sq)
            for t in toks:
                if t == '"':
                    in_dq = not in_dq
                elif t == "'":
                    in_sq = not in_sq
    return out


# ----------------------------------------------------------------------
# 8.  Self-contained DIALOGUE detection (no upstream NLP needed)
# ----------------------------------------------------------------------
def mark_dialogue_by_quotes(df_map):
    """Set DIALOGUE from the speech marks themselves, using the same
    quote-state logic as the segmenter. A token is dialogue if it lies
    between a matched pair of straight double quotes, the quote marks
    included. The original aes function only sets DIALOGUE from a
    DialogueSpansJSON column produced by an upstream NLP step; when that
    column is absent it marks nothing, so this restores it."""
    df = df_map.copy()
    if "DIALOGUE" not in df.columns:
        df["DIALOGUE"] = False
    idcol = "Identifier" if "Identifier" in df.columns else "ID"
    out = pd.Series(False, index=df.index)
    for _, g in df.sort_values([idcol, "CorrSentenceID", "corr_index"],
                               kind="mergesort").groupby(idcol, sort=False):
        in_dq = False
        open_idx = None
        for idx in g.index:
            tok = _tok_str(g.at[idx, "corr_token"])
            if tok == '"':
                if not in_dq:                  # opening quote
                    in_dq = True
                    open_idx = idx
                    out.at[idx] = True
                else:                          # closing quote
                    in_dq = False
                    out.at[idx] = True
                    open_idx = None
            elif in_dq:
                out.at[idx] = True
        # an unclosed quote: leave its trailing span marked (best effort)
    df["DIALOGUE"] = out.reindex(df.index).fillna(False).astype(bool)
    return df


# ----------------------------------------------------------------------
# 9.  Model-driven TITLE / ENDING / OTHER marking (self-contained)
# ----------------------------------------------------------------------
import json as _json

def _sid3(x):
    try:
        return f"{int(x):03d}"
    except Exception:
        return "000"

def _is_wordish_tok(tok):
    return bool(re.search(r"[A-Za-z0-9]", _tok_str(tok)))

def mark_artifacts_by_model(df_map, texts):
    """Apply title_detector output (TitleWordCount / EndingWordCount /
    OtherSpansJSON on `texts`) to the word map. Runs AFTER aes (which
    resets these columns), then rebuilds SentenceRef so the title is its
    own sentence 0 and the real first sentence is no longer contaminated.

    No-op if the columns are absent: the existing heuristic stays in
    charge, nothing is deleted. Caps guard against a bad model count
    swallowing the whole script."""
    if texts is None or "TitleWordCount" not in getattr(texts, "columns", []):
        return df_map
    df = df_map.copy()
    for c in ("TITLE", "TextualArtifact"):
        if c not in df.columns:
            df[c] = False if c == "TITLE" else ""
    idcol = "Identifier" if "Identifier" in df.columns else "ID"
    tdf = texts.copy()
    tcol = "Identifier" if "Identifier" in tdf.columns else "ID"
    tdf[tcol] = tdf[tcol].astype(str)
    by = tdf.set_index(tcol)

    for ID, g in df.groupby(idcol, sort=False):
        ID = str(ID)
        if ID not in by.index:
            continue
        row = by.loc[ID]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        tw, ew = row.get("TitleWordCount", 0), row.get("EndingWordCount", 0)
        if tw == ENGINE or ew == ENGINE:           # mock run
            continue
        g = g.sort_values("corr_index", kind="mergesort")
        idxs = g.index.tolist()
        word_idxs = [i for i in idxs if _is_wordish_tok(g.at[i, "corr_token"])]
        n_words = len(word_idxs)

        # ----- TITLE: first tw words (+ any punctuation up to them) -----
        try:
            tw = int(tw)
        except Exception:
            tw = 0
        if 0 < tw <= 20 and tw < n_words:          # caps: short, not whole
            last_title_word = word_idxs[tw - 1]
            cut = idxs.index(last_title_word)
            title_idx = idxs[:cut + 1]
            df.loc[title_idx, "TITLE"] = True
            df.loc[title_idx, "TextualArtifact"] = "TITLE"
            df.loc[title_idx, "CorrSentenceID"] = 0

        # ----- ENDING: last ew words (+ trailing punctuation) -----
        try:
            ew = int(ew)
        except Exception:
            ew = 0
        if 0 < ew <= 10 and ew < n_words:
            first_end_word = word_idxs[-ew]
            cut = idxs.index(first_end_word)
            end_idx = idxs[cut:]
            # don't overwrite a title
            end_idx = [i for i in end_idx
                       if df.at[i, "TextualArtifact"] != "TITLE"]
            df.loc[end_idx, "TextualArtifact"] = "ENDING"

        # ----- OTHER: only clearly non-story, located by text -----
        spans = row.get("OtherSpansJSON", "[]")
        try:
            spans = _json.loads(spans) if isinstance(spans, str) else []
        except Exception:
            spans = []
        if spans and {"corr_start", "corr_end"} <= set(g.columns):
            # reconstruct the script's corrected text once
            toks = g["corr_token"].map(_tok_str).tolist()
            txt = detok_v2(toks)
            low = txt.lower()
            for sp in spans:
                frag = str(sp.get("text", "")).strip().lower()
                if len(frag) < 3:
                    continue
                pos = low.find(frag)
                if pos < 0:
                    continue                        # cautious: not found
                st, en = pos, pos + len(frag)
                # crude char->token map via order (offsets may be stale)
                acc = 0
                for i in idxs:
                    t = _tok_str(g.at[i, "corr_token"])
                    j = txt.lower().find(t.lower(), acc)
                    if j < 0:
                        continue
                    acc = j + len(t)
                    if j >= st and j < en and \
                       df.at[i, "TextualArtifact"] not in ("TITLE",):
                        df.at[i, "TextualArtifact"] = "OTHER"

    # rebuild SentenceRef so a title sits at sentence 0 cleanly
    df["SentenceRef"] = (df[idcol].astype(str) + "_s"
                         + df["CorrSentenceID"].map(_sid3))
    df.loc[df["TITLE"] == True, "Sentence Boundaries"] = "Title"
    return df


# ----------------------------------------------------------------------
# 10.  Paragraph-break token (request 2)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 11.  Section 8: boundary-retained text
# ----------------------------------------------------------------------
# Per-token columns added to the word map:
#   StudentTerminalMark    — student's terminal at this alignment position
#                            ("." "?" "!" or "" if student had none here)
#   CorrectedTerminalMark  — terminal used in boundary-retained text
#                            ("." "?" "!" or "" if no boundary here at all)
#   CapitalSource          — 'student' | 'machine-added' | 'NA'
#   BoundarySource         — 'student' | 'machine-added' | 'NA'
#
# 'student' = the student placed the mark / capital themselves.
# 'machine-added' = the corrector introduced it; the student had nothing there.
#
# Locked worked example (BUILD_BRIEF §8):
#   Raw:      i went. to the shop and i saw a big dog? the dog barked
#   Clean:    I went to the shop and I saw a big dog. The dog barked.
#   Retained: I went. To the shop and I saw a big dog. The dog barked.
#
# The `.` after "went" is BoundarySource=student; "To" is CapitalSource=machine-added.
# The `?`→`.` at "dog" is BoundarySource=student (student put a mark, corrector
# normalised its type). The final `.` is BoundarySource=machine-added.

_TERM_SET = {".", "?", "!"}


def _add_boundary_retained_columns(wm):
    """Add the four Section 8 per-token columns to the word map.
    Must be called after enrich() (needs TokenCategory) and before
    inject_paragraph_breaks() (needs _ap for sort order)."""
    df = wm.copy()
    df["StudentTerminalMark"] = ""
    df["CorrectedTerminalMark"] = ""
    df["CapitalSource"] = "NA"
    df["BoundarySource"] = "NA"
    idcol = "Identifier" if "Identifier" in df.columns else "ID"

    for _, g in df.groupby(idcol, sort=False):
        # Sort by alignment position so deletes are not pushed to the end.
        if "_ap" in g.columns:
            g_s = g.sort_values("_ap", kind="mergesort")
        else:
            g_s = g
        idxs = g_s.index.tolist()

        # Pass 1: mark every position that carries a terminal in boundary-retained.
        # A position is a terminal in boundary-retained when:
        #   - student wrote a terminal here (raw_token in TERMINALS), OR
        #   - corrector inserted/kept a terminal here (corr_token in TERMINALS)
        for idx in idxs:
            rt = _tok_str(df.at[idx, "raw_token"])
            ct = _tok_str(df.at[idx, "corr_token"])
            rt_is = rt in _TERM_SET
            ct_is = ct in _TERM_SET
            if rt_is or ct_is:
                df.at[idx, "StudentTerminalMark"] = rt if rt_is else ""
                # Prefer the corrected mark (may normalise ? or ! to .);
                # if corrector removed the student's mark, keep it as ".".
                df.at[idx, "CorrectedTerminalMark"] = ct if ct_is else "."
                df.at[idx, "BoundarySource"] = (
                    "student" if rt_is else "machine-added")

        # Pass 2: CapitalSource for words that are sentence-initial in
        # boundary-retained (i.e., the first word after each retained terminal).
        after_boundary = True   # script-start counts as sentence-initial
        for idx in idxs:
            rt = _tok_str(df.at[idx, "raw_token"])
            ctm = df.at[idx, "CorrectedTerminalMark"]
            cat = str(df.at[idx, "TokenCategory"]) if "TokenCategory" in df.columns else ""
            is_wordish = cat in ("word", "contraction")
            if after_boundary and is_wordish:
                first_raw = next((c for c in rt if c.isalpha()), "")
                df.at[idx, "CapitalSource"] = (
                    "student" if first_raw.isupper() else "machine-added")
                after_boundary = False
            if ctm:     # non-empty CorrectedTerminalMark → boundary follows
                after_boundary = True

    return df


def build_boundary_retained_texts(wm, texts=None,
                                  corr_col="Corrected text (8)",
                                  id_col="Research ID"):
    """Return {Identifier: text} for 'Corrected text (boundaries retained)'.

    Reads the four Section 8 columns from the word map (added by
    _add_boundary_retained_columns). Paragraph-break rows (_ap=NaN) are
    sorted to the end within each script and skipped during text construction.

    When `texts` is supplied the function re-injects newlines from the clean
    corrected text so the boundary-retained version has the same paragraph
    and speech-break structure as `Corrected text (8)`. Without `texts` the
    output is a flat string (old behaviour, backward-compatible)."""
    idcol = "Identifier" if "Identifier" in wm.columns else "ID"

    # Build clean corrected text lookup keyed by Identifier
    corr_by_id = {}
    if texts is not None:
        _ik = id_col if id_col in getattr(texts, "columns", []) else "Identifier"
        for _, r in texts.iterrows():
            corr_by_id[str(r[_ik])] = str(r.get(corr_col, ""))

    out = {}
    for ident, g in wm.groupby(idcol, sort=False):
        if "_ap" in g.columns:
            ap_num = pd.to_numeric(g["_ap"], errors="coerce")
            g_s = g.assign(_ap_num=ap_num).sort_values("_ap_num", kind="mergesort")
        else:
            g_s = g

        # Each entry: (text_to_emit, corr_start, corr_end)
        # corr_start / corr_end are None for deleted terminal tokens.
        emit = []
        for _, r in g_s.iterrows():
            ctm = str(r.get("CorrectedTerminalMark", ""))
            cap_src = str(r.get("CapitalSource", "NA"))
            ct = _tok_str(r.get("corr_token"))
            op = str(r.get("op", "equal"))
            cs = r.get("corr_start")
            ce = r.get("corr_end")
            if ctm and ctm != "NA":
                emit.append((ctm, cs, ce))
                continue
            if op in ("delete", "paragraph"):
                continue
            tok = ct
            if not tok:
                continue
            # Re-capitalise sentence-initial words (covers both 'student' and
            # 'machine-added': when the corrected token is lower-cased because
            # in the clean text it was mid-sentence, we must re-raise it here).
            if cap_src in ("student", "machine-added"):
                tok = re.sub(r"[A-Za-z]",
                             lambda m: m.group().upper(), tok, count=1)
            emit.append((tok, cs, ce))

        base = detok_v2([t for t, _, _ in emit])
        clean = corr_by_id.get(str(ident))

        if not clean or not emit:
            out[str(ident)] = base
            continue

        # Count newlines between adjacent pairs of emitted tokens using
        # their corr positions in the clean corrected text.  Pairs where
        # either side has no corr position (deleted terminals) are skipped;
        # in practice the only newlines that matter (paragraph / speech
        # breaks) appear between regular equal/insert tokens.
        nl_after = [0] * len(emit)
        for i in range(len(emit) - 1):
            ce_i = emit[i][2]
            cs_j = emit[i + 1][1]
            if ce_i is None or cs_j is None:
                continue
            try:
                ce_i_int = int(float(ce_i))
                cs_j_int = int(float(cs_j))
            except (TypeError, ValueError):
                continue
            if cs_j_int > ce_i_int:
                nl_after[i] = clean[ce_i_int:cs_j_int].count("\n")

        if not any(nl_after):
            out[str(ident)] = base
            continue

        # Walk the base string in token order, locate each emitted token,
        # and splice in the recorded newlines (absorbing the trailing space).
        result = []
        pos = 0
        for i, (tok, _, _) in enumerate(emit):
            idx = base.find(tok, pos)
            if idx < 0:                 # safety fallback
                result.append(tok)
                if nl_after[i]:
                    result.append("\n" * nl_after[i])
                continue
            result.append(base[pos:idx + len(tok)])
            pos = idx + len(tok)
            if nl_after[i]:
                if pos < len(base) and base[pos] == " ":
                    pos += 1            # absorb the space before newlines
                result.append("\n" * nl_after[i])
        result.append(base[pos:])
        out[str(ident)] = "".join(result).strip()

    return out


def inject_paragraph_breaks(df_map, texts, id_col="Research ID",
                            raw_col="Raw text"):
    """Insert a structural token wherever the RAW text has a line break.

    TokenCategory is 'speech break' when a dialogue mark sits on either
    side of the newline (the line before closes with `"` after the
    dialogue, or the line after opens with `"`); otherwise 'paragraph
    break'. TokenSubcat is 'new line' in both cases. The token carries
    no spelling/punctuation/capitalisation/SS judgement and is never
    counted as an edit; it only records where the break is and which
    kind it is.

    Note: the r5 sample raw text contains no line breaks, so this will
    rarely fire there by design; it is built for inputs that do."""
    df = df_map.copy()
    idcol = "Identifier" if "Identifier" in df.columns else "ID"
    raw_by_id = {str(i): str(t) for i, t in
                 zip(texts[id_col].astype(str), texts[raw_col])}
    parts = []
    for ID, g in df.groupby(idcol, sort=False):
        g = g.sort_values("raw_index", kind="mergesort").reset_index(drop=True)
        raw = raw_by_id.get(str(ID), "")
        nls = [m for m, ch in enumerate(raw) if ch == "\n"]
        if not nls:
            parts.append(g)
            continue
        # collapse a run of consecutive newlines into a single break
        # (carry the run's start and end-exclusive so we can look at
        # what sits on each side of the WHOLE run, not just one \n)
        runs, run_start, prev = [], nls[0], nls[0]
        for p in nls[1:]:
            if p == prev + 1:
                prev = p
            else:
                runs.append((run_start, prev + 1))
                run_start, prev = p, p
        runs.append((run_start, prev + 1))

        rows = [r._asdict() if hasattr(r, "_asdict") else dict(r)
                for r in g.to_dict("records")]
        for r_start, r_end in sorted(runs, reverse=True):
            # classify: speech break vs paragraph break (raw-side rule)
            before = raw[:r_start].rstrip()
            after = raw[r_end:].lstrip()
            speech_side_before = bool(before) and before[-1] == '"'
            speech_side_after = bool(after) and after[0] == '"'
            cat = ("speech break"
                   if (speech_side_before or speech_side_after)
                   else "paragraph break")

            after_idx = g.index[g["raw_start"].fillna(-1) >= r_start]
            ins_at = int(after_idx[0]) if len(after_idx) else len(rows)
            blank = {c: "NA" for c in g.columns}
            blank[idcol] = ID
            blank["raw_token"] = "\\n"
            blank["raw_start"] = r_start
            blank["raw_end"] = r_end
            blank["op"] = "paragraph"
            blank["TokenCategory"] = cat
            blank["TokenSubcat"] = "new line"
            if "_ap" in g.columns:
                try:
                    if ins_at < len(rows):
                        next_ap = pd.to_numeric(pd.Series([rows[ins_at].get("_ap")]), errors="coerce").iloc[0]
                        prev_ap = pd.to_numeric(pd.Series([rows[ins_at - 1].get("_ap") if ins_at > 0 else None]), errors="coerce").iloc[0]
                        if pd.notna(next_ap) and pd.notna(prev_ap):
                            blank["_ap"] = (prev_ap + next_ap) / 2.0
                        elif pd.notna(next_ap):
                            blank["_ap"] = next_ap - 0.5
                    else:
                        prev_ap = pd.to_numeric(pd.Series([rows[-1].get("_ap")]), errors="coerce").iloc[0] if rows else None
                        if pd.notna(prev_ap):
                            blank["_ap"] = prev_ap + 0.5
                except Exception as exc:
                    print(f"  [warn] _ap interpolation failed for {ID} "
                          f"at raw_start={r_start}: {exc}")
            for b in ("TITLE", "DIALOGUE"):
                if b in g.columns:
                    blank[b] = False
            if "CorrSentenceID" in g.columns and ins_at < len(rows):
                blank["CorrSentenceID"] = rows[ins_at].get("CorrSentenceID")
            rows.insert(ins_at, blank)
        parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True)


def _fix_delete_sentence_ids(df_map):
    """Change 7. Delete rows (raw_token present, corr_token absent) have a
    blank corr_index, so assign_corr_sentence_ids_v2 sorts them to the
    end of the script and they pick up the FINAL sentence's id, even
    though physically they sit inside an earlier sentence. Re-walk the
    script in raw_index order and copy the preceding non-delete row's
    CorrSentenceID onto each delete row, so grouping by CorrSentenceID
    attributes the delete's error to the right sentence."""
    df = df_map.copy()
    idcol = "Identifier" if "Identifier" in df.columns else "ID"
    for sid, g in df.groupby(idcol, sort=False):
        g_raw = (g.assign(_ri=pd.to_numeric(g["raw_index"], errors="coerce"))
                  .sort_values("_ri", kind="mergesort", na_position="last"))
        prev_csid = None
        for idx in g_raw.index:
            op = str(g_raw.loc[idx, "op"])
            csid = g_raw.loc[idx, "CorrSentenceID"]
            if op == "delete":
                if prev_csid is not None:
                    df.at[idx, "CorrSentenceID"] = prev_csid
            else:
                if pd.notna(csid) and str(csid) != "":
                    prev_csid = csid
    return df


def _merge_punctuation_only_sentences(wm, sent):
    """Change 10. Sentences whose CorrectedSentence contains no
    alphabetic characters are segmentation artefacts — usually an
    orphaned terminal mark that got broken off into its own row when
    the student wrote it with extra whitespace. Merge each into the
    previous valid sentence in the same script by updating wm tokens'
    SentenceRef and CorrSentenceID, then drop the empty sentence row.

    First-in-script orphans (rare) have nowhere to merge into, so
    they are simply dropped. The corresponding wm tokens then have a
    dangling SentenceRef; downstream code already tolerates that.

    Mutates wm in place. Returns the filtered sent dataframe."""
    import re as _re

    def _has_letters(s):
        return bool(_re.search(r"[A-Za-z]", str(s)))

    sent = sent.copy()
    drop_refs = []
    merge_map = {}              # bad SentenceRef -> previous good SentenceRef
    for _sid, g in sent.groupby("Identifier", sort=False):
        g = g.sort_values("SentenceRef")
        prev_ref = None
        for _, r in g.iterrows():
            sref = r["SentenceRef"]
            if not _has_letters(r["CorrectedSentence"]):
                if prev_ref is not None:
                    merge_map[sref] = prev_ref
                drop_refs.append(sref)
            else:
                prev_ref = sref

    if merge_map:
        for bad_sref, good_sref in merge_map.items():
            good_csid_series = wm.loc[wm["SentenceRef"] == good_sref,
                                       "CorrSentenceID"]
            if good_csid_series.empty:
                continue
            good_csid = good_csid_series.iloc[0]
            mask = wm["SentenceRef"] == bad_sref
            wm.loc[mask, "SentenceRef"] = good_sref
            wm.loc[mask, "CorrSentenceID"] = good_csid
        # refresh CorrectedSentence and RawSentence on merge targets so
        # the now-merged punctuation appears in the sentence text
        target_refs = set(merge_map.values())
        corr_txt = restate_sentences(wm, token_col="corr_token")
        raw_txt = restate_sentences(wm, token_col="raw_token")
        for tref in target_refs:
            if tref in corr_txt:
                sent.loc[sent["SentenceRef"] == tref,
                         "CorrectedSentence"] = corr_txt[tref]
            if tref in raw_txt:
                sent.loc[sent["SentenceRef"] == tref,
                         "RawSentence"] = raw_txt[tref]

    if drop_refs:
        sent = sent[~sent["SentenceRef"].isin(drop_refs)].reset_index(drop=True)
    return sent
