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
    r"|[A-Za-z]+(?:['\u2019][A-Za-z]+)+['\u2019]?"   # true contraction: letter ' letter
    r"|[A-Za-z]*s['\u2019](?![A-Za-z])"              # plural possessive: dogs'
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

_APOS = ("'", "\u2019")

def _has_internal_apostrophe(tok):
    if not tok or len(tok) < 3:
        return False
    # letter ' letter  (don't, Let's)  OR  letters'  (dogs')
    if re.search(r"[A-Za-z]['\u2019][A-Za-z]", tok):
        return True
    if re.fullmatch(r"[A-Za-z]*s['\u2019]", tok):
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
    "'": "Apostrophe", "\u2019": "Apostrophe",
    '"': "Quotation mark", "\u201c": "Quotation mark",
    "\u201d": "Quotation mark",
    "(": "Bracket", ")": "Bracket", "[": "Bracket", "]": "Bracket",
    "-": "Hyphen", "\u2014": "Dash", "\u2013": "Dash",
    "...": "Ellipsis", "\u2026": "Ellipsis",
}

# Tiny starter homophone set -- deliberately small; the engine refines this.
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
    if re.search(r"[A-Za-z]", t) and re.fullmatch(r"[A-Za-z'\u2019]+", t):
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
              raw_col="Raw text", corr_col="Corrected text (8)",
              boundary_mode="rules"):
    """Run the fixed mechanical layer + schema on a texts dataframe.

    Returns df_map with the script identifier in a column called
    'Identifier'.  Your original aes_canonical is not modified, and the
    IDeas criterion column 'ID' (if present) is left untouched.

    boundary_mode:
      'rules'     (default) -- segment on corrector terminals, then apply the
                  rule-based detectors (Rule 1 here; Rules 2/3 + BR1 in the
                  sentence layer).
      'corrector' -- Branch B: segment on the INTENDED terminals (corrector
                  marks + restored student marks) and skip Rule 1. The other
                  detectors are skipped in run_sentence_layer.
    """
    # Proper quote-aware segmentation now handles this (see section 7);
    # the old constant-patching hack is removed.

    t = texts.copy()
    if id_col not in t.columns:
        raise KeyError(f"id_col {id_col!r} not found. Columns: {list(t.columns)}")
    t["Identifier"] = t[id_col].astype(str)

    # word map -- _ap (alignment position, per-script) is kept for Section 8
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

    if boundary_mode == "corrector":
        wm = assign_intended_sentence_ids_v2(wm)     # Branch B: intended terminals
        # Rule 1 (BR3) is a detector -- skipped in corrector-driven mode.
    else:
        wm = assign_corr_sentence_ids_v2(wm)
        wm = _apply_rule1_sentence_splits(wm)        # BR3: Rule 1 post-seg splits
    wm = _fix_delete_sentence_ids(wm)                # Change 7
    # CB5: mark_title_and_dialogue runs BEFORE the sacred-terminal guard so
    # that the heuristic evaluates the corrector's original last sentence
    # rather than a guard-created fragment that may have a higher CorrSentenceID
    # even though it sits mid-script in story order.
    wm = aes.mark_title_and_dialogue(wm, t_aes)
    if boundary_mode == "corrector":
        wm = _apply_sacred_terminal_guard_wm(wm)    # CB5: student-terminal guard
    wm = _merge_attribution_fragments(wm)            # C1: attribution exemption
    wm = _merge_orphaned_reporting_clauses(wm)       # BR5: orphan reporting clause merge
    wm = mark_dialogue_by_quotes(wm)                 # self-contained DIALOGUE
    wm = mark_artifacts_by_model(wm, t_aes)          # title/ending/other
    wm = _apply_allcaps_title_split(wm)              # CB6-TITLE A: prose-fused ALL-CAPS prefix
    wm = _apply_cutoff_detection(wm)                 # CB5/CB6: CUTOFF detection
    wm = aes.add_sentence_boundary_flags(wm)

    wm = wm.drop(columns=["ID"], errors="ignore")   # drop the alias
    wm = enrich(wm)                                  # uses 'Identifier'
    # WC3: error column fixes -- must run AFTER enrich() which creates
    # TokenCategory, Spell error, SS class from _enrich_row.
    wm = _fix_proper_noun_spell_errors(wm)
    wm = _fix_extra_word_contradictions(wm)
    wm = _split_ss_other(wm)
    wm = _add_boundary_retained_columns(wm)          # Section 8
    wm = inject_paragraph_breaks(wm, texts, id_col=id_col, raw_col=raw_col)
    _idcol = "Identifier" if "Identifier" in wm.columns else "ID"
    wm = (wm.assign(_ap_sort=pd.to_numeric(wm["_ap"], errors="coerce"))
            .sort_values([_idcol, "_ap_sort"], kind="mergesort")
            .drop(columns="_ap_sort")
            .reset_index(drop=True))
    # WA2: Dtype unification -- convert bool columns to string before export
    # so downstream string comparisons (== 'True') and CSV round-trips are safe.
    for _col in ("DIALOGUE", "TITLE"):
        if _col in wm.columns and wm[_col].dtype == bool:
            wm[_col] = wm[_col].map({True: "True", False: "False"}).astype(str)
    return wm


# ----------------------------------------------------------------------
# C1 -- Dialogue-attribution exemption (merge incorrectly split attribution)
# ----------------------------------------------------------------------
_REPORTING_VERBS_C1 = {
    "said", "asked", "replied", "answered", "exclaimed", "whispered",
    "shouted", "called", "cried", "muttered", "added", "continued",
    "began", "told", "yelled", "demanded", "repeated", "suggested",
    "agreed", "declared", "announced", "mumbled", "remarked", "observed",
    "insisted", "admitted", "confessed", "sighed", "gasped", "screamed",
    "hissed", "growled", "snapped", "grumbled", "murmured", "mentioned",
    "noted", "wondered", "queried",
}
_ATTRIB_PRONOUNS_C1 = {"he", "she", "it", "they", "i", "we", "you"}

def _sentence_is_attribution(corr_tokens, max_words=12):
    """
    Return True if the token list forms a dialogue attribution tag:
      Pattern A: [Name|Pronoun] [lowercase reporting verb] [optional more]
      Pattern B: [lowercase reporting verb] [Name|determiner] [optional more]
    Only considers sentences <= max_words real words.
    """
    words = [str(t).lower().strip(",.!?\"'") for t in corr_tokens
             if str(t).strip() not in ("", ",", ".", "!", "?", "'", '"')]
    if not words or len(words) > max_words:
        return False
    # Pattern A: word1 is name/pronoun, word2 is reporting verb
    if len(words) >= 2:
        w1, w2 = words[0], words[1]
        if w2 in _REPORTING_VERBS_C1:
            # w1 is pronoun or capitalised name
            if w1 in _ATTRIB_PRONOUNS_C1:
                return True
            if len(w1) >= 2 and w1[0].isupper() and w1[1:].islower():
                # Re-check from the original (not lowercased) to confirm capital
                orig_words = [str(t).strip(",.!?\"'") for t in corr_tokens
                              if str(t).strip() not in ("", ",", ".", "!", "?", "'", '"')]
                if orig_words and orig_words[0][0].isupper():
                    return True
    # Pattern B: word1 is reporting verb, word2 is name/determiner
    if len(words) >= 2:
        w1 = words[0]
        if w1 in _REPORTING_VERBS_C1:
            return True
    return False


# ----------------------------------------------------------------------
# BR12 -- Ellipsis masking helpers (prevent Rule 1 false splits at ellipsis)
# -----------------------------------------------------------------------
_ELLIPSIS_RE = re.compile(r'\.{2,}|...')
_ELLIPSIS_SENTINEL = '\x01ELLIPSIS\x01'
_ELLIPSIS_BOUNDARY_RE = re.compile(r'\.\.\.\s+([A-Z])')


def _mask_ellipsis(text):
    """Replace ellipsis sequences with a non-triggering sentinel."""
    return _ELLIPSIS_RE.sub(_ELLIPSIS_SENTINEL, text)


def _unmask_ellipsis(text):
    """Restore ellipsis sentinel to '...'."""
    return text.replace(_ELLIPSIS_SENTINEL, '...')


# ----------------------------------------------------------------------
# BR3 -- Rule 1 post-segmentation sentence splitting
# -----------------------------------------------------------------------
_RULE1_TRIGGER_BR3 = re.compile(
    # BR3-fix-C (v2): the closing-quote group is made fully optional as a GROUP
    # so that the required \s+ is not starved when no quote is present.
    #   Plain ". C" : [.!?] + (group absent) + \s+ eats space + C \u2713
    #   "! \" C"    : [.!?] + ( \s* + quote(s) ) + \s+ eats space + C \u2713
    #   "!\"\" C"   : [.!?] + ( quotes ) + \s+ + C \u2713
    # The original "fix C v1" used \s*[quote]{0,3}\s+ which let \s* consume
    # the only space, leaving \s+ unsatisfied for plain ". C" / "? C" patterns.
    r'[.!?](\s*[\u201c\u201d\u2018\u2019"\']{1,3})?\s+[\u201c\u201d\u2018\u2019"\']{0,2}(?=[A-Z0-9])'
)
_ABBREV_BR3 = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|No)\.(?=\s)",
    re.IGNORECASE
)
_SCENE_SHIFT_BR3 = re.compile(
    r"[.!?]\s*\d+\s+(?:weeks?|days?|months?|years?|hours?|minutes?)\s+"
    r"(?:later|earlier|before|after)\b",
    re.IGNORECASE
)


def _apply_rule1_sentence_splits(df_map):
    """
    BR3: After sentence ID assignment, find sentences whose corrected text
    contains a Rule 1 trigger (terminal mark + capital/digit) and split them
    by reassigning CorrSentenceID to sub-groups of tokens.

    Safe: only reassigns IDs, never adds or removes token rows.
    """
    try:
        df = df_map.copy()
        idcol = "Identifier" if "Identifier" in df.columns else "ID"
        new_ids = df["CorrSentenceID"].copy()

        # BR9: initialise the origin-sentence-ID metadata column
        br3_origins = pd.Series(pd.NA, index=df.index, dtype="object")

        # Track the highest used CorrSentenceID per script to generate new IDs
        max_sid = {}
        for ident, g in df.groupby(idcol, sort=False):
            sids = pd.to_numeric(g["CorrSentenceID"], errors="coerce").dropna()
            max_sid[ident] = int(sids.max()) if len(sids) > 0 else 0

        for ident, g in df.groupby(idcol, sort=False):
            for csid, sg in g.groupby("CorrSentenceID", sort=True):
                try:
                    if pd.isna(csid):
                        continue
                    # Reconstruct corrected sentence text for this sentence
                    sg_sorted = sg.sort_values("corr_index", kind="mergesort")
                    corr_tokens = sg_sorted["corr_token"].fillna("").astype(str).tolist()
                    corr_text = " ".join(t for t in corr_tokens if t)

                    # BR12: mask ellipsis sequences to prevent Rule 1 from splitting at them
                    corr_text_masked = _mask_ellipsis(corr_text)

                    # Fast check: does the text even have a potential Rule 1 trigger?
                    has_rule1 = _RULE1_TRIGGER_BR3.search(corr_text_masked)
                    # BR12: also check for ellipsis-as-boundary (ellipsis + capital = new sentence)
                    has_ellipsis_boundary = _ELLIPSIS_BOUNDARY_RE.search(corr_text)
                    if not has_rule1 and not has_ellipsis_boundary:
                        continue

                    # Check abbreviation exemptions (on masked text to avoid spurious matches)
                    abbrev_positions = set()
                    for m in _ABBREV_BR3.finditer(corr_text_masked):
                        abbrev_positions.add(m.end())

                    # Build token start positions for mapping regex matches to tokens
                    cumulative_pos = 0
                    token_start_positions = []
                    for tok in corr_tokens:
                        token_start_positions.append(cumulative_pos)
                        cumulative_pos += len(tok) + 1  # +1 for space

                    # Build quote-count prefix sums for odd-quote exemption (Fix F)
                    # Count all opening/closing quote characters in the masked text
                    # cumulatively so we can test in O(1) whether a match position
                    # sits inside an open quote span.
                    _ALL_QUOTES_RE = re.compile(
                        r'[""''"\'`]'
                    )
                    quote_positions = [qm.start()
                                       for qm in _ALL_QUOTES_RE.finditer(corr_text_masked)]

                    def _quote_count_before(pos):
                        """Count quote chars before position pos (binary search)."""
                        lo, hi = 0, len(quote_positions)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if quote_positions[mid] < pos:
                                lo = mid + 1
                            else:
                                hi = mid
                        return lo

                    # Find split points from Rule 1 trigger (on masked text)
                    split_token_indices = []
                    if has_rule1:
                        for m in _RULE1_TRIGGER_BR3.finditer(corr_text_masked):
                            trigger_pos = m.end()
                            # Skip if abbreviation nearby
                            if any(abs(trigger_pos - ap) <= 2 for ap in abbrev_positions):
                                continue
                            # Fix F -- odd-quote-count exemption: if an odd number of
                            # quote chars precede the trigger, we are INSIDE an open
                            # dialogue span.  Splitting here would fragment the speech
                            # into pieces (the ALL-CAPS cascade bug).  Skip it.
                            if _quote_count_before(m.start()) % 2 == 1:
                                continue
                            # Find which token starts at or after trigger_pos
                            next_token_idx = None
                            for ti, tsp in enumerate(token_start_positions):
                                if tsp >= trigger_pos:
                                    next_token_idx = ti
                                    break
                            if next_token_idx is None or next_token_idx == 0:
                                continue
                            # Check there's substantial content on both sides
                            before_count = next_token_idx
                            after_count = len(corr_tokens) - next_token_idx
                            if before_count < 3 or after_count < 1:
                                continue
                            split_token_indices.append(next_token_idx)

                    # BR12: also find split points from ellipsis-as-boundary (on original text)
                    if has_ellipsis_boundary:
                        for m in _ELLIPSIS_BOUNDARY_RE.finditer(corr_text):
                            trigger_pos = m.start(1)  # capital letter position
                            # Find which token starts at or after trigger_pos
                            next_token_idx = None
                            for ti, tsp in enumerate(token_start_positions):
                                if tsp >= trigger_pos:
                                    next_token_idx = ti
                                    break
                            if next_token_idx is None or next_token_idx == 0:
                                continue
                            before_count = next_token_idx
                            after_count = len(corr_tokens) - next_token_idx
                            if before_count < 3 or after_count < 1:
                                continue
                            split_token_indices.append(next_token_idx)

                    if not split_token_indices:
                        continue

                    # Create new CorrSentenceID for each split segment after the first
                    split_points = [0] + sorted(set(split_token_indices)) + [len(corr_tokens)]
                    current_csid = int(csid)

                    for seg_idx, (seg_start, seg_end) in enumerate(
                            zip(split_points[:-1], split_points[1:])):
                        if seg_idx == 0:
                            target_csid = current_csid
                        else:
                            max_sid[ident] += 1
                            target_csid = max_sid[ident]

                        seg_rows = sg_sorted.iloc[seg_start:seg_end]
                        new_ids.loc[seg_rows.index] = target_csid
                        # BR9: record the origin sentence ID for non-first segments
                        if seg_idx > 0:
                            br3_origins.loc[seg_rows.index] = current_csid

                except Exception as _e:
                    # Any per-sentence failure: skip this sentence, keep original IDs
                    continue

        df["CorrSentenceID"] = new_ids
        # BR9: store split-origin metadata column
        df["_br3_origin_sid"] = br3_origins
        return df
    except Exception as _outer:
        # Outer failure: return original unchanged
        return df_map


# ----------------------------------------------------------------------
# CB5 -- Sacred-terminal guard (word-map level)
# After assign_intended_sentence_ids_v2 + _fix_delete_sentence_ids, force
# a boundary at every student-origin terminal mark left mid-sentence.
# DELETE rows (corr_index=NaN) were sorted to end by assign_intended so
# they were skipped; iterating in raw_index order fixes that.
# -----------------------------------------------------------------------

_STG_SINGLE_CAP_RE = re.compile(r'\b[A-Z]\s*$')
_STG_ABBREV_WORDS_RE = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|No)\s*$', re.IGNORECASE)
_STG_DIGIT_END_RE = re.compile(r'\d\s*$')
_STG_DIGIT_START_RE = re.compile(r'^\s*\d')


def _stg_is_exempt_period(text_before, first_raw_after):
    """True if '.' here is an abbreviation or decimal, not a sentence end."""
    if _STG_SINGLE_CAP_RE.search(text_before):     # H.  R.  S.
        return True
    if _STG_ABBREV_WORDS_RE.search(text_before):   # Mr.  Dr.  etc.
        return True
    if (first_raw_after and                        # decimal: 3.14
            _STG_DIGIT_END_RE.search(text_before) and
            _STG_DIGIT_START_RE.match(first_raw_after)):
        return True
    return False


def _apply_sacred_terminal_guard_wm(df_map):
    """CB5: Force sentence boundaries at terminal marks that the segmenter
    missed. Catches:
      - Student terminals (raw_token in .?!) that the corrector removed or
        changed to a comma, leaving them invisible to the corrector segmenter.
      - Corrector-kept or corrector-inserted terminals (corr_token in .?!)
        where aes._is_terminal_token returned False because the next token is
        a digit ("stairs. 1 hour later") or the preceding token is a single
        alpha letter ("am I?").

    Must run after _fix_delete_sentence_ids. Sets _stg_origin_sid on
    split-off rows for BoundaryProvenance='student-terminal-guard' tagging.
    """
    df = df_map.copy()
    idcol = "Identifier" if "Identifier" in df.columns else "ID"
    new_ids = df["CorrSentenceID"].copy()
    stg_origin: pd.Series = pd.Series(pd.NA, index=df.index, dtype="object")

    max_sid: dict = {}
    for ident, g in df.groupby(idcol, sort=False):
        sids = pd.to_numeric(g["CorrSentenceID"], errors="coerce").dropna()
        max_sid[ident] = int(sids.max()) if len(sids) else 0

    for ident, g in df.groupby(idcol, sort=False):
        for csid, sg in g.groupby("CorrSentenceID", sort=True):
            if pd.isna(csid):
                continue
            # Sort tokens so each appears at its natural position in the text.
            # INSERT rows (NaN raw_index, real corr_index) sort by corr_index
            # so they appear adjacent to the tokens they were inserted between,
            # rather than piling up at the end where they would land in the
            # wrong segment after a guard split.
            # DELETE rows (real raw_index, NaN corr_index) sort by raw_index.
            # EQUAL/REPLACE rows sort by raw_index (≈ corr_index for these).
            _ri = pd.to_numeric(sg["raw_index"], errors="coerce")
            _ci_col = (sg["corr_index"]
                       if "corr_index" in sg.columns
                       else pd.Series(pd.NA, index=sg.index))
            _ci = pd.to_numeric(_ci_col, errors="coerce")
            # INSERT: use corr_index; real rows: use raw_index; fallback: inf
            _sort_key = _ri.where(_ri.notna(), _ci).fillna(1e15)
            sg_raw = sg.iloc[_sort_key.values.argsort(kind="mergesort")]

            idxs = sg_raw.index.tolist()
            raw_toks = sg_raw["raw_token"].fillna("").astype(str).tolist()
            corr_toks = sg_raw["corr_token"].fillna("").astype(str).tolist()
            # True for each position where the token is a corrector INSERT
            # (no student raw_index)
            _raw_idx_num = pd.to_numeric(
                sg_raw["raw_index"], errors="coerce").tolist()
            is_insert = [pd.isna(v) for v in _raw_idx_num]

            # Punctuation / quote chars that don't count as sentence content
            _STG_NONWORD = set('.!?,;:"\'')

            total_corr_words = sum(
                1 for i, t in enumerate(corr_toks)
                if t.strip() and t.strip() not in _STG_NONWORD
                and not is_insert[i]
            )
            if total_corr_words < 3:
                continue

            split_positions: list = []
            text_so_far = ""
            last_split_pos = 0   # track for per-segment before_words count

            for i, (rt, ct) in enumerate(zip(raw_toks, corr_toks)):
                rt_s = rt.strip()
                ct_s = ct.strip()

                # INSERT tokens: raw_token is the string "nan"; skip them for
                # text_so_far accumulation but let them participate in terminal
                # detection via ct_s.
                if is_insert[i]:
                    # INSERT terminal (corrector added a ./?/!) can fire; skip
                    # non-terminals silently.
                    if ct_s not in _TERM_SET:
                        continue
                    # INSERT terminal: treat same as real terminal below
                    rt_s = ""  # no raw form

                # Fire on either: student terminal (raw) OR corrected terminal (corr)
                is_terminal = rt_s in _TERM_SET or ct_s in _TERM_SET
                if not is_terminal:
                    if rt_s and rt_s != "nan":
                        text_so_far += rt_s + " "
                    continue

                # The terminal mark string for guard / text purposes
                term_s = rt_s if rt_s in _TERM_SET else ct_s

                # Skip if no real (student-written) content words remain after
                # this terminal.  Ignores corrector-INSERT tokens (e.g. "as",
                # a connecting word the corrector added) and all punctuation /
                # quote chars so a closing " or , does not fool the guard into
                # splitting a dialogue-attribution sentence.
                remaining_corr_real = [
                    corr_toks[j].strip()
                    for j in range(i + 1, len(corr_toks))
                    if corr_toks[j].strip()
                    and corr_toks[j].strip() not in _STG_NONWORD
                    and not is_insert[j]
                ]
                if not remaining_corr_real:
                    text_so_far += term_s + " "
                    continue

                # Need >= 2 real (non-INSERT) words since the last split
                before_words = len([
                    j for j in range(last_split_pos, i)
                    if not is_insert[j]
                    and raw_toks[j].strip()
                    and raw_toks[j].strip() not in _STG_NONWORD
                    and raw_toks[j].strip() != "nan"
                ])
                if before_words < 2:
                    text_so_far += term_s + " "
                    continue

                # Abbreviation / decimal guard for period
                if term_s == ".":
                    next_raw = next(
                        (raw_toks[j].strip()
                         for j in range(i + 1, len(raw_toks))
                         if raw_toks[j].strip() and not is_insert[j]),
                        "",
                    )
                    if _stg_is_exempt_period(text_so_far, next_raw):
                        text_so_far += term_s + " "
                        continue

                split_positions.append(i)
                last_split_pos = i + 1
                text_so_far += term_s + " "

            if not split_positions:
                continue

            current_csid = int(csid)
            boundaries = sorted(set(split_positions))
            starts = [0] + [p + 1 for p in boundaries]
            ends   = [p + 1 for p in boundaries] + [len(idxs)]

            for seg_idx, (seg_start, seg_end) in enumerate(zip(starts, ends)):
                if seg_idx == 0:
                    target_csid = current_csid
                else:
                    max_sid[ident] += 1
                    target_csid = max_sid[ident]
                seg_idxs = idxs[seg_start:seg_end]
                new_ids.loc[seg_idxs] = target_csid
                if seg_idx > 0:
                    stg_origin.loc[seg_idxs] = current_csid

    df["CorrSentenceID"] = new_ids
    df["_stg_origin_sid"] = stg_origin
    return df


# ----------------------------------------------------------------------
# BR5 -- Orphaned reporting clause merge
# -----------------------------------------------------------------------
_REPORTING_VERBS_BR5 = {
    # Past tense (existing)
    "said", "asked", "replied", "answered", "exclaimed", "whispered",
    "shouted", "called", "cried", "muttered", "added", "continued",
    "began", "told", "yelled", "demanded", "repeated", "suggested",
    "agreed", "declared", "announced", "mumbled", "remarked", "observed",
    "insisted", "admitted", "confessed", "sighed", "gasped", "screamed",
    "hissed", "growled", "snapped", "grumbled", "murmured", "mentioned",
    "noted", "wondered", "queried", "laughed", "responded", "responds",
    "replies", "gasps", "cried", "chuckled", "grinned", "smiled",
    "questioned", "whined",
    # Present tense (3rd person singular) -- BR10
    "says", "asks", "answers", "exclaims", "whispers",
    "shouts", "calls", "mutters", "adds", "continues",
    "tells", "yells", "demands", "repeats", "suggests",
    "agrees", "declares", "announces", "mumbles", "remarks",
    "observes", "insists", "admits", "confesses", "sighs",
    "screams", "hisses", "growls", "snaps",
    "grumbles", "murmurs", "mentions", "notes", "wonders",
    "queries", "laughs", "whines", "questions",
    "cries", "rings",
}


def _is_orphaned_reporting_br5(tokens):
    """
    BR5: Return True if this token sequence looks like an orphaned
    reporting clause: short (<= 8 tokens), contains a reporting verb.
    """
    if not tokens or len(tokens) > 8:
        return False
    words_lower = {w.strip(",.!?\"'""''").lower()
                   for w in tokens if w.strip()}
    return bool(words_lower & _REPORTING_VERBS_BR5)


def _merge_orphaned_reporting_clauses(df_map):
    """
    BR5: After sentence splitting, find sentences that are orphaned
    reporting clauses (short, contain a reporting verb) whose previous
    sentence in the same script ends with a closing quote or dialogue.

    Merges the orphaned sentence into the previous one by reassigning
    CorrSentenceID. Sets MergedFromOversplit column on merged rows.
    """
    try:
        df = df_map.copy()
        idcol = "Identifier" if "Identifier" in df.columns else "ID"
        new_ids = df["CorrSentenceID"].copy()

        if "MergedFromOversplit" not in df.columns:
            df["MergedFromOversplit"] = "False"

        for ident, g in df.groupby(idcol, sort=False):
            # Get sentences in order
            sids = sorted(
                pd.to_numeric(g["CorrSentenceID"], errors="coerce").dropna().unique()
            )
            if len(sids) < 2:
                continue

            for i in range(1, len(sids)):
                try:
                    curr_sid = sids[i]
                    prev_sid = sids[i - 1]

                    curr_rows = g[g["CorrSentenceID"] == curr_sid].sort_values(
                        "corr_index", kind="mergesort")
                    prev_rows = g[g["CorrSentenceID"] == prev_sid].sort_values(
                        "corr_index", kind="mergesort")

                    if curr_rows.empty or prev_rows.empty:
                        continue

                    # Get tokens of current sentence
                    curr_tokens = curr_rows["corr_token"].fillna("").astype(str).tolist()

                    # Check if current sentence is an orphaned reporting clause
                    if not _is_orphaned_reporting_br5(curr_tokens):
                        continue

                    # Check if previous sentence ends with a closing quote or dialogue
                    prev_tokens = prev_rows["corr_token"].fillna("").astype(str).tolist()
                    prev_text = " ".join(t for t in prev_tokens if t)
                    prev_last = prev_tokens[-1].strip() if prev_tokens else ""
                    has_quote_end = (
                        prev_last in ('"', "'", "\u201d", "\u2019", ".", "!", "?") or
                        prev_last == "''" or  # doubled apostrophe
                        '"' in prev_text or
                        "\u2019" in prev_text or
                        "\u201c" in prev_text or
                        "\u201d" in prev_text or
                        "\u2018" in prev_text or
                        "\u2019" in prev_text or
                        "\u2019\u2019" in prev_text  # doubled apostrophe in text
                    )

                    if not has_quote_end:
                        continue

                    # BR5-fix-E: Don't merge when the previous sentence has
                    # COMPLETE dialogue (terminal mark immediately before closing
                    # quote in the token stream). E.g. [... '!' '"'] means the
                    # dialogue ended with '!"' \u2014 a complete unit. Merging the
                    # attribution back would undo a correct BR3 split.
                    # Pattern: second-to-last token is ! ? . AND last token is a
                    # closing quote char.
                    _CLOSING_QUOTE_TOKS = {'"', "'", '\u201d', '\u2019', "''"}
                    _TERMINAL_MARK_TOKS = {'!', '?', '.'}
                    non_empty_prev = [t.strip() for t in prev_tokens if t.strip()]
                    if (len(non_empty_prev) >= 2 and
                            non_empty_prev[-1] in _CLOSING_QUOTE_TOKS and
                            non_empty_prev[-2] in _TERMINAL_MARK_TOKS):
                        continue  # complete dialogue \u2014 do not merge attribution

                    # Merge: reassign current sentence rows to previous sentence ID
                    new_ids.loc[curr_rows.index] = prev_sid
                    df.loc[curr_rows.index, "MergedFromOversplit"] = "True"
                except Exception:
                    continue

        df["CorrSentenceID"] = new_ids
        return df
    except Exception:
        return df_map


def _merge_attribution_fragments(df_map):
    """
    C1 post-processing: merge sentences that consist solely of a dialogue
    attribution tag back into the previous sentence.
    A sentence is an attribution if it is <= 12 words and matches an
    attribution pattern (subject + reporting verb, or reporting verb + subject).
    Only fires when the PREVIOUS sentence ended with a dialogue close-quote.
    """
    df = df_map.copy()
    idcol = "Identifier" if "Identifier" in df.columns else "ID"

    for ID, g in df.groupby(idcol, sort=False):
        g = g.sort_values("corr_index", kind="mergesort")
        sentences = {}
        for idx, row in g.iterrows():
            csid = row.get("CorrSentenceID")
            if pd.isna(csid):
                continue
            sentences.setdefault(csid, []).append((idx, str(row.get("corr_token", "") or "")))

        sids = sorted(sentences.keys())
        merges = {}  # csid → merge_into_csid
        for k, csid in enumerate(sids):
            if k == 0:
                continue
            prev_csid = sids[k - 1]
            toks_this = [t for _, t in sentences[csid]]
            toks_prev = [t for _, t in sentences[prev_csid]]

            # Only fire if previous sentence ended with a closing quote or terminal
            prev_ends = [t.strip() for t in toks_prev[-3:] if t.strip()]
            prev_closed_quote = any(t in ('"', "'", '\u201c', '\u201d', "\u2018", "\u2019") for t in prev_ends)

            if prev_closed_quote and _sentence_is_attribution(toks_this):
                merges[csid] = prev_csid

        if merges:
            # Apply merges: update CorrSentenceID for rows of merged sentences
            for csid, target_csid in merges.items():
                row_idxs = [idx for idx, _ in sentences[csid]]
                df.loc[row_idxs, "CorrSentenceID"] = target_csid

    # Rebuild SentenceRef after merge changes
    if "SentenceRef" in df.columns:
        _idcol2 = "Identifier" if "Identifier" in df.columns else "ID"
        df["SentenceRef"] = (df[_idcol2].astype(str) + "_s"
                             + df["CorrSentenceID"].map(_sid3))
    return df


# ----------------------------------------------------------------------
# WC3 -- Error column fixes
# ----------------------------------------------------------------------

def _fix_proper_noun_spell_errors(df):
    """WC3a: Mark Spell error=TRUE for proper-noun replace ops left as NA."""
    mask = (
        (df["op"] == "replace") &
        (df["TokenCategory"] == "word") &
        (df["raw_token"].astype(str).str.lower() !=
         df["corr_token"].astype(str).str.lower()) &
        (df.get("Spell error", pd.Series(["NA"] * len(df))) == "NA")
    )
    if mask.any():
        df = df.copy()
        df.loc[mask, "Spell error"] = "TRUE"
    return df


def _fix_extra_word_contradictions(df):
    """WC3b: 'extra word' + op=equal is a contradiction -- relabel to NA."""
    if "SS class" not in df.columns:
        return df
    mask = (df["SS class"] == "extra word") & (df["op"] == "equal")
    if mask.any():
        df = df.copy()
        df.loc[mask, "SS class"] = "NA"
        if "SS subtype" in df.columns:
            df.loc[mask, "SS subtype"] = "NA"
        if "SS error" in df.columns:
            df.loc[mask, "SS error"] = "FALSE"
    return df


def _classify_ss_other_sub(raw, corr):
    """WC3c: Disambiguate 'other' SS class into four named sub-classes."""
    raw_l = str(raw or "").lower()
    corr_l = str(corr or "").lower()
    _BOUNDARY_SPLITS = {
        "everyday", "everytime", "everynight", "atleast", "infront",
        "alot", "alright", "alright", "infact", "aswell",
    }
    if raw_l in _BOUNDARY_SPLITS:
        return "word_boundary_split"
    if corr_l.startswith(raw_l) or corr_l.endswith(raw_l):
        if len(corr_l) > len(raw_l) + 1:
            return "compound_morpheme"
    if abs(len(raw_l) - len(corr_l)) > 4:
        return "token_alignment_artefact"
    return "wrong_word_substitution"


def _split_ss_other(df):
    """WC3c: Replace 'other' SS class with specific sub-class."""
    if "SS class" not in df.columns:
        return df
    mask = df["SS class"] == "other"
    if not mask.any():
        return df
    df = df.copy()
    for idx in df[mask].index:
        df.loc[idx, "SS class"] = _classify_ss_other_sub(
            df.loc[idx, "raw_token"], df.loc[idx, "corr_token"]
        )
    return df


# ----------------------------------------------------------------------
# B1 -- BoundaryProvenance rollup (sentence-level column)
# ----------------------------------------------------------------------

def _rollup_boundary_provenance(wm, sent_df):
    """
    B1: Add BoundaryProvenance column to sent_df by examining the terminal
    mark row for each sentence in wm.
    Values: 'student', 'student-changed', 'machine-readin', 'none'
    """
    if "BoundarySource" not in wm.columns:
        sent_df = sent_df.copy()
        sent_df["BoundaryProvenance"] = "none"
        return sent_df

    terminals = set(getattr(aes, "TERMINALS", {'.', '!', '?'}))
    by_ref = {}
    for ref, g in wm.groupby("SentenceRef", sort=False):
        term_rows = g[g["corr_token"].astype(str).isin(terminals)]
        if term_rows.empty:
            by_ref[ref] = "none"
            continue
        last = term_rows.iloc[-1]
        bs = str(last.get("BoundarySource", "") or "")
        op = str(last.get("op", "") or "")
        if bs == "student" and op == "equal":
            by_ref[ref] = "student"
        elif bs == "student":
            by_ref[ref] = "student-changed"
        elif bs == "machine-added":
            by_ref[ref] = "machine-readin"
        else:
            by_ref[ref] = "none"

    sent_df = sent_df.copy()
    sent_df["BoundaryProvenance"] = sent_df["SentenceRef"].map(by_ref).fillna("none")
    return sent_df


# ----------------------------------------------------------------------
# SB1 -- Tense drift conservative tighten (exemption rules)
# ----------------------------------------------------------------------

_MODAL_RE = re.compile(
    r"\b(will|would|could|should|might|may|can|shall|'ll|'d|gonna|going\s+to)\b",
    re.IGNORECASE,
)
_PRES_PERF_RE = re.compile(
    r"\b(has|have|had|having|'s|'ve|'d)\s+\w+(ed|en|t)\b",
    re.IGNORECASE,
)


def _sent_is_fully_quoted(text):
    t = str(text or "").strip()
    return bool(
        (t.startswith('"') and t.endswith('"')) or
        (t.startswith("'") and t.endswith("'")) or
        (t.startswith('\u201c') and t.endswith('\u201d'))
    )


def _sent_is_modal_only(text):
    stripped = _MODAL_RE.sub("", str(text or ""))
    past_m = re.findall(r"\b\w+ed\b|\bwas\b|\bwere\b|\bhad\b", stripped, re.I)
    pres_m = re.findall(r"\b\w+s\b|\bis\b|\bare\b|\bam\b", stripped, re.I)
    return not past_m and not pres_m


def _sent_is_pres_perf_only(text):
    if not _PRES_PERF_RE.search(str(text or "")):
        return False
    stripped = _PRES_PERF_RE.sub("", str(text or ""))
    past_simple = re.findall(
        r"\b(was|were|did|went|came|saw|got|made|ran|took|found|said)\b",
        stripped, re.I,
    )
    return not past_simple


def _apply_tense_drift_exemptions(raw_label, sentence_text):
    """SB1: Apply three exemption rules to a raw Tense drift label."""
    if raw_label != "Incorrect":
        return raw_label
    if _sent_is_fully_quoted(sentence_text):
        return "NA"
    if _sent_is_modal_only(sentence_text):
        return "Correct"
    if _sent_is_pres_perf_only(sentence_text):
        return "Correct"
    return raw_label


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


# ---------------------------------------------------------------------------
# H1 -- Per-script dominant tense for inter-sentence Tense drift detection
# ---------------------------------------------------------------------------

# Past-tense finite verb markers (expanded from _PAST for verb-level voting)
_H1_PAST_AUX   = {'was', 'were', 'had', 'did'}
_H1_PAST_IRREG = {
    'went', 'came', 'saw', 'got', 'made', 'took', 'thought', 'knew',
    'said', 'told', 'felt', 'gave', 'found', 'left', 'ran', 'sat',
    'stood', 'heard', 'held', 'kept', 'meant', 'paid', 'put', 'read',
    'sent', 'set', 'spoke', 'stuck', 'taught', 'won', 'wrote', 'broke',
    'brought', 'bought', 'caught', 'chose', 'drank', 'drove', 'ate',
    'fell', 'flew', 'forgot', 'grew', 'lost', 'rose', 'shook', 'sang',
    'spent', 'sprang', 'stole', 'swam', 'threw', 'wore', 'began',
    'awoke', 'tore', 'hid', 'rode', 'slept', 'fought', 'led', 'fled',
    # common regular-looking past forms that overlap _PAST
    'looked', 'walked', 'turned', 'asked', 'called', 'wanted',
    'could', 'would',
}
_H1_PRES_AUX  = {'am', 'is', 'are', 'do', 'does', 'has', 'have'}
_H1_PRES_BASE = {
    'go', 'come', 'see', 'get', 'make', 'take', 'think', 'know',
    'say', 'tell', 'feel', 'give', 'find', 'leave', 'run', 'sit',
    'stand', 'hear', 'hold', 'keep', 'mean', 'pay', 'put', 'read',
    'send', 'set', 'speak', 'teach', 'win', 'write', 'break',
    'bring', 'buy', 'catch', 'choose', 'drink', 'drive', 'eat',
    'fall', 'fly', 'forget', 'grow', 'lose', 'rise', 'shake', 'sing',
    'spend', 'spring', 'steal', 'swim', 'throw', 'wear', 'begin',
    'awake', 'tear', 'hide', 'ride', 'sleep', 'fight', 'lead', 'flee',
    'want', 'need', 'like', 'love', 'hate', 'wish', 'hope', 'try',
    'live', 'work', 'play', 'walk', 'talk', 'look', 'watch', 'listen',
    'says', 'goes', 'runs', 'gets', 'comes', 'wants',   # 3sg-s forms
}
_H1_PAST_ED_RE = re.compile(r'^[a-z]{3,}ed$')


def _h1_token_tense(word, token_subcat):
    """Classify one token's tense contribution: 'past', 'present', or None.

    Only fires on verb / auxiliary verb tokens.
    """
    subcat = str(token_subcat or "").lower()
    if subcat not in ('verb', 'auxiliary verb'):
        return None
    w = re.sub(r"[^a-z]", "", str(word or "").lower())
    if not w:
        return None

    if w in _H1_PAST_AUX:
        return 'past'
    if w in _H1_PRES_AUX:
        return 'present'
    if w in _H1_PAST_IRREG:
        return 'past'
    if _H1_PAST_ED_RE.match(w) and subcat == 'verb':
        return 'past'
    if w in _H1_PRES_BASE:
        return 'present'
    # 3sg present: base form + s
    if (w.endswith('s') and not w.endswith('ss') and len(w) > 2
            and w[:-1] in _H1_PRES_BASE):
        return 'present'
    return None


def _h1_compute_dominant_tense(script_wm):
    """Determine the dominant tense for one script via verb-level majority vote.

    Excludes tokens inside dialogue (DIALOGUE column = True/true/1).

    Returns: 'past' | 'present' | 'mixed_legitimate' | 'indeterminate'
    """
    past_n = present_n = 0
    for _, row in script_wm.iterrows():
        dlg = str(row.get('DIALOGUE', '')).strip().lower()
        if dlg in ('true', '1', 'yes'):
            continue
        t = _h1_token_tense(
            row.get('corr_token', ''),
            row.get('TokenSubcat', ''),
        )
        if t == 'past':
            past_n += 1
        elif t == 'present':
            present_n += 1

    total = past_n + present_n
    if total < 5:
        return 'indeterminate'

    ratio = past_n / total
    if ratio >= 0.70:
        return 'past'
    if ratio <= 0.30:
        return 'present'
    return 'mixed_legitimate'


def _h1_sentence_tense(sentence_wm):
    """Dominant tense within a single sentence (excluding dialogue tokens).

    Returns: 'past' | 'present' | 'mixed' | None
    """
    past_n = present_n = 0
    for _, row in sentence_wm.iterrows():
        dlg = str(row.get('DIALOGUE', '')).strip().lower()
        if dlg in ('true', '1', 'yes'):
            continue
        t = _h1_token_tense(
            row.get('corr_token', ''),
            row.get('TokenSubcat', ''),
        )
        if t == 'past':
            past_n += 1
        elif t == 'present':
            present_n += 1
    if past_n == 0 and present_n == 0:
        return None
    if past_n > 0 and present_n == 0:
        return 'past'
    if present_n > 0 and past_n == 0:
        return 'present'
    return 'mixed'   # intra-sentence shift → SS territory, not Tense drift


def _h1_assign_tense_drift(sent_text, sentence_wm, dominant_tense):
    """Assign Tense drift for one sentence: 'Correct', 'Incorrect', or 'NA'.

    Applies the SB1 exemption rules then checks against dominant_tense.
    """
    # Exemption 1: fully in quotes → dialogue uses speaker's own tense
    if _sent_is_fully_quoted(sent_text):
        return 'NA'
    # Exemption 2: only modal/future verbs
    if _sent_is_modal_only(sent_text):
        return 'Correct'
    # Exemption 3: present perfect aspect only
    if _sent_is_pres_perf_only(sent_text):
        return 'Correct'

    # If dominant is indeterminate, don't flag
    if dominant_tense == 'indeterminate':
        return 'NA'

    sent_tense = _h1_sentence_tense(sentence_wm)
    if sent_tense is None:
        return 'NA'
    if sent_tense == 'mixed':
        # Intra-sentence tense shift is SS territory, not Tense drift
        return 'Correct'
    # mixed_legitimate script: sub-episode structure, be lenient
    if dominant_tense == 'mixed_legitimate':
        return 'Correct'

    return 'Correct' if sent_tense == dominant_tense else 'Incorrect'

# ---------------------------------------------------------------------------
# SA1 -- Punc internal helpers
# ---------------------------------------------------------------------------
from collections import Counter as _Counter

# Marks that can appear INSIDE a sentence (not at the boundary)
_INTERNAL_MARKS = {',', ';', ':', '"', "'", '(', ')', '-', '[', ']'}

_MARK_NAMES = {
    ',': 'comma', "'": 'apostrophe', '"': 'doublequote',
    '-': 'dash', ';': 'semicolon', ':': 'colon',
    '(': 'paren', ')': 'paren', '[': 'bracket', ']': 'bracket',
}

def _strip_boundary_marks(text):
    """Strip the terminal mark and wrapping quotes, leaving the sentence body."""
    body = str(text or "")
    body = re.sub(r'^[\s"\']*', '', body)
    body = re.sub(r'[\s"\']*$', '', body)
    if body and body[-1] in '.!?':
        body = body[:-1]
    return body

def _extract_internal_marks(text):
    """Return a Counter of internal marks in the sentence body."""
    body = _strip_boundary_marks(text)
    return _Counter(c for c in body if c in _INTERNAL_MARKS)

def _has_placement_error(raw, corr):
    """
    Detect cases where mark counts are equal but positions differ
    (e.g. comma outside quote vs inside).  Simplified heuristic.
    """
    raw_pat = re.findall(r'[",.\'!?]{2,}', str(raw or ""))
    corr_pat = re.findall(r'[",.\'!?]{2,}', str(corr or ""))
    return raw_pat != corr_pat

def _classify_punc_internal(raw_text, corr_text):
    """
    Compare internal marks between raw and corrected versions.

    Returns (status, subtype, counters_str):
      status       : 'Correct', 'Incorrect', or 'NA'
      subtype      : 'missing comma', 'extra comma', 'missing apostrophe',
                     'multiple', 'placement', or '' for Correct/NA
      counters_str : compact 'key=value,...' string (empty when Correct/NA)
    """
    raw_marks  = _extract_internal_marks(raw_text)
    corr_marks = _extract_internal_marks(corr_text)

    # No internal marks in either version → NA
    if not raw_marks and not corr_marks:
        return 'NA', '', ''

    if raw_marks == corr_marks:
        if _has_placement_error(raw_text, corr_text):
            return 'Incorrect', 'placement', 'placement_error=1'
        return 'Correct', '', ''

    # Compute per-mark differences (corr − raw > 0 means student missed them)
    per_mark = {}
    subtypes_seen = []
    for mark in set(raw_marks.keys()) | set(corr_marks.keys()):
        diff = corr_marks[mark] - raw_marks[mark]
        if diff == 0:
            continue
        # Normalise open/close parens/brackets to single name
        name = _MARK_NAMES.get(mark, 'other')
        if diff > 0:
            key = f'{name}_missed'
            subtypes_seen.append(f'missing {name}')
        else:
            key = f'{name}_extra'
            subtypes_seen.append(f'extra {name}')
        per_mark[key] = per_mark.get(key, 0) + abs(diff)

    subtype = subtypes_seen[0] if len(subtypes_seen) == 1 else 'multiple'
    counters_str = ','.join(f'{k}={v}' for k, v in per_mark.items())
    return 'Incorrect', subtype, counters_str


# ----------------------------------------------------------------------
# BR1 -- Long run-on length fallback
# -----------------------------------------------------------------------
_LENGTH_TRIGGER_TOKENS = 60
_MIN_SUB_SENTENCE_TOKENS = 20
# BR-fix-B (v2): simple CCJJ coordinators (and/but/so/or) are SKIPPED when
# the first half already has >= 25 tokens -- student narrative "and-chains" are
# stylistically intended and splitting late at 'and/but/so' creates sub-
# sentences that begin with a coordinator (looks wrong). Only split at simple
# coordinators when the coordinator is early (before < 25) so both halves
# are balanced. Complex subordinating conjunctions keep the 20-token threshold.
_SIMPLE_COORDINATORS_BR1 = {'and', 'but', 'so', 'or'}
_COMPLEX_COORDINATORS_BR1 = {
    'then', 'that', 'when', 'because',
    'after', 'before', 'while', 'until', 'where', 'which',
}
_COORDINATORS_BR1 = _SIMPLE_COORDINATORS_BR1 | _COMPLEX_COORDINATORS_BR1
_MAX_FIRST_HALF_SIMPLE_COORD = 25  # skip simple coordinators at positions >= 25


def _find_long_run_on_split(tokens, effective_len=None):
    """
    BR1: For a sentence >= _LENGTH_TRIGGER_TOKENS tokens with no comma-splice,
    find the best split point: coordinator closest to midpoint where both
    halves have >= _MIN_SUB_SENTENCE_TOKENS tokens.

    BR-fix-B (v2): simple CCJJ coordinators (and/but/so/or) are REJECTED when
    the first half would be >= _MAX_FIRST_HALF_SIMPLE_COORD (25) tokens.
    This prevents "And/But/So" over-splits that produce second sub-sentences
    starting with a coordinator.

    BR-fix-D: accepts `effective_len` to override len(tokens) for the early-
    return guard, so tokenizer-counted 60-token sentences aren't missed even
    when whitespace-split gives 57-59 tokens.

    Returns (token_index_before_split, reason_str) or (None, None).
    The token at the returned index starts the second sentence.
    """
    n = effective_len if effective_len is not None else len(tokens)
    if n < _LENGTH_TRIGGER_TOKENS:
        return None, None

    split_candidates = []
    for i, tok in enumerate(tokens):
        word = tok.lower().strip(",.!?'\"")
        before = i
        after = len(tokens) - i
        if word in _SIMPLE_COORDINATORS_BR1:
            # BR-fix-B (v2): skip simple coordinators where first half >= 25
            # (late coordinator splits create second-half sentences starting
            # with 'And/But/So', which is stylistically wrong for student prose)
            if (before < _MAX_FIRST_HALF_SIMPLE_COORD and
                    before >= _MIN_SUB_SENTENCE_TOKENS and
                    after >= _MIN_SUB_SENTENCE_TOKENS):
                split_candidates.append(i)
        elif word in _COMPLEX_COORDINATORS_BR1:
            # Standard: both halves >= 20 tokens
            if before >= _MIN_SUB_SENTENCE_TOKENS and after >= _MIN_SUB_SENTENCE_TOKENS:
                split_candidates.append(i)

    if not split_candidates:
        return None, None

    midpoint = len(tokens) // 2
    best = min(split_candidates, key=lambda i: abs(i - midpoint))
    return best, f"long_fused_{n}_tokens"


# ----------------------------------------------------------------------
# BR2 -- RunOnSuspect -> action (convert flag to actual splits)
# -----------------------------------------------------------------------

def _find_comma_splice_position(sentence_text):
    """Find the token index of the first comma-splice position."""
    _splice = re.compile(r",\s*(I|we|he|she|they|it|you)\s+\w+", re.IGNORECASE)
    tokens = sentence_text.split()
    m = _splice.search(sentence_text)
    if not m:
        return None
    before_splice = sentence_text[:m.start()].split()
    # Split BEFORE the pronoun (the comma belongs with the first sentence)
    return len(before_splice)  # index of pronoun token


def _split_run_on_sentence(sent_row):
    """
    BR2: Split a RunOnSuspect=True sentence into two sentence rows.

    Returns list of one or two row dicts. If split is not possible, returns
    the original row unchanged.
    """
    try:
        corr = str(sent_row.get("CorrectedSentence", "") or "")
        tokens = corr.split()

        # Get split point
        split_pt = sent_row.get("RunOnSplitPoint")
        reason = str(sent_row.get("RunOnReason", "") or "")

        if split_pt is None and "comma_splice" in reason:
            split_pt = _find_comma_splice_position(corr)

        if split_pt is None or not isinstance(split_pt, (int, float)):
            return [sent_row]

        split_pt = int(split_pt)
        # BR-fix-A: require at least 5 tokens on the left to prevent
        # stranded-pronoun fragments like "Hopes high, I." (split_pt=3).
        # Previously 3, which allowed exactly-3 splits to pass the guard.
        if split_pt < 5 or split_pt > len(tokens) - 3:
            return [sent_row]  # Too close to edge -- skip

        first_tokens = tokens[:split_pt]
        second_tokens = tokens[split_pt:]

        # First sentence: capture original splice mark, remove it, add period
        first_raw = " ".join(first_tokens)
        original_splice_mark = ""
        if first_raw and first_raw[-1] in ",;:":
            original_splice_mark = first_raw[-1]
            first_text = first_raw[:-1].rstrip()
        else:
            first_text = first_raw.rstrip()

        if not first_text.endswith((".", "!", "?")):
            first_text += "."

        # Second sentence: capture original first word, then capitalise
        second_raw = " ".join(second_tokens)
        original_first_word = second_tokens[0] if second_tokens else ""
        if second_raw and second_raw[0].islower():
            second_text = second_raw[0].upper() + second_raw[1:]
        else:
            second_text = second_raw

        # Build two rows
        orig_ref = str(sent_row.get("SentenceRef", ""))
        row_a = dict(sent_row)
        row_a["CorrectedSentence"] = first_text
        row_a["SentenceRef"] = orig_ref + "a"
        row_a["BoundaryProvenance"] = "machine-runon"
        row_a["OriginalSpliceMark"] = original_splice_mark
        row_a["RunOnSplitInsertion"] = (original_splice_mark == "")
        row_a["RunOnSuspect"] = "False"  # Resolved
        # BR6: recompute TokensInSentence from actual text, not inherited parent count
        row_a["TokensInSentence"] = len(first_text.split())

        row_b = dict(sent_row)
        row_b["CorrectedSentence"] = second_text
        row_b["SentenceRef"] = orig_ref + "b"
        row_b["BoundaryProvenance"] = "machine-runon"
        row_b["OriginalFirstWord"] = original_first_word
        row_b["OriginalSpliceMark"] = ""
        row_b["RunOnSplitInsertion"] = False
        row_b["RunOnSuspect"] = "False"  # Resolved
        # BR6: recompute TokensInSentence from actual text, not inherited parent count
        row_b["TokensInSentence"] = len(second_text.split())

        return [row_a, row_b]
    except Exception:
        return [sent_row]


def apply_run_on_splits_to_sentences(sent_df):
    """
    BR2: Convert all RunOnSuspect=True sentences into split pairs.
    Returns a new sentence DataFrame with split sentences inserted.

    New columns added:
    - OriginalSpliceMark: the comma/semicolon at the splice point (or '')
    - RunOnSplitInsertion: True if a terminal mark was machine-inserted
    - OriginalFirstWord: original lowercase first word of second sub-sentence
    - MergedFromOversplit: False (placeholder; BR5 sets True at word-map level)
    """
    try:
        rows = []
        for _, row in sent_df.iterrows():
            if str(row.get("RunOnSuspect", "False")) != "True":
                rows.append(dict(row))
            else:
                rows.extend(_split_run_on_sentence(row))

        result = pd.DataFrame(rows).reset_index(drop=True)

        # Ensure preservation columns exist
        for col in ("OriginalSpliceMark", "OriginalFirstWord"):
            if col not in result.columns:
                result[col] = ""
        for col in ("RunOnSplitInsertion",):
            if col not in result.columns:
                result[col] = False

        return result
    except Exception:
        return sent_df


# ----------------------------------------------------------------------
# BR8 -- Recursive run-on splitting (up to _MAX_SPLIT_DEPTH passes)
# -----------------------------------------------------------------------
_MAX_SPLIT_DEPTH = 4


def _redetect_run_on(sent_df):
    """
    Re-run the comma-splice and length-based run-on detectors on a
    sentence DataFrame, producing fresh RunOnSuspect/RunOnSplitPoint/
    RunOnReason columns. Used by BR8's recursive split logic.
    """
    _splice_re = re.compile(r",\s*(I|we|he|she|they|it|you)\s+\w+", re.IGNORECASE)
    flags, points, reasons = [], [], []
    for _, row in sent_df.iterrows():
        corr = str(row.get("CorrectedSentence", "") or "")
        n_toks = int(row.get("TokensInSentence", 0) or 0)
        tokens = corr.split()
        splices = _splice_re.findall(corr)

        is_run_on = (len(splices) >= 2 or
                     (len(splices) == 1 and n_toks >= 40) or
                     (len(splices) == 1 and n_toks >= 20))
        split_pt = None
        reason = ""

        if is_run_on and splices:
            reason = "comma_splice"
            m = _splice_re.search(corr)
            if m:
                before_splice = corr[:m.start()].split()
                # Fix-A-complement: use same formula as _find_comma_splice_position
                # (no +1). The +1 caused split_pt to land ON the pronoun token,
                # putting it in the first half and creating stranded fragments.
                split_pt = len(before_splice)

        # BR-fix-D (same as in enrich_sentences): use max to catch tokenizer/split discrepancy
        effective_toks = max(len(tokens), n_toks)
        if not is_run_on and effective_toks >= _LENGTH_TRIGGER_TOKENS:
            quote_chars = corr.count('"') + corr.count('"') + corr.count('"')
            if quote_chars < 4:
                # Pass effective_toks so the inner guard uses the tokenizer count,
                # not the whitespace-split count (BR-fix-D)
                pt, rsn = _find_long_run_on_split(tokens, effective_len=effective_toks)
                if pt is not None:
                    is_run_on = True
                    split_pt = pt
                    reason = rsn

        flags.append("True" if is_run_on else "False")
        points.append(split_pt)
        # Preserve existing RunOnReason for non-run-on rows (e.g. fused_clause_pattern
        # set by Rule 3 should not be overwritten with empty string)
        if is_run_on:
            reasons.append(reason)
        else:
            orig = str(row.get("RunOnReason", "") or "")
            reasons.append(orig if orig else "")

    sent_df = sent_df.copy()
    sent_df["RunOnSuspect"] = flags
    sent_df["RunOnSplitPoint"] = points
    sent_df["RunOnReason"] = reasons
    return sent_df


def apply_run_on_splits_recursive(sent_df, depth=0):
    """
    BR8: Recursively apply run-on splits up to _MAX_SPLIT_DEPTH times.
    After each split pass, re-detect run-on conditions on the new rows
    and split again if needed.
    """
    if depth >= _MAX_SPLIT_DEPTH:
        return sent_df

    # Check if any rows need splitting
    if not (sent_df["RunOnSuspect"].astype(str) == "True").any():
        return sent_df

    # Apply one round of splits
    result = apply_run_on_splits_to_sentences(sent_df)

    # Re-detect on all rows (split rows now have smaller text)
    result = _redetect_run_on(result)

    # Recurse if still needed
    return apply_run_on_splits_recursive(result, depth=depth + 1)


# ----------------------------------------------------------------------
# BR4 -- Rule 3: fused-clause detector
# -----------------------------------------------------------------------
# Catches no-comma fused run-ons: two or more independent clauses
# concatenated without any internal punctuation. Fires on sentences >= 20
# tokens where a subject + finite verb appears after a non-separator,
# non-coordinator content word.
#
# Rule order: Rule 1 → Rule 2 → Rule 3 → BR1 (length fallback)
# Rule 3 runs before BR8 recursion, which then handles sub-sentences.
# -----------------------------------------------------------------------

_R3_HARD_COORDINATORS = frozenset({'and', 'but', 'or', 'nor'})

_R3_SUBORDINATORS = frozenset({
    'when', 'while', 'because', 'if', 'although', 'though', 'since',
    'before', 'after', 'until', 'till', 'as', 'unless', 'whereas',
    'whenever', 'wherever', 'whilst', 'once',
})

_R3_SUBJECT_PRONOUNS = frozenset({
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'this', 'that', 'these', 'those', 'there',
})

_R3_REPORTING_VERBS = frozenset({
    'said', 'asked', 'replied', 'answered', 'exclaimed', 'whispered',
    'shouted', 'called', 'cried', 'muttered', 'added', 'continued',
    'began', 'told', 'yelled', 'demanded', 'repeated', 'suggested',
    'agreed', 'declared', 'announced', 'mumbled', 'remarked', 'observed',
    'insisted', 'admitted', 'confessed', 'sighed', 'gasped', 'screamed',
    'hissed', 'growled', 'snapped', 'grumbled', 'murmured', 'mentioned',
    'noted', 'wondered', 'queried', 'laughed', 'responded', 'whines',
    'questioned', 'says', 'asks', 'replies', 'answers', 'exclaims',
    'whispers', 'shouts', 'calls', 'cries', 'mutters', 'adds',
    'continues', 'tells', 'yells', 'demands', 'repeats', 'suggests',
    'agrees', 'declares', 'announces', 'mumbles', 'remarks', 'observes',
    'insists', 'admits', 'confesses', 'sighs', 'gasps', 'screams',
    'hisses', 'growls', 'snaps', 'grumbles', 'murmurs', 'mentions',
    'notes', 'wonders', 'queries', 'laughs', 'responds', 'questions',
})

_R3_SEPARATOR_MARKS = frozenset({'.', '!', '?', ',', ';', ':', '--', '-', '...'})
_R3_QUOTE_MARKS = frozenset({'"', '"', '"', "''", "'"})

_R3_MIN_SENT_TOKENS = 20  # sentence-level minimum; below this Rule 3 doesn't fire
_R3_MIN_FRAGMENT_TOKENS = 8  # minimum tokens in any resulting fragment
# Raised 5->8: a 5-7 token "fragment" is rarely a genuine independent clause;
# the smaller threshold produced dangling tails like "The line was so long."


def _r3_is_finite_verb(token_subcat):
    """True for verb and auxiliary verb subcategories."""
    return token_subcat in ('verb', 'auxiliary verb')


def _r3_is_coordinator(text, at_sent_start):
    """R3a: true when text is a coordinator that vetoes splits before it.

    `so` at sentence start is a connective adverb (not a coordinator).
    `so` mid-sentence is a coordinator.
    """
    t = text.lower().strip()
    if t in _R3_HARD_COORDINATORS:
        return True
    if t == 'so' and not at_sent_start:
        return True
    return False


_R3_VERB_RUN_SUBCATS = frozenset({'verb', 'auxiliary verb', 'adverb'})


def _r3_coordinator_in_window(tokens, i, prior_candidates):
    """R3a (windowed): veto the split when a coordinator joins directly to the
    candidate through a run of verbs/adverbs -- i.e. the candidate is the
    complement of a coordinated predicate, not a fused new sentence.

    Walk backward from i-1 over a contiguous run of verb/auxiliary/adverb
    tokens (a coordinated predicate sharing the earlier subject). If the token
    immediately before that run is a coordinator (and/but/or/nor, or
    mid-sentence "so"), veto.

    Example (vetoed): "...raced to the same ride but realised the line was so
    long" -- Pattern D fires on "the line was", but the run back is
    ["realised"] (a bare coordinated verb with no subject of its own) and the
    token before it is "but" -> veto. Checking only tokens[i-1] (="realised")
    missed the "but".

    Counter-example (NOT vetoed): "...shops and bought milk I came home the dog
    barked" -- Pattern B fires on "I came"; tokens[i-1] is "milk" (a noun, not a
    verb), so the verb-run is empty and the token there ("milk") is not a
    coordinator. The earlier "and" coordinates "went...and bought" and is
    correctly out of scope, so the genuine fused boundary survives.

    Also handles the case where the coordinator sits immediately before the
    candidate (empty verb-run), preserving the original single-token veto.
    """
    prior_set = set(prior_candidates)
    j = i - 1
    while j > 0:
        sub = tokens[j].get('TokenSubcat', '')
        t = str(tokens[j].get('corr_token', '') or '').strip().lower()
        if not t:
            j -= 1
            continue
        if t in _R3_SEPARATOR_MARKS:
            return False
        if j in prior_set:
            return False
        # Coordinator-by-text is checked BEFORE the verb/adverb run-skip:
        # "so" is POS-tagged as an adverb, so skipping the run first would walk
        # past it and miss the veto. Any coordinator found while traversing the
        # coordinated-predicate run vetoes the split.
        if t in _R3_HARD_COORDINATORS:
            return True
        if t == 'so':
            # mid-sentence "so" is a coordinator (loop only visits j >= 1, so a
            # "so" reached here is never the sentence's first token)
            return True
        if sub in _R3_VERB_RUN_SUBCATS:
            j -= 1
            continue
        break  # non-verb token that isn't a coordinator -- predicate run ends
    return False


_R3_RESTRICTED_PRONOUNS = frozenset({'it', 'that', 'this', 'these', 'those', 'there'})
# Adjective excluded: "gone deaf it's..." -- adjective ends the preceding clause, so
# "it" after an adjective genuinely starts a new clause and should NOT be suppressed.
_R3_NOUN_SUBCATS = frozenset({'noun', 'proper noun', 'pronoun', 'determiner', 'numeral'})


def _r3_refine_with_subordinator(tokens, candidate_pos):
    """R3b: walk backward from candidate_pos. If a subordinator is found
    with no separator or coordinator between, move the split to before it.

    Stops at: separator marks, coordinators (and/but/or/nor/so).
    This prevents crossing coordinator-joined phrases (e.g. "tried as hard
    as he could" -- the `as` belongs with `tried`, not the following clause).
    Only moves back if the resulting fragment still meets the MIN threshold.
    """
    for j in range(candidate_pos - 1, -1, -1):
        t = tokens[j].get('corr_token', '').strip().lower()
        if not t:
            continue
        if t in _R3_SEPARATOR_MARKS:
            break
        if t in _R3_HARD_COORDINATORS or t == 'so':
            break  # don't cross coordinator boundaries
        if t in _R3_SUBORDINATORS:
            if j >= _R3_MIN_FRAGMENT_TOKENS:
                return j
            break
    return candidate_pos


def find_fused_clause_boundaries_r3(tokens):
    """Scan a token list (dicts with corr_token, TokenSubcat, TokenCategory)
    for fused-clause boundary candidates.

    Returns sorted list of token indices where a split should be inserted.
    Indices reference positions in the passed token list.

    Key guards:
    - Fragment guard: i >= _R3_MIN_FRAGMENT_TOKENS (no tiny leading fragments)
    - Coordinator veto R3a: and/but/or/nor/so vetoes the split before them
    - Subordinator guard: if prev token is subordinator, inside sub-clause
    - Relative pronoun guard: `that`/`which`/`who` etc. after a noun = relative clause
    - Restricted pronouns (it/that/this/these/those/there): only fire when prev
      is NOT a noun/adjective/pronoun (avoids relative-clause false positives)
    """
    candidates = []
    n = len(tokens)
    in_quote = False

    for i, tok in enumerate(tokens):
        text = str(tok.get('corr_token', '') or '').strip()
        text_lower = text.lower()
        subcat = tok.get('TokenSubcat', '')
        cat = tok.get('TokenCategory', '')

        # Track quote state (suppress splits inside dialogue)
        if text in _R3_QUOTE_MARKS:
            in_quote = not in_quote
            continue

        if not text:
            continue
        if i == 0:
            continue  # first token always starts the sentence

        if in_quote:
            continue

        # Fragment guard -- checked first for all patterns
        if i < _R3_MIN_FRAGMENT_TOKENS:
            continue

        # What immediately precedes this position?
        prev_text = str(tokens[i - 1].get('corr_token', '') or '').strip()
        prev_lower = prev_text.lower()
        prev_subcat = tokens[i - 1].get('TokenSubcat', '')
        at_prev_sent_start = (i - 1 == 0)

        # If preceded by a separator, already bounded -- skip
        if prev_text in _R3_SEPARATOR_MARKS:
            continue

        # R3a (windowed): a coordinator anywhere between the previous clause
        # boundary and this position vetoes the split -- the candidate verb/
        # clause is coordinated with the preceding predicate, not fused.
        if _r3_coordinator_in_window(tokens, i, candidates):
            continue

        # Guard: if the preceding token is a subordinator, the subject-verb
        # pair is inside a subordinate clause -- do NOT mark as fused boundary
        if prev_lower in _R3_SUBORDINATORS:
            continue

        # Also suppress if preceded by relative-clause pronouns
        if prev_lower in ('who', 'which', 'that', 'whom', 'whose'):
            continue

        # NOTE: subordinators are NEVER direct split triggers (spec R3b).
        # They only move an existing Pattern B/C/D split BACKWARD via refinement.
        if text_lower in _R3_SUBORDINATORS:
            continue

        # Pattern B: subject pronoun + finite verb
        if text_lower in _R3_SUBJECT_PRONOUNS:
            if i + 1 < n:
                next_tok = tokens[i + 1]
                next_subcat = next_tok.get('TokenSubcat', '')
                if _r3_is_finite_verb(next_subcat):
                    # Restricted pronouns: only fire when NOT after a noun/adj/pronoun
                    # (avoids relative-clause false positives like "junk that filled")
                    if text_lower in _R3_RESTRICTED_PRONOUNS:
                        if prev_subcat in _R3_NOUN_SUBCATS:
                            continue
                    candidates.append(i)
            continue

        # Pattern C: proper noun + finite verb (not a reporting verb)
        if subcat == 'proper noun' and cat == 'word':
            if i + 1 < n:
                next_tok = tokens[i + 1]
                next_subcat = next_tok.get('TokenSubcat', '')
                next_text_lower = str(next_tok.get('corr_token', '') or '').lower().strip()
                if _r3_is_finite_verb(next_subcat):
                    if next_text_lower in _R3_REPORTING_VERBS:
                        continue  # dialogue attribution -- don't split
                    candidates.append(i)
            continue

        # Pattern D: determiner + noun + finite verb
        if subcat == 'determiner' and cat == 'word':
            if i + 2 < n:
                noun_tok = tokens[i + 1]
                verb_tok = tokens[i + 2]
                if noun_tok.get('TokenSubcat', '') in ('noun', 'proper noun'):
                    if _r3_is_finite_verb(verb_tok.get('TokenSubcat', '')):
                        candidates.append(i)

    # R3b: refine each candidate by backing up to any preceding subordinator
    raw_refined = []
    seen = set()
    for c in sorted(candidates):
        r = _r3_refine_with_subordinator(tokens, c)
        if r not in seen and r >= _R3_MIN_FRAGMENT_TOKENS:
            raw_refined.append(r)
            seen.add(r)

    # Final pass: remove candidates that would create too-small fragments
    # (catches cases where two candidates are very close together or the
    #  last fragment would be tiny)
    filtered = []
    prev_pos = 0
    for r in sorted(raw_refined):
        if r - prev_pos >= _R3_MIN_FRAGMENT_TOKENS:
            filtered.append(r)
            prev_pos = r
    # Check last fragment
    if filtered and len(tokens) - filtered[-1] < _R3_MIN_FRAGMENT_TOKENS:
        filtered.pop()

    return filtered


def _r3_tokens_to_text(tokens):
    """Join corr_token values into sentence text, add period if needed."""
    parts = [str(t.get('corr_token', '') or '').strip() for t in tokens
             if str(t.get('corr_token', '') or '').strip()]
    text = ' '.join(parts).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ('.', '!', '?'):
        text += '.'
    return text


def apply_rule3_splits(sent_df, wm):
    """BR4: Apply Rule 3 fused-clause detection.

    Uses word_map TokenSubcat for structural boundary detection.
    Produces multiple sub-sentences from a fused run-on.
    Sub-sentences get a/b/c suffix on SentenceRef.
    Called before BR8 so recursion handles any remaining run-ons.
    """
    if wm is None or wm.empty:
        return sent_df

    # Build token lookup: SentenceRef → ordered list of token dicts
    by_ref = {}
    for ref, g in wm.groupby('SentenceRef', sort=False):
        corrected = (g[g['corr_index'].notna() &
                       (g['corr_token'].astype(str).str.strip() != '')]
                     .sort_values('corr_index'))
        if not corrected.empty:
            by_ref[ref] = corrected[['corr_token', 'TokenCategory',
                                     'TokenSubcat']].to_dict('records')

    rows = []
    for _, sent_row in sent_df.iterrows():
        ref = str(sent_row.get('SentenceRef', ''))
        tokens = by_ref.get(ref)

        if not tokens or len(tokens) < _R3_MIN_SENT_TOKENS:
            rows.append(dict(sent_row))
            continue

        candidates = find_fused_clause_boundaries_r3(tokens)

        if not candidates:
            rows.append(dict(sent_row))
            continue

        # Build sub-sentence fragments
        split_points = [0] + candidates + [len(tokens)]
        fragments = []
        for k in range(len(split_points) - 1):
            frag_tokens = tokens[split_points[k]:split_points[k + 1]]
            # Skip empty or too-small fragments
            word_tokens = [t for t in frag_tokens
                           if str(t.get('corr_token', '') or '').strip()]
            if len(word_tokens) < _R3_MIN_FRAGMENT_TOKENS:
                # Too small -- absorb into the previous fragment if possible
                if fragments:
                    fragments[-1] = fragments[-1] + frag_tokens
                else:
                    fragments.append(frag_tokens)
            else:
                fragments.append(frag_tokens)

        if len(fragments) <= 1:
            rows.append(dict(sent_row))
            continue

        # Build rows for each fragment
        suffixes = [chr(ord('a') + k) for k in range(len(fragments))]
        orig_ref = ref
        for k, (frag_toks, suffix) in enumerate(zip(fragments, suffixes)):
            sub_text = _r3_tokens_to_text(frag_toks)
            new_row = dict(sent_row)
            new_row['CorrectedSentence'] = sub_text
            new_row['TokensInSentence'] = len(sub_text.split())
            new_row['SentenceRef'] = orig_ref + suffix
            new_row['BoundaryProvenance'] = 'machine-runon'
            new_row['RunOnReason'] = 'fused_clause_pattern'
            new_row['OriginalSpliceMark'] = ''
            new_row['RunOnSplitInsertion'] = (k == 0)
            original_words = [str(t.get('corr_token', '') or '').strip()
                              for t in frag_toks
                              if str(t.get('corr_token', '') or '').strip()]
            new_row['OriginalFirstWord'] = original_words[0] if original_words else ''
            new_row['RunOnSuspect'] = 'False'  # BR8 will re-detect if needed
            rows.append(new_row)

    result = pd.DataFrame(rows).reset_index(drop=True)
    for col in ('OriginalSpliceMark', 'OriginalFirstWord', 'RunOnSplitInsertion',
                'RunOnReason', 'RunOnSplitPoint'):
        if col not in result.columns:
            result[col] = ''
    return result


# ----------------------------------------------------------------------
# BR13 -- SoundEffect detection
# -----------------------------------------------------------------------
def _is_sound_effect(text):
    """
    Detect onomatopoeia / sound effects / vocal exclamations.
    Conservative: short (<=3 words), either mostly uppercase OR has heavy
    letter repetition, ends with ! or multiple !!.
    """
    t = str(text or "").strip()
    if not t:
        return False
    words = t.rstrip("!?.").split()
    if not words or len(words) > 3:
        return False

    has_terminal_emphasis = t.rstrip().endswith('!')

    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    cap_ratio = sum(1 for c in letters if c.isupper()) / len(letters)

    # Heavy letter repetition (WOOOOOO -- very few unique chars)
    has_repetition = any(
        len(w) >= 4 and len(set(w.lower())) <= max(2, len(w) // 3)
        for w in words
    )

    # Triple consecutive repeat (Aaargh → 'aaa', Eeeeek → 'eee')
    # catches mixed-case exclamations missed by has_repetition
    has_triple_repeat = any(
        bool(re.search(r'(.)\1\1', w.lower()))
        for w in words
    )

    # Single short mostly-caps word with ! terminal
    is_short_caps = (
        len(words) == 1 and
        len(words[0].rstrip("!?.")) <= 8 and
        cap_ratio >= 0.6 and
        has_terminal_emphasis
    )

    return has_repetition or has_triple_repeat or is_short_caps


# ----------------------------------------------------------------------
# BR9 -- Normalise BR3-split SentenceRefs to use a/b suffix convention
# -----------------------------------------------------------------------
def _apply_br9_renaming(sent_df, wm_with_br3_origin):
    """
    BR9: Rename BR3-split SentenceRefs from integer format to a/b suffix.
    wm_with_br3_origin is the word map with _br3_origin_sid column.
    """
    if "_br3_origin_sid" not in wm_with_br3_origin.columns:
        return sent_df

    df = sent_df.copy()
    idcol = "Identifier" if "Identifier" in wm_with_br3_origin.columns else "ID"

    for ident, wm_g in wm_with_br3_origin.groupby(idcol, sort=False):
        # Find rows with BR3 split origin
        br3_splits = wm_g[wm_g["_br3_origin_sid"].notna()].copy()
        if br3_splits.empty:
            continue

        # Group by origin sentence ID
        for origin_sid, grp in br3_splits.groupby("_br3_origin_sid"):
            try:
                origin_sid_int = int(origin_sid)
            except (TypeError, ValueError):
                continue

            # The origin sentence gets suffix 'a'
            origin_sref = f"{ident}_s{origin_sid_int:03d}"
            new_origin_sref = f"{origin_sref}a"

            # Each split gets suffix 'b', 'c', etc.
            new_sids = sorted(pd.to_numeric(grp["CorrSentenceID"],
                                            errors="coerce").dropna().unique())
            for i, new_sid in enumerate(new_sids):
                try:
                    new_sid_int = int(new_sid)
                except (TypeError, ValueError):
                    continue
                new_sref = f"{ident}_s{new_sid_int:03d}"
                suffix = chr(ord('b') + i)  # b, c, d...
                renamed_sref = f"{origin_sref}{suffix}"

                # Update SentenceRef in sentence layer
                df.loc[df["SentenceRef"] == origin_sref, "SentenceRef"] = new_origin_sref
                df.loc[df["SentenceRef"] == new_sref, "SentenceRef"] = renamed_sref

    return df


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

    def _is_dialogue_group(grp):
        """WA2 converts DIALOGUE bool→str ('True'/'False'). Handle both."""
        col = grp.get("DIALOGUE", pd.Series(["False"]))
        return bool((col.astype(str).str.lower() == "true").any())

    # H1 -- per-script dominant tense via verb-level majority vote
    # (non-dialogue tokens only; 4 states: past / present /
    #  mixed_legitimate / indeterminate)
    script_dominant_tense = {}
    for ident, g in wm.groupby("Identifier", sort=False):
        script_dominant_tense[ident] = _h1_compute_dominant_tense(g)

    # Keep legacy script_tense for any other callers that may reference it
    script_tense = {k: ("past" if v in ("past", "mixed_legitimate", "indeterminate")
                        else "present")
                    for k, v in script_dominant_tense.items()}

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
                # A2b: if the raw sentence has ANY terminal mark the student
                # did supply punctuation, just misplaced -- use a narrower label.
                _raw_sent = str(row.get("RawSentence", "") or "")
                cat = ("wrong end boundary"
                       if re.search(r"[.!?]", _raw_sent)
                       else "missing all")
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
        # BR13: detect SoundEffect sentences
        if is_art:
            col5.append("NA")
        elif _is_sound_effect(str(row.get("CorrectedSentence", "") or "")):
            col5.append("SoundEffect")
        else:
            col5.append(ENGINE)

        # ---- Pronoun column (engine) ----
        if is_art or g is None:
            pron_v.append("NA"); pron_s.append("NA")
        else:
            pron_v.append(ENGINE); pron_s.append(ENGINE)

        # ---- Tense drift (H1: per-script dominant tense + SB1 exemptions) ----
        if is_art or g is None:
            tense_v.append("NA")
        else:
            dominant = script_dominant_tense.get(ident, 'indeterminate')
            corr_text = str(row.get("CorrectedSentence", "") or "")
            tense_v.append(
                _h1_assign_tense_drift(corr_text, g, dominant))
        prev_ident = ident

    # BR13: SoundEffect sentences are exempt from SS internal and tense drift scoring
    for i, stype in enumerate(col5):
        if stype == "SoundEffect":
            col1[i] = "NA"
            tense_v[i] = "NA"

    s["SS internal"] = col1
    s["Punc boundary"] = col2
    s["Grammar subtype (most serious)"] = col3
    s["Punc boundary subtype"] = col4
    s["Sentence type"] = col5
    s["Pronoun ref"] = pron_v
    s["Pronoun ref subtype"] = pron_s
    s["Tense drift"] = tense_v

    # SA1 -- Punc internal (three new columns)
    pi_status, pi_subtype, pi_counters = [], [], []
    for _, row in s.iterrows():
        st, sub, cnt = _classify_punc_internal(
            row.get("RawSentence", ""), row.get("CorrectedSentence", "")
        )
        pi_status.append(st)
        pi_subtype.append(sub)
        pi_counters.append(cnt)
    s["Punc internal"] = pi_status
    s["Punc internal subtype"] = pi_subtype
    s["Punc internal counters"] = pi_counters

    # B1 -- BoundaryProvenance rollup
    s = _rollup_boundary_provenance(wm, s)

    # D1 + BR1 -- RunOnSuspect flag (comma-splice detection + length fallback)
    _splice_re = re.compile(r",\s*(I|we|he|she|they|it|you)\s+\w+", re.IGNORECASE)
    run_on_flags = []
    run_on_split_points = []
    run_on_reasons = []
    for _, row in s.iterrows():
        corr = str(row.get("CorrectedSentence", "") or "")
        n_toks = int(row.get("TokensInSentence", 0) or 0)
        tokens = corr.split()
        splices = _splice_re.findall(corr)

        # Existing D1 comma-splice detection
        is_run_on = (len(splices) >= 2 or
                     (len(splices) == 1 and n_toks >= 40) or
                     (len(splices) == 1 and n_toks >= 20))
        split_pt = None
        reason = ""

        if is_run_on and splices:
            reason = "comma_splice"
            m = _splice_re.search(corr)
            if m:
                before_splice = corr[:m.start()].split()
                split_pt = len(before_splice)  # index of pronoun token

        # BR1 length-based fallback (only if not already flagged)
        # BR-fix-D: use max(whitespace-split count, TokensInSentence) so that
        # sentences like BGRRHYPQ_s008 (corr.split()=59, tokenize()=60) are
        # caught even when the tokenizer and whitespace-split disagree by 1-3.
        effective_toks = max(len(tokens), n_toks)
        if not is_run_on and effective_toks >= _LENGTH_TRIGGER_TOKENS:
            # Skip if mostly dialogue (lots of quote marks)
            quote_chars = (corr.count('"') + corr.count('"') +
                           corr.count('"'))
            if quote_chars < 4:
                # BR-fix-D: pass effective_toks so the inner guard uses the
                # tokenizer count, not the whitespace-split count
                pt, rsn = _find_long_run_on_split(tokens, effective_len=effective_toks)
                if pt is not None:
                    is_run_on = True
                    split_pt = pt
                    reason = rsn

        run_on_flags.append("True" if is_run_on else "False")
        run_on_split_points.append(split_pt)
        run_on_reasons.append(reason if is_run_on else "")

    s["RunOnSuspect"] = run_on_flags
    s["RunOnSplitPoint"] = run_on_split_points
    s["RunOnReason"] = run_on_reasons

    # BR5 -- MergedFromOversplit rollup from word map to sentence layer
    # _merge_orphaned_reporting_clauses sets this on word-map rows; roll up here.
    if "MergedFromOversplit" in wm.columns:
        merged_flags = []
        for _, row in s.iterrows():
            ref = row.get("SentenceRef", "")
            g = by_ref.get(ref)
            if g is not None and "MergedFromOversplit" in g.columns:
                is_merged = (g["MergedFromOversplit"].astype(str) == "True").any()
            else:
                is_merged = False
            merged_flags.append("True" if is_merged else "False")
        s["MergedFromOversplit"] = merged_flags
    else:
        s["MergedFromOversplit"] = "False"

    return s

# ---------------------------------------------------------------------------
# CB6 -- Broadened CUTOFF detection
# ---------------------------------------------------------------------------

_CB6_CONTINUATION_WORDS = frozenset({
    # Subordinators requiring a clause
    "because", "although", "though", "since", "while", "unless", "if", "as",
    "whereas", "until", "when",
    # Coordinators ("and/but ending = almost always cutoff")
    "and", "but", "or", "nor",
    # Prepositions strongly requiring NP
    # ("about" excluded -- stranded preposition in "knew about." is complete)
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into",
    "onto", "over", "under", "through", "during",
    # Articles / possessives (cannot stand alone)
    "a", "an", "the", "my", "your", "his", "her", "its", "our", "their",
    "some", "any",
    # Modals / auxiliaries requiring VP
    # (do/does/did excluded -- can end sentences emphatically: "we really did.")
    "will", "would", "should", "can", "could", "might", "may", "must",
    "shall", "is", "are", "was", "were", "has", "have", "had",
    "be", "been", "being",
    # REMOVED: "after", "before", "so", "yet", "this", "that", "these", "those"
    # REMOVED: "do", "does", "did" -- can be sentence-final emphatic auxiliaries
})

# Stylistic one-word closers that are NOT cutoffs
_CB6_CLOSER_ALLOW = frozenset({
    "the end", "fin", "finis", "done", "over", "finished",
    "goodbye", "goodnight", "farewell",
})

# Common abbreviations that are not mid-word fragments
_CB6_ABBREVS = frozenset({
    "USA", "UK", "US", "UN", "EU", "FBI", "CIA", "NASA", "CEO", "PM",
    "AM", "PhD", "MD", "BA", "MA", "TV", "ID", "OK",
})

_CB6_PUNCT = set('.!?,;:\'"')
_CB6_TERMINALS_RE = re.compile(r'[.!?]|\.\.\.')
# CB6b: stylistic trailing ellipsis (3+ dots at end, optionally after ?/!)
_CB6_TRAILING_ELLIPSIS_RE = re.compile(r'\.{3,}\s*$')
# CB6b: leading closing-quote attribution fragment, e.g. `," explained X.`
# produced when the sacred-terminal-guard splits extended dialogue at a
# student terminal mark. The lone quote here CLOSES a quote opened earlier,
# so it must not be read as an unclosed (cutoff) quote.
_CB6_LEADING_CLOSE_QUOTE_RE = re.compile(r'^\s*[,.]\s*["\u201c\u201d]')


def _cb6_count_quotes(text: str) -> int:
    """CB6b: count dialogue double-quote characters (ASCII and curly).
    Straight single quotes are skipped -- too easily confused with
    apostrophes in contractions."""
    t = str(text or "")
    return t.count('"') + t.count('\u201d') + t.count('\u201c')


def _cb6_is_cutoff_sent(text: str) -> bool:
    """CB6 broadened CUTOFF at sentence level: fires if any indicator is true.

    Indicators (positional gate - last sentence - enforced by caller):
      r0. 0 content words, or 1 content word with no terminal
      r1. Last content word is a continuation word
      r2. Unclosed quotation -- odd within-sentence double-quote count, last
          char not a closing quote. CB6b: skipped for leading closing-quote
          attribution fragments (e.g. `," explained X.`).
      r3. Exactly 1 content word not on closer allow-list -- CB6b: skipped
          if the sentence ends with ?/! or has stylistic trailing ellipsis
      r4. Last content token is ALL-CAPS alpha, length 1-3, not known abbrev
      r5. No terminal AND <=8 content words
    """
    t = str(text or "").strip()
    if not t:
        return True

    has_terminal = bool(_CB6_TERMINALS_RE.search(t))
    content = re.findall(r'\S+', t)
    content = [w for w in content if re.search(r'[A-Za-z0-9]', w)]
    n_words = len(content)
    last_word = re.sub(r'[^\w]', '', content[-1]).lower() if content else ""

    # r0: 0 words, or 1 word no terminal
    if n_words == 0:
        return True
    if n_words == 1 and not has_terminal:
        return True

    # r1: ends on continuation word
    if last_word in _CB6_CONTINUATION_WORDS:
        return True

    # r2: unclosed double-quotation mark (within-sentence odd count, last
    # char not a closing quote). CB6b: skip leading closing-quote attribution
    # fragments -- their lone quote closes dialogue opened in a prior
    # sentence, so the fragment is not truncated.
    quote_count = _cb6_count_quotes(t)
    last_char = t[-1]
    if (quote_count % 2 == 1
            and last_char not in {'"', '\u201d'}
            and not _CB6_LEADING_CLOSE_QUOTE_RE.match(t)):
        return True

    # r3: exactly 1 content word, not on closer allow-list -- CB6b: skip
    # stylistic short questions/exclamations and deliberate trailing ellipsis.
    if n_words == 1 and last_word not in _CB6_CLOSER_ALLOW:
        ends_q_or_excl = last_char in {'?', '!'}
        has_trailing_ellipsis = bool(_CB6_TRAILING_ELLIPSIS_RE.search(t))
        if not ends_q_or_excl and not has_trailing_ellipsis:
            return True

    # r4: ALL-CAPS final word, length 1-3, not known abbreviation
    last_word_raw = re.sub(r'[^\w]', '', content[-1]) if content else ""
    if (last_word_raw.isupper() and last_word_raw.isalpha()
            and 1 <= len(last_word_raw) <= 3
            and last_word_raw not in _CB6_ABBREVS):
        return True

    # r5: no terminal AND <=8 content words
    if not has_terminal and n_words <= 8:
        return True

    return False


def _apply_allcaps_title_split(wm):
    """CB6-TITLE Option A: detect a prose-fused title in the first body sentence.

    If the first non-TITLE sentence starts with >=2 consecutive ALL-CAPS
    alphabetic tokens (each >=2 chars) followed by mixed-case prose, split
    the ALL-CAPS prefix into a new CorrSentenceID=0 TITLE segment and leave
    the body at its original CorrSentenceID.

    Runs after mark_artifacts_by_model so model-detected titles take
    priority and are not re-processed."""
    df = wm.copy()
    idcol = "Identifier" if "Identifier" in df.columns else "ID"
    if "TITLE" not in df.columns:
        df["TITLE"] = False
    if "TextualArtifact" not in df.columns:
        df["TextualArtifact"] = ""

    for ident, g in df.groupby(idcol, sort=False):
        g_sorted = g.sort_values("corr_index", kind="mergesort")
        sids = pd.to_numeric(g_sorted["CorrSentenceID"], errors="coerce")
        body_sids = sorted(s for s in sids.dropna().unique() if s != 0)
        if not body_sids:
            continue
        first_body_sid = body_sids[0]
        grp = g_sorted[sids == first_body_sid]
        if grp.empty:
            continue
        # Skip if already fully tagged TITLE
        if (grp["TextualArtifact"].fillna("") == "TITLE").all():
            continue

        toks = grp["corr_token"].fillna("").astype(str).tolist()
        tok_idxs = grp.index.tolist()

        # Walk tokens to find consecutive ALL-CAPS alphabetic prefix
        allcaps_positions = []   # positions in toks list of ALL-CAPS word tokens
        in_prefix = True
        body_word_found = False

        for i, tok in enumerate(toks):
            t = tok.strip()
            if not t or t in _CB6_PUNCT:
                continue  # skip punctuation between words
            # Word token
            if in_prefix and t.isupper() and t.isalpha() and len(t) >= 2:
                allcaps_positions.append(i)
            else:
                in_prefix = False
                if allcaps_positions:
                    body_word_found = True
                break

        if len(allcaps_positions) < 2 or not body_word_found:
            continue  # need >=2 ALL-CAPS tokens with body following

        # Everything up to and including the last ALL-CAPS token → TITLE
        last_allcaps_pos = allcaps_positions[-1]
        prefix_df_idxs = tok_idxs[:last_allcaps_pos + 1]
        df.loc[prefix_df_idxs, "TITLE"] = True
        df.loc[prefix_df_idxs, "TextualArtifact"] = "TITLE"
        df.loc[prefix_df_idxs, "CorrSentenceID"] = 0
        df.loc[prefix_df_idxs, "SentenceRef"] = str(ident) + "_s000"

    return df


def _apply_cutoff_detection(wm):
    """CB5: Mark the story-last sentence of each script as CUTOFF using the
    basic aes._is_cutoff rule (0 words, or 1 word with no terminal).

    CB6 broadened indicators run separately at sentence level in
    _apply_cb6_cutoff_extended. CUTOFF (priority 5) overrides ENDING (3)
    but not TITLE (4)."""
    df = wm.copy()
    if "TextualArtifact" not in df.columns:
        df["TextualArtifact"] = ""
    idcol = "Identifier" if "Identifier" in df.columns else "ID"
    _ri_num = pd.to_numeric(df.get("raw_index", pd.Series(pd.NA, index=df.index)),
                            errors="coerce")
    _ap_num = pd.to_numeric(df.get("_ap", pd.Series(pd.NA, index=df.index)),
                            errors="coerce")
    for ident, g in df.groupby(idcol, sort=False):
        g_ri = _ri_num.loc[g.index]
        real_mask = g_ri.notna()
        g_ap = _ap_num.loc[g.index]
        real_ap = g_ap[real_mask]
        if real_ap.empty:
            continue
        max_ap = real_ap.max()
        last_tok_idx = real_ap[real_ap == max_ap].index[0]
        last_csid = g.at[last_tok_idx, "CorrSentenceID"]
        if pd.isna(last_csid):
            continue
        grp = g[g["CorrSentenceID"] == last_csid]
        if grp["TextualArtifact"].isin(["TITLE", "CUTOFF"]).any():
            continue
        toks = grp["corr_token"].fillna("").astype(str).tolist()
        if aes._is_cutoff(toks):
            df.loc[grp.index, "TextualArtifact"] = "CUTOFF"
    return df


def _apply_cb6_cutoff_extended(sent_df: "pd.DataFrame") -> "pd.DataFrame":
    """CB6: Apply broadened CUTOFF indicators to the story-last sentence using
    CorrectedSentence text. Runs at sentence level after _apply_sentence_artifact_tags.

    Skips sentences already tagged CUTOFF or TITLE. Iterates in SentenceRef
    order and finds the last sentence that is neither CUTOFF nor TITLE.
    CUTOFF (priority 5) overrides ENDING (3) but not TITLE (4)."""
    df = sent_df.copy()
    if "TextualArtifact" not in df.columns:
        df["TextualArtifact"] = ""

    for ident, g in df.groupby("Identifier", sort=False):
        g_sorted = g.sort_values("SentenceRef")
        idxs = g_sorted.index.tolist()
        if not idxs:
            continue
        last_idx = None
        for idx in reversed(idxs):
            ta = str(df.at[idx, "TextualArtifact"] or "")
            if ta not in ("CUTOFF", "TITLE"):
                last_idx = idx
                break
        if last_idx is None:
            continue
        text = str(df.at[last_idx, "CorrectedSentence"] or "")
        if _cb6_is_cutoff_sent(text):
            df.at[last_idx, "TextualArtifact"] = "CUTOFF"

    return df


def _apply_stg_boundary_provenance(sent_df, wm):
    """CB5: Override BoundaryProvenance to 'student-terminal-guard' for
    sentences that were created by _apply_sacred_terminal_guard_wm."""
    if "_stg_origin_sid" not in wm.columns:
        return sent_df
    df = sent_df.copy()
    idcol = "Identifier" if "Identifier" in wm.columns else "ID"
    guard_refs: set = set()
    for ident, g in wm.groupby(idcol, sort=False):
        guard_rows = g[g["_stg_origin_sid"].notna()]
        if guard_rows.empty:
            continue
        for ref in guard_rows["SentenceRef"].dropna().unique():
            guard_refs.add(str(ref))
    if guard_refs:
        mask = df["SentenceRef"].isin(guard_refs)
        df.loc[mask, "BoundaryProvenance"] = "student-terminal-guard"
    return df


# ---------------------------------------------------------------------------
# CB5 -- Sentence-level TextualArtifact tagging (TIMESKIP, ENDING, OPENING)
# ---------------------------------------------------------------------------

_TIMESKIP_PATTERNS = [
    re.compile(
        r'\b(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|a\s+few|several)\s+'
        r'(?:second|minute|hour|day|week|month|year)s?\s+'
        r'(?:later|earlier|after|ago)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\blater\s+that\s+(?:night|morning|afternoon|evening|day)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\bthe\s+(?:next|following)\s+(?:day|morning|night|week|month|year)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:meanwhile|moments\s+later|soon\s+after|hours\s+passed'
        r'|days\s+went\s+by)\b',
        re.IGNORECASE,
    ),
    re.compile(r'^\s*\d+\s+\w+\s+(?:later|earlier)\s*:', re.IGNORECASE),
]
_TIMESKIP_MAX_WORDS = 4  # bare markers are short; full sentences are longer


def _is_bare_timeskip(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    if not any(p.search(t) for p in _TIMESKIP_PATTERNS):
        return False
    words = re.findall(r'[A-Za-z0-9]+', t)
    return len(words) <= _TIMESKIP_MAX_WORDS


_ENDING_PATTERNS = [
    # "the end" only at the END of the sentence (followed by punctuation or EOS)
    re.compile(r'\bthe\s+end\s*[.!?]*\s*$', re.IGNORECASE),
    re.compile(r'\bhappily\s+ever\s+after\b', re.IGNORECASE),
    re.compile(r'\blived\s+happily\s+ever\s+after\b', re.IGNORECASE),
    re.compile(r"\band\s+that'?s?\s+how\b.*\bend(?:s|ed)?\b", re.IGNORECASE),
    re.compile(r"\bthat'?s?\s+the\s+end\s+of\b", re.IGNORECASE),
    re.compile(r'\bthe\s+end\s+of\s+the\s+story\b', re.IGNORECASE),
]
# CB5b: negation words that suppress ENDING phrase match
_ENDING_NEGATORS_RE = re.compile(
    r"\b(?:not|isn'?t|wasn'?t|won'?t|shouldn'?t|never|no|hardly)\b",
    re.IGNORECASE,
)


def _is_ending_phrase(text: str) -> bool:
    t = str(text or "")
    if not any(p.search(t) for p in _ENDING_PATTERNS):
        return False
    # CB5b: skip if negated within ~3 tokens before any ending phrase
    for p in _ENDING_PATTERNS:
        m = p.search(t)
        if m:
            window = t[max(0, m.start() - 40):m.start()]
            if _ENDING_NEGATORS_RE.search(window):
                return False
    return True


_OPENING_PHRASES = [
    re.compile(r'once\s+upon\s+a\s+time\b', re.IGNORECASE),
    re.compile(r'^one\s+day\b', re.IGNORECASE),
    re.compile(r'^long\s+ago\b', re.IGNORECASE),
    re.compile(r'^it\s+was\s+a\s+dark\s+and\s+stormy\s+night\b', re.IGNORECASE),
    re.compile(r'^in\s+a\s+land\s+far\s+away\b', re.IGNORECASE),
]


def _is_opening_phrase(text: str) -> bool:
    t = str(text or "").strip()
    return any(p.search(t) for p in _OPENING_PHRASES)


# Precedence: CUTOFF(5) > TITLE(4) > ENDING(3) > OPENING(2) > TIMESKIP(1) > ""(0)
_ARTIFACT_PRIORITY = {
    "CUTOFF": 5, "TITLE": 4, "ENDING": 3, "OPENING": 2, "TIMESKIP": 1, "": 0
}


def _apply_sentence_artifact_tags(sent_df: "pd.DataFrame") -> "pd.DataFrame":
    """CB5/CB5b: Tag TIMESKIP, ENDING, OPENING in the TextualArtifact column.

    OPENING Option C (both): first non-TITLE segment OR classic-opener phrase.
      CB5b: phrase OPENING only fires in the first 3 sentences of the script.
    ENDING: recognised story-closer phrases.
      CB5b: phrase ENDING only fires in the last 2 sentences; negation suppresses.
    TIMESKIP: bare time-marker fragment, <= _TIMESKIP_MAX_WORDS words.
    Existing TITLE/CUTOFF tags are never overridden.
    """
    df = sent_df.copy()
    if "TextualArtifact" not in df.columns:
        df["TextualArtifact"] = ""

    def _can_set(current: str, new_tag: str) -> bool:
        return (_ARTIFACT_PRIORITY.get(new_tag, 0) >
                _ARTIFACT_PRIORITY.get(str(current or ""), 0))

    for ident, g in df.groupby("Identifier", sort=False):
        g_sorted = g.sort_values("SentenceRef")
        idxs = g_sorted.index.tolist()

        # First non-TITLE sentence index (for OPENING positional rule)
        first_non_title_idx = next(
            (i for i in idxs
             if str(df.at[i, "TextualArtifact"] or "") not in ("TITLE",)),
            None,
        )

        # CB5b ENDING position guard: exclude guard-created and CUTOFF/TITLE
        # sentences so that "The end." isn't pushed out of window by garbled
        # guard fragments that follow it.
        non_guard_idxs = [
            i for i in idxs
            if str(df.at[i, "BoundaryProvenance"] or "") != "student-terminal-guard"
            and str(df.at[i, "TextualArtifact"] or "") not in ("CUTOFF", "TITLE")
        ]
        n_ng = len(non_guard_idxs)

        for pos, idx in enumerate(idxs):
            text = str(df.at[idx, "CorrectedSentence"] or "")
            current = str(df.at[idx, "TextualArtifact"] or "")

            # TIMESKIP (lowest priority -- apply first so higher can override)
            if _can_set(current, "TIMESKIP") and _is_bare_timeskip(text):
                df.at[idx, "TextualArtifact"] = "TIMESKIP"
                current = "TIMESKIP"

            # ENDING -- CB5b: phrase rule only fires in the last 2 non-guard sentences
            if _can_set(current, "ENDING") and _is_ending_phrase(text):
                if idx in non_guard_idxs:
                    nc_pos = non_guard_idxs.index(idx)
                    if nc_pos >= n_ng - 2:
                        df.at[idx, "TextualArtifact"] = "ENDING"
                        current = "ENDING"

            # OPENING (Option C: positional OR phrase)
            # CB5b: phrase rule only fires in the first 3 sentences (pos 0-2)
            if _can_set(current, "OPENING"):
                is_positional = (idx == first_non_title_idx)
                is_phrase_in_window = (_is_opening_phrase(text) and pos <= 2)
                if is_positional or is_phrase_in_window:
                    df.at[idx, "TextualArtifact"] = "OPENING"

    return df


def run_sentence_layer(texts, id_col="Research ID",
                       raw_col="Raw text", corr_col="Corrected text (8)",
                       precomputed_wm=None, boundary_mode="rules"):
    """Full path: fixed word layer -> aes.run_step9 -> six sentence columns.
    Returns sent_df with the script id in 'Identifier'.

    Pass precomputed_wm = an already engine-enriched word map (word class
    and SS class filled) so the sentence grammar-subtype rollup resolves
    instead of staying TBD(engine).

    boundary_mode='corrector' selects Branch B: the precomputed_wm must have
    been built with run_layer(boundary_mode='corrector'), and the rule-based
    detectors (Rule 3 + BR8/Rule 2/BR1) are skipped here so the corrector's
    intended terminal marks are the sole source of boundaries."""
    wm = (precomputed_wm if precomputed_wm is not None
          else run_layer(texts, id_col=id_col, raw_col=raw_col,
                          corr_col=corr_col, boundary_mode=boundary_mode))

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
    # A2c: normalise awkward mixed-terminal patterns (e.g. "..?" → "?",
    # "..!" → "!", "...?" → "...?").  Three or more dots before ? or ! are
    # kept as an ellipsis + terminal; fewer are collapsed to the terminal alone.
    def _normalise_terminals(text):
        """Single-pass normaliser: N dots immediately before ? or !
        → '...' + terminal (N>=3) or just the terminal (N<3)."""
        if not isinstance(text, str):
            return text
        def _fix(m):
            ndots = len(m.group(1))
            return ("..." if ndots >= 3 else "") + m.group(2)
        return re.sub(r"(\.+)([?!])", _fix, text)
    sent["CorrectedSentence"] = sent["CorrectedSentence"].apply(
        _normalise_terminals)
    # CB5: strip connector-comma artifacts: ", ." at end of sentence
    # arises when the corrector adds a trailing comma before its sentence
    # terminal; after the sacred-terminal guard splits the sentence the
    # comma ends up at the end of the new (second) segment.
    sent["CorrectedSentence"] = sent["CorrectedSentence"].str.replace(
        r",\s*\.\s*$", ".", regex=True)
    # Change 10 -- merge segmentation-artefact sentences (content is
    # punctuation only, no alphabetic characters) into the previous
    # valid sentence in the same script. Mutates wm in place so the
    # word_map and sentences stay aligned on SentenceRef.
    sent = _merge_punctuation_only_sentences(wm, sent)
    # carry the script-level flags onto every sentence row
    cmap = t.set_index("Identifier")
    for col in ["ScriptClass", "ScriptClassReason", "Concern1",
                "Concern1Reason"]:
        sent[col] = sent["Identifier"].map(cmap[col])
    result = enrich_sentences(wm, sent)
    # BR9: normalise BR3-split SentenceRefs to use a/b suffix convention
    result = _apply_br9_renaming(result, wm_a)
    if boundary_mode == "corrector":
        # CB5: tag guard-created sentences and apply structural artifact tags
        result = _apply_stg_boundary_provenance(result, wm_a)
        result = _apply_sentence_artifact_tags(result)
        result = _apply_cb6_cutoff_extended(result)   # CB6: broadened CUTOFF at sentence level
        return result
    # Rules mode: Rule 3 + BR8 (comma-splice / BR1) -- not the production path
    result = apply_rule3_splits(result, wm_a)
    result = apply_run_on_splits_recursive(result)
    result = _apply_sentence_artifact_tags(result)
    return result


# ----------------------------------------------------------------------
# WB1 -- Contraction POS deterministic override
# Runs AFTER spaCy fills TokenSubcat but BEFORE the chunker.
# Called from run_all.py between Stage 4 and Stage 5c.
# ----------------------------------------------------------------------

def apply_contraction_pos_override(df):
    """
    WB1: For contraction rows, override TokenSubcat from a deterministic
    lookup table (contractions_lookup.py). Preserves rows not in the
    lookup -- they pass through unchanged.
    """
    try:
        from contractions_lookup import CONTRACTION_POS
    except ImportError:
        return df   # table not found -- skip gracefully

    contr_mask = df["TokenCategory"] == "contraction"
    if not contr_mask.any():
        return df

    df = df.copy()
    contractions = df.loc[contr_mask].copy()
    keys = contractions["corr_token"].astype(str).str.lower()
    new_subcats = keys.map(lambda k: CONTRACTION_POS.get(k, (None,))[0])

    hits = new_subcats.notna()
    if hits.any():
        df.loc[contractions[hits].index, "TokenSubcat"] = new_subcats[hits].values
    return df


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
_HARD_CLOSERS = {")", "]", "}", "\u201d", "\u2019", "\u00bb"}  # unambiguous

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


def assign_intended_sentence_ids_v2(df_map):
    """Branch B (corrector-driven) segmentation.

    Identical state machine to assign_corr_sentence_ids_v2, but a position is
    treated as a sentence terminal when EITHER:
      - the corrector wrote a terminal there (normal corrector boundary), OR
      - the student wrote a terminal there that the corrector downgraded to a
        comma / removed (raw_token in TERMINALS, corr_token not) -- the CB2
        "student terminal marks are always boundaries" rule.

    This is the union used by _add_boundary_retained_columns (rt_is OR ct_is),
    i.e. it segments on the *intended* text's terminal marks. Used only when
    boundary_mode='corrector'; the rule-based detectors are disabled in that
    mode, so these terminal marks ARE the boundaries.
    """
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
        corr_toks = (g["corr_token"] if "corr_token" in g.columns
                     else g["raw_token"]).map(_tok_str).tolist()
        raw_toks = (g["raw_token"] if "raw_token" in g.columns
                    else g["corr_token"]).map(_tok_str).tolist()
        sids = []
        sid = 1
        in_dq = False
        in_sq = False
        pending = False
        for i, raw in enumerate(corr_toks):
            t = raw.strip()
            prev = corr_toks[i - 1].strip() if i > 0 else ""
            nxt = corr_toks[i + 1].strip() if i + 1 < len(corr_toks) else ""

            is_dq = (t == '"')
            is_sq = (t == "'")
            dq_closer = is_dq and in_dq
            sq_closer = is_sq and in_sq
            is_closer = (t in _HARD_CLOSERS) or dq_closer or sq_closer

            if pending:
                if is_closer:
                    sids.append(sid)
                    if is_dq:
                        in_dq = not in_dq
                    elif is_sq:
                        in_sq = not in_sq
                    continue
                sid += 1
                pending = False
                sids.append(sid)
            else:
                sids.append(sid)

            if is_dq:
                in_dq = not in_dq
            elif is_sq:
                in_sq = not in_sq

            corr_terminal = aes._is_terminal_token(t, prev, nxt)
            student_terminal = (raw_toks[i].strip() in _TERM_SET
                                and t not in _TERM_SET)
            if corr_terminal or student_terminal:
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

    return re.sub(r"\.\s*\.\s*\.", "...", "".join(parts)).strip()


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
    # Always reset -- our model-driven logic must override the aes heuristic
    # (aes.mark_title_and_dialogue runs first and may tag short scripts as
    # TITLE even when the model returns TitleWordCount=0, e.g. KVSMRXTR).
    # CB5: preserve CUTOFF -- it is deterministic, not from the model.
    df["TITLE"] = False
    df.loc[df["TextualArtifact"] != "CUTOFF", "TextualArtifact"] = ""
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

    # SA2 -- catch ALL-CAPS short titles at body positions 1-3 that the model
    # missed (e.g. SHWJRFLY "A STORY" at _s001).  Runs BEFORE SentenceRef
    # rebuild so raw_token grouping is still on the original CorrSentenceID.
    try:
        from title_detector import is_likely_title_at_body_start
        for _ID, _g in df.groupby(idcol, sort=False):
            _g_sorted = _g.sort_values("corr_index", kind="mergesort")
            for _csid, _sg in _g_sorted.groupby("CorrSentenceID", sort=True):
                try:
                    _sid_int = int(_csid)
                except Exception:
                    continue
                # Only check positions 1-3 that are not already marked TITLE
                if _sid_int < 1 or _sid_int > 3:
                    continue
                if (_sg["TextualArtifact"] == "TITLE").any():
                    continue
                # Reconstruct the raw text for this sentence
                _raw_toks = _sg["raw_token"].map(_tok_str).tolist()
                _raw_text = detok_v2(_raw_toks)
                if is_likely_title_at_body_start(_raw_text, _sid_int):
                    df.loc[_sg.index, "TITLE"] = True
                    df.loc[_sg.index, "TextualArtifact"] = "TITLE"
    except ImportError:
        pass   # title_detector not available; skip SA2 heuristic

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
#   StudentTerminalMark    -- student's terminal at this alignment position
#                            ("." "?" "!" or "" if student had none here)
#   CorrectedTerminalMark  -- terminal used in boundary-retained text
#                            ("." "?" "!" or "" if no boundary here at all)
#   CapitalSource          -- 'student' | 'machine-added' | 'NA'
#   BoundarySource         -- 'student' | 'machine-added' | 'NA'
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
    script in raw_index order and copy the nearest non-delete row's
    CorrSentenceID onto each delete row: the previous non-delete row
    when one exists, otherwise the next one (handles start-of-script
    delete runs where no preceding row exists yet)."""
    import bisect
    df = df_map.copy()
    idcol = "Identifier" if "Identifier" in df.columns else "ID"
    for sid, g in df.groupby(idcol, sort=False):
        g_raw = (g.assign(_ri=pd.to_numeric(g["raw_index"], errors="coerce"))
                  .sort_values("_ri", kind="mergesort", na_position="last"))
        # Collect (list-position, csid) for every non-delete row in order
        nd_positions = []
        nd_csids = []
        for list_pos, idx in enumerate(g_raw.index):
            op = str(g_raw.loc[idx, "op"])
            csid = g_raw.loc[idx, "CorrSentenceID"]
            if op != "delete" and pd.notna(csid) and str(csid) != "":
                nd_positions.append(list_pos)
                nd_csids.append(csid)
        if not nd_positions:
            continue
        for list_pos, idx in enumerate(g_raw.index):
            if str(g_raw.loc[idx, "op"]) != "delete":
                continue
            # Find insertion point in nd_positions
            ins = bisect.bisect_right(nd_positions, list_pos)
            if ins > 0:
                csid = nd_csids[ins - 1]   # previous non-delete (normal case)
            else:
                csid = nd_csids[0]          # no previous -- use next (start-of-script)
            df.at[idx, "CorrSentenceID"] = csid
    return df


def _merge_punctuation_only_sentences(wm, sent):
    """Change 10. Sentences whose CorrectedSentence contains no
    alphabetic characters are segmentation artefacts -- usually an
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
