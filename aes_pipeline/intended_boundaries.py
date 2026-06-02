"""
CB2 — Build the 'Corrected intended' text column.

Takes the raw text and the corrected text (post-CB1 strengthened corrector)
and produces a third column that represents the student's intended sentence
boundaries: student terminal marks are always honoured as boundaries; commas
and fused gaps defer to the corrector's boundary placement.

Mark-flavour is resolved conservatively:
  - Student '.' + corrector '?' + sentence IS interrogative  -> use '?'
  - Student '.' + corrector '?' + sentence NOT interrogative -> keep '.'
  - Student '.' + corrector '!'                              -> keep '.' (reject stylistic)
  - Student '?' or '!'                                       -> keep student's mark
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TERMINAL_MARKS = frozenset({'.', '?', '!'})

QUESTION_STARTERS = frozenset({
    'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose', 'whom',
    'can', 'could', 'will', 'would', 'do', 'does', 'did', 'is', 'are',
    'was', 'were', 'has', 'have', 'had', 'should', 'shall', 'may', 'might',
    'am', 'whichever', 'whoever',
})

# Abbreviations whose trailing dot must NOT be treated as a sentence boundary
_ABBREVS = frozenset({
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'e.g', 'i.e',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
})

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _trailing_mark(token: str) -> str:
    """Return the trailing terminal mark of *token*, or '' if none."""
    t = token.rstrip()
    if t and t[-1] in TERMINAL_MARKS:
        return t[-1]
    return ''


def _trailing_punct(token: str) -> str:
    """Return the last non-space character of *token* (comma, mark, etc.)."""
    t = token.rstrip()
    return t[-1] if t else ''


def _strip_trailing_mark(token: str) -> str:
    """Remove a trailing terminal mark from *token* (preserves other punctuation)."""
    t = token.rstrip()
    if t and t[-1] in TERMINAL_MARKS:
        return t[:-1]
    return t


def _swap_mark(token: str, new_mark: str) -> str:
    """Replace the trailing terminal mark of *token* with *new_mark*."""
    t = token.rstrip()
    if t and t[-1] in TERMINAL_MARKS:
        return t[:-1] + new_mark
    # No existing terminal mark -- append
    return t + new_mark


def _restore_terminal_mark(token: str, mark: str) -> str:
    """
    Ensure *token* ends with *mark*.

    If the token currently ends with a comma or other non-terminal punct,
    the comma is replaced with the mark.  If it already ends with a
    different terminal mark it is swapped.  Otherwise the mark is appended.
    """
    t = token.rstrip()
    if not t:
        return mark
    last = t[-1]
    if last == ',':
        return t[:-1] + mark
    if last in TERMINAL_MARKS:
        return t[:-1] + mark
    return t + mark


# ---------------------------------------------------------------------------
# Deterministic interrogative check
# ---------------------------------------------------------------------------

def is_interrogative(sentence_text: str) -> bool:
    """
    Return True if *sentence_text* is structurally a question.

    Checks whether the first content word is a question-word / auxiliary.
    Model-independent and deterministic.
    """
    words = sentence_text.strip().split()
    if not words:
        return False
    first = re.sub(r'[.,!?";\'‘’“”]', '', words[0]).lower()
    return first in QUESTION_STARTERS


# ---------------------------------------------------------------------------
# Mark-flavour resolver
# ---------------------------------------------------------------------------

def resolve_mark_flavour(
    corr_tok: str,
    student_mark: str,
    corr_mark: str,
    sentence_context: str,
) -> str:
    """
    Decide which terminal mark to use when student and corrector disagree.

    Conservative: keep the student's mark unless it is grammatically wrong.
    """
    # Student '.' + corrector '?' -- accept only if genuinely interrogative
    if student_mark == '.' and corr_mark == '?':
        if is_interrogative(sentence_context):
            return _swap_mark(corr_tok, '?')
        return _swap_mark(corr_tok, '.')

    # Student '.' + corrector '!' -- reject stylistic upgrade
    if student_mark == '.' and corr_mark == '!':
        return _swap_mark(corr_tok, '.')

    # Student '?' or '!' -- honour the student's choice
    if student_mark in ('?', '!'):
        return _swap_mark(corr_tok, student_mark)

    # Default -- keep corrector's mark
    return corr_tok


# ---------------------------------------------------------------------------
# Tokeniser -- splits on whitespace but keeps trailing punctuation attached
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Split *text* into whitespace-delimited tokens."""
    return text.split() if text.strip() else []


# ---------------------------------------------------------------------------
# Simple word-level aligner (Longest Common Subsequence)
# ---------------------------------------------------------------------------

def _lcs_align(raw_tokens: list[str], corr_tokens: list[str]) -> list[dict]:
    """
    Align raw and corrected tokens using a simple LCS-based approach.

    Returns a list of alignment records:
      {'raw_token': str, 'corr_token': str}

    Tokens that appear only in one stream are represented with the other
    set to ''.  This is a heuristic suitable for lightly-corrected text
    where the word order is preserved.
    """
    n, m = len(raw_tokens), len(corr_tokens)

    # Build LCS table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ri = re.sub(r'[^a-z]', '', raw_tokens[i - 1].lower())
            ci = re.sub(r'[^a-z]', '', corr_tokens[j - 1].lower())
            if ri and ci and ri == ci:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Trace back
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            ri = re.sub(r'[^a-z]', '', raw_tokens[i - 1].lower())
            ci = re.sub(r'[^a-z]', '', corr_tokens[j - 1].lower())
            if ri and ci and ri == ci:
                alignment.append({'raw_token': raw_tokens[i - 1], 'corr_token': corr_tokens[j - 1]})
                i -= 1; j -= 1
                continue
        if j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            alignment.append({'raw_token': '', 'corr_token': corr_tokens[j - 1]})
            j -= 1
        else:
            alignment.append({'raw_token': raw_tokens[i - 1], 'corr_token': ''})
            i -= 1

    alignment.reverse()
    return alignment


# ---------------------------------------------------------------------------
# Context helper -- sentence preceding a given alignment position
# ---------------------------------------------------------------------------

def _get_preceding_sentence(alignment: list[dict], pos: int) -> str:
    """
    Return the corrected text of the sentence that ends at *pos*.

    Scans backwards from *pos* to find the start of the current sentence
    (i.e. the previous terminal mark or the beginning of the stream).
    The previous-sentence's terminal token is NOT included.
    """
    parts = []
    for k in range(pos, -1, -1):
        ct = alignment[k]['corr_token']
        # Stop before adding a token from a prior sentence
        if k < pos and ct and _trailing_mark(ct) in TERMINAL_MARKS:
            break
        if ct:
            parts.append(ct)
    parts.reverse()
    return ' '.join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_intended_boundaries(
    raw_text: str,
    corrected_text: str,
    word_alignment: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Build the 'Corrected intended' text.

    Parameters
    ----------
    raw_text       : The student's original text.
    corrected_text : The corrector's output (post-CB1).
    word_alignment : Optional pre-built alignment list (dicts with
                     'raw_token' and 'corr_token').  If None it is
                     computed here via LCS.

    Returns
    -------
    intended_text  : The intended-boundary text (str).
    provenance     : List of per-token provenance dicts:
                       'boundary'          : True/False -- is this a sentence end?
                       'provenance'        : 'student' | 'machine-runon' | 'none'
                       'original_mark'     : what the student wrote ('.' ',' '' etc.)
                       'mark_corrected'    : True if mark flavour was changed
    """
    raw_tokens = _tokenise(raw_text)
    corr_tokens = _tokenise(corrected_text)

    if word_alignment is None:
        alignment = _lcs_align(raw_tokens, corr_tokens)
    else:
        alignment = word_alignment

    intended_tokens: list[str] = []
    provenance_records: list[dict] = []

    for pos, align in enumerate(alignment):
        raw_tok = align.get('raw_token', '')
        corr_tok = align.get('corr_token', '')

        raw_mark = _trailing_mark(raw_tok)
        raw_punct = _trailing_punct(raw_tok)  # could be comma
        corr_mark = _trailing_mark(corr_tok)

        out_tok = corr_tok
        prov = 'none'
        orig_mark = ''
        mark_corrected = False

        if raw_mark in TERMINAL_MARKS:
            # Student wrote a terminal mark -- it is always a boundary
            orig_mark = raw_mark
            if corr_mark not in TERMINAL_MARKS:
                # Corrector downgraded or removed the student's terminal -- restore
                out_tok = _restore_terminal_mark(corr_tok, raw_mark)
                prov = 'student'
                mark_corrected = False
            elif corr_mark != raw_mark:
                # Both terminal but different flavour -- apply mark-flavour rule
                sentence_ctx = _get_preceding_sentence(alignment, pos)
                resolved = resolve_mark_flavour(corr_tok, raw_mark, corr_mark, sentence_ctx)
                mark_corrected = (resolved != corr_tok)
                out_tok = resolved
                prov = 'student'
            else:
                # Same mark -- keep corrected token as-is
                prov = 'student'
        elif raw_punct == ',':
            # Student wrote a comma -- defer to corrector's boundary
            orig_mark = ','
            if corr_mark in TERMINAL_MARKS:
                prov = 'machine-runon'
                out_tok = corr_tok
            else:
                prov = 'none'
        else:
            # Student wrote nothing (fused gap) -- take corrector's boundary
            orig_mark = ''
            if corr_mark in TERMINAL_MARKS:
                prov = 'machine-runon'
                out_tok = corr_tok
            else:
                prov = 'none'

        if out_tok:  # skip pure-insertion tokens with no corrected form if empty
            intended_tokens.append(out_tok)

        is_boundary = _trailing_mark(out_tok) in TERMINAL_MARKS if out_tok else False
        provenance_records.append({
            'corr_token': corr_tok,
            'out_token': out_tok,
            'boundary': is_boundary,
            'provenance': prov,
            'original_mark': orig_mark,
            'mark_corrected': mark_corrected,
        })

    intended_text = ' '.join(t for t in intended_tokens if t)
    return intended_text, provenance_records


# ---------------------------------------------------------------------------
# Convenience: segment the intended text into sentences
# ---------------------------------------------------------------------------

def segment_intended_text(intended_text: str) -> list[str]:
    """
    Split *intended_text* on terminal marks (., ?, !).

    Respects common abbreviations so "Mr. Smith went" doesn't split.
    Returns a list of non-empty sentence strings, each including its
    terminal mark.
    """
    # Protect abbreviation dots by temporarily replacing them
    protected = intended_text
    for abbrev in sorted(_ABBREVS, key=len, reverse=True):
        protected = re.sub(
            r'\b' + re.escape(abbrev) + r'\.',
            abbrev.replace('.', '\x00') + '\x01',
            protected,
            flags=re.IGNORECASE,
        )

    # Split on terminal marks -- keep the mark with the sentence
    parts = re.split(r'(?<=[.?!])\s+', protected)
    sentences = []
    for part in parts:
        # Restore abbreviation placeholders
        restored = part.replace('\x00', '.').replace('\x01', '.')
        s = restored.strip()
        if s:
            sentences.append(s)
    return sentences
