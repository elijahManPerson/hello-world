"""
pa_scorer.py
============
Pa (paragraph) scoring — Phase Pa, Patches Pa2 + Pa3.

Pa2: Robust HTML paragraph counting (regex-based; handles all markup variants
     observed in the corpus without BeautifulSoup dependency).

Pa3: Four-state Pa_source logic:
     - HTML present                      → pipeline_computed (HTML available)
     - No HTML, marker Pa given          → marker_assigned
     - No HTML, no marker Pa             → unavailable
     - HTML present but parse failed     → html_parse_failed (fallback: marker if present)
"""

import re
from html import unescape


# ---------------------------------------------------------------------------
# Pa2 — Robust paragraph counter
# ---------------------------------------------------------------------------

# Block-level tags that open a paragraph
_BLOCK_OPEN = re.compile(
    r'<\s*(p|li)\b[^>]*>',
    re.IGNORECASE
)

# Block-level closes — includes the corpus typo <p/P>
_BLOCK_CLOSE = re.compile(
    r'<\s*/\s*(p|li)\b[^>]*>|<\s*p\s*/\s*P\s*>',
    re.IGNORECASE
)

# Inline tags — stripped without paragraph effect
_INLINE_TAG = re.compile(
    r'<\s*/\?\s*(span|em|strong|u|br|i|b)\b[^>]*\s*/?>',
    re.IGNORECASE
)

# Catch-all tag remover (used after inline stripping)
_ANY_TAG = re.compile(r'<[^>]+')


def _strip_inline(html):
    return _INLINE_TAG.sub('', html)


def _decode(text):
    """Decode HTML entities; replace non-breaking space with plain space."""
    return unescape(text).replace('\xa0', ' ')


def count_paragraph_breaks(html_text):
    """
    Count paragraph BREAKS (= paragraphs - 1) in HTML.

    Rules (Pa2 spec):
    - Each <p>...</p> pair with non-whitespace content = 1 paragraph
    - Each <li>...</li> pair with content = 1 paragraph
    - Bare text with no block tags = 1 paragraph -> 0 breaks
    - Empty <p></p> pairs don't count
    - Inline tags (span/em/strong/u/br/i/b) are stripped; no paragraph effect
    - &nbsp; decoded to space
    - <p/P> malformed close tag tolerated

    Returns int (>= 0) or None if html_text is empty/None.
    """
    if not html_text or not str(html_text).strip():
        return None

    text = _strip_inline(str(html_text))

    opens = list(_BLOCK_OPEN.finditer(text))
    closes = list(_BLOCK_CLOSE.finditer(text))

    if not opens:
        # Bare text -- count as 1 paragraph -> 0 breaks
        stripped = _ANY_TAG.sub('', text)
        stripped = _decode(stripped).strip()
        return 0 if not stripped else 0   # 1 paragraph -> 0 breaks

    paragraph_count = 0
    for open_match in opens:
        # Find the next close at or after this open's end position.
        # Use >= so that <p></p> (immediately adjacent) is correctly paired.
        close_match = next(
            (c for c in closes if c.start() >= open_match.end()),
            None
        )
        if close_match is None:
            content = text[open_match.end():]
        else:
            content = text[open_match.end():close_match.start()]

        content = _ANY_TAG.sub('', content)
        content = _decode(content).strip()
        if content:
            paragraph_count += 1

    # breaks = paragraphs - 1 (a single paragraph has 0 breaks)
    return max(0, paragraph_count - 1)


# ---------------------------------------------------------------------------
# Pa3 — Four-state Pa_source logic
# ---------------------------------------------------------------------------

PA_SOURCE_PIPELINE   = 'pipeline_computed (HTML available)'
PA_SOURCE_MARKER     = 'marker_assigned'
PA_SOURCE_UNAVAILABLE = 'unavailable'
PA_SOURCE_PARSE_FAIL  = 'html_parse_failed'


def add_pa_columns(texts_df, html_col="Raw HTML", pa_col="Pa"):
    """
    Add Pa_source and ParagraphBreaks columns to texts_df.
    Overwrites pa_col with pipeline-computed value when HTML is present.

    Four-state logic (Pa3):
    1. HTML present -> pipeline_computed; Pa = min(2, breaks)
    2. No HTML, marker Pa given -> marker_assigned; Pa unchanged
    3. No HTML, no marker Pa -> unavailable; Pa = ''
    4. HTML present but parse returned None -> html_parse_failed; falls back to marker
    """
    texts_df = texts_df.copy()

    pa_source_vals = []
    para_break_vals = []
    pa_vals = []

    for _, row in texts_df.iterrows():
        html = row.get(html_col, "")
        marker_pa = str(row.get(pa_col, "")).strip()

        has_html = bool(html and str(html).strip())
        has_marker = bool(marker_pa and marker_pa not in ("nan", "NA", "None"))

        if has_html:
            breaks = count_paragraph_breaks(html)
            if breaks is not None:
                pa = min(2, breaks)
                pa_source_vals.append(PA_SOURCE_PIPELINE)
                para_break_vals.append(breaks)
                pa_vals.append(pa)
            else:
                # Parse returned None (shouldn't happen for non-empty HTML, but guard it)
                if has_marker:
                    pa_source_vals.append(PA_SOURCE_MARKER)
                    para_break_vals.append(None)
                    try:
                        pa_vals.append(int(float(marker_pa)))
                    except (ValueError, TypeError):
                        pa_vals.append('')
                else:
                    pa_source_vals.append(PA_SOURCE_PARSE_FAIL)
                    para_break_vals.append(None)
                    pa_vals.append('')
        elif has_marker:
            pa_source_vals.append(PA_SOURCE_MARKER)
            para_break_vals.append(None)
            try:
                pa_vals.append(int(float(marker_pa)))
            except (ValueError, TypeError):
                pa_vals.append('')
        else:
            pa_source_vals.append(PA_SOURCE_UNAVAILABLE)
            para_break_vals.append(None)
            pa_vals.append('')

    texts_df["Pa_source"] = pa_source_vals
    texts_df["ParagraphBreaks"] = para_break_vals
    texts_df[pa_col] = pa_vals
    return texts_df
