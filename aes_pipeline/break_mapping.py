"""
break_mapping.py
================
Phase BK1 -- deterministic break-mapping foundation.

Parses each script's HTML into an ordered list of blocks, then maps every
sentence to the block it sits in. Adds five new columns to the sentences
table (plus one audit column):

  - ParagraphID            integer per script (1, 2, 3...)
  - BlockType              paragraph / line_break / list_item
  - PrecededByBreak        none / paragraph / line_break / list_item
  - BlockStyle             normal / title_styled
  - BlockPosition          first / middle / last / only
  - BlockMatchConfidence   high / low      (audit, not used by Step 2)

This is STEP 1 -- plumbing only, no break-correctness judgements. The
True/Missing/Superfluous classifications are Step 2, layered on top.

Match strategy
--------------
Sentence text won't match the HTML character-for-character (spelling was
corrected). We match on the first few words of the RAW sentence text,
lowercased and whitespace-normalised. This handles the overwhelming
majority of cases because the first few words are usually short common
words that the corrector leaves alone. Where a match isn't confident we
keep the sentence in the current block (no crash) and flag the row with
BlockMatchConfidence='low' for audit.
"""
from __future__ import annotations

import re
from html import unescape

import pandas as pd

# Block-level open tags carrying inner text
_BR_RE = re.compile(r'<\s*br\s*/?\s*>', re.IGNORECASE)
_TAG_RE = re.compile(r'<[^>]+')

# Title-style signals
_CENTER_RE = re.compile(r'text-align\s*:\s*center', re.IGNORECASE)
_STRONG_RE = re.compile(r'<\s*strong\b', re.IGNORECASE)
_BIGFONT_RE = re.compile(r'font-size\s*:\s*(1[6-9]|[2-9][0-9])px',
                         re.IGNORECASE)

# A <p> or <li> block plus its inner content, tolerant of missing close
# tags (corpus has malformed `<p/P` artefacts). We capture inner text up
# to either the matching close tag or the next block-open / end of input.
_BLOCK_ITER_RE = re.compile(
    r'<\s*(p|li)\b([^>]*)>(.*?)'
    r'(?=</\s*(?:p|li)\s*[Pp]?\s*>|<\s*(?:p|li)\b|$)',
    re.IGNORECASE | re.DOTALL,
)


def _strip_to_text(s: str) -> str:
    """HTML -> plain text: drop all tags, unescape entities, normalise NBSPs
    and whitespace to single spaces."""
    if not s:
        return ""
    t = _TAG_RE.sub('', s)
    t = unescape(t).replace('\xa0', ' ').replace('​', '')
    return ' '.join(t.split())


def parse_html_blocks(html: str) -> list:
    """Parse HTML into an ordered list of block dicts.

    Each block: {text, block_type, styled, order}.
      block_type: 'paragraph' | 'list_item' | 'line_break_segment'
      styled: True if the block carries title-like styling
    Handles: <p>, <li>, <br> splits, bare text (no tags = one paragraph),
    &nbsp; decoding, malformed close tags.
    """
    if not html or not str(html).strip():
        return []

    blocks: list = []
    order = 0
    matches = list(_BLOCK_ITER_RE.finditer(str(html)))

    if not matches:
        # Bare text -- single paragraph
        text = _strip_to_text(html)
        if text:
            blocks.append({
                'text': text, 'block_type': 'paragraph',
                'styled': False, 'order': 0,
            })
        return blocks

    for m in matches:
        tag = m.group(1).lower()
        attrs = m.group(2) or ''
        inner = m.group(3) or ''
        block_type = 'list_item' if tag == 'li' else 'paragraph'
        styled = bool(
            _CENTER_RE.search(attrs)
            or _STRONG_RE.search(inner)
            or _BIGFONT_RE.search(inner)
            or _BIGFONT_RE.search(attrs)
        )

        # Split on <br> into segments (soft line breaks within a paragraph)
        segments = _BR_RE.split(inner)
        # Filter out segments that are empty after stripping
        seg_texts = [_strip_to_text(s) for s in segments]
        seg_texts = [t for t in seg_texts if t]
        if not seg_texts:
            continue

        if len(seg_texts) == 1:
            blocks.append({
                'text': seg_texts[0], 'block_type': block_type,
                'styled': styled, 'order': order,
            })
            order += 1
        else:
            # Multiple <br>-separated segments
            for t in seg_texts:
                blocks.append({
                    'text': t, 'block_type': 'line_break_segment',
                    'styled': styled, 'order': order,
                })
                order += 1

    return blocks


def _norm(t: str) -> str:
    return ' '.join(str(t or '').lower().split())


def _find_block(blocks_norm: list, sent_text: str, prev_block: int):
    """Find the smallest block index >= prev_block whose normalised text
    contains the first N words of sent_text. Tries head lengths 4 -> 3 -> 2,
    so a single off-by-one word divergence (e.g. corrector swapped "with"
    for "by") still aligns.

    Returns (block_idx, confidence). Confidence is 'high' if a >=3-word head
    matched, otherwise 'low'. If no head matches anywhere from prev_block
    onward, returns (prev_block, 'low') -- the sentence stays in the
    current block rather than crashing or jumping wildly."""
    words = _norm(sent_text).split()
    if not words:
        return prev_block, 'low'
    n_blocks = len(blocks_norm)
    for n_head in (4, 3, 2):
        head = ' '.join(words[:n_head])
        if not head:
            continue
        for b in range(prev_block, n_blocks):
            if head in blocks_norm[b]:
                return b, ('high' if n_head >= 3 else 'low')
    return prev_block, 'low'


def _public_block_type(block_type: str) -> str:
    """External (column) label: collapse 'line_break_segment' -> 'line_break'."""
    return 'line_break' if block_type == 'line_break_segment' else block_type


def map_sentences_to_blocks(sent_df: pd.DataFrame,
                            texts_df: pd.DataFrame,
                            id_col: str = "Research ID",
                            html_col: str = "Raw HTML") -> pd.DataFrame:
    """Add BK1 columns to sentence rows by aligning sentence text to HTML
    blocks. Mutates a copy of sent_df. Sentences whose script has no HTML
    fall back to a single virtual block (ParagraphID=1, BlockPosition=only,
    BlockMatchConfidence=low)."""
    df = sent_df.copy()

    # Initialise new columns
    df['ParagraphID'] = 0
    df['BlockType'] = 'paragraph'
    df['PrecededByBreak'] = 'none'
    df['BlockStyle'] = 'normal'
    df['BlockPosition'] = 'only'
    df['BlockMatchConfidence'] = 'high'

    # Build html lookup keyed by Identifier (== string of id_col)
    html_lookup = {}
    if html_col in texts_df.columns:
        for _, r in texts_df.iterrows():
            html_lookup[str(r[id_col])] = str(r.get(html_col, '') or '')

    for ident, g in df.groupby('Identifier', sort=False):
        html = html_lookup.get(str(ident), '')
        blocks = parse_html_blocks(html)
        n_blocks = len(blocks)

        # Sort sentences in narrative order
        g_sorted = g.sort_values('SentenceRef')
        idxs = g_sorted.index.tolist()

        if n_blocks == 0:
            # No HTML / unparseable -- one virtual block, low confidence
            for idx in idxs:
                df.at[idx, 'ParagraphID'] = 1
                df.at[idx, 'BlockType'] = 'paragraph'
                df.at[idx, 'PrecededByBreak'] = 'none'
                df.at[idx, 'BlockStyle'] = 'normal'
                df.at[idx, 'BlockPosition'] = 'only'
                df.at[idx, 'BlockMatchConfidence'] = 'low'
            continue

        # Pre-normalise every block's text once
        blocks_norm = [_norm(b['text']) for b in blocks]

        # Walk sentences in narrative order; monotonic cursor across blocks.
        prev_block = 0
        prev_chosen = -1   # last block actually assigned (for break detection)

        for idx in idxs:
            row = df.loc[idx]
            # Prefer raw text for matching (HTML carries the raw spelling);
            # fall back to corrected if raw is empty.
            sent_text = (str(row.get('RawSentence') or '')
                         or str(row.get('CorrectedSentence') or ''))

            block_idx, confidence = _find_block(blocks_norm, sent_text,
                                                prev_block)

            blk = blocks[block_idx]
            block_type = blk['block_type']
            para_id = block_idx + 1

            # PrecededByBreak: first sentence of any newly-entered block
            # carries the block's entry break; later sentences in the same
            # block do not. Use prev_chosen so we detect transitions even
            # if the very first sentence didn't land in block 0.
            if block_idx != prev_chosen and block_idx > 0:
                preceded = _public_block_type(block_type)
            else:
                preceded = 'none'

            # Block position within script
            if n_blocks == 1:
                pos = 'only'
            elif block_idx == 0:
                pos = 'first'
            elif block_idx == n_blocks - 1:
                pos = 'last'
            else:
                pos = 'middle'

            df.at[idx, 'ParagraphID'] = para_id
            df.at[idx, 'BlockType'] = _public_block_type(block_type)
            df.at[idx, 'PrecededByBreak'] = preceded
            df.at[idx, 'BlockStyle'] = ('title_styled' if blk['styled']
                                        else 'normal')
            df.at[idx, 'BlockPosition'] = pos
            df.at[idx, 'BlockMatchConfidence'] = confidence

            prev_block = block_idx
            prev_chosen = block_idx

    # Compact ParagraphIDs per script so they are contiguous (1..K) and
    # recompute BlockPosition from the actually-used block sequence. Empty
    # HTML blocks (e.g. `<p>&nbsp;</p>` spacers) or list items the sentence
    # layer didn't extract leave gaps in the raw block index; downstream
    # consumers expect 1, 2, 3... with no holes.
    for ident, g in df.groupby('Identifier', sort=False):
        g_sorted = g.sort_values('SentenceRef')
        used = []
        seen = set()
        for pid in g_sorted['ParagraphID']:
            if pid not in seen:
                used.append(int(pid))
                seen.add(int(pid))
        remap = {old: new for new, old in enumerate(used, start=1)}
        k = len(used)
        for idx, pid in zip(g_sorted.index, g_sorted['ParagraphID']):
            new_pid = remap.get(int(pid), int(pid))
            df.at[idx, 'ParagraphID'] = new_pid
            if k == 1:
                df.at[idx, 'BlockPosition'] = 'only'
            elif new_pid == 1:
                df.at[idx, 'BlockPosition'] = 'first'
            elif new_pid == k:
                df.at[idx, 'BlockPosition'] = 'last'
            else:
                df.at[idx, 'BlockPosition'] = 'middle'

    return df
