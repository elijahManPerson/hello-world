"""
word_map_chunker_v3.py
=======================
v3 chunker with SFL group structure, three-tier lexical classification,
and no default-to-precise fallback.

Changes from v2:
  - Three-tier lexical precision (precise / specific / simple) replaces
    the previous two-tier (precise / simple). The new `specific` tier
    captures words with no simpler one-word equivalent but which are
    uncommon (wraith, ennui, petrichor).
  - load_lexicon now returns four sets (precise, specific, simple, phrases).
  - chunk_and_classify accepts specific_set as a new positional argument.
  - classify_precision no longer defaults unknown words to 'precise'.
    With the LLM-classified lexicon giving 97%+ explicit coverage on
    real student writing, the residual unknown set is dominated by
    tokenization edge cases (contractions, numerals, single letters)
    rather than vocabulary gaps. Returning None for these lets
    downstream scoring distinguish "no signal" from "signal: precise".

Changes from v1:
  - Adverbs that modify verbs now form their own adverbial_group (Circumstance)
    instead of joining the verb group. (Adverbs modifying adjectives inside an
    NG stay in the NG as degree modifiers.)
  - Adds columns: GrammaticalType, LexicalPrecision, CohesionRole, UsedAppropriately

GroupType values:
  noun_group       = participant (NG)
  verb_group       = process (VG)
  adverbial_group  = circumstance realized by adverb(s)
  null             = floating function word or punctuation

The preposition + nominal complement is NOT bundled into one "prepositional
group" - prepositions float as grammatical items and the nominal inside is
its own NG, so we preserve participant identity inside circumstances.
"""

import pandas as pd
import re
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# POS classification
# ---------------------------------------------------------------------------

LEXICAL_TAGS = {'noun', 'proper noun', 'verb', 'adjective', 'adverb', 'numeral'}
GRAMMATICAL_TAGS = {
    'pronoun', 'determiner', 'preposition', 'auxiliary verb',
    'coordinating conjunction', 'subordinating conjunction',
    'interjection', 'other',
}

NG_HEAD_TAGS = {'noun', 'proper noun', 'pronoun'}
VG_HEAD_TAGS_PREFERRED = {'verb'}
VG_HEAD_TAGS_FALLBACK = {'auxiliary verb'}

RELATIVE_PRONOUN_TOKENS = {'who', 'whom', 'whose', 'which', 'that', 'where', 'when'}


def is_lexical(token_subcat, token_category):
    if token_category == 'punctuation':
        return None
    if token_subcat in LEXICAL_TAGS:
        return True
    if token_subcat in GRAMMATICAL_TAGS:
        return False
    return None


# ---------------------------------------------------------------------------
# Sentence chunking with SFL group types
# ---------------------------------------------------------------------------

NG_INTENSIFIERS = {
    "very", "really", "just", "too", "so", "quite", "rather", "fairly",
    "pretty", "extremely", "incredibly", "absolutely", "almost", "barely",
    "hardly", "scarcely", "only", "even",
}


def chunk_sentence(tokens):
    """Return per-token group assignments for one sentence.

    A token gets:
      - group_id_suffix (e.g. 'NG1', 'VG2', 'CG1' for adverbial_group)
      - group_type ('noun_group', 'verb_group', 'adverbial_group', None)
      - group_role ('head', 'modifier', 'attribute', None)
    """
    n = len(tokens)
    assignments = [{'group_id_suffix': None,
                    'group_type': None,
                    'group_role': None} for _ in range(n)]

    ng_count = 0
    vg_count = 0
    cg_count = 0  # adverbial / Circumstance
    current = None

    def open_g(gtype, start_idx):
        nonlocal ng_count, vg_count, cg_count
        if gtype == 'noun_group':
            ng_count += 1
            suffix = f'NG{ng_count}'
        elif gtype == 'verb_group':
            vg_count += 1
            suffix = f'VG{vg_count}'
        else:  # adverbial_group
            cg_count += 1
            suffix = f'CG{cg_count}'
        return {'type': gtype, 'id_suffix': suffix, 'indices': [start_idx]}

    def close_g(grp):
        if grp is None or not grp['indices']:
            return
        head_idx = _pick_head(tokens, grp['indices'], grp['type'])
        for idx in grp['indices']:
            assignments[idx]['group_id_suffix'] = grp['id_suffix']
            assignments[idx]['group_type'] = grp['type']
            assignments[idx]['group_role'] = 'head' if idx == head_idx else 'modifier'

    i = 0
    while i < n:
        tok = tokens[i]
        cat = tok['TokenCategory']
        pos = tok['TokenSubcat']

        if cat == 'punctuation':
            close_g(current)
            current = None
            i += 1
            continue

        # ---- Noun group starters/continuers ----
        if pos == 'determiner':
            close_g(current)
            current = open_g('noun_group', i)

        elif pos == 'adjective':
            if current and current['type'] == 'verb_group':
                # Predicative adj after copula - close VG, start NG (Rule 1 fixes later)
                close_g(current)
                current = open_g('noun_group', i)
            elif current and current['type'] == 'noun_group':
                current['indices'].append(i)
            else:
                close_g(current)
                current = open_g('noun_group', i)

        elif pos == 'numeral':
            if current and current['type'] == 'noun_group':
                current['indices'].append(i)
            else:
                close_g(current)
                current = open_g('noun_group', i)

        elif pos in ('noun', 'proper noun'):
            if current and current['type'] == 'noun_group':
                current['indices'].append(i)
            else:
                close_g(current)
                current = open_g('noun_group', i)

        elif pos == 'pronoun':
            close_g(current)
            current = open_g('noun_group', i)

        # ---- Verb group starters/continuers ----
        elif pos in ('verb', 'auxiliary verb'):
            if current and current['type'] == 'verb_group':
                current['indices'].append(i)
            else:
                close_g(current)
                current = open_g('verb_group', i)

        # ---- "Other" tokens: 'to', 'not', some contractions ----
        elif pos == 'other':
            if current and current['type'] == 'verb_group':
                current['indices'].append(i)
            else:
                close_g(current)
                current = open_g('verb_group', i)

        # ---- Adverbs: NEW LOGIC for SFL ----
        elif pos == 'adverb':
            next_pos = _peek_next_word_pos(tokens, i)
            if next_pos in ('adjective', 'adverb'):
                # Degree modifier inside NG — only intensifiers allowed (WD2)
                word_lower = str(tok.get('corr_token', '')).lower()
                if current and current['type'] == 'noun_group' and word_lower in NG_INTENSIFIERS:
                    current['indices'].append(i)
                else:
                    # Non-intensifier or no current NG — start adverbial group
                    close_g(current)
                    current = open_g('adverbial_group', i)
            else:
                # SFL change: separate adverbial_group instead of joining VG
                close_g(current)
                current = open_g('adverbial_group', i)
                # Look ahead for more adverbs in a chain
                while i + 1 < n and tokens[i+1]['TokenSubcat'] == 'adverb':
                    next_next = _peek_next_word_pos(tokens, i + 1)
                    if next_next in ('adjective', 'adverb'):
                        break  # next adverb is modifying an adj, stop here
                    i += 1
                    current['indices'].append(i)

        elif pos in ('preposition', 'subordinating conjunction', 'interjection'):
            close_g(current)
            current = None

        elif pos == 'coordinating conjunction':
            next_pos = _peek_next_word_pos(tokens, i)
            if current and current['type'] == 'noun_group' and \
               next_pos in ('adjective', 'noun', 'proper noun', 'numeral'):
                current['indices'].append(i)
            elif current and current['type'] == 'verb_group' and \
                 next_pos in ('verb', 'auxiliary verb'):
                current['indices'].append(i)
            elif current and current['type'] == 'adverbial_group' and \
                 next_pos == 'adverb':
                current['indices'].append(i)
            else:
                close_g(current)
                current = None

        else:
            close_g(current)
            current = None

        i += 1

    close_g(current)
    return assignments


def _peek_next_word_pos(tokens, i):
    for j in range(i + 1, len(tokens)):
        if tokens[j]['TokenCategory'] == 'word':
            return tokens[j]['TokenSubcat']
    return None


def _pick_head(tokens, indices, group_type):
    if group_type == 'noun_group':
        for idx in reversed(indices):
            if tokens[idx]['TokenSubcat'] in NG_HEAD_TAGS:
                return idx
        return indices[-1]
    elif group_type == 'verb_group':
        for idx in reversed(indices):
            if tokens[idx]['TokenSubcat'] in VG_HEAD_TAGS_PREFERRED:
                return idx
        for idx in reversed(indices):
            if tokens[idx]['TokenSubcat'] in VG_HEAD_TAGS_FALLBACK:
                return idx
        return indices[-1]
    else:  # adverbial_group
        for idx in reversed(indices):
            if tokens[idx]['TokenSubcat'] == 'adverb':
                return idx
        return indices[-1]


# ---------------------------------------------------------------------------
# Rule 1: attribute attachment
# ---------------------------------------------------------------------------

def apply_rule_1_attributes(tokens, assignments):
    groups = []
    seen = set()
    for i, a in enumerate(assignments):
        if a['group_id_suffix'] is None:
            continue
        key = a['group_id_suffix']
        if key in seen:
            continue
        seen.add(key)
        indices = [j for j, b in enumerate(assignments) if b['group_id_suffix'] == key]
        head_idx = next((j for j in indices if assignments[j]['group_role'] == 'head'), indices[-1])
        groups.append({'id': key, 'type': a['group_type'], 'indices': indices, 'head_idx': head_idx})

    def is_attr_candidate(grp):
        if grp['type'] != 'noun_group':
            return False
        tags = [tokens[i]['TokenSubcat'] for i in grp['indices']]
        ATTR_OK = {'adjective', 'adverb', 'coordinating conjunction'}
        return all(p in ATTR_OK for p in tags) and len(tags) > 0

    for k, grp in enumerate(groups):
        if not is_attr_candidate(grp):
            continue
        prev = groups[:k]
        if len(prev) < 2:
            continue
        last_prev = prev[-1]
        second_last = prev[-2]
        target = None
        if last_prev['type'] == 'verb_group' and second_last['type'] == 'noun_group':
            target = second_last
        elif last_prev['type'] == 'noun_group' and len(prev) >= 3:
            third_last = prev[-3]
            if second_last['type'] == 'verb_group' and third_last['type'] == 'noun_group':
                target = last_prev
        if target is None:
            continue
        for idx in grp['indices']:
            assignments[idx]['group_id_suffix'] = target['id']
            assignments[idx]['group_type'] = 'noun_group'
            assignments[idx]['group_role'] = 'attribute'


# ---------------------------------------------------------------------------
# Rule 3: clause depth
# ---------------------------------------------------------------------------

def assign_clause_depth(tokens):
    depth = 0
    depths = [0] * len(tokens)
    prev_word_pos = None
    for i, tok in enumerate(tokens):
        pos = tok['TokenSubcat']
        word = str(tok.get('corr_token', '')).lower()
        if pos == 'subordinating conjunction':
            depth += 1
        elif pos == 'pronoun' and word in RELATIVE_PRONOUN_TOKENS:
            if prev_word_pos in ('noun', 'proper noun'):
                depth += 1
        depths[i] = depth
        if tok['TokenCategory'] == 'word':
            prev_word_pos = pos
    return depths


# ---------------------------------------------------------------------------
# Grammatical type classification (for Cohesion)
# ---------------------------------------------------------------------------

# Connectives - adverbs that function as conjunctions/text-organising
# Note: conj_additive includes both clausal ("I ran, and he followed") and
# phrasal ("apples and oranges") coordination. Downstream consumers should
# distinguish via: phrasal = CohesionRole=="conj_additive" AND GroupType notna()
CONNECTIVES_TEMPORAL = {'then', 'next', 'after', 'before', 'while', 'when',
                        'finally', 'meanwhile', 'soon', 'later', 'afterwards',
                        'eventually', 'suddenly', 'now', 'first', 'second',
                        'third', 'lastly', 'until', 'till', 'once'}
CONNECTIVES_CAUSAL = {'because', 'so', 'therefore', 'consequently', 'thus',
                      'since', 'hence'}
CONNECTIVES_ADVERSATIVE = {'but', 'however', 'although', 'though', 'yet',
                           'nevertheless', 'whereas', 'instead', 'still',
                           'otherwise', 'despite', 'except'}
CONNECTIVES_ADDITIVE = {'and', 'also', 'too', 'as well', 'furthermore',
                        'moreover', 'in addition', 'besides'}

DEMONSTRATIVE_PRONOUNS = {'this', 'that', 'these', 'those'}


def classify_grammatical_type(token_text, token_subcat):
    """Return (grammatical_type, cohesion_role) for a grammatical token."""
    word = str(token_text).lower().strip()

    if token_subcat == 'pronoun':
        if word in DEMONSTRATIVE_PRONOUNS:
            return ('demonstrative_pronoun', 'reference')
        else:
            return ('pronoun', 'reference')

    if token_subcat == 'determiner':
        if word in DEMONSTRATIVE_PRONOUNS:  # this/that/these/those used as det
            return ('demonstrative_det', 'reference')
        if word in ('the',):
            return ('definite_article', None)  # could be reference but uncertain
        return ('determiner', None)

    if token_subcat == 'coordinating conjunction':
        if word in CONNECTIVES_ADVERSATIVE:
            return ('coord_conjunction', 'conj_adversative')
        if word in CONNECTIVES_CAUSAL:
            return ('coord_conjunction', 'conj_causal')
        return ('coord_conjunction', 'conj_additive')

    if token_subcat == 'subordinating conjunction':
        if word == "that":   # complementiser — not a cohesion device
            return ('sub_conjunction', None)
        if word in CONNECTIVES_TEMPORAL:
            return ('sub_conjunction', 'conj_temporal')
        if word in CONNECTIVES_CAUSAL:
            return ('sub_conjunction', 'conj_causal')
        if word in CONNECTIVES_ADVERSATIVE:
            return ('sub_conjunction', 'conj_adversative')
        return ('sub_conjunction', 'conj_other')

    if token_subcat == 'preposition':
        return ('preposition', None)

    if token_subcat == 'auxiliary verb':
        return ('auxiliary_verb', None)

    if token_subcat == 'interjection':
        # WM3: interjections that are discourse markers get cohesion roles
        role = _H2B_ADVERB_ROLES.get(word)
        return ('interjection', role)

    if token_subcat == 'other':
        if word == 'not':
            return ('negation', None)
        if word == 'to':
            return ('infinitive_to', None)
        return ('other', None)

    return (None, None)


def classify_lexical_connective(token_text, token_subcat):
    """For lexical adverbs/interjections that work as connectives, return cohesion_role."""
    if token_subcat not in ('adverb', 'interjection'):  # WM3: added interjection
        return None
    word = str(token_text).lower().strip()
    if word in CONNECTIVES_TEMPORAL:
        return 'conj_temporal'
    if word in CONNECTIVES_CAUSAL:
        return 'conj_causal'
    if word in CONNECTIVES_ADVERSATIVE:
        return 'conj_adversative'
    if word in CONNECTIVES_ADDITIVE - {'and'}:
        return 'conj_additive'
    # H2b: 7 additional adverbs missed by the original routing
    if word in _H2B_ADVERB_ROLES:
        return _H2B_ADVERB_ROLES[word]
    return None


# H2b — 7 missed adverbs extended routing
_H2B_ADVERB_ROLES = {
    'anyway':   'conj_adversative',
    'actually': 'conj_adversative',
    'indeed':   'conj_additive',
    'rather':   'conj_adversative',
    'else':     'conj_adversative',   # mostly "or else"
    'already':  'conj_temporal',
    'further':  'conj_additive',
}


# ---------------------------------------------------------------------------
# H2 — ConnectiveSophistication lookup tables
# ---------------------------------------------------------------------------

_BASIC_CONNECTIVES = {
    # Coordinating
    'and', 'but', 'or', 'so',
    # Temporal
    'then', 'now', 'when', 'before', 'after', 'next', 'first',
    'soon', 'once', 'while', 'until', 'till',
    # Duration/state
    'still', 'yet',
    # Additive
    'also', 'too', 'either', 'both', 'plus',
    # Causal
    'because',
    # Conditional
    'if',
    # Adversative
    'though',
    # Comparison
    'like',
}

_ADVANCED_CONNECTIVES = {
    # Temporal
    'meanwhile', 'eventually', 'later', 'suddenly', 'finally',
    'subsequently', 'previously', 'simultaneously', 'formerly',
    'thereafter', 'presently', 'currently',
    # Adversative
    'however', 'although', 'instead', 'despite', 'otherwise',
    'except', 'nevertheless', 'conversely', 'regardless',
    'notwithstanding',
    # Causal
    'therefore', 'consequently', 'thus', 'hence', 'accordingly',
    'since',
    # Additive
    'furthermore', 'moreover', 'besides', 'additionally',
    'similarly', 'nor',
    # H2b adverbs that are advanced
    'actually', 'indeed', 'rather',
}


def classify_connective_sophistication(token, cohesion_role):
    """Return 'basic', 'advanced', or '' for a token.

    Only fires on tokens that have a non-reference CohesionRole
    (already classified as a connective). conj_other items (wh-words,
    edge cases) are left unlabelled.

    Handles pd.NA / None safely — never raises boolean-value-of-NA error.
    """
    # Safely convert: pd.NA, None, and NaN all → empty string
    cr = '' if (cohesion_role is None or cohesion_role is pd.NA
                or str(cohesion_role) in ('nan', '<NA>')) else str(cohesion_role).strip()
    if not cr or cr.startswith('reference') or cr == 'conj_other':
        return ''
    t = str(token).lower().strip()
    if t in _BASIC_CONNECTIVES:
        return 'basic'
    if t in _ADVANCED_CONNECTIVES:
        return 'advanced'
    return ''


# ---------------------------------------------------------------------------
# Lexical precision (simple vs precise) using mined lexicon
# ---------------------------------------------------------------------------

def load_lexicon(lexicon_dir):
    """Load the precise/specific/simple/phrases lexicons from text files.

    Returns four sets. The specific tier is new in v3: words with no
    simpler one-word equivalent but which are uncommon/sophisticated
    (wraith, ennui, petrichor). See export_lexicon_to_files.py for
    how these files are produced from the LLM-classified lexicon.
    """
    lex_dir = Path(lexicon_dir)
    precise = set()
    specific = set()
    simple = set()
    phrases = set()
    p_file  = lex_dir / 'lexicon_precise.txt'
    sp_file = lex_dir / 'lexicon_specific.txt'
    s_file  = lex_dir / 'lexicon_simple.txt'
    f_file  = lex_dir / 'lexicon_phrases.txt'
    if p_file.exists():
        precise  = {l.strip().lower() for l in p_file.read_text().splitlines() if l.strip()}
    if sp_file.exists():
        specific = {l.strip().lower() for l in sp_file.read_text().splitlines() if l.strip()}
    if s_file.exists():
        simple   = {l.strip().lower() for l in s_file.read_text().splitlines() if l.strip()}
    if f_file.exists():
        phrases  = {l.strip().lower() for l in f_file.read_text().splitlines() if l.strip()}
    return precise, specific, simple, phrases


def classify_precision(token_text, token_subcat, precise_set, specific_set, simple_set):
    """Return 'precise', 'specific', 'simple', or None for a lexical token.

    Three-tier classification, in priority order:
      1. On the precise list -> precise (simpler equivalent exists)
      2. On the specific list -> specific (no simpler equivalent but rare)
      3. On the simple list -> simple (common everyday word)
      4. Proper noun -> precise (specific names are precise by convention)
      5. Default -> None (let downstream decide; do NOT default to precise)

    The default flipped from 'precise' to None in v3. With the LLM
    classified lexicon (~102K entries) and 97%+ explicit coverage on
    real student writing, the residual default-fallback is overwhelmingly
    tokenization edge cases (contractions like I'll/they're, numerals,
    single letters) rather than vocabulary gaps. Defaulting those to
    'precise' would inflate the precise count with non-vocabulary tokens.
    Returning None lets downstream scoring treat them as unclassified.
    """
    if token_subcat not in LEXICAL_TAGS:
        return None
    word = str(token_text).lower().strip()
    if not word:
        return None
    word = re.sub(r"'s$", "", word)
    if token_subcat == 'numeral':
        return 'simple'
    if word in precise_set:
        return 'precise'
    if word in specific_set:
        return 'specific'
    if word in simple_set:
        return 'simple'
    return None


# ---------------------------------------------------------------------------
# Used appropriately (from op column)
# ---------------------------------------------------------------------------

def classify_used_appropriately(op_value):
    """Map op to UsedAppropriately.

    op = 'equal' -> TRUE (student wrote exactly the right token)
    op = 'replace' -> FALSE (student wrote something else, corrector replaced)
    op = 'insert' -> FALSE (student didn't write this token at all - omission)
    other -> None
    """
    if op_value == 'equal':
        return True
    if op_value in ('replace', 'insert'):
        return False
    return None


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def chunk_and_classify(df, precise_set, specific_set, simple_set, phrases_set):
    """Apply chunking + all classification layers to the word map DataFrame.

    Signature changed in v3: accepts specific_set as a new positional
    argument between precise_set and simple_set. Callers must pass all
    four lexicons (phrases_set is loaded but not yet used in matching).
    """
    df = df.copy()
    work_mask = df['corr_index'].notna()
    work_df = df[work_mask].copy()

    # Initialise new columns
    for col in ['GroupID', 'GroupType', 'GroupRole', 'GrammaticalType',
                'LexicalPrecision', 'CohesionRole']:
        df[col] = pd.Series([pd.NA] * len(df), dtype='object')
    df['ClauseDepth'] = pd.Series([pd.NA] * len(df), dtype='Int64')
    df['IsLexical'] = pd.Series([pd.NA] * len(df), dtype='boolean')
    df['UsedAppropriately'] = pd.Series([pd.NA] * len(df), dtype='boolean')

    # Process each sentence
    grouped = work_df.groupby(['Identifier', 'CorrSentenceID'], sort=False)
    for (script_id, sent_id), sent_rows in grouped:
        sent_rows = sent_rows.sort_values('corr_index')
        original_indices = sent_rows.index.tolist()
        tokens = sent_rows[['corr_token', 'TokenCategory', 'TokenSubcat']].to_dict('records')

        assignments = chunk_sentence(tokens)
        apply_rule_1_attributes(tokens, assignments)
        depths = assign_clause_depth(tokens)

        for k, orig_idx in enumerate(original_indices):
            a = assignments[k]
            tok = tokens[k]
            tok_text = tok['corr_token']
            tok_subcat = tok['TokenSubcat']
            tok_cat = tok['TokenCategory']

            # Group columns
            if a['group_id_suffix'] is not None:
                df.at[orig_idx, 'GroupID'] = f'{script_id}_s{sent_id:03d}_{a["group_id_suffix"]}'
                df.at[orig_idx, 'GroupType'] = a['group_type']
                df.at[orig_idx, 'GroupRole'] = a['group_role']

            df.at[orig_idx, 'ClauseDepth'] = depths[k]

            # IsLexical
            lex = is_lexical(tok_subcat, tok_cat)
            df.at[orig_idx, 'IsLexical'] = lex

            # Lexical precision (three-tier in v3)
            if lex is True:
                df.at[orig_idx, 'LexicalPrecision'] = classify_precision(
                    tok_text, tok_subcat, precise_set, specific_set, simple_set
                )
                # Check if lexical adverb is a connective (Cohesion)
                lex_cohesion = classify_lexical_connective(tok_text, tok_subcat)
                if lex_cohesion is not None:
                    df.at[orig_idx, 'CohesionRole'] = lex_cohesion
                # WD1: when CohesionRole is set, precision is null (D-W3)
                if df.at[orig_idx, 'CohesionRole'] is not pd.NA and df.at[orig_idx, 'CohesionRole']:
                    df.at[orig_idx, 'LexicalPrecision'] = pd.NA
            elif lex is False:
                # Grammatical token: classify type and cohesion role
                gtype, crole = classify_grammatical_type(tok_text, tok_subcat)
                df.at[orig_idx, 'GrammaticalType'] = gtype
                if crole is not None:
                    df.at[orig_idx, 'CohesionRole'] = crole

    # UsedAppropriately from op (across all rows in corrected text)
    df.loc[work_mask, 'UsedAppropriately'] = work_df['op'].apply(classify_used_appropriately)

    # H2 — ConnectiveSophistication (basic / advanced / '')
    # Applied to every row; only fires where CohesionRole is a connective.
    # Use vectorised zip (not apply) to avoid pd.NA boolean-context errors.
    if 'CohesionRole' in df.columns:
        _tokens = df['corr_token'].fillna('').astype(str)
        _roles  = df['CohesionRole'].fillna('').astype(str).replace({'nan': '', '<NA>': ''})
        df['ConnectiveSophistication'] = [
            classify_connective_sophistication(t, r)
            for t, r in zip(_tokens, _roles)
        ]

    # WM2 — PrepGroupID / PrepGroupRole (flat, additive — doesn't touch GroupID)
    df = assign_prep_groups(df)

    return df


# ---------------------------------------------------------------------------
# WM2 — Prepositional phrase grouper
# ---------------------------------------------------------------------------

_PREP_PHRASE_BREAKERS = frozenset({
    'verb', 'auxiliary verb',
    'coordinating conjunction', 'subordinating conjunction',
})


def assign_prep_groups(wm):
    """Assign PrepGroupID and PrepGroupRole to prepositional phrases.

    For each preposition, finds the immediately following noun-group
    complement and wraps both into a prep_group.  Flat schema: new
    PrepGroupID / PrepGroupRole columns sit alongside the existing
    GroupID columns without altering them.
    """
    wm = wm.copy()
    wm['PrepGroupID']   = ''
    wm['PrepGroupRole'] = ''

    prep_group_counter = 0

    # Build a lookup: (Identifier, raw_index) → DataFrame index
    idx_lookup = {}
    for df_idx, row in wm.iterrows():
        ident = row.get('Identifier', '')
        ri    = row.get('raw_index', None)
        if ident and ri is not None:
            idx_lookup[(ident, ri)] = df_idx

    for ident in wm['Identifier'].unique():
        script = wm[wm['Identifier'] == ident].sort_values('raw_index')
        rows = script.to_dict('records')

        i = 0
        while i < len(rows):
            row = rows[i]
            if row.get('TokenSubcat') != 'preposition':
                i += 1
                continue

            # Scan ahead for the complement
            j = i + 1
            complement_raw_indices = []

            # Skip non-word tokens
            while j < len(rows) and rows[j].get('TokenCategory') != 'word':
                j += 1

            # Collect complement tokens up to and including the noun-group head
            while j < len(rows):
                tok = rows[j]
                if tok.get('TokenCategory') != 'word':
                    j += 1
                    continue
                pos = tok.get('TokenSubcat', '')
                if pos == 'preposition':
                    break  # new preposition → new phrase
                if pos in _PREP_PHRASE_BREAKERS:
                    break
                complement_raw_indices.append(tok['raw_index'])
                # Stop at noun-group head or standalone pronoun
                if (tok.get('GroupRole') == 'head' and
                        tok.get('GroupType') == 'noun_group'):
                    j += 1
                    break
                if pos == 'pronoun':
                    j += 1
                    break
                j += 1

            if complement_raw_indices:
                prep_group_counter += 1
                gid = f'pg_{ident}_{prep_group_counter}'

                # Write preposition head
                df_idx = idx_lookup.get((ident, row['raw_index']))
                if df_idx is not None:
                    wm.at[df_idx, 'PrepGroupID']   = gid
                    wm.at[df_idx, 'PrepGroupRole'] = 'head'

                # Write complement tokens
                for ri in complement_raw_indices:
                    df_idx = idx_lookup.get((ident, ri))
                    if df_idx is not None:
                        wm.at[df_idx, 'PrepGroupID']   = gid
                        wm.at[df_idx, 'PrepGroupRole'] = 'complement'

                i = j
            else:
                i += 1

    return wm
