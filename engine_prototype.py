"""
engine_prototype.py
===================
Fills the two deferred columns:
  - TokenSubcat   (word class for word/contraction rows)
  - SS class      (grammatical subtype of sentence-structure errors)

Two interchangeable engines, same input and output, so you can run both on
the same scripts and compare. Neither runs in the build sandbox (no API,
no internet); both are written to run in your Colab / machine.

  fill_with_claude(df, model="claude-haiku-4-5-20251001")
  fill_with_spacy(df)                      # free, local, needs: pip install spacy + model

Both accept mock=True to show the wiring and the exact prompt/IO with no
cost and no install, so you can sanity-check the plumbing first.

Cost shape (Haiku-class, your r5 corpus): only the ~350 real edits per run
are sent for SS-class, not all 6,000 tokens. Word class is done locally by
spaCy for free. That keeps the API spend a rounding error next to the
correction call you already pay for. Confirm live rates at the Anthropic
pricing page.
"""

import json, re

ENGINE = "TBD(engine)"

# ----------------------------------------------------------------------
# Shared: which rows still need each engine
# ----------------------------------------------------------------------
def _rows_needing_wordclass(df):
    return df.index[df["TokenSubcat"] == ENGINE].tolist()

def _rows_needing_ssclass(df):
    return df.index[df["SS class"] == ENGINE].tolist()


# ----------------------------------------------------------------------
# Engine A — Claude, cheap model, one structured call per script
# ----------------------------------------------------------------------
_TAG_SYSTEM = """You label tokens from a student's corrected narrative.
Return STRICT JSON only, a list with one object per item id, no prose:

[{"id": 0, "word_class": "...", "edit_kind": "...", "spelling_kind": "...",
  "ss_class": "..."}]

word_class (ALWAYS give this for a word or contraction): one of
noun, proper noun, verb, auxiliary verb, adjective, adverb, pronoun,
determiner, preposition, coordinating conjunction,
subordinating conjunction, numeral, interjection, other.

For items that include "student_wrote" and "corrected_to" (an edit):

edit_kind: "spelling" if the student clearly meant the corrected word but
mis-spelt it, including homophones and slips; "word-form" if the change is
grammatical, a different form of the word or a different word that changes
the sentence (e.g. ate->eat, run->running, a missing or extra word).

If edit_kind is "spelling": give spelling_kind = one of
"Incorrect" (a genuine spelling error, the student did not know it),
"Typo" (a not-plausible-as-spelling mechanical slip, e.g. a letter
reversal or stray key, the student plainly knows the word),
"Homophone" (a real word that sounds like the target: their/there).
Leave ss_class "".

If edit_kind is "word-form": give ss_class = one of
verb tense, verb agreement, noun number, noun other, adjective, adverb,
preposition, determiner, conjunction, pronoun, word order, missing word,
extra word, other. Leave spelling_kind "".

For an inserted or deleted word, edit_kind is "word-form" and give the
ss_class (missing word / extra word, or the grammatical type)."""

def _claude_items(df, idxs):
    items = []
    for i in idxs:
        r = df.loc[i]
        it = {"id": int(i),
              "token": str(r.get("corr_token") or r.get("raw_token") or "")}
        op = r.get("op")
        if op != "equal":
            it["student_wrote"] = str(r.get("raw_token") or "")
            it["corrected_to"] = str(r.get("corr_token") or "")
            it["change"] = op
        items.append(it)
    return items


def _rows_needing_editkind(df):
    m = (df["op"].astype(str) == "replace") & (
        df["Spell error"].astype(str) == "TRUE")
    return df.index[m].tolist()

def fill_with_claude(df, model="claude-haiku-4-5-20251001",
                     api_key=None, mock=False):
    df = df.copy()
    need = sorted(set(_rows_needing_wordclass(df))
                  | set(_rows_needing_ssclass(df))
                  | set(_rows_needing_editkind(df)))
    if not need:
        return df

    # batch by script so context stays small and cacheable
    groups = {}
    for i in need:
        groups.setdefault(str(df.loc[i, "Identifier"]), []).append(i)

    if mock:
        any_id = next(iter(groups))
        print("MOCK — first script payload that would be sent:")
        print(json.dumps(_claude_items(df, groups[any_id][:6]), indent=2))
        print("\nMOCK — system prompt:\n", _TAG_SYSTEM[:300], "...")
        return df

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    SPELL_KINDS = {"Incorrect", "Typo", "Homophone"}
    for sid, idxs in groups.items():
        items = _claude_items(df, idxs)
        msg = client.messages.create(
            model=model, max_tokens=2048,
            system=[{"type": "text", "text": _TAG_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user",
                       "content": json.dumps({"items": items})}],
        )
        txt = next((b.text for b in msg.content if b.type == "text"), "[]")
        txt = re.sub(r"```json|```", "", txt).strip()
        try:
            tags = json.loads(txt)
        except Exception:
            continue
        for t in tags:
            i = t.get("id")
            if i not in df.index:
                continue
            # 1) word class
            if df.at[i, "TokenSubcat"] == ENGINE and t.get("word_class"):
                df.at[i, "TokenSubcat"] = t["word_class"]
            op = str(df.at[i, "op"])
            ek = t.get("edit_kind")
            # 2) insert/delete still needing an SS class
            if df.at[i, "SS class"] == ENGINE and t.get("ss_class"):
                df.at[i, "SS class"] = t["ss_class"]
            # 3) a replace: resolve spelling vs word-form (request 3)
            if op == "replace" and str(df.at[i, "Spell error"]) == "TRUE":
                if ek == "word-form":
                    # it is sentence structure, not spelling
                    df.at[i, "Spell error"] = "NA"
                    df.at[i, "Spell subtype"] = "NA"
                    df.at[i, "Spell class"] = "NA"
                    df.at[i, "SS error"] = "TRUE"
                    df.at[i, "SS subtype"] = "Changed"
                    df.at[i, "SS class"] = t.get("ss_class") or "other"
                elif ek == "spelling":
                    sk = t.get("spelling_kind")
                    if sk in SPELL_KINDS:        # engine widens Typo here
                        df.at[i, "Spell class"] = sk
                    # SS stays NA (already set by the cheap layer)
    return df


# ----------------------------------------------------------------------
# Engine B — local spaCy, free, deterministic
# ----------------------------------------------------------------------
_SPACY_POS = {"NOUN": "noun", "PROPN": "proper noun", "VERB": "verb",
              "AUX": "auxiliary verb", "ADJ": "adjective", "ADV": "adverb",
              "PRON": "pronoun", "DET": "determiner", "ADP": "preposition",
              "CCONJ": "coordinating conjunction",
              "SCONJ": "subordinating conjunction",
              "NUM": "numeral", "INTJ": "interjection",
              "PART": "other", "SYM": "other", "X": "other"}

def fill_with_spacy(df, mock=False):
    df = df.copy()
    if mock:
        print("MOCK — spaCy path: pip install spacy && "
              "python -m spacy download en_core_web_sm")
        print("It tags word_class for every word/contraction locally, free.")
        print("SS class still needs rules or Claude; spaCy gives POS, not "
              "error type, so pair it with Engine A for SS class.")
        return df
    import spacy
    from spacy.tokens import Doc
    nlp = spacy.load("en_core_web_sm")
    for sid, g in df.groupby("Identifier", sort=False):
        toks = [(i, str(df.loc[i, "corr_token"] or df.loc[i, "raw_token"] or ""))
                for i in g.index
                if df.loc[i, "TokenSubcat"] == ENGINE]
        toks = [(i, t) for i, t in toks if t]   # drop empties
        if not toks:
            continue
        # Build a Doc from our pre-tokenised list so spaCy does NOT
        # retokenise. Without this, contractions like "I'm" make spaCy
        # emit more tokens than ours, and the zip below misaligns every
        # downstream row in the script.
        words = [t for _, t in toks]
        doc = Doc(nlp.vocab, words=words)
        for pipe_name, pipe in nlp.pipeline:
            doc = pipe(doc)
        assert len(doc) == len(toks), \
            f"alignment failed for {sid}: {len(doc)} vs {len(toks)}"
        for (i, _), tk in zip(toks, doc):
            df.at[i, "TokenSubcat"] = _SPACY_POS.get(tk.pos_, "other")
    return df


# ======================================================================
# Word-class engine B: Claude (Haiku) — selectable alternative to spaCy
# ======================================================================

_WORDCLASS_SYSTEM = """\
Classify each token's word class.

"c" must be exactly one of:
  noun, proper noun, verb, auxiliary verb, adjective, adverb,
  pronoun, determiner, preposition, coordinating conjunction,
  subordinating conjunction, numeral, interjection, other.

Hard cases:
  "to" before a verb → other (infinitive marker, not preposition)
  "to" before a noun/pronoun → preposition
  Lowercased proper nouns (character names, place names) → proper noun
  "but" joining two clauses → coordinating conjunction
  "but" meaning "only/except" → preposition
  "as" introducing a clause → subordinating conjunction
  "as" meaning "in the role of" → preposition

Return a JSON array, one object per input item, order preserved:
  [{"id": <integer>, "c": "<word class>"}, ...]
No prose, no markdown fences."""

_VALID_WORDCLASS = set(_SPACY_POS.values()) | {"other"}


def fill_wordclass_with_claude(df, texts=None,
                               model="claude-haiku-4-5-20251001",
                               api_key=None, mock=False,
                               corr_col="Corrected text (8)",
                               id_col="Research ID",
                               chunk_size=400):
    """Fill TokenSubcat for word/contraction rows using Claude.

    Drop-in replacement for fill_with_spacy when the Claude engine is
    selected. Chunks long scripts so the response never truncates silently.
    Per-chunk failures are logged and skipped; the rest of the script
    continues.
    """
    import anthropic, json, textwrap
    df = df.copy()
    if mock:
        print("MOCK — Claude word-class path:")
        print(f"  model: {model}")
        print("  Per script: sends full corrected text + word/contraction items")
        print(f"  14-way enum prompt, chunks of {chunk_size} tokens")
        print("  Returns [{'id': 0, 'c': 'noun'}, ...] per chunk")
        return df

    client = anthropic.Anthropic(api_key=api_key)

    # Build a lookup from Identifier → corrected text if supplied
    corr_lookup = {}
    if texts is not None:
        for _, row in texts.iterrows():
            corr_lookup[str(row[id_col])] = str(row.get(corr_col, ""))

    for sid, g in df.groupby("Identifier", sort=False):
        idxs = [i for i in g.index
                if df.loc[i, "TokenCategory"] in ("word", "contraction")
                and df.loc[i, "TokenSubcat"] == ENGINE]
        if not idxs:
            continue

        script_text = corr_lookup.get(str(sid), "")
        items = [{"id": pos, "tok": str(df.loc[i, "corr_token"] or
                                        df.loc[i, "raw_token"] or "")}
                 for pos, i in enumerate(idxs)]

        # chunk to avoid silent truncation
        for chunk_start in range(0, len(items), chunk_size):
            chunk = items[chunk_start: chunk_start + chunk_size]
            user_msg = (
                f"Script:\n{script_text}\n\n"
                f"Tokens:\n{json.dumps(chunk, ensure_ascii=False)}"
            )
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=len(chunk) * 12 + 200,
                    system=_WORDCLASS_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text.strip()
                parsed = json.loads(raw)
                id_to_class = {item["id"]: item["c"] for item in parsed}
                for pos, i in enumerate(idxs[chunk_start: chunk_start + chunk_size]):
                    wc = id_to_class.get(chunk_start + pos)
                    if wc and wc in _VALID_WORDCLASS:
                        df.at[i, "TokenSubcat"] = wc
            except Exception as exc:
                print(f"  [wordclass] {sid} chunk {chunk_start}: {exc} — skipped")
    return df


if __name__ == "__main__":
    import pandas as pd, sys
    sys.path.insert(0, "/home/claude")
    print("This is a prototype module. In Colab:")
    print("  from engine_prototype import fill_with_claude, fill_with_spacy")
    print("  df = fill_with_spacy(df)              # free word classes")
    print("  df = fill_with_claude(df, mock=True)  # see the SS-class call")


# ======================================================================
# Sentence-level engine: Sentence type + Pronoun reference
# ======================================================================
# Pronoun severity, editable one line, same idea as SS_SEVERITY in mech_v2.
# Most serious first. No-antecedent is the worst cohesion break (no tie at
# all), then a tie to the wrong thing, then an unclear tie.
PRONOUN_SEVERITY = ["No antecedent", "Wrong referent", "Ambiguous referent"]

_SENT_TYPES = ("Simple", "Compound", "Complex", "Compound-Complex",
               "Complex - Projected/speech", "Fragment")

_SENT_SYSTEM = """You analyse student narrative sentences. You are given the \
whole script for context and a numbered list of its sentences. For EACH \
sentence id return two judgements.

1) sentence_type, exactly one of:
- "Simple": one independent clause.
- "Compound": two or more independent clauses joined by a coordinator or \
semicolon, no dependent clause.
- "Complex": one independent clause plus one or more dependent clauses.
- "Compound-Complex": compound plus at least one dependent clause.
- "Complex - Projected/speech": a clause complex built on projection, that \
is reported or direct speech or thought, for example  she said that ...  or \
 "...," she said.  Use this whenever projection is the main link, even if \
the sentence is otherwise complex.
- "Fragment": grammatically incomplete unit — missing the main subject, the \
main verb, or both, and not standing in for one by convention. Counts as an \
incorrect simple sentence in the rubric. Examples that ARE fragments: \
"Out in the dark.", "Because he was tired.", "After they left."  Examples \
that are NOT fragments and should be labelled Simple instead: stylistic \
interjections and one-word exclamations ("Wow!", "Yes!", "Hello."), \
imperatives where the subject is understood ("Stop!", "Run."), and dialogue \
tags that pair with the speech ("she said.", "Andrew asked curiously.").

2) pronoun, judged using the WHOLE script for context:
- verdict "Correct": pronouns have clear recoverable referents.
- verdict "NA": no pronouns to judge.
- verdict "Incorrect" with subtype one of:
  - "No antecedent": a pronoun has nothing in the text it can refer to.
  - "Wrong referent": a pronoun points to the wrong thing.
  - "Ambiguous referent": two or more possible referents, unclear which.
If more than one pronoun problem, report the single most serious in this \
order: No antecedent, then Wrong referent, then Ambiguous referent.

Return STRICT JSON only, a list, no prose:
[{"id": 0, "sentence_type": "...", "pronoun_verdict": "...", \
"pronoun_subtype": "..."}]
pronoun_subtype is "" unless verdict is "Incorrect"."""


def _sentence_items(sent_df_one_script):
    g = sent_df_one_script.sort_values("SentenceRef")
    items, refs = [], []
    for n, (_, r) in enumerate(g.iterrows()):
        # skip artifacts / non-scorable: those stay NA, not sent to model
        if str(r.get("Sentence type")) != ENGINE:
            continue
        items.append({"id": n, "sentence": str(r["CorrectedSentence"])})
        refs.append((n, r["SentenceRef"]))
    return items, refs, g


_SENT_CHUNK = 20        # max sentences per API call
_SENT_TOK_PER = 150    # token budget per sentence in the response
_SENT_TOK_OVERHEAD = 400


def _rescue_partial_json(txt):
    """Extract as many complete objects as possible from a truncated JSON array."""
    txt = txt.strip()
    if not txt.startswith("["):
        return "[]"
    # find the rightmost complete object boundary
    for end in (txt.rfind("}]"), txt.rfind("},")):
        if end > 0:
            candidate = txt[:end + (2 if txt[end:end+2] == "}]" else 1)] + "]"
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass
    return "[]"


def fill_sentence_engine(sent_df, model="claude-sonnet-4-6",
                         api_key=None, mock=False,
                         chunk_size=10, max_retries=3):
    """Fill Sentence type, Pronoun ref, Pronoun ref subtype.

    Script context is built from all non-artifact sentences (reviewer
    issue 2): titles and endings are excluded; the model judges pronoun
    reference against story sentences only.

    Robust against silent drop-out (Change 6):
      - Chunk-local ids 0..k-1 per call, so the model cannot drift on
        positional id matching when it omits items.
      - max_retries attempts per chunk with linear backoff.
      - If the response is missing ids that were sent, those ids are
        retried in a smaller focused follow-up call before being marked
        FAILED(engine).
      - Real failures are printed with script id and reason, never
        silently swallowed as TBD(engine)."""
    df = sent_df.copy()

    import anthropic, time
    client = anthropic.Anthropic(api_key=api_key)

    def _apply_tag(ref, t):
        st = t.get("sentence_type")
        if st in _SENT_TYPES:
            df.loc[df["SentenceRef"] == ref, "Sentence type"] = st
            # Change 9 — Fragment is an incorrect simple sentence per
            # the rubric, so override SS internal regardless of edit count.
            if st == "Fragment":
                df.loc[df["SentenceRef"] == ref, "SS internal"] = "Incorrect"
        pv = t.get("pronoun_verdict")
        if pv in ("Correct", "Incorrect", "NA"):
            df.loc[df["SentenceRef"] == ref, "Pronoun ref"] = pv
            df.loc[df["SentenceRef"] == ref, "Pronoun ref subtype"] = (
                t.get("pronoun_subtype", "") if pv == "Incorrect" else "NA")

    def _call(items, script_text, mt):
        msg = client.messages.create(
            model=model, max_tokens=mt,
            system=[{"type": "text", "text": _SENT_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": json.dumps(
                {"script": script_text, "sentences": items})}],
        )
        txt = next((b.text for b in msg.content if b.type == "text"), "[]")
        if msg.stop_reason == "max_tokens":
            txt = _rescue_partial_json(txt)
        txt = re.sub(r"```json|```", "", txt).strip()
        parsed = json.loads(txt)
        return {t["id"]: t for t in parsed
                if isinstance(t, dict) and "id" in t}

    for ident, g in df.groupby("Identifier", sort=False):
        g_sorted = g.sort_values("SentenceRef")
        # collect refs that still need filling (ENGINE-typed sentences only)
        needs = [(str(r["CorrectedSentence"]), r["SentenceRef"])
                 for _, r in g_sorted.iterrows()
                 if str(r.get("Sentence type")) == ENGINE]
        # context: all non-artifact sentences (reviewer issue 2, narrower form)
        script_text = " ".join(
            str(r["CorrectedSentence"])
            for _, r in g_sorted.iterrows()
            if str(r.get("TextualArtifact", "")) == ""
            and str(r.get("CorrectedSentence", "")).strip()
        )
        if not needs:
            continue

        if mock:
            print(f"MOCK script {ident}: {len(needs)} sentences would be "
                  f"sent in chunks of {chunk_size} with non-artifact script "
                  f"context.")
            print("first item:", json.dumps(
                {"id": 0, "sentence": needs[0][0]}))
            print("system prompt head:\n", _SENT_SYSTEM[:240], "...\n")
            return df

        for start in range(0, len(needs), chunk_size):
            chunk = needs[start:start + chunk_size]
            # chunk-local ids 0..k-1 — no positional drift across chunks
            items = [{"id": i, "sentence": s} for i, (s, _) in enumerate(chunk)]
            refs = [ref for _, ref in chunk]
            mt = min(len(items) * 300 + 600, 8192)

            tags, last_err = {}, None
            for attempt in range(max_retries):
                try:
                    tags = _call(items, script_text, mt)
                    break
                except Exception as e:
                    last_err = e
                    if attempt < max_retries - 1:
                        time.sleep(1 + attempt)
            if not tags and last_err is not None:
                print(f"  [sentence engine] {ident} chunk start={start} "
                      f"failed after {max_retries} attempts: "
                      f"{type(last_err).__name__}: {last_err}")
                for ref in refs:
                    df.loc[df["SentenceRef"] == ref,
                           "Sentence type"] = "FAILED(engine)"
                continue

            # apply what came back; collect any missing for a focused retry
            missing = []
            for i, ref in enumerate(refs):
                t = tags.get(i)
                if t is None:
                    missing.append((i, ref))
                else:
                    _apply_tag(ref, t)

            if missing:
                miss_items = [{"id": i, "sentence": chunk[i][0]}
                              for i, _ in missing]
                miss_mt = min(len(miss_items) * 400 + 800, 8192)
                try:
                    retry_tags = _call(miss_items, script_text, miss_mt)
                except Exception as e:
                    print(f"  [sentence engine] {ident} retry of "
                          f"{len(missing)} missing items failed: "
                          f"{type(e).__name__}: {e}")
                    retry_tags = {}
                for i, ref in missing:
                    t = retry_tags.get(i)
                    if t is None:
                        df.loc[df["SentenceRef"] == ref,
                               "Sentence type"] = "FAILED(engine)"
                    else:
                        _apply_tag(ref, t)
    return df
