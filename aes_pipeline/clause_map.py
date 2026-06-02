"""
Optional add-on -- Clause map (fine-grained clause/phrase layer).

OFF by default. Turn on with the stage toggle in run_all.py
(CONFIG["stage_clauses"] = "run"). Needs an api_key.

Produces a NEW table, clause_map.csv, sitting between sentences and
word_map: one row per UNIT, where a unit is either a clause or a phrase
that stands outside any clause. It does not modify the sentence rows.

Every CLAUSE becomes a unit, including clauses embedded inside another
(relative, content/noun, finite subordinate, and non-finite clauses that
act as adjuncts or carry their own subject). A to-infinitive or -ing/-ed
that is merely the object or complement of the main verb ("wanted to go",
"liked living there") stays IN the main clause and is not extracted. When a
clause is embedded, the host keeps its remaining words with the embedded
clause removed, so no word appears in two units. Verbless phrases that do a
job inside a clause stay absorbed; only material standing OUTSIDE every
clause (a fronted or absolute phrase, an appositive, a vocative, or a whole
verbless fragment) becomes a Phrase unit. Because embedded clauses are
pulled out, the unit texts no longer simply concatenate back into the
sentence, but no word is double-counted.

Columns
-------
    Identifier     script id (links to texts / sentences / word_map)
    SentenceRef    the sentence this unit belongs to
    UnitIndex      order within the sentence, 0-based
    UnitText       the span itself
    UnitKind       "Clause" | "Phrase"
    Structure      see _STRUCTURE_CLAUSE / _STRUCTURE_PHRASE
    Position       "Initial" | "Medial" | "Final"
    Function       "(main)" or a meaning relation (see _FUNCTIONS)

This layer is explanatory and teacher-facing. Clause segmentation is a
genuine judgement call at the edges, so expect more run-to-run wobble than
the Sentence type column. Do NOT feed it into scoring without a validation
pass. It is, however, a useful cross-check on Sentence type (two
independents and no subordinate should read Compound, and so on).
"""
from __future__ import annotations

import json
import re

# Canonical controlled vocabularies -----------------------------------------
_STRUCTURE_CLAUSE = ("Independent", "Coordinate", "Subordinate",
                     "Relative", "Content", "Non-finite", "Verbless")
_STRUCTURE_PHRASE = ("Prepositional", "Adverbial", "Noun")
_POSITIONS = ("Initial", "Medial", "Final")
_FUNCTIONS = ("(main)", "Cause", "Condition", "Time", "Contrast",
              "Concession", "Addition", "Elaboration", "Projection",
              "Comparison", "Vocative")

# Sentence types that are NOT worth clause-mapping (excluded artifacts and
# non-genuine writing). Everything else with a real type is mapped.
_SKIP_TYPES = {"SoundEffect", "Interjection", "TBD(engine)",
               "FAILED(engine)", "nan", ""}

_CM_SYSTEM = """You break student narrative sentences into their grammatical \
units, one unit per CLAUSE plus one unit for each PHRASE that stands outside \
every clause. You are given the whole script for context and a numbered list \
of its sentences.

EXTRACT EVERY CLAUSE as its own unit, including clauses that are EMBEDDED \
inside another clause. This is the rule most often missed, so attend to it:
- a relative clause modifying a noun ("the dog THAT BARKED") is its own unit;
- a content/noun clause filling a slot ("she knew THAT SHE WAS LATE", "three \
reasons WHY I HATE THIS") is its own unit;
- a finite subordinate adverbial clause ("because she was tired", "while \
they ran") is its own unit;
- a non-finite clause is its own unit ONLY when it is an ADJUNCT or carries \
its own subject: a fronted or trailing purpose/result/participial phrase \
("TO WIN THE RACE, she trained", "she trained TO WIN THE RACE", \
"EXHAUSTED, he sat down", "HIM BEING LATE, we left").

Do NOT split out a to-infinitive or -ing/-ed that is simply the object or \
complement of the main verb. Those stay PART OF the main clause: "I wanted \
to go home", "Jordan liked living there", "it was time to shine", "she \
hoped to find peace" are each ONE clause. Treat the verb-plus-complement as \
a single predicate. If you are unsure whether a non-finite is an adjunct or \
a complement, leave it inside the main clause.

If the sentence contains a finite verb (or an adjunct/own-subject non-finite \
as above) that you have not placed in its own unit, you have under-segmented; \
fix it. But never split out a bare verb complement.

When a clause is embedded inside another, give the embedded clause its own \
words, and give the HOST clause the remaining words with the embedded clause \
removed, so no word appears in two units. Example: "Let me give you three \
reasons why I hate NAPLAN" gives host "Let me give you three reasons" \
(Independent) and "why I hate NAPLAN" (Content). Order units by where each \
one begins; an interrupting embedded clause takes position "Medial".

A PHRASE unit (no verb) is emitted ONLY for verbless material standing \
OUTSIDE every clause: a fronted or absolute adverbial/prepositional phrase \
("In the morning, ..."), an appositive ("my brother, a doctor, ..."), a \
vocative ("Marcus, ..."), or a whole verbless fragment ("A ring for five \
million dollars."). A noun phrase or prepositional phrase doing a job INSIDE \
a clause (subject, object, ordinary modifier) stays part of that clause and \
is NOT a separate unit. Only ever split out clauses, which have verbs; never \
split a verbless phrase out of its clause.

For each unit return: text, kind, structure, position, function.

kind: "Clause" or "Phrase".

structure for a Clause, exactly one of:
- "Independent": a main clause (the first/only main clause).
- "Coordinate": a main clause joined to a previous main clause by a \
coordinator (and, but, or, so, nor, yet) or a semicolon.
- "Subordinate": a finite adverbial clause (because, when, if, although, \
while, since, as, until ...).
- "Relative": a clause modifying a noun ("the dog that barked").
- "Content": a noun clause filling a clause slot ("she knew that she was \
late"). Set function to Projection ONLY when it is genuine reported or \
direct speech or thought introduced by a saying or thinking verb (say, ask, \
tell, shout, think, wonder, realise, and the like). A plain content clause \
that is not speech or thought is not Projection.
- "Non-finite": a to-infinitive, -ing, or -ed clause that is an ADJUNCT or \
carries its own subject ("To win the race, she trained", "Exhausted, he sat \
down"). A to-infinitive or -ing/-ed that is merely the object or complement \
of the main verb ("wanted to go", "liked living there") is NOT extracted; it \
stays in the main clause.
- "Verbless": a clause-like unit with no verb ("when in doubt", "his hat in \
his hand").

structure for a Phrase, exactly one of:
- "Prepositional": ("in the morning", "to the shops").
- "Adverbial": a fronted/absolute adverbial not headed by a preposition.
- "Noun": an appositive or a vocative noun phrase, or a verbless NP fragment.

position (of the unit within the whole sentence): "Initial", "Medial" \
(interrupting), or "Final".

function:
- main clauses (Independent, Coordinate): "(main)".
- everything else, the meaning relation, exactly one of: "Cause", \
"Condition", "Time", "Contrast", "Concession", "Addition", "Elaboration", \
"Projection", "Comparison", "Vocative". Use "Projection" for reported or \
direct speech/thought. Use "Vocative" for direct address. Appositives are \
"Elaboration".

Return ONLY a raw JSON array, no prose, no markdown fences. Your entire \
response must start with [ and end with ]. Example: \
[{"id": 0, "units": [{"text": "...", "kind": "Clause", "structure": "...", \
"position": "...", "function": "..."}]}]"""


def _rescue_partial_json(txt):
    txt = txt.strip()
    if not txt.startswith("["):
        return "[]"
    for end in (txt.rfind("}]"), txt.rfind("},")):
        if end > 0:
            candidate = txt[:end + (2 if txt[end:end + 2] == "}]" else 1)] + "]"
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass
    return "[]"


def build_clause_map(sent_df, model="claude-sonnet-4-6", api_key=None,
                     mock=False, chunk_size=8, max_retries=3):
    """Return a clause_map dataframe (one row per unit). sent_df is not changed.

    Mirrors engine_prototype.fill_sentence_engine: per-script grouping,
    chunk-local ids, focused retry for dropped ids, cost tracking.
    """
    import pandas as pd

    rows = []  # collected unit rows -> dataframe at the end

    def _emit(ident, ref, units):
        for idx, u in enumerate(units):
            if not isinstance(u, dict):
                continue
            rows.append({
                "Identifier": ident,
                "SentenceRef": ref,
                "UnitIndex": idx,
                "UnitText": str(u.get("text", "")).strip(),
                "UnitKind": u.get("kind", ""),
                "Structure": u.get("structure", ""),
                "Position": u.get("position", ""),
                "Function": u.get("function", ""),
            })

    cols = ["Identifier", "SentenceRef", "UnitIndex", "UnitText",
            "UnitKind", "Structure", "Position", "Function"]

    if mock:
        print("MOCK clause_map: would map sentences with a real Sentence "
              f"type in chunks of {chunk_size}.")
        print("system prompt head:\n", _CM_SYSTEM[:240], "...")
        return pd.DataFrame(columns=cols)

    import anthropic
    import time
    client = anthropic.Anthropic(api_key=api_key)

    def _call(items, script_text, mt):
        msg = client.messages.create(
            model=model, max_tokens=mt,
            system=[{"type": "text", "text": _CM_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": json.dumps(
                {"script": script_text, "sentences": items})}],
        )
        try:
            from cost_tracker import tracker
            tracker.record("clauses", msg.usage, model=model)
        except Exception:
            pass
        txt = next((b.text for b in msg.content if b.type == "text"), "[]")
        if msg.stop_reason == "max_tokens":
            txt = _rescue_partial_json(txt)
        txt = re.sub(r"```json|```", "", txt).strip()
        parsed = json.loads(txt)
        return {t["id"]: t for t in parsed
                if isinstance(t, dict) and "id" in t}

    df = sent_df

    def _is_mappable(r):
        st = str(r.get("Sentence type"))
        if st in _SKIP_TYPES:
            return False
        if str(r.get("TextualArtifact", "")) in (
                "TITLE", "ENDING", "CUTOFF", "TIMESKIP", "TIMESLIP"):
            return False
        return bool(str(r.get("CorrectedSentence", "")).strip())

    for ident, g in df.groupby("Identifier", sort=False):
        g_sorted = g.sort_values("SentenceRef")
        needs = [(str(r["CorrectedSentence"]), r["SentenceRef"])
                 for _, r in g_sorted.iterrows() if _is_mappable(r)]
        script_text = " ".join(
            str(r["CorrectedSentence"])
            for _, r in g_sorted.iterrows()
            if str(r.get("TextualArtifact", "")) == ""
            and str(r.get("CorrectedSentence", "")).strip())
        if not needs:
            continue

        for start in range(0, len(needs), chunk_size):
            chunk = needs[start:start + chunk_size]
            items = [{"id": i, "sentence": s} for i, (s, _) in enumerate(chunk)]
            refs = [ref for _, ref in chunk]
            mt = min(len(items) * 450 + 800, 8192)

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
                print(f"  [clause_map] {ident} chunk start={start} failed after "
                      f"{max_retries} attempts: {type(last_err).__name__}: {last_err}")
                continue

            missing = []
            for i, ref in enumerate(refs):
                t = tags.get(i)
                if t is None:
                    missing.append((i, ref))
                else:
                    _emit(ident, ref, t.get("units", []))

            if missing:
                miss_items = [{"id": i, "sentence": chunk[i][0]} for i, _ in missing]
                miss_mt = min(len(miss_items) * 550 + 800, 8192)
                try:
                    retry_tags = _call(miss_items, script_text, miss_mt)
                except Exception as e:
                    print(f"  [clause_map] {ident} retry of {len(missing)} "
                          f"missing failed: {type(e).__name__}: {e}")
                    retry_tags = {}
                for i, ref in missing:
                    t = retry_tags.get(i)
                    if t is not None:
                        _emit(ident, ref, t.get("units", []))

    return pd.DataFrame(rows, columns=cols)
