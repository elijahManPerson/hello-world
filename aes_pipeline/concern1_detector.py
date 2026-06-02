"""
concern1_detector.py
=====================
Welfare triage for student scripts (the "Script of Concern 1" flag).

WHAT THIS IS
  An advisory signal that puts a script in front of a human, with a reason,
  because something in it may reflect a child dealing with something real:
  bullying, depression, mental-health difficulty, abuse, neglect, violence,
  suicide or self-harm, or illegal activity.

WHAT THIS IS NOT
  - Not a determination about a child.
  - Not a mandated report and not a substitute for your safeguarding policy
    or normal pastoral care.
  - Not a guarantee. False negatives are possible. "No concern" means the
    model saw nothing, NOT that the child is safe. Normal duty of care still
    stands on every script, flagged or not.

DESIGN RULES (do not weaken these without a deliberate decision)
  - Advisory only. The output never alters any writing score. It lives in
    its own column, read by humans, not by the scoring layer.
  - Sensitivity-biased. When genuinely unsure, it leans to "Possible", not
    "No concern". A human glance is cheap; a miss is not.
  - Runs on the RAW student text, never the corrected text, because
    correction can sanitise or mask a disclosure.
  - Reason is a short paraphrase plus a location pointer ONLY. The model is
    instructed never to return the child's words verbatim, so a possible
    disclosure does not travel verbatim everywhere the dataset goes.
  - Strongest model by default. At this volume cost is negligible and this
    is the judgement where quality matters most.
  - The cheap pass never sets this. Only this detector does, and even its
    "No concern" does not clear the human's responsibility.

VALIDATION
  This column cannot be proven on held-out data the way punctuation can:
  there is no ground truth for welfare concern and it would be wrong to
  manufacture some. Validate it by reading real flagged outputs with your
  safeguarding lead, in your own environment, with a human always deciding.

  Use mock=True to inspect the exact prompt and output shape at zero cost.
"""

import json
import re

ENGINE = "TBD(engine)"

LEVELS = ("No concern", "Possible", "Salient")

CATEGORIES = (
    "bullying", "depression", "mental health", "abuse", "neglect",
    "violence", "suicide or self-harm", "illegal activity",
)

# The heart of the detector. Review this wording carefully before any run;
# it encodes the triage framing, the sensitivity bias and the
# fiction-versus-lived nuance, and it forbids verbatim child text.
_SYSTEM = """You support a child-welfare triage process for school writing. \
A human reviews everything you flag; you do not decide anything and you do \
not take any action. Your only job is to judge whether a piece of student \
writing should be put in front of an adult for a possible welfare concern, \
and to describe why in a way that protects the child's privacy.

Concern types: bullying, depression, mental health, abuse, neglect, \
violence, suicide or self-harm, illegal activity.

These are narrative tasks, so dark, sad or violent content is often just \
imaginative writing to a prompt and is NOT in itself a concern. Weigh \
whether the writing reads as prompt-driven fiction (a story with invented \
characters and events, consistent with a creative task) or as something \
lived (first-person, specific, persistent, addressed as the writer's own \
experience, or oddly out of step with a story task). Treat this as a \
signal, not a verdict.

Bias toward sensitivity. If you are genuinely unsure, choose "Possible" \
rather than "No concern". A human glancing at a false alarm costs seconds; \
a missed disclosure does not.

Levels:
- "No concern": nothing, or only clearly prompt-driven fiction with no \
lived signals.
- "Possible": ambiguous, or mild but real-sounding allusion, or fiction \
that carries some first-person or distress signals.
- "Salient": specific and serious, reads as lived rather than invented, \
for example explicit suicidal or self-harm content or an account of abuse \
or neglect that does not read as a story.

PRIVACY RULE, MANDATORY: never quote the student's words. Give a short \
neutral paraphrase and a location pointer only (for example "self-harm \
allusion, final paragraph"). Do not reproduce phrases from the text.

Return STRICT JSON only, no prose:
{"level": "...", "categories": ["..."], "paraphrase": "...", \
"location": "...", "reads_as": "prompt-driven fiction | possibly lived | \
unclear", "rationale": "one neutral sentence"}

If nothing is present: level "No concern", categories [], paraphrase "", \
location "", reads_as "prompt-driven fiction", rationale "".
"""


def assess_concern1(raw_text, model="claude-opus-4-7",
                    api_key=None, mock=False):
    """Assess one raw script. Returns a dict with level, categories,
    paraphrase, location, reads_as, rationale. mock=True returns the
    prompt and a structural example with no API call."""
    text = str(raw_text or "").strip()

    if mock:
        return {
            "_mock": True,
            "system_prompt": _SYSTEM,
            "user_payload": text[:300] + (" ..." if len(text) > 300 else ""),
            "example_output_shape": {
                "level": "Possible",
                "categories": ["depression"],
                "paraphrase": "[neutral paraphrase of the allusion]",
                "location": "second paragraph",
                "reads_as": "unclear",
                "rationale": "[one neutral sentence]",
            },
        }

    if not text:
        return {"level": "No concern", "categories": [], "paraphrase": "",
                "location": "", "reads_as": "prompt-driven fiction",
                "rationale": ""}

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model, max_tokens=600,
        system=[{"type": "text", "text": _SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": text}],
    )
    try:
        from cost_tracker import tracker
        tracker.record("welfare", msg.usage, model=model)
    except Exception:
        pass
    out = next((b.text for b in msg.content if b.type == "text"), "{}")
    out = re.sub(r"```json|```", "", out).strip()
    try:
        r = json.loads(out)
    except Exception:
        # On any parse failure, fail SAFE: surface for human review.
        return {"level": "Possible", "categories": [],
                "paraphrase": "model output unparseable, review manually",
                "location": "", "reads_as": "unclear",
                "rationale": "automatic fallback: could not parse model output"}
    if r.get("level") not in LEVELS:
        r["level"] = "Possible"          # fail safe, never silently No concern
    return r


def run_concern1(texts, id_col="Research ID", raw_col="Raw text",
                 model="claude-opus-4-7", api_key=None, mock=False):
    """Fill Concern1 / Concern1Reason on a texts frame. Concern1 holds the
    level; Concern1Reason holds the paraphrase, pointer, categories and the
    fiction read as a compact JSON string. No verbatim child text is stored.
    The scoring layer must never read these columns."""
    df = texts.copy()
    df["Identifier"] = df[id_col].astype(str)
    levels, reasons = [], []
    for raw in df[raw_col].astype(str):
        r = assess_concern1(raw, model=model, api_key=api_key, mock=mock)
        if r.get("_mock"):
            levels.append(ENGINE)
            reasons.append("MOCK: see assess_concern1(text, mock=True)")
            continue
        levels.append(r.get("level", "Possible"))
        reasons.append(json.dumps({
            "categories": r.get("categories", []),
            "paraphrase": r.get("paraphrase", ""),
            "location": r.get("location", ""),
            "reads_as": r.get("reads_as", "unclear"),
            "rationale": r.get("rationale", ""),
        }, ensure_ascii=False))
    df["Concern1"] = levels
    df["Concern1Reason"] = reasons
    return df


if __name__ == "__main__":
    print("Concern 1 welfare detector. Inspect the prompt with:")
    print("  from concern1_detector import assess_concern1")
    print("  print(assess_concern1('some text', mock=True)['system_prompt'])")
