"""
corrector_claude.py
===================
Drop-in corrector for aes_canonical.run_pipeline() that calls the Claude API.

Usage:
    import os
    from corrector_claude import make_claude_corrector
    from aes_canonical import run_pipeline
    import pandas as pd

    corrector = make_claude_corrector()          # reads ANTHROPIC_API_KEY from env
    df = pd.read_csv("your_data.csv")
    result = run_pipeline(df, corrector=corrector, id_col="Research ID")
    result["sent_df"].to_csv("output_sent_df.csv", index=False)

The corrector is light-touch: it fixes spelling, punctuation, capitalisation and
basic grammar while preserving every student's vocabulary and voice.  It does NOT
rewrite sentences, change word choice, or add/remove content.  This keeps the
token-level diff (built_word_map) meaningful.

It ALWAYS attempts a correction on every script, including off-topic,
copied-prompt and very rough Year 3 writing. The only input returned
unchanged is one that is literally empty or has no letters at all. Being
light-touch is what keeps "more aggressive coverage" safe: it reaches every
script without rewriting any child's voice.

Prompt caching is applied to the system prompt so repeated calls across a batch
share the same cached prefix (~90% cheaper after the first call).
"""

import anthropic
import re
import time
import logging

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert proofreader for primary and secondary school student writing, including very young writers (Year 3) whose work is phonetic, rough and full of errors.

Your task is to lightly correct the student's raw text. Apply ONLY these corrections:
- Fix spelling errors (e.g. "recieve" -> "receive", "finaly" -> "finally", and heavy phonetic spelling such as "wrire" -> "write", "plaace" -> "place", "adout" -> "about")
- Fix capitalisation (sentence starts, proper nouns like "I")
- Fix basic subject-verb agreement errors
- Fix punctuation, INCLUDING ensuring every sentence and fragment ends with a
  terminal mark (. ? or !)

SENTENCE BOUNDARIES -- IMPORTANT:
- Every complete sentence and every fragment must end with a terminal mark.
- If the student has run two or more sentences together with no punctuation
  between them (a fused run-on), insert full stops to separate them into
  distinct sentences. Do this for EVERY boundary, not just the first.
- If the student joined two independent clauses with only a comma (a comma
  splice), separate them. When the two clauses are CLOSELY RELATED (one
  continues, elaborates, contrasts with, or is a consequence of the other),
  PREFER a semicolon, or a colon or dash where it suits the relationship,
  rather than a full stop. Use a full stop only when the two clauses are
  distinct, unrelated statements. A genuine fused run-on with no mark at all
  still takes full stops, as above.
- Splitting or rejoining run-ons and splices in this way is REQUIRED and is
  NOT considered "restructuring". You are adding or changing punctuation, not
  changing words.

DIALOGUE ATTRIBUTION MUST STAY ATTACHED. When dialogue ends inside a
quote ("...!" she said. or "...!" laughed Joy.), the reporting clause
must remain attached to the dialogue as one sentence. Do not insert a
sentence boundary between the closing quote and the attribution. Lowercase
reporting verbs (said, whispered, laughed, responded) are the signal that
an attribution follows.

DO NOT:
- Change the student's vocabulary or word choices
- Reorder words or rephrase sentences (this is what "do not restructure" means)
- Add new content or remove existing content
- "Improve" the writing style
- Change a grammatically correct terminal mark for emphasis (e.g. do not change
  a correct full stop into an exclamation mark)
- Alter dialogue or direct speech beyond punctuation
- Add a closing quotation mark to dialogue that has no closing quote in the raw text

ALWAYS attempt a correction on EVERY input. This includes writing that is
off topic, writing that looks like a copied or garbled task prompt, and
very rough young-student writing. Even when the text is messy or not a
story, still fix the spelling, punctuation and capitalisation you can,
while keeping the student's own words and meaning. Do not return the text
unchanged just because it is messy, off topic or prompt-like.

Return ONLY the corrected text with no explanation, preamble or commentary.
The ONLY case where you return the input unchanged is when it is literally
empty or has no alphabetic characters at all."""


def make_claude_corrector(
    model: str = "claude-opus-4-7",
    api_key: str | None = None,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
) -> callable:
    """
    Return a corrector function  str -> str  suitable for aes_canonical.run_pipeline().

    Parameters
    ----------
    model       : Claude model ID.  Defaults to claude-opus-4-7.
    api_key     : Anthropic API key.  If None, read from ANTHROPIC_API_KEY env var.
    max_retries : Number of retries on rate-limit or server errors.
    retry_base_delay : Initial backoff in seconds (doubled each retry).
    """
    client = anthropic.Anthropic(api_key=api_key)

    def _corrector(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        if not text:
            return ""

        attempt = 0
        delay = retry_base_delay
        while True:
            try:
                with client.messages.stream(
                    model=model,
                    max_tokens=4096,
                    thinking={"type": "adaptive"},
                    system=[
                        {
                            "type": "text",
                            "text": _SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": text,
                        }
                    ],
                ) as stream:
                    final = stream.get_final_message()

                try:
                    from cost_tracker import tracker
                    tracker.record("correct", final.usage, model=model)
                except Exception:
                    pass

                corrected = next(
                    (b.text for b in final.content if b.type == "text"), text
                )
                return corrected.strip()

            except anthropic.RateLimitError as e:
                attempt += 1
                if attempt > max_retries:
                    logger.error("Rate limit exceeded after %d retries", max_retries)
                    raise
                retry_after = int(
                    getattr(e.response, "headers", {}).get("retry-after", delay)
                )
                logger.warning(
                    "Rate limited. Waiting %ss (attempt %d/%d)",
                    retry_after,
                    attempt,
                    max_retries,
                )
                time.sleep(retry_after)
                delay *= 2

            except anthropic.APIStatusError as e:
                attempt += 1
                if e.status_code < 500 or attempt > max_retries:
                    raise
                logger.warning(
                    "Server error %d. Retrying in %ss (attempt %d/%d)",
                    e.status_code,
                    delay,
                    attempt,
                    max_retries,
                )
                time.sleep(delay)
                delay *= 2

    _corrector.__name__ = f"claude_corrector_{model.replace('-', '_')}"
    return _corrector
