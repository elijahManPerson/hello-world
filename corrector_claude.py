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

Prompt caching is applied to the system prompt so repeated calls across a batch
share the same cached prefix (~90% cheaper after the first call).
"""

import anthropic
import re
import time
import logging

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert proofreader for primary and secondary school student writing.

Your task is to lightly correct the student's raw text. Apply ONLY these corrections:
- Fix spelling errors (e.g. "recieve" -> "receive", "finaly" -> "finally")
- Fix punctuation errors (missing full stops, commas, apostrophes)
- Fix capitalisation (sentence starts, proper nouns like "I")
- Fix basic subject-verb agreement errors
- Fix run-on sentences by adding the minimum punctuation needed

DO NOT:
- Change the student's vocabulary or word choices
- Rewrite, restructure or paraphrase sentences
- Add new content or remove existing content
- "Improve" the writing style
- Alter dialogue or direct speech beyond punctuation

Return ONLY the corrected text with no explanation, preamble or commentary.
If the input is empty or contains no recognisable prose (e.g. only symbols or
digits), return it unchanged."""


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
