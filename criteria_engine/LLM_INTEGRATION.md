# Claude Haiku scoring backend (Mode 3)

The production path: **Claude Haiku reads a script and returns per-criterion
scores + evidence; the engine sums them to the tentative total (the mark) and
runs the fit layer.** The criterion predictor stays the boss; the total
predictor is only a sanity check.

## "A Haiku trained model" = in-context calibration

There is no customer fine-tuning endpoint for Haiku, so "trained on the data" is
implemented as **in-context calibration**: a spread of the human-marked gold
scripts (`gold_clean_365_with_year_proxy.csv`, one or more per total band) is
embedded in a cached system prompt as exemplars. Haiku marks new scripts to
match those exemplars. This needs no training run and updates instantly when the
gold set changes.

## How it's wired (per the Claude API guidance)

`criteria_engine/llm_backend.py`:

- **Model:** `claude-haiku-4-5` (`DEFAULT_MODEL`). `HaikuScorer` is an alias of
  `ClaudeScorer`.
- **Structured outputs:** the response is constrained by a `json_schema` where
  each criterion is an integer `enum` over its exact range (`AU` 0–6, `Pa` 0–2,
  …). Haiku *cannot* return an out-of-range or non-integer score.
- **Prompt caching:** the rubric + exemplars are a stable prefix in `system`
  with a `cache_control` breakpoint; the per-script text is the volatile suffix.
  Each repeat scoring then reads the cached prefix (~0.1× input cost).
- **Thinking:** left off — Haiku 4.5 does not take the adaptive-thinking /
  `effort` parameters (those are Opus/Sonnet 4.6+).
- **SDK:** the official `anthropic` SDK, imported lazily so the core engine stays
  dependency-free. `pip install anthropic` to use this backend.

The JSONL cache key matches the handover's `aes_precision_cache.jsonl` shape
(`version|model|sha1(text)`), and evidence units keep the
`{label, start, end, text, confidence}` schema.

## Usage

```python
from criteria_engine.llm_backend import ClaudeScorer, JsonlCache, load_gold_exemplars

# Calibrate on the gold data (more per band = stronger calibration + real caching)
exemplars = load_gold_exemplars("gold_clean_365_with_year_proxy.csv", per_band=3)

scorer = ClaudeScorer(
    exemplars=exemplars,
    cache=JsonlCache("aes_haiku_cache.jsonl"),   # reuse across runs
)

result = scorer.predict(my_script, year_level=5)
mark = result["tentative_total"]          # 0–47, the criterion sum
print(mark, result["profile_fit"], result["review_flags"])
```

`scorer.score(text)` returns just the `CriterionPrediction` (scores / evidence /
units) without the fit layer.

## Notes

- **Caching threshold:** Haiku's minimum cacheable prefix is ~4096 tokens. One
  short exemplar per band (`per_band=1`) is ~1.5K tokens and won't cache; use
  `per_band=3`+ (or longer exemplars) so the prefix crosses the threshold and the
  cache actually activates. Verify with `usage.cache_read_input_tokens`.
- **Gold data is not committed.** `load_gold_exemplars` reads a path you provide;
  student scripts stay out of the repo.
- **Validate the mark.** Run the LLM scores through `cross_validate` against a
  held-out gold fold (the handover ships `gold_365_fold_assignment.csv`) — this is
  where Haiku should beat the text-feature prior on the criteria it can't read
  (Spell/Pun/SS exact match; see `VALIDATION.md`).
- **Year level** is provisional in the handover (`YearProxy_*`) and passed only
  as context to the user message; it is not used for calibration.
