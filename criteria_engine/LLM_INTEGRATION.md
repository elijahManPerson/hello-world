# LLM integration (Mode 3) — prep notes

The production path: an LLM reads the script and returns **per-criterion scores
+ evidence**; the engine preserves the ranges, sums to the tentative total, and
runs the fit layer. The criterion predictor stays the boss; the total predictor
is only a sanity check.

## What already exists in Drive

`aes_precision_cache.jsonl` (`6.Results`, updated **Feb 2026**) shows the
direction is already underway:

- Model: **gpt-4o**.
- 336 cache entries (46 populated), keyed `precision_units_v1|gpt-4o|<sha1>`.
- Each entry holds **evidence units**: `{label, start, end, text, confidence}`,
  e.g. `{"label": "precise_lexis", "text": "Floating chairs", "confidence": 0.9}`.

So the existing work extracts *evidence spans* (currently lexical-precision
units). That is exactly the engine's `evidence=` concept, one level finer.

## What the scaffold provides (`llm_backend.py`)

- `JsonlCache` — same key shape (`version|model|sha1(text)`) as the precision
  cache, so this can sit alongside / extend it.
- `EvidenceUnit` — same `{label, start, end, text, confidence}` schema.
- `build_rubric_prompt()` — a starter prompt that asks for each criterion **in
  its range**, plus a rationale per criterion, plus evidence units. It includes
  the engine's `BAND_PROFILE` anchors to nudge the model toward internally
  consistent profiles.
- `LLMScorer` — abstract; implement one method, `_call_model(prompt) -> dict`.
- `.predict()` — scores with the LLM then calls `predict_criteria(...)`, so the
  fit layer flags any profile the model produces that violates the workbook
  patterns (e.g. `TS=4` with `AU≤3`).

## To finish wiring (the remaining ~1 step)

1. Subclass `LLMScorer` and implement `_call_model` against your client
   (gpt-4o, to match the existing pipeline). Parse the JSON response into
   `{"scores": {...}, "evidence": {...}, "units": [...]}`.
2. Point `JsonlCache` at a cache file (reuse the precision-cache folder).
3. Validate: run `cross_validate` with these LLM scores supplied as
   `criterion_scores` against a gold Drive set — this is where we expect the big
   lift over the text-feature prior on **Voc** and **Pun** (the two criteria the
   prior cannot read; see `VALIDATION.md`).

## Why this is the right target

The trained `.keras` models top out around r ≈ 0.88 on total and the BERT/
embedding variants regressed (see PR description). The LLM path is the only one
that reads *meaning*, and it natively produces the evidence the fit layer wants.
The engine's job is unchanged: preserve the rubric and catch implausible
profiles.

> Deliberately not included: a hard dependency on any API SDK, or live calls.
> The scaffold is import-safe and offline; only `_call_model` reaches the network.
