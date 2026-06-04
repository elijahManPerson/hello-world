# Claude scoring backend + training on the gold data (Mode 3)

The production path: **Claude reads a script and returns per-criterion scores +
evidence; the engine sums them to the tentative total (the mark) and runs the
fit layer.** The criterion predictor stays the boss; the total predictor is only
a sanity check.

Default model is **`claude-sonnet-4-6`** (`SonnetScorer`). `HaikuScorer` pins
Haiku 4.5; any model can be set via `model=`.

## Can you "train it on your data"?

**You cannot fine-tune Claude itself** — there's no fine-tuning endpoint for
Sonnet/Haiku. Two mechanisms give you genuine "trained on my data" behaviour
with a frozen API model, and they stack:

1. **In-context calibration** — `load_gold_exemplars()` embeds a per-band spread
   of your human-marked gold scripts in a cached system prompt, so Claude marks
   new scripts to match them.
2. **A trained calibration layer** (`calibration.GoldCalibrator`) — learns a
   per-criterion correction from *(model score → your human mark)* on a training
   fold and remaps the model's scores at inference.

### What the calibrator buys you (measured, held-out)

Trained on the 365 gold scripts with the handover's A/B fold split (train on one
fold, score the other — the calibrator never sees the test fold), wrapping the
**text-feature prior** as the base scorer:

| Total mark | before | after |
|---|---|---|
| MAE | 4.40 | **3.96** |
| within-4 | 60.8% | **66.6%** |
| correlation | 0.889 | 0.883 |

Biggest per-criterion exact-match gains: **Spelling 31.5% → 52.1%**, Vocabulary
40.5% → 50.1%, Character/Setting 46.3% → 57.0%, Punctuation 43.8% → 46.8%. The
same calibrator wraps the Sonnet scorer unchanged — it corrects *whatever* the
base scorer produces.

> The text-feature prior is just a convenient, API-key-free base to prove the
> calibration. The real lift comes from wrapping **Sonnet** (which reads meaning)
> with this calibrator.

## How it's wired (per the Claude API guidance)

`criteria_engine/llm_backend.py`:

- **Structured outputs:** a `json_schema` where each criterion is an integer
  `enum` over its exact range (AU 0–6, Pa 0–2, …) — the model can't return an
  out-of-range or non-integer score.
- **Prompt caching:** rubric + exemplars are a stable prefix in `system` with a
  `cache_control` breakpoint; the per-script text is the volatile suffix.
- **Adaptive thinking:** on for Sonnet 4.6 / Opus 4.x, off for Haiku 4.5 (which
  doesn't take the parameter).
- **SDK:** the official `anthropic` SDK, imported lazily. `pip install anthropic`.

## Usage

```python
from criteria_engine.llm_backend import SonnetScorer, JsonlCache, load_gold_exemplars
from criteria_engine.calibration import GoldCalibrator

exemplars  = load_gold_exemplars("gold_clean_365_with_year_proxy.csv", per_band=3)
calibrator = GoldCalibrator.load("calibrator.json")   # trained on your gold

scorer = SonnetScorer(
    exemplars=exemplars,
    calibrator=calibrator,
    cache=JsonlCache("aes_sonnet_cache.jsonl"),
)
result = scorer.predict(my_script, year_level=5)
mark = result["tentative_total"]      # 0–47, calibrated to your data
```

Train and save the calibrator:

```bash
python examples/train_calibrator.py \
    --gold gold_clean_365_with_year_proxy.csv \
    --folds gold_365_fold_assignment.csv \
    --out calibrator.json \
    --predictor sonnet      # or "prior" (no API key) / "haiku"
```

## Notes

- **Caching threshold:** Haiku/Sonnet cache prefixes from ~1–4K tokens
  (model-dependent). Use `per_band=3`+ so the rubric+exemplars prefix is large
  enough to cache; verify with `usage.cache_read_input_tokens`.
- **Gold data is not committed.** Exemplars and the calibrator are loaded from
  paths you supply; student scripts stay out of the repo. The saved
  `calibrator.json` is aggregate lookup tables, not student text.
- **Validate end-to-end.** With an API key, run `--predictor sonnet` to get the
  honest before/after mark agreement for the real model.
