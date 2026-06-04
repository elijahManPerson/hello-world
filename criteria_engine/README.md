# Criteria Prediction Engine for AES Cross-Validation

A two-layer engine for NAPLAN-style narrative scoring that **preserves the
workbook's original criterion ranges and total scale exactly**, then adds a
profile *fit* check on top.

It is the implementation of the design discussion: predict each criterion
inside its own range, sum to a tentative total, and use "fit" as a consistency
check — *given the other evidence, does this criterion score look plausible,
high, low, or suspicious?* — rather than as a replacement score.

## The two layers

1. **Criteria predictor (the boss).** Each criterion is predicted inside its
   original range. The tentative total is always `sum(criteria)` — the engine
   never invents a new 0–100 scale.

   ```
   tentative_total = AU + TS + ID + CS/PD + Voc + Coh + Pa + SS + Pun + Spell
   ```

2. **Fit layer (sanity check).** The predicted/observed profile is compared
   against the typical profile for its total band, the hard "suspicious
   pattern" rules are applied, and a fit label + confidence + review flags are
   returned. The fit layer **never changes a score**.

## Score ranges (preserved exactly)

| Criterion | Range | | Criterion | Range |
|-----------|-------|-|-----------|-------|
| AU (Audience) | 0–6 | | Coh (Cohesion) | 0–4 |
| TS (Text Structure) | 0–4 | | Pa (Paragraphing) | 0–2 |
| ID (Ideas) | 0–5 | | SS (Sentence Structure) | 0–6 |
| CS/PD (Character/Setting) | 0–4 | | Pun (Punctuation) | 0–5 |
| Voc (Vocabulary) | 0–5 | | Spell (Spelling) | 0–6 |
| | | | **Total** | **0–47** |

Workbook invariant (held across all 9,695 scripts):
`Total == sum of the ten criteria`.

## Install / requirements

Pure standard library — no dependencies. Python 3.9+.

## Python API

```python
from criteria_engine import predict_criteria, assess_scores, cross_validate, audit_fit

# Predict criteria from raw text, then fit-check.
predict_criteria(text=my_script)

# Predict from length only (when you have WordCount but not the text).
predict_criteria(word_count=480)

# Fit-check existing human / LLM / model criterion scores. This is the core
# cross-validation use: it judges plausibility without re-predicting.
assess_scores({"AU": 4, "TS": 3, "ID": 3, "CS/PD": 3, "Voc": 3,
               "Coh": 2, "Pa": 1, "SS": 4, "Pun": 2, "Spell": 5})

# Have an LLM/NN produce the criterion scores; let the engine fit-check them.
predict_criteria(text=my_script,
                 criterion_scores=llm_scores,   # dict of the ten criteria
                 evidence=llm_evidence)          # optional per-criterion notes

# Cross-validate text-feature predictions against gold workbook rows.
cross_validate(rows)   # exact / within-1 / MAE per criterion; total MAE + corr

# Audit how often gold/model profiles trip the suspicious-pattern rules.
audit_fit(rows)
```

## Command line

```bash
python -m criteria_engine score   --text "Once upon a time..."
python -m criteria_engine score   --word-count 480
python -m criteria_engine assess  --scores AU=4,TS=3,ID=3,CS/PD=3,Voc=3,Coh=2,Pa=1,SS=4,Pun=2,Spell=5
python -m criteria_engine validate --csv workbook.csv
python -m criteria_engine audit    --csv workbook.csv
```

`validate`/`audit` accept the workbook schema directly: a `Raw text` or
`WordCount` column plus the criterion columns (`Pun, Spell, Voc, SS, AU, TS,
ID, CS/PD, Coh, Pa`; `Vocabulary`/`Punctuation`/… aliases also accepted).

## Fit labels (section 6)

| Label | Meaning |
|-------|---------|
| Strong fit | Profile matches the band anchor; no contradictions. |
| Mostly fits | One to three small (±1) residuals. |
| Uneven profile | A two-point residual, or four+ one-point residuals. |
| Criterion conflict | A three-point residual or a rare co-occurrence. |
| Review needed | A hard rule (0 cases in the workbook) is violated. |

## Hard "suspicious pattern" rules (section 7)

Co-occurrences with **0 cases** in the workbook → **Review needed**:

* `TS = 4` with `AU ≤ 3`
* `TS = 4` with `CS/PD ≤ 2`
* `Pa = 2` with `Coh ≤ 1`

Very rare co-occurrences → **lower confidence** (Criterion conflict):

* `Voc ≥ 4` with `SS ≤ 2` (~0.18%)
* `Pun ≥ 4` with `SS ≤ 2` (~0.10%)
* `Spell ≥ 5` with `Pun ≤ 1` (~0.29%)

## Output shape

`predict_criteria` returns `criteria_predictions` (score/range/confidence/
evidence per criterion), `tentative_total`, `total_band`, `expected_profile`,
`actual_profile`, `profile_fit`, `confidence`, `review_flags`,
`family_coherence`, and a `text_prior` block (the length-only sanity check on
the total). `assess_scores` returns the fit report alone.

## Where the LLM plugs in

The text-feature predictor is a **prior**, not a verdict. Pass real per-criterion
scores and evidence from an LLM rubric pass via `criterion_scores=` /
`evidence=`; the engine then sums them, preserves the ranges, and runs the same
fit layer. This is the recommended production path — *the criterion predictor is
the boss; the total predictor is only a sanity check.*

## Module map

| Module | Responsibility |
|--------|----------------|
| `ranges.py` | Criterion ranges, names, families, clamping. |
| `features.py` | Surface text-feature extraction. |
| `priors.py` | Word-count prior table; total-band expected profiles. |
| `fit.py` | Residual classification, suspicious rules, fit labels. |
| `engine.py` | `predict_criteria`, `assess_scores`. |
| `crossval.py` | `cross_validate`, `audit_fit`, CSV loading. |
| `cli.py` | `python -m criteria_engine` entry point. |
| `keras_backend.py` | Optional: score with the trained `model_*.keras` models (needs a saved TF-IDF vectorizer — see file header). |
| `llm_backend.py` | Claude scorer (Mode 3, **Sonnet** default) — produces the mark; see `LLM_INTEGRATION.md`. |
| `calibration.py` | `GoldCalibrator` — trains a per-criterion correction on your gold marks. |

## Backends & validation

- **Validation on real data:** `examples/validate_drive_csv.py <csv>` runs
  `cross_validate` + `audit_fit` against a gold-scored CSV. Results on 220 real
  scripts are in `VALIDATION.md` (text prior: total r ≈ 0.91, within-1 ≥91% on
  every criterion; fit layer: 0 hard-rule fires, 1 soft fire at its predicted rarity).
- **Keras backend:** `KerasCriterionScorer` reconstructs the notebook's exact
  feature pipeline and feeds the fit layer. Requires a saved `tfidf_vectorizer.pkl`
  (the original run saved only the scaler) — the notebook now has a cell to save it.
- **Claude backend:** `SonnetScorer` (default `claude-sonnet-4-6`) / `HaikuScorer`
  score a script and return the mark via the fit layer — schema-constrained to
  in-range scores, prompt-cached rubric+exemplars, adaptive thinking where
  supported. Needs `pip install anthropic`. See `LLM_INTEGRATION.md`.
- **Training on your data:** `GoldCalibrator` learns a per-criterion correction
  from your gold marks (you can't fine-tune Claude itself). Fold-validated on the
  365 gold scripts it cut total-mark MAE 4.40 → 3.96 and Spelling exact-match
  31.5% → 52.1%. Train via `examples/train_calibrator.py`; wrap any scorer with
  `calibrator=`.

## Tests

```bash
python criteria_engine/tests/test_engine.py      # or: python -m pytest criteria_engine/tests
```
