# Validation on real data

Primary run: **220 real human-marked scripts** from
`1.Training Data/Data for Testing avg.csv` (averaged marker scores, rounded to
integers for the engine). Reproduce with:

```bash
python examples/validate_drive_csv.py "Data for Testing avg.csv"
```

## 1. `cross_validate` — text-feature prior vs gold (n = 220)

| Metric (total) | Value |
|----------------|-------|
| Correlation | **0.911** |
| MAE | 3.36 |
| Within 2 | 46.8% |
| Within 4 | 73.2% |

Per criterion:

| Criterion | Exact % | Within-1 % | MAE |
|-----------|--------:|-----------:|----:|
| AU | 57.7 | 94.1 | 0.48 |
| TS | 65.9 | 97.7 | 0.36 |
| ID | 60.0 | 97.3 | 0.43 |
| CS/PD | 60.9 | 97.7 | 0.41 |
| Voc | 55.0 | 94.5 | 0.51 |
| Coh | 66.8 | 99.5 | 0.34 |
| Pa | 70.5 | 100.0 | 0.30 |
| SS | 47.7 | 93.2 | 0.59 |
| Pun | 45.0 | 91.8 | 0.64 |
| Spell | 35.9 | 92.7 | 0.71 |

**Reading it:** the word-count prior tracks the total strongly (r ≈ 0.91) but is
only roughly calibrated (within-2 ≈ 47%) — as a length prior should be.
Within-1 is 91–100% for **every** criterion. The hardest to pin *exactly* are
the mechanical/surface ones — **Spell, Pun, SS** (lowest exact %, highest MAE) —
which makes sense: surface features don't capture spelling/punctuation accuracy.
These are the criteria the LLM layer should lift most.

> Note: an earlier 21-record sample suggested Voc/Pun were badly weak (within-1
> ~60%). That was small-sample noise — on the full 220-record set Voc within-1 is
> 94.5%. The robust weak spots are Spell/Pun/SS on *exact* match.

## 2. `audit_fit` — fit layer over the gold profiles (n = 220)

| | |
|---|---|
| **Hard-rule fires** (impossible patterns) | **0 / 220** |
| Soft-rule fires | 1 (`Spell_high_Pun_low`, ≈0.45% — matches the ~0.29% rarity) |
| Fit labels | 11 Strong · 112 Mostly fits · 95 Uneven · 2 Criterion conflict |

**Reading it:** zero hard-rule fires across 220 independent real scripts
confirms the "0 cases in the workbook" patterns (`TS=4 & AU≤3`, etc.) really do
not occur in genuine marking — the hard rules won't false-alarm. The one soft
fire lands right at its predicted rarity. The actionable-flag rate (Criterion
conflict + hard) is ~1%; "Uneven" (≈43%, medium confidence) reflects that real
writing is genuinely uneven and is informational, not a review trigger.
