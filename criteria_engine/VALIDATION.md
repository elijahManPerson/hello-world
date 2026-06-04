# Validation on real data

Run against **21 real human-marked scripts** from
`1.Training Data/Data for Testing avg short.csv` (averaged marker scores,
rounded to integers for the engine). Reproduce with:

```bash
python examples/validate_drive_csv.py "Data for Testing avg short.csv"
```

## 1. `cross_validate` — text-feature prior vs gold

| Metric (total) | Value |
|----------------|-------|
| Correlation | **0.947** |
| MAE | 5.48 |
| Within 2 | 33.3% |
| Within 4 | 52.4% |

Per criterion:

| Criterion | Exact % | Within-1 % | MAE |
|-----------|--------:|-----------:|----:|
| AU | 61.9 | 95.2 | 0.43 |
| TS | 71.4 | 100.0 | 0.29 |
| ID | 52.4 | 95.2 | 0.52 |
| CS/PD | 66.7 | 100.0 | 0.33 |
| Voc | **9.5** | **61.9** | **1.33** |
| Coh | 66.7 | 100.0 | 0.33 |
| Pa | 66.7 | 100.0 | 0.33 |
| SS | 42.9 | 100.0 | 0.57 |
| Pun | **9.5** | **57.1** | **1.33** |
| Spell | 38.1 | 95.2 | 0.67 |

**Reading it:** the word-count prior ranks scripts well (r ≈ 0.95) but is not
calibrated on absolute total (within-2 only 33%) — exactly what a length prior
should do. It is strong on most criteria (within-1 ≥ 95% for 8 of 10) but
**cannot read Vocabulary or Punctuation** from surface features. Those two need
the real model / LLM layer.

## 2. `audit_fit` — fit layer over the gold profiles

| | |
|---|---|
| Hard-rule fires (impossible patterns) | **0** |
| Fit labels | 3 Strong · 12 Mostly fits · 5 Uneven · 1 Criterion conflict |

**Reading it:** the suspicious-pattern rules do **not** false-alarm on genuine
human marks (0 hard fires), and ~71% of real scripts sit cleanly in their band
profile, with a sensible minority flagged uneven — real writing is uneven.

> Caveat: 21 records is a small set. Point the script at the larger Drive test
> sets (`Data for Testing avg.csv`, ~200 records) for a fuller picture.
