"""Train a calibration layer on the gold marks.

You can't fine-tune Claude itself, but you *can* learn a correction from your
365 human-marked scripts and apply it to whatever scorer you use (the Claude
backend or the text-feature prior). For each criterion, the calibrator learns,
from a training fold, the mean gold score among items that received a given
predicted score, then rounds and clamps it. At inference it remaps each raw
criterion score through that learned table.

This is genuinely "trained on your data" and works with a frozen API model:
fit on one fold, validate on the held-out fold (the handover ships
``gold_365_fold_assignment.csv``).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

from .ranges import CRITERIA, clamp, total as profile_total
from .priors import total_band


class GoldCalibrator:
    """Per-criterion score-to-mark correction learned from gold data."""

    def __init__(self, maps: dict[str, dict[int, int]] | None = None):
        # maps: criterion -> {raw_score: corrected_score}
        self.maps = maps or {}

    def fit(self, pairs: list[tuple[dict, dict]]) -> "GoldCalibrator":
        """Learn from (predicted_profile, gold_profile) pairs."""
        acc: dict[str, dict[int, list[int]]] = {c: defaultdict(list) for c in CRITERIA}
        for pred, gold in pairs:
            for c in CRITERIA:
                acc[c][clamp(c, pred[c])].append(int(gold[c]))
        self.maps = {
            c: {raw: clamp(c, round(sum(golds) / len(golds)))
                for raw, golds in table.items()}
            for c, table in acc.items()
        }
        return self

    def apply(self, scores: dict[str, float]) -> dict[str, int]:
        """Remap a raw criterion profile through the learned correction."""
        out = {}
        for c in CRITERIA:
            raw = clamp(c, scores[c])
            out[c] = self.maps.get(c, {}).get(raw, raw)
        return out

    def to_dict(self) -> dict:
        # JSON keys must be strings; convert back on load.
        return {c: {str(k): v for k, v in table.items()} for c, table in self.maps.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "GoldCalibrator":
        return cls({c: {int(k): int(v) for k, v in table.items()} for c, table in d.items()})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "GoldCalibrator":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# --- Training / evaluation helpers ---------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def build_pairs(records, predict_fn, gold_fn) -> list[tuple[dict, dict]]:
    """Run ``predict_fn`` over records, pair predictions with gold profiles."""
    pairs = []
    for rec in records:
        gold = gold_fn(rec)
        if gold is None:
            continue
        pairs.append((predict_fn(rec), gold))
    return pairs


def evaluate_marks(pairs: list[tuple[dict, dict]], calibrator: GoldCalibrator | None = None) -> dict:
    """Agreement of (optionally calibrated) predictions vs gold marks."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    per = {c: {"exact": 0, "within1": 0, "abserr": 0.0} for c in CRITERIA}
    pred_totals, gold_totals = [], []
    t_abserr = t_within2 = t_within4 = 0
    for pred, gold in pairs:
        prof = calibrator.apply(pred) if calibrator else {c: clamp(c, pred[c]) for c in CRITERIA}
        for c in CRITERIA:
            d = abs(prof[c] - gold[c])
            per[c]["abserr"] += d
            per[c]["exact"] += d == 0
            per[c]["within1"] += d <= 1
        pt, gt = profile_total(prof), profile_total(gold)
        pred_totals.append(pt)
        gold_totals.append(gt)
        td = abs(pt - gt)
        t_abserr += td
        t_within2 += td <= 2
        t_within4 += td <= 4
    return {
        "n": n,
        "per_criterion": {
            c: {
                "exact_pct": round(100 * per[c]["exact"] / n, 1),
                "within1_pct": round(100 * per[c]["within1"] / n, 1),
                "mae": round(per[c]["abserr"] / n, 3),
            }
            for c in CRITERIA
        },
        "total": {
            "mae": round(t_abserr / n, 3),
            "within2_pct": round(100 * t_within2 / n, 1),
            "within4_pct": round(100 * t_within4 / n, 1),
            "correlation": round(_pearson(pred_totals, gold_totals), 3),
        },
    }


def foldwise_calibrate(records, predict_fn, gold_fn, fold_fn) -> dict:
    """Two-fold honest evaluation: train on one fold, score the other.

    Returns pooled before/after metrics plus a calibrator fit on ALL records
    (for production use).
    """
    by_fold: dict[str, list] = defaultdict(list)
    for rec in records:
        by_fold[fold_fn(rec)].append(rec)

    pooled_uncal: list[tuple[dict, dict]] = []
    pooled_cal: list[tuple[dict, dict]] = []
    for held_out, items in by_fold.items():
        train = [r for f, rs in by_fold.items() if f != held_out for r in rs]
        cal = GoldCalibrator().fit(build_pairs(train, predict_fn, gold_fn))
        test_pairs = build_pairs(items, predict_fn, gold_fn)
        pooled_uncal.extend(test_pairs)
        pooled_cal.extend((cal.apply(p), g) for p, g in test_pairs)

    full = GoldCalibrator().fit(build_pairs(records, predict_fn, gold_fn))
    return {
        "before": evaluate_marks(pooled_uncal),
        "after": evaluate_marks([(p, g) for p, g in pooled_cal]),
        "calibrator": full,
    }
