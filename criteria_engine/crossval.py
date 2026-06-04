"""Cross-validation harness for the criteria prediction engine.

Two complementary jobs:

1. ``cross_validate`` - predict criteria from text features and compare
   against gold workbook scores. Reports the same agreement metrics the
   prototype reported: exact agreement, within-1, within-2, MAE, and the
   correlation of the summed total with the gold total.

2. ``audit_fit`` - run the fit layer over a set of *gold* (or model) scores
   and report how often each suspicious pattern fires. On clean workbook data
   the hard rules should fire ~0% of the time; a non-trivial rate on model
   output is a signal the model is producing impossible profiles.

Input records are plain dicts so the harness has no hard pandas dependency.
A record needs the gold criterion scores plus either ``text`` or
``WordCount``/``word_count``. Column aliases match the workbook schema.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict

from .ranges import CRITERIA, total as profile_total
from .engine import predict_criteria, assess_scores
from .fit import SUSPICIOUS_RULES

# Workbook column aliases -> canonical criterion key.
_ALIASES: dict[str, str] = {
    "AU": "AU", "TS": "TS", "ID": "ID", "CS/PD": "CS/PD", "CS_PD": "CS/PD",
    "Voc": "Voc", "Vocabulary": "Voc", "Coh": "Coh", "Cohesion": "Coh",
    "Pa": "Pa", "Paragraphing": "Pa",
    "SS": "SS", "SentenceStructure": "SS",
    "Pun": "Pun", "Punctuation": "Pun",
    "Spell": "Spell", "Spelling": "Spell",
}


def _gold_profile(record: dict) -> dict[str, int] | None:
    """Extract a complete gold criterion profile from a record, or None."""
    out: dict[str, int] = {}
    for key, val in record.items():
        canon = _ALIASES.get(str(key).strip())
        if canon and canon not in out:
            try:
                out[canon] = int(round(float(val)))
            except (TypeError, ValueError):
                return None
    return out if all(c in out for c in CRITERIA) else None


def _record_text_and_wc(record: dict) -> tuple[str | None, int | None]:
    text = record.get("text") or record.get("Raw text") or record.get("raw_text")
    wc = record.get("word_count") or record.get("WordCount")
    try:
        wc = int(round(float(wc))) if wc is not None else None
    except (TypeError, ValueError):
        wc = None
    return text, wc


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


def cross_validate(records: list[dict]) -> dict:
    """Predict criteria from text features and score against gold labels."""
    per_crit_exact = defaultdict(int)
    per_crit_within1 = defaultdict(int)
    per_crit_abserr = defaultdict(float)
    n = 0

    pred_totals: list[float] = []
    gold_totals: list[float] = []
    total_abserr = 0.0
    total_within2 = 0
    total_within4 = 0

    skipped = 0
    for rec in records:
        gold = _gold_profile(rec)
        if gold is None:
            skipped += 1
            continue
        text, wc = _record_text_and_wc(rec)
        if text is None and wc is None:
            skipped += 1
            continue

        result = predict_criteria(text=text, word_count=wc)
        pred = result["actual_profile"]

        n += 1
        for c in CRITERIA:
            d = abs(pred[c] - gold[c])
            per_crit_abserr[c] += d
            if d == 0:
                per_crit_exact[c] += 1
            if d <= 1:
                per_crit_within1[c] += 1

        pt, gt = profile_total(pred), profile_total(gold)
        pred_totals.append(pt)
        gold_totals.append(gt)
        td = abs(pt - gt)
        total_abserr += td
        total_within2 += td <= 2
        total_within4 += td <= 4

    if n == 0:
        return {"n": 0, "skipped": skipped, "error": "no scorable records"}

    per_criterion = {
        c: {
            "exact_pct": round(100 * per_crit_exact[c] / n, 1),
            "within1_pct": round(100 * per_crit_within1[c] / n, 1),
            "mae": round(per_crit_abserr[c] / n, 3),
        }
        for c in CRITERIA
    }

    return {
        "n": n,
        "skipped": skipped,
        "per_criterion": per_criterion,
        "total": {
            "mae": round(total_abserr / n, 3),
            "within2_pct": round(100 * total_within2 / n, 1),
            "within4_pct": round(100 * total_within4 / n, 1),
            "correlation": round(_pearson(pred_totals, gold_totals), 3),
        },
    }


def audit_fit(records: list[dict]) -> dict:
    """Run the fit layer over gold/model profiles; report flag rates."""
    rule_hits = defaultdict(int)
    fit_labels = defaultdict(int)
    n = 0
    skipped = 0
    for rec in records:
        gold = _gold_profile(rec)
        if gold is None:
            skipped += 1
            continue
        n += 1
        report = assess_scores(gold)
        fit_labels[report["fit"]] += 1
        for rule in SUSPICIOUS_RULES:
            if rule.fires(gold):
                rule_hits[rule.name] += 1

    if n == 0:
        return {"n": 0, "skipped": skipped, "error": "no scorable records"}

    return {
        "n": n,
        "skipped": skipped,
        "fit_label_distribution": {
            k: {"count": v, "pct": round(100 * v / n, 2)}
            for k, v in sorted(fit_labels.items())
        },
        "suspicious_rule_rate": {
            rule.name: {
                "severity": rule.severity,
                "count": rule_hits[rule.name],
                "pct": round(100 * rule_hits[rule.name] / n, 3),
            }
            for rule in SUSPICIOUS_RULES
        },
    }


def load_csv(path: str) -> list[dict]:
    """Load workbook-style rows from a CSV into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
