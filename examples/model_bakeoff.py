"""Opus vs Sonnet (vs Haiku) bake-off on a gold sample.

For each model: score a stratified sample of gold scripts once, fit a per-model
GoldCalibrator on the fold split, and report held-out mark agreement before and
after calibration. This is the principled way to decide whether Opus's raw-
quality edge survives calibration and justifies its cost.

Requires ANTHROPIC_API_KEY in the environment (the SDK reads it) and
`pip install anthropic`. Costs real API calls: ~(sample x models) requests.

    python examples/model_bakeoff.py \
        --gold gold_clean_365_with_year_proxy.csv \
        --folds gold_365_fold_assignment.csv \
        --sample 60 --models sonnet,opus
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from criteria_engine import CRITERIA, load_gold_exemplars
from criteria_engine.priors import total_band
from criteria_engine.calibration import foldwise_calibrate
from criteria_engine.llm_backend import SonnetScorer, HaikuScorer, OpusScorer, JsonlCache

SCORERS = {"sonnet": SonnetScorer, "haiku": HaikuScorer, "opus": OpusScorer}


def load_records(gold_path, folds_path):
    folds = {r["Research ID"]: r.get("Fold", "A")
             for r in csv.DictReader(open(folds_path, encoding="utf-8-sig"))} if folds_path else {}
    out = []
    for row in csv.DictReader(open(gold_path, encoding="utf-8-sig")):
        rid = row.get("Research ID", "")
        try:
            gold = {c: int(float(row[c])) for c in CRITERIA}
            wc = int(float(row["WordCount"]))
        except (ValueError, KeyError):
            continue
        text = (row.get("Raw text") or "").strip()
        if wc < 20 or not text:
            continue
        out.append({"id": rid, "text": text, "wc": wc, "gold": gold,
                    "total": int(float(row.get("Total", 0) or 0)),
                    "fold": folds.get(rid, "A")})
    return out


def stratified_sample(records, n, seed=0):
    rng = random.Random(seed)
    by_band = defaultdict(list)
    for r in records:
        by_band[total_band(r["total"])].append(r)
    picked = []
    bands = list(by_band)
    i = 0
    while len(picked) < n and any(by_band.values()):
        b = bands[i % len(bands)]
        if by_band[b]:
            picked.append(by_band[b].pop(rng.randrange(len(by_band[b]))))
        i += 1
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--folds")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--models", default="sonnet,opus")
    ap.add_argument("--exemplars", action="store_true",
                    help="include gold exemplars in the prompt (per_band=2)")
    ap.add_argument("--cache-dir", default=".")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set in the environment.")

    records = load_records(args.gold, args.folds)
    sample = stratified_sample(records, args.sample)
    print(f"{len(records)} gold rows; scoring a stratified sample of {len(sample)}")

    exemplars = load_gold_exemplars(args.gold, per_band=2) if args.exemplars else None

    rows = []
    for name in [m.strip() for m in args.models.split(",")]:
        scorer = SCORERS[name](exemplars=exemplars,
                               cache=JsonlCache(os.path.join(args.cache_dir, f"bakeoff_{name}.jsonl")))
        print(f"\nscoring with {scorer.model} ...")
        preds = {}
        for i, rec in enumerate(sample, 1):
            preds[rec["id"]] = scorer.score(rec["text"], year_level=None).scores
            print(f"  {i}/{len(sample)}", end="\r")
        print()

        res = foldwise_calibrate(
            sample,
            predict_fn=lambda rec: preds[rec["id"]],
            gold_fn=lambda rec: rec["gold"],
            fold_fn=lambda rec: rec["fold"],
        )
        b, a = res["before"]["total"], res["after"]["total"]
        rows.append((name, scorer.model, b, a))

    print("\n=== held-out TOTAL mark agreement ===")
    print(f"{'model':18}{'stage':8}{'MAE':>7}{'within2':>9}{'within4':>9}{'corr':>7}")
    for name, model, b, a in rows:
        print(f"{model:18}{'raw':8}{b['mae']:>7}{b['within2_pct']:>9}{b['within4_pct']:>9}{b['correlation']:>7}")
        print(f"{'':18}{'calib':8}{a['mae']:>7}{a['within2_pct']:>9}{a['within4_pct']:>9}{a['correlation']:>7}")


if __name__ == "__main__":
    raise SystemExit(main())
