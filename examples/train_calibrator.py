"""Train a GoldCalibrator on the gold marks and report held-out accuracy.

By default it trains the calibrator over the **text-feature prior** (runs with
no API key), which is useful for sanity-checking the calibration itself. Point
``--predictor sonnet`` at a real ClaudeScorer once ANTHROPIC_API_KEY is set to
calibrate the Claude marks instead.

    python examples/train_calibrator.py \
        --gold gold_clean_365_with_year_proxy.csv \
        --folds gold_365_fold_assignment.csv \
        --out calibrator.json

The fold split gives an honest estimate: the calibrator is trained on one fold
and scored on the other, then a final calibrator is fit on all rows and saved.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from criteria_engine import predict_criteria, CRITERIA
from criteria_engine.calibration import foldwise_calibrate


def load_records(gold_path: str, folds_path: str | None) -> list[dict]:
    folds = {}
    if folds_path:
        for r in csv.DictReader(open(folds_path, encoding="utf-8-sig")):
            folds[r["Research ID"]] = r.get("Fold", "A")
    records = []
    for row in csv.DictReader(open(gold_path, encoding="utf-8-sig")):
        rid = row.get("Research ID", "")
        try:
            gold = {c: int(float(row[c])) for c in CRITERIA}
            wc = int(float(row["WordCount"]))
        except (ValueError, KeyError):
            continue
        records.append({
            "id": rid, "text": row.get("Raw text", ""), "wc": wc,
            "gold": gold, "fold": folds.get(rid, "A"),
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--folds")
    ap.add_argument("--out", default="calibrator.json")
    ap.add_argument("--predictor", choices=["prior", "sonnet", "haiku"], default="prior")
    args = ap.parse_args()

    records = load_records(args.gold, args.folds)
    print(f"loaded {len(records)} gold records")

    if args.predictor == "prior":
        def predict_fn(rec):
            return predict_criteria(text=rec["text"], word_count=rec["wc"])["actual_profile"]
    else:
        from criteria_engine.llm_backend import SonnetScorer, HaikuScorer
        scorer = (SonnetScorer if args.predictor == "sonnet" else HaikuScorer)()
        def predict_fn(rec):
            return scorer.score(rec["text"]).scores

    res = foldwise_calibrate(
        records, predict_fn, gold_fn=lambda r: r["gold"], fold_fn=lambda r: r["fold"],
    )
    b, a = res["before"]["total"], res["after"]["total"]
    print("\nTOTAL mark agreement on held-out folds:")
    print(f"  before calibration: MAE {b['mae']}  within2 {b['within2_pct']}%  within4 {b['within4_pct']}%  r {b['correlation']}")
    print(f"  after  calibration: MAE {a['mae']}  within2 {a['within2_pct']}%  within4 {a['within4_pct']}%  r {a['correlation']}")

    res["calibrator"].save(args.out)
    print(f"\nsaved calibrator trained on all {len(records)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
