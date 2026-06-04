"""Validate the engine against a gold-scored CSV (e.g. a Drive test set).

The CSV needs the ten criterion columns (workbook names/aliases accepted) plus
either ``Raw text`` or ``WordCount``. Runs both reports:

  * cross_validate - text-feature prior vs gold (exact / within-1 / MAE; total
    MAE + correlation),
  * audit_fit      - fit layer over the gold profiles (label distribution and
    how often the suspicious-pattern rules fire).

Usage:
    python examples/validate_drive_csv.py "Data for Testing avg.csv"
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from criteria_engine import cross_validate, audit_fit, load_csv, CRITERIA


def main(path: str) -> int:
    rows = load_csv(path)
    print(f"loaded {len(rows)} rows from {path}\n")

    cv = cross_validate(rows)
    print("=== cross_validate (text-feature prior vs gold) ===")
    print("n:", cv.get("n"), "skipped:", cv.get("skipped"))
    if cv.get("n"):
        print("total:", json.dumps(cv["total"]))
        print("per-criterion (exact% / within1% / MAE):")
        for c in CRITERIA:
            pc = cv["per_criterion"][c]
            print(f"  {c:6} {pc['exact_pct']:5}  {pc['within1_pct']:5}  {pc['mae']}")

    af = audit_fit(rows)
    print("\n=== audit_fit (fit layer over gold profiles) ===")
    if af.get("n"):
        print("fit labels:", json.dumps(
            {k: v["count"] for k, v in af["fit_label_distribution"].items()}))
        fired = {n: d["count"] for n, d in af["suspicious_rule_rate"].items() if d["count"]}
        print("suspicious-rule fires:", json.dumps(fired) if fired else "none")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/validate_drive_csv.py <csv_path>")
    raise SystemExit(main(sys.argv[1]))
