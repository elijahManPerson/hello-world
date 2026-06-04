"""Command-line interface for the criteria prediction engine.

    python -m criteria_engine score --text "Once upon a time..."
    python -m criteria_engine score --word-count 480
    python -m criteria_engine assess --scores AU=4,TS=3,ID=3,CS/PD=3,Voc=3,Coh=2,Pa=1,SS=4,Pun=2,Spell=5
    python -m criteria_engine validate --csv workbook.csv
    python -m criteria_engine audit --csv workbook.csv
"""

from __future__ import annotations

import argparse
import json
import sys

from .ranges import CRITERIA
from .engine import predict_criteria, assess_scores
from .crossval import cross_validate, audit_fit, load_csv


def _parse_scores(s: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for pair in s.split(","):
        key, _, val = pair.partition("=")
        out[key.strip()] = float(val)
    missing = [c for c in CRITERIA if c not in out]
    if missing:
        raise SystemExit(f"missing criteria: {', '.join(missing)}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="criteria_engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score", help="predict criteria from text/length")
    p_score.add_argument("--text")
    p_score.add_argument("--word-count", type=int)
    p_score.add_argument("--year-level", type=int)

    p_assess = sub.add_parser("assess", help="fit-check existing criterion scores")
    p_assess.add_argument("--scores", required=True)

    p_val = sub.add_parser("validate", help="cross-validate against gold CSV")
    p_val.add_argument("--csv", required=True)

    p_aud = sub.add_parser("audit", help="audit fit/flag rates over a CSV")
    p_aud.add_argument("--csv", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "score":
        if args.text is None and args.word_count is None:
            raise SystemExit("provide --text or --word-count")
        result = predict_criteria(
            text=args.text, word_count=args.word_count, year_level=args.year_level
        )
    elif args.cmd == "assess":
        result = assess_scores(_parse_scores(args.scores))
    elif args.cmd == "validate":
        result = cross_validate(load_csv(args.csv))
    elif args.cmd == "audit":
        result = audit_fit(load_csv(args.csv))
    else:  # pragma: no cover
        parser.error("unknown command")

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
