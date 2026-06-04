"""Walkthrough of the criteria prediction engine.

    python examples/demo.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from criteria_engine import predict_criteria, assess_scores, cross_validate, audit_fit


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# 1. Predict criteria from raw text (criterion predictor is the boss).
section("1. Predict criteria from text, then fit-check")
story = (
    "The storm had been building all afternoon. By the time Mia reached the "
    "old lighthouse, rain lashed the cliffs and the sea boiled below. She "
    "pushed the heavy door open and stepped inside, shaking the water from "
    "her coat. A single lantern flickered on the stairs. Someone had been "
    "here, and recently. Mia climbed, counting each step, her breath loud in "
    "the narrow dark. At the top she found a notebook, its pages curled with "
    "damp, and in it the answer she had crossed the island to find."
) * 2
result = predict_criteria(text=story, script_id="DEMO_0001")
print(json.dumps(result, indent=2))

# 2. Fit-check existing human / LLM / model criterion scores (Part 1 use).
section("2. Fit-check supplied scores (the AES consistency check)")
print(json.dumps(
    assess_scores(
        {"AU": 4, "TS": 3, "ID": 3, "CS/PD": 3, "Voc": 3,
         "Coh": 2, "Pa": 1, "SS": 4, "Pun": 2, "Spell": 5},
        script_id="DEMO_0002",
    ),
    indent=2,
))

# 3. A suspicious profile that trips a hard rule (0 cases in the workbook).
section("3. Suspicious profile (TS high, AU low) -> Review needed")
print(json.dumps(
    assess_scores(
        {"AU": 3, "TS": 4, "ID": 3, "CS/PD": 4, "Voc": 3,
         "Coh": 3, "Pa": 1, "SS": 4, "Pun": 3, "Spell": 4},
        script_id="DEMO_0003",
    ),
    indent=2,
))

# 4. Cross-validate text-feature predictions against gold rows.
section("4. Cross-validate against gold rows")
rows = [
    {"WordCount": 480, "AU": 4, "TS": 3, "ID": 3, "CS/PD": 3, "Voc": 3,
     "Coh": 3, "Pa": 1, "SS": 4, "Pun": 3, "Spell": 4},
    {"WordCount": 90, "AU": 2, "TS": 1, "ID": 2, "CS/PD": 1, "Voc": 2,
     "Coh": 2, "Pa": 0, "SS": 2, "Pun": 1, "Spell": 2},
    {"WordCount": 700, "AU": 5, "TS": 3, "ID": 4, "CS/PD": 4, "Voc": 4,
     "Coh": 3, "Pa": 2, "SS": 4, "Pun": 3, "Spell": 5},
]
print(json.dumps(cross_validate(rows), indent=2))

# 5. Audit fit / suspicious-rule rates over the same rows.
section("5. Audit fit + suspicious-rule rates")
print(json.dumps(audit_fit(rows), indent=2))
