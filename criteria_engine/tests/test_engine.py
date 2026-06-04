"""Tests for the criteria prediction engine.

Run with:  python -m pytest criteria_engine/tests
       or:  python criteria_engine/tests/test_engine.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from criteria_engine.ranges import CRITERIA, RANGES, TOTAL_MAX, clamp, total
from criteria_engine.priors import (
    total_band,
    BAND_PROFILE,
    expected_profile_for_total,
)
from criteria_engine.fit import assess_criterion_fit
from criteria_engine.engine import predict_criteria, assess_scores
from criteria_engine.crossval import cross_validate, audit_fit


def test_total_max_is_47():
    assert TOTAL_MAX == 47
    assert sum(RANGES.values()) == 47


def test_band_profiles_respect_ranges():
    # Every expected profile value must sit inside its criterion range.
    for band, profile in BAND_PROFILE.items():
        for c in CRITERIA:
            assert 0 <= profile[c] <= RANGES[c], (band, c)


def test_total_band_edges():
    assert total_band(0) == "0-9"
    assert total_band(9) == "0-9"
    assert total_band(10) == "10-16"
    assert total_band(28) == "23-28"
    assert total_band(29) == "29-34"
    assert total_band(47) == "41-47"


def test_clamp_respects_range():
    assert clamp("Pa", 5) == 2
    assert clamp("AU", -3) == 0
    assert clamp("TS", 2.6) == 3


def test_total_preserved_as_sum():
    profile = {"AU": 4, "TS": 3, "ID": 3, "CS/PD": 3, "Voc": 3,
               "Coh": 2, "Pa": 1, "SS": 4, "Pun": 2, "Spell": 5}
    assert total(profile) == 30
    report = assess_criterion_fit(profile)
    assert report.raw_total == 30
    assert report.total_band == "29-34"


def test_worked_example_mostly_fits():
    # The design's worked example: three one-point residuals -> "Mostly fits",
    # confidence high.
    profile = {"AU": 4, "TS": 3, "ID": 3, "CS/PD": 3, "Voc": 3,
               "Coh": 2, "Pa": 1, "SS": 4, "Pun": 2, "Spell": 5}
    report = assess_criterion_fit(profile)
    assert report.fit == "Mostly fits"
    assert report.confidence == "high"
    # Cohesion and Spelling deviations should be flagged.
    joined = " ".join(report.review_flags)
    assert "Cohesion" in joined
    assert "Spelling" in joined


def test_perfect_band_profile_is_strong_fit():
    profile = dict(expected_profile_for_total(31))  # the 29-34 anchor
    report = assess_criterion_fit(profile)
    assert report.fit == "Strong fit"
    assert report.review_flags == []


def test_hard_rule_ts_high_au_low():
    # TS=4 with AU<=3 has 0 cases in the workbook -> Review needed.
    profile = {"AU": 3, "TS": 4, "ID": 3, "CS/PD": 4, "Voc": 3,
               "Coh": 3, "Pa": 1, "SS": 4, "Pun": 3, "Spell": 4}
    report = assess_criterion_fit(profile)
    assert report.fit == "Review needed"
    assert "TS_high_AU_low" in report.fired_rules


def test_hard_rule_pa_high_coh_low():
    profile = {"AU": 4, "TS": 3, "ID": 3, "CS/PD": 3, "Voc": 3,
               "Coh": 1, "Pa": 2, "SS": 4, "Pun": 3, "Spell": 4}
    report = assess_criterion_fit(profile)
    assert "Pa_high_Coh_low" in report.fired_rules
    assert report.fit == "Review needed"


def test_soft_rule_lowers_confidence():
    # Spell>=5 with Pun<=1 is rare -> Criterion conflict, not Review needed.
    profile = {"AU": 4, "TS": 2, "ID": 3, "CS/PD": 2, "Voc": 3,
               "Coh": 2, "Pa": 1, "SS": 3, "Pun": 1, "Spell": 5}
    report = assess_criterion_fit(profile)
    assert "Spell_high_Pun_low" in report.fired_rules
    assert report.fit == "Criterion conflict"


def test_predict_from_text_preserves_ranges():
    text = ("It was a dark and stormy night. The old house creaked. "
            "Sarah opened the door slowly, her heart pounding. ") * 30
    result = predict_criteria(text=text)
    assert result["score_range_preserved"] is True
    assert result["total_range"] == "0-47"
    for c in CRITERIA:
        s = result["criteria_predictions"][c]["score"]
        assert 0 <= s <= RANGES[c]
    assert 0 <= result["tentative_total"] <= 47
    # Tentative total must equal the criterion sum.
    assert result["tentative_total"] == sum(
        result["criteria_predictions"][c]["score"] for c in CRITERIA
    )


def test_predict_by_word_count_only():
    result = predict_criteria(word_count=480)
    assert result["tentative_total"] == sum(
        result["actual_profile"][c] for c in CRITERIA
    )
    assert result["total_band"] in {"23-28", "29-34"}


def test_supplied_scores_are_fit_checked():
    scores = {"AU": 6, "TS": 4, "ID": 5, "CS/PD": 4, "Voc": 5,
              "Coh": 4, "Pa": 2, "SS": 6, "Pun": 4, "Spell": 5}
    result = predict_criteria(text="short", criterion_scores=scores)
    assert result["tentative_total"] == 45
    assert result["actual_profile"]["AU"] == 6
    assert "supplied" in result["basis"]


def test_assess_scores_round_trips():
    scores = {"AU": 4, "TS": 3, "ID": 3, "CS/PD": 3, "Voc": 3,
              "Coh": 3, "Pa": 1, "SS": 4, "Pun": 3, "Spell": 4}
    out = assess_scores(scores, script_id="ID_000001")
    assert out["script_id"] == "ID_000001"
    assert out["raw_total"] == 31
    assert out["fit"] == "Strong fit"


def test_cross_validate_metrics_shape():
    rows = [
        {"WordCount": 480, "AU": 4, "TS": 3, "ID": 3, "CS/PD": 3, "Voc": 3,
         "Coh": 2, "Pa": 1, "SS": 3, "Pun": 2, "Spell": 4},
        {"WordCount": 80, "AU": 2, "TS": 1, "ID": 2, "CS/PD": 1, "Voc": 2,
         "Coh": 2, "Pa": 0, "SS": 2, "Pun": 1, "Spell": 2},
    ]
    metrics = cross_validate(rows)
    assert metrics["n"] == 2
    assert set(metrics["per_criterion"]) == set(CRITERIA)
    assert "mae" in metrics["total"]
    assert "correlation" in metrics["total"]


def test_audit_fit_clean_profiles_no_hard_flags():
    # Profiles taken from the band anchors should not trip hard rules.
    rows = []
    for band, profile in BAND_PROFILE.items():
        row = dict(profile)
        row["WordCount"] = 300
        rows.append(row)
    audit = audit_fit(rows)
    hard = {"TS_high_AU_low", "TS_high_CSPD_low", "Pa_high_Coh_low"}
    for name in hard:
        assert audit["suspicious_rule_rate"][name]["count"] == 0


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
