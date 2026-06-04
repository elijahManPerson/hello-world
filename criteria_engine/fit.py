"""The fit layer: profile sanity checking.

Given a criterion profile, this module:
  * sums it to a tentative total and finds the total band (sections 5A/5B),
  * compares each criterion to the band's expected profile (section 5C),
  * applies the hard "suspicious pattern" rules observed as 0 / near-0 cases
    in the workbook (section 7),
  * reports family coherence (section 8),
  * and returns a fit label + confidence + human-readable review flags
    (sections 6 and 10).

The fit layer never changes a score. It only judges whether a score "fits"
the rest of the evidence: plausible, high, low, or suspicious.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ranges import CRITERIA, FULL_NAMES, FAMILIES, total as profile_total
from .priors import total_band, expected_profile_for_total

# Criteria with a small range where a one-point gap is more meaningful
# (section 5: "For Pa 0-2, a difference of 1 can be meaningful").
_SMALL_RANGE = {"Pa"}


@dataclass
class SuspiciousRule:
    """A named co-occurrence the workbook shows as impossible or very rare."""
    name: str
    severity: str               # "hard" (0 cases) or "soft" (very rare)
    message: str
    predicate: callable

    def fires(self, p: dict[str, int]) -> bool:
        return self.predicate(p)


# Section 7. "hard" = 0 observed cases -> Review needed.
#            "soft" = observed but very rare -> lower confidence.
SUSPICIOUS_RULES: list[SuspiciousRule] = [
    SuspiciousRule(
        "TS_high_AU_low", "hard",
        "Text Structure is high (4) while Audience is low (<=3) - "
        "0 such cases in the workbook. Text structure may be over-scored.",
        lambda p: p["TS"] == 4 and p["AU"] <= 3,
    ),
    SuspiciousRule(
        "TS_high_CSPD_low", "hard",
        "Text Structure is high (4) while Character/Setting is low (<=2) - "
        "0 such cases in the workbook. Narrative structure may be over-scored.",
        lambda p: p["TS"] == 4 and p["CS/PD"] <= 2,
    ),
    SuspiciousRule(
        "Pa_high_Coh_low", "hard",
        "Paragraphing is high (2) while Cohesion is low (<=1) - "
        "0 such cases in the workbook. Paragraphing may be over-scored.",
        lambda p: p["Pa"] == 2 and p["Coh"] <= 1,
    ),
    SuspiciousRule(
        "Voc_high_SS_low", "soft",
        "Vocabulary is high (>=4) while Sentence Structure is low (<=2) - "
        "~0.18% of cases. Vocabulary may be over-scored or SS under-scored.",
        lambda p: p["Voc"] >= 4 and p["SS"] <= 2,
    ),
    SuspiciousRule(
        "Pun_high_SS_low", "soft",
        "Punctuation is high (>=4) while Sentence Structure is low (<=2) - "
        "~0.10% of cases. Review carefully.",
        lambda p: p["Pun"] >= 4 and p["SS"] <= 2,
    ),
    SuspiciousRule(
        "Spell_high_Pun_low", "soft",
        "Spelling is high (>=5) while Punctuation is low (<=1) - "
        "~0.29% of cases. Lower confidence.",
        lambda p: p["Spell"] >= 5 and p["Pun"] <= 1,
    ),
]


@dataclass
class FitReport:
    raw_total: int
    total_band: str
    expected_profile: dict[str, int]
    actual_profile: dict[str, int]
    residuals: dict[str, int]
    fit: str
    confidence: str
    review_flags: list[str] = field(default_factory=list)
    family_coherence: dict[str, str] = field(default_factory=dict)
    fired_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw_total": self.raw_total,
            "total_band": self.total_band,
            "expected_profile": self.expected_profile,
            "actual_profile": self.actual_profile,
            "residuals": self.residuals,
            "fit": self.fit,
            "confidence": self.confidence,
            "review_flags": self.review_flags,
            "family_coherence": self.family_coherence,
            "fired_rules": self.fired_rules,
        }


def _direction_word(resid: int) -> str:
    mag = abs(resid)
    adj = "lower" if resid < 0 else "higher"
    if mag == 1:
        return f"slightly {adj}"
    if mag == 2:
        return f"noticeably {adj}"
    return f"much {adj}"


def _residual_message(crit: str, resid: int) -> str:
    return (
        f"{FULL_NAMES[crit]} is {_direction_word(resid)} than typical "
        f"for this total band."
    )


def _severity(crit: str, resid: int) -> int:
    """Effective severity of a residual (0=fit, 1=normal, 2=misfit, 3+=strong).

    Small-range criteria escalate one level: for Paragraphing a one-point gap
    is treated as a possible misfit rather than normal variation.
    """
    mag = abs(resid)
    if crit in _SMALL_RANGE and mag >= 1:
        mag += 1
    return mag


def _family_coherence(residuals: dict[str, int]) -> dict[str, str]:
    """Label each family by how internally consistent its residuals are."""
    out: dict[str, str] = {}
    for fam, members in FAMILIES.items():
        spread = max(residuals[m] for m in members) - min(
            residuals[m] for m in members
        )
        if spread <= 1:
            out[fam] = "coherent"
        elif spread == 2:
            out[fam] = "mild tension"
        else:
            out[fam] = "incoherent"
    return out


def assess_criterion_fit(profile: dict[str, int]) -> FitReport:
    """Run the full fit check on a criterion profile (sections 5-10)."""
    actual = {c: int(profile[c]) for c in CRITERIA}
    raw_total = profile_total(actual)
    band = total_band(raw_total)
    expected = expected_profile_for_total(raw_total)
    residuals = {c: actual[c] - expected[c] for c in CRITERIA}

    # Tally residual severities.
    ones = sum(1 for c in CRITERIA if _severity(c, residuals[c]) == 1)
    twos = sum(1 for c in CRITERIA if _severity(c, residuals[c]) == 2)
    threes = sum(1 for c in CRITERIA if _severity(c, residuals[c]) >= 3)

    # Suspicious patterns.
    hard_fired = [r for r in SUSPICIOUS_RULES if r.severity == "hard" and r.fires(actual)]
    soft_fired = [r for r in SUSPICIOUS_RULES if r.severity == "soft" and r.fires(actual)]

    # Fit label (section 6), calibrated to the worked example in the design
    # (three one-point residuals -> "Mostly fits").
    if hard_fired:
        fit = "Review needed"
        confidence = "low"
    elif threes >= 1:
        fit = "Criterion conflict"
        confidence = "low-to-medium"
    elif soft_fired:
        fit = "Criterion conflict"
        confidence = "low-to-medium"
    elif twos >= 1 or ones >= 4:
        fit = "Uneven profile"
        confidence = "medium"
    elif ones >= 1:
        fit = "Mostly fits"
        confidence = "high"
    else:
        fit = "Strong fit"
        confidence = "high"

    # Review flags: every >=1-point residual, plus any suspicious patterns.
    flags = [
        _residual_message(c, residuals[c])
        for c in CRITERIA
        if abs(residuals[c]) >= 1
    ]
    flags.extend(r.message for r in hard_fired)
    flags.extend(r.message for r in soft_fired)

    return FitReport(
        raw_total=raw_total,
        total_band=band,
        expected_profile=expected,
        actual_profile=actual,
        residuals=residuals,
        fit=fit,
        confidence=confidence,
        review_flags=flags,
        family_coherence=_family_coherence(residuals),
        fired_rules=[r.name for r in hard_fired + soft_fired],
    )
