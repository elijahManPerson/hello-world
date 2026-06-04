"""Canonical rubric schema for the AES criteria prediction engine.

These ranges are taken directly from the workbook and MUST be preserved
exactly. The engine never invents a new total scale: the tentative total is
always ``sum(criteria)`` and every criterion is clamped to its own range.

Workbook invariant (verified across all 9,695 scripts):
    Total == AU + TS + ID + CS/PD + Voc + Coh + Pa + SS + Pun + Spell
"""

from __future__ import annotations

# Canonical criterion order. Every profile (list form) in this package uses
# exactly this order, so lists and dicts can be converted unambiguously.
CRITERIA: tuple[str, ...] = (
    "AU",      # Audience
    "TS",      # Text Structure
    "ID",      # Ideas
    "CS/PD",   # Character & Setting / Persuasive Devices
    "Voc",     # Vocabulary
    "Coh",     # Cohesion
    "Pa",      # Paragraphing
    "SS",      # Sentence Structure
    "Pun",     # Punctuation
    "Spell",   # Spelling
)

# Inclusive maximum for each criterion (minimum is always 0).
RANGES: dict[str, int] = {
    "AU": 6,
    "TS": 4,
    "ID": 5,
    "CS/PD": 4,
    "Voc": 5,
    "Coh": 4,
    "Pa": 2,
    "SS": 6,
    "Pun": 5,
    "Spell": 6,
}

# Total range is the sum of the criterion maxima: 6+4+5+4+5+4+2+6+5+6 = 47.
TOTAL_MAX: int = sum(RANGES.values())  # 47

# Human-readable names for evidence strings and review flags.
FULL_NAMES: dict[str, str] = {
    "AU": "Audience",
    "TS": "Text Structure",
    "ID": "Ideas",
    "CS/PD": "Character/Setting",
    "Voc": "Vocabulary",
    "Coh": "Cohesion",
    "Pa": "Paragraphing",
    "SS": "Sentence Structure",
    "Pun": "Punctuation",
    "Spell": "Spelling",
}

# "Range" label shown in output, e.g. {"AU": "0-6"}.
RANGE_LABELS: dict[str, str] = {c: f"0-{RANGES[c]}" for c in CRITERIA}

# Criteria grouped into families that should move together (section 8).
# Used by the fit layer to report which families are internally coherent.
FAMILIES: dict[str, tuple[str, ...]] = {
    "global_quality": ("AU", "TS", "ID", "CS/PD", "Voc", "Coh", "SS"),
    "narrative_development": ("TS", "ID", "CS/PD", "AU"),
    "language_control": ("SS", "Pun", "Spell"),
    "organisation": ("TS", "Coh", "Pa"),
    "mechanical_surface": ("Pun", "Spell", "SS"),
}


def clamp(criterion: str, score: float) -> int:
    """Clamp ``score`` to the inclusive [0, RANGES[criterion]] integer range."""
    hi = RANGES[criterion]
    return int(max(0, min(hi, round(score))))


def as_list(profile: dict[str, float]) -> list[int]:
    """Convert a criterion dict to a list in canonical CRITERIA order."""
    return [clamp(c, profile[c]) for c in CRITERIA]


def as_dict(values: list[float]) -> dict[str, int]:
    """Convert a canonical-order list back to a clamped criterion dict."""
    if len(values) != len(CRITERIA):
        raise ValueError(f"expected {len(CRITERIA)} values, got {len(values)}")
    return {c: clamp(c, v) for c, v in zip(CRITERIA, values)}


def total(profile: dict[str, float]) -> int:
    """Tentative total = clamped sum of the criterion profile."""
    return sum(clamp(c, profile[c]) for c in CRITERIA)
