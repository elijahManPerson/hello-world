"""Prior profiles transcribed from the workbook analysis.

Two tables:

1. WORDCOUNT_PRIOR  - section 2 of the criteria-predictor design. Maps a word
   count band to a likely criterion profile. This is the *first guess* before
   any semantic evidence.

2. BAND_PROFILE     - section 4 of the fit design. Maps a total-score band to
   the typical criterion profile observed for that total. This is the anchor
   the fit layer compares an actual profile against.

Both are in canonical CRITERIA order:
    AU, TS, ID, CS/PD, Voc, Coh, Pa, SS, Pun, Spell
"""

from __future__ import annotations

from .ranges import CRITERIA, as_dict
from .features import TextFeatures

# --- Total-score bands -----------------------------------------------------

# (inclusive_upper_bound, label)
_BAND_EDGES: list[tuple[int, str]] = [
    (9, "0-9"),
    (16, "10-16"),
    (22, "17-22"),
    (28, "23-28"),
    (34, "29-34"),
    (40, "35-40"),
    (47, "41-47"),
]

BAND_LABELS: tuple[str, ...] = tuple(label for _, label in _BAND_EDGES)


def total_band(raw_total: int) -> str:
    """Return the total band label for a raw total (section 5, Step B)."""
    for upper, label in _BAND_EDGES:
        if raw_total <= upper:
            return label
    return _BAND_EDGES[-1][1]


# Expected criterion profile per total band (section 4 table).
#                       AU TS ID CSPD Voc Coh Pa SS Pun Spell
_BAND_PROFILE_RAW: dict[str, list[int]] = {
    "0-9":   [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    "10-16": [2, 1, 2, 1, 2, 2, 0, 2, 1, 2],
    "17-22": [3, 2, 2, 2, 2, 2, 0, 2, 2, 3],
    "23-28": [3, 2, 3, 2, 2, 2, 1, 3, 2, 4],
    "29-34": [4, 3, 3, 3, 3, 3, 1, 4, 3, 4],
    "35-40": [5, 3, 4, 4, 4, 3, 2, 4, 3, 5],
    "41-47": [6, 4, 5, 4, 5, 4, 2, 5, 4, 5],
}

BAND_PROFILE: dict[str, dict[str, int]] = {
    band: dict(zip(CRITERIA, vals)) for band, vals in _BAND_PROFILE_RAW.items()
}


def expected_profile_for_total(raw_total: int) -> dict[str, int]:
    """Expected criterion profile for the band containing ``raw_total``."""
    return dict(BAND_PROFILE[total_band(raw_total)])


# --- Word-count prior table ------------------------------------------------

# (inclusive_word_count_upper, profile, typical_total)
#                              AU TS ID CSPD Voc Coh Pa SS Pun Spell
_WORDCOUNT_PRIOR_RAW: list[tuple[int, list[int], int]] = [
    (50,   [1, 1, 1, 1, 1, 1, 0, 1, 1, 1], 8),
    (100,  [2, 1, 2, 1, 2, 2, 0, 2, 1, 2], 15),
    (150,  [2, 2, 2, 2, 2, 2, 0, 2, 2, 3], 18),
    (200,  [3, 2, 2, 2, 2, 2, 0, 2, 2, 3], 20),
    (250,  [3, 2, 3, 2, 2, 2, 1, 2, 2, 3], 22),
    (300,  [3, 2, 3, 2, 2, 2, 1, 3, 2, 3], 23),
    (350,  [3, 2, 3, 2, 2, 2, 1, 3, 2, 4], 24),
    (400,  [3, 2, 3, 3, 3, 2, 1, 3, 2, 4], 26),
    (500,  [4, 3, 3, 3, 3, 2, 1, 3, 2, 4], 28),
    (650,  [4, 3, 3, 3, 3, 3, 1, 4, 3, 4], 31),
    (800,  [4, 3, 4, 3, 4, 3, 1, 4, 3, 5], 33),
    (10**9, [5, 3, 4, 3, 4, 3, 2, 4, 3, 5], 35),
]


def wordcount_band(word_count: int) -> tuple[dict[str, int], int]:
    """Return (prior_profile, typical_total) for a word count."""
    for upper, profile, typ in _WORDCOUNT_PRIOR_RAW:
        if word_count <= upper:
            return dict(zip(CRITERIA, profile)), typ
    last = _WORDCOUNT_PRIOR_RAW[-1]
    return dict(zip(CRITERIA, last[1])), last[2]


def predict_from_surface_features(
    features: TextFeatures, year_level: int | None = None
) -> dict[str, int]:
    """Text-feature-only criterion prior (section 5, ``predict_from_surface``).

    Starts from the word-count band profile, then applies a few small,
    bounded, individually defensible nudges from other surface features.
    Nudges never move a criterion by more than one point and are always
    re-clamped to the criterion range. This is a prior, not a verdict.
    """
    profile, _typical = wordcount_band(features.word_count)
    profile = dict(profile)

    if features.word_count == 0:
        return as_dict([0] * len(CRITERIA))

    # Vocabulary tracks the rate of longer, more mature words.
    if features.long_word_rate >= 0.20:
        profile["Voc"] += 1
    elif features.long_word_rate <= 0.05:
        profile["Voc"] -= 1

    # Sentence Structure: very long run-on "sentences" suggest weak boundary
    # control; short controlled sentences suggest the opposite.
    if features.avg_sentence_length >= 35:
        profile["SS"] -= 1
    elif 8 <= features.avg_sentence_length <= 22 and features.word_count >= 120:
        profile["SS"] += 1

    # Punctuation: any use of internal punctuation variety nudges Pun up from a
    # bare-minimum prior; complete absence of end marks nudges it down.
    if features.punctuation_variety >= 4 and features.comma_density > 0:
        profile["Pun"] += 1
    elif features.endmark_density == 0:
        profile["Pun"] -= 1

    return as_dict([profile[c] for c in CRITERIA])
