"""Criteria prediction engine.

The criterion predictor is the boss (design section "My recommended rule"):
predict each criterion inside its original range, sum to a tentative total,
then run the fit layer as a sanity check. The total predictor is never the
boss; it only asks "given this text, does this total look plausible?".

Sources of a per-criterion score, in priority order:
  1. ``criterion_scores`` passed in (e.g. from the workbook NN model or an LLM
     rubric pass) - treated as the prediction to be fit-checked.
  2. text-feature prior from :mod:`priors` - used when no scores are supplied.

An optional ``evidence`` mapping (criterion -> short string) is threaded into
the output so the prediction is inspectable.
"""

from __future__ import annotations

from .ranges import CRITERIA, RANGE_LABELS, clamp, total as profile_total
from .features import extract_text_features, TextFeatures
from .priors import predict_from_surface_features, wordcount_band, total_band
from .fit import assess_criterion_fit, FitReport

# Default per-criterion confidence for a text-feature-only prediction,
# reflecting how well each criterion is predicted from surface signal alone
# (Audience/Ideas/Spelling read more reliably than Paragraphing/Punctuation).
_DEFAULT_CRIT_CONFIDENCE: dict[str, str] = {
    "AU": "medium-high",
    "TS": "medium",
    "ID": "medium-high",
    "CS/PD": "medium",
    "Voc": "medium",
    "Coh": "high",
    "Pa": "medium",
    "SS": "medium",
    "Pun": "medium",
    "Spell": "medium-high",
}


def _prior_evidence(crit: str, features: TextFeatures) -> str:
    """A short, honest text-feature rationale for a prior prediction."""
    wc = features.word_count
    if crit == "AU":
        return f"Sustained-ness inferred from length (~{wc} words) and control."
    if crit == "TS":
        return "Narrative shape inferred from length and sentence count."
    if crit == "ID":
        return f"Development inferred from text volume (~{wc} words)."
    if crit == "CS/PD":
        return "Character/setting detail inferred from descriptive volume."
    if crit == "Voc":
        return f"Word maturity from long-word rate {features.long_word_rate}."
    if crit == "Coh":
        return "Sequencing inferred from length and connective opportunity."
    if crit == "Pa":
        return f"Paragraphing inferred from line breaks ({features.line_count} lines)."
    if crit == "SS":
        return f"Boundary control from avg sentence length {features.avg_sentence_length}."
    if crit == "Pun":
        return (
            f"Punctuation from variety {features.punctuation_variety}, "
            f"comma density {features.comma_density}."
        )
    if crit == "Spell":
        return "Spelling inferred from length band (no error model in prior)."
    return "Text-feature prior."


def predict_criteria(
    text: str | None = None,
    *,
    word_count: int | None = None,
    year_level: int | None = None,
    criterion_scores: dict[str, float] | None = None,
    evidence: dict[str, str] | None = None,
    script_id: str | None = None,
) -> dict:
    """Predict each criterion, sum to a tentative total, then fit-check.

    Parameters
    ----------
    text:
        Raw script text. Optional if ``word_count`` is supplied.
    word_count:
        Overrides/supplies word count when raw text isn't available.
    criterion_scores:
        Externally supplied per-criterion scores (NN model or LLM). When
        present these are the prediction; the prior is only a fallback.
    evidence:
        Optional per-criterion rationale strings to surface in the output.
    script_id:
        Optional identifier echoed into the output.
    """
    features = extract_text_features(text)
    if word_count is not None:
        # Re-derive the surface prior at the supplied word count.
        features = TextFeatures(
            char_count=features.char_count,
            word_count=int(word_count),
            line_count=features.line_count,
            sentence_count=features.sentence_count,
            avg_sentence_length=features.avg_sentence_length,
            avg_word_length=features.avg_word_length,
            long_word_rate=features.long_word_rate,
            type_token_ratio=features.type_token_ratio,
            comma_density=features.comma_density,
            endmark_density=features.endmark_density,
            punctuation_variety=features.punctuation_variety,
        )

    prior = predict_from_surface_features(features, year_level)

    if criterion_scores is not None:
        predicted = {c: clamp(c, criterion_scores[c]) for c in CRITERIA}
        basis = "supplied criterion scores (fit-checked)"
    else:
        predicted = prior
        basis = "text features only (prior)"

    fit: FitReport = assess_criterion_fit(predicted)

    ev = evidence or {}
    criteria_predictions = {
        c: {
            "score": predicted[c],
            "range": RANGE_LABELS[c],
            "confidence": _DEFAULT_CRIT_CONFIDENCE[c],
            "evidence": ev.get(c, _prior_evidence(c, features)),
        }
        for c in CRITERIA
    }

    _prior_profile, text_typical_total = wordcount_band(features.word_count)
    text_fit = _text_total_fit(text_typical_total, fit.raw_total)

    out: dict = {
        "criteria_predictions": criteria_predictions,
        "tentative_total": fit.raw_total,
        "total_range": "0-47",
        "score_range_preserved": True,
        "total_band": fit.total_band,
        "expected_profile": fit.expected_profile,
        "actual_profile": fit.actual_profile,
        "profile_fit": fit.fit,
        "confidence": fit.confidence,
        "review_flags": fit.review_flags,
        "family_coherence": fit.family_coherence,
        "text_prior": {
            "predicted_profile": prior,
            "typical_total_for_length": text_typical_total,
            "text_total_fit": text_fit,
        },
        "basis": basis,
    }
    if script_id is not None:
        out = {"script_id": script_id, **out}
    return out


def _text_total_fit(text_typical_total: int, raw_total: int) -> str:
    """Does the criterion sum sit near what the text length alone predicts?"""
    gap = abs(raw_total - text_typical_total)
    same_band = total_band(raw_total) == total_band(text_typical_total)
    if gap <= 3 or same_band:
        return "plausible"
    if gap <= 7:
        return "somewhat high" if raw_total > text_typical_total else "somewhat low"
    return "implausible-for-length"


def assess_scores(
    criterion_scores: dict[str, float], *, script_id: str | None = None
) -> dict:
    """Pure consistency check on existing criterion scores (design Part 1).

    Use this for AES cross-validation: feed in human or LLM criterion scores
    and get back whether each score is plausible / high / low / suspicious,
    without any text-feature prediction.
    """
    fit = assess_criterion_fit({c: clamp(c, criterion_scores[c]) for c in CRITERIA})
    result = fit.to_dict()
    if script_id is not None:
        result = {"script_id": script_id, **result}
    return result
