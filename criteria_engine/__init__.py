"""Criteria prediction engine for AES cross-validation.

A two-layer engine for NAPLAN-style narrative scoring that preserves the
workbook's original criterion ranges (Total 0-47) exactly:

  1. Criteria predictor (the boss): predict each criterion inside its own
     range, then sum to a tentative total.
  2. Fit layer (sanity check): does each criterion score fit the rest of the
     profile - plausible, high, low, or suspicious?

Quick start
-----------
    from criteria_engine import predict_criteria, assess_scores, cross_validate

    # Predict + fit-check from raw text:
    predict_criteria(text=my_script)

    # Fit-check existing human/LLM/model criterion scores:
    assess_scores({"AU":4,"TS":3,"ID":3,"CS/PD":3,"Voc":3,
                   "Coh":2,"Pa":1,"SS":4,"Pun":2,"Spell":5})

    # Cross-validate predictions against gold workbook rows:
    cross_validate(rows)
"""

from .ranges import CRITERIA, RANGES, TOTAL_MAX, RANGE_LABELS, FAMILIES
from .features import extract_text_features, TextFeatures
from .priors import (
    total_band,
    expected_profile_for_total,
    predict_from_surface_features,
    wordcount_band,
    BAND_PROFILE,
)
from .fit import assess_criterion_fit, FitReport, SUSPICIOUS_RULES
from .engine import predict_criteria, assess_scores
from .crossval import cross_validate, audit_fit, load_csv
from .llm_backend import (
    LLMScorer,
    ClaudeScorer,
    HaikuScorer,
    JsonlCache,
    EvidenceUnit,
    load_gold_exemplars,
    build_rubric_system_prompt,
    build_user_message,
)
from .keras_backend import KerasCriterionScorer

__all__ = [
    "CRITERIA", "RANGES", "TOTAL_MAX", "RANGE_LABELS", "FAMILIES",
    "extract_text_features", "TextFeatures",
    "total_band", "expected_profile_for_total",
    "predict_from_surface_features", "wordcount_band", "BAND_PROFILE",
    "assess_criterion_fit", "FitReport", "SUSPICIOUS_RULES",
    "predict_criteria", "assess_scores",
    "cross_validate", "audit_fit", "load_csv",
    "LLMScorer", "ClaudeScorer", "HaikuScorer", "JsonlCache", "EvidenceUnit",
    "load_gold_exemplars", "build_rubric_system_prompt", "build_user_message",
    "KerasCriterionScorer",
]

__version__ = "0.1.0"
