"""PREP scaffold for the LLM scoring backend (production path, Mode 3).

This lays the groundwork to let an LLM read a script, return per-criterion
scores *and* evidence, and feed both into the fit layer via
:func:`criteria_engine.predict_criteria`. It is intentionally a scaffold: the
actual model call is left as one abstract method so it can be wired to whatever
client you use (the existing pipeline uses ``gpt-4o``).

Designed to match the existing ``aes_precision_cache.jsonl`` shape:
  * cache key  : ``"{version}|{model}|{sha1(text)}"``
  * evidence   : units of ``{label, start, end, text, confidence}``

Nothing here imports an API SDK at module load, so the core engine stays
dependency-free.
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
from dataclasses import dataclass, asdict

from .ranges import CRITERIA, RANGE_LABELS, RANGES, clamp
from .priors import BAND_PROFILE
from .engine import predict_criteria

CACHE_VERSION = "criterion_scores_v1"


@dataclass
class EvidenceUnit:
    """A span of evidence, mirroring the precision-cache unit schema."""
    label: str
    start: int
    end: int
    text: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CriterionPrediction:
    scores: dict[str, int]                 # the ten criteria, clamped
    evidence: dict[str, str]               # criterion -> short rationale
    units: list[EvidenceUnit]              # optional fine-grained spans


def text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


class JsonlCache:
    """Append-only JSONL cache keyed like the existing precision cache."""

    def __init__(self, path: str, model: str, version: str = CACHE_VERSION):
        self.path = path
        self.model = model
        self.version = version
        self._mem: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._mem[rec["key"]] = rec

    def key(self, text: str) -> str:
        return f"{self.version}|{self.model}|{text_hash(text)}"

    def get(self, text: str) -> dict | None:
        return self._mem.get(self.key(text))

    def put(self, text: str, payload: dict) -> None:
        rec = {"key": self.key(text), **payload}
        self._mem[rec["key"]] = rec
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


def build_rubric_prompt(text: str, year_level: int | None = None) -> str:
    """Starter prompt: ask for each criterion IN ITS RANGE, plus evidence.

    Refine against the official NAPLAN marking guide before production use.
    The band anchors below are the engine's BAND_PROFILE, included so the model
    is nudged toward internally consistent profiles.
    """
    ranges = "\n".join(f"  - {c} ({RANGE_LABELS[c]})" for c in CRITERIA)
    anchors = "\n".join(
        f"  total {band}: " + ", ".join(f"{c}={BAND_PROFILE[band][c]}" for c in CRITERIA)
        for band in BAND_PROFILE
    )
    yl = f"\nYear level: {year_level}" if year_level is not None else ""
    return (
        "You are marking a narrative writing script against the NAPLAN-style "
        "rubric. Score EACH criterion strictly within its range, and justify "
        "each score with a short rationale grounded in the text.\n\n"
        f"Criteria and ranges:\n{ranges}\n\n"
        f"Typical (not mandatory) profiles by total band, for consistency:\n{anchors}\n"
        f"{yl}\n\n"
        "Return JSON: {\"scores\": {criterion: int, ...}, "
        "\"evidence\": {criterion: str, ...}, "
        "\"units\": [{\"label\": str, \"start\": int, \"end\": int, "
        "\"text\": str, \"confidence\": float}]}\n\n"
        "Script:\n\"\"\"\n" + (text or "") + "\n\"\"\"\n"
    )


class LLMScorer(abc.ABC):
    """Abstract scorer. Implement :meth:`_call_model` for your client."""

    def __init__(self, model: str, cache: JsonlCache | None = None):
        self.model = model
        self.cache = cache

    @abc.abstractmethod
    def _call_model(self, prompt: str) -> dict:
        """Call the LLM and return the parsed JSON dict (scores/evidence/units).

        Wire this to your client (the existing pipeline uses gpt-4o). It should
        return a dict shaped like the JSON described in ``build_rubric_prompt``.
        """
        raise NotImplementedError

    def score(self, text: str, year_level: int | None = None) -> CriterionPrediction:
        if self.cache is not None:
            cached = self.cache.get(text)
            if cached is not None and "payload" in cached:
                return _to_prediction(cached["payload"])

        raw = self._call_model(build_rubric_prompt(text, year_level))
        pred = _to_prediction(raw)

        if self.cache is not None:
            self.cache.put(text, {"payload": {
                "scores": pred.scores,
                "evidence": pred.evidence,
                "units": [u.to_dict() for u in pred.units],
            }})
        return pred

    def predict(self, text: str, year_level: int | None = None, **kwargs) -> dict:
        """Score with the LLM, then run the fit layer (the full Mode-3 path)."""
        pred = self.score(text, year_level=year_level)
        return predict_criteria(
            text=text,
            year_level=year_level,
            criterion_scores=pred.scores,
            evidence=pred.evidence,
            **kwargs,
        )


def _to_prediction(raw: dict) -> CriterionPrediction:
    scores = {c: clamp(c, float(raw.get("scores", {}).get(c, 0))) for c in CRITERIA}
    evidence = {c: str(raw.get("evidence", {}).get(c, "")) for c in CRITERIA}
    units = [EvidenceUnit(**u) for u in raw.get("units", []) if isinstance(u, dict)]
    return CriterionPrediction(scores=scores, evidence=evidence, units=units)
