"""Claude scoring backend — produces a mark for a script.

The production path (Mode 3): Claude reads a script and returns per-criterion
scores + evidence, which the engine sums to a tentative total (the mark) and
runs through the fit layer.

"Trained on the data" is achieved two ways (you cannot fine-tune Claude itself):
  * **in-context calibration** — a sample of the human-marked gold scripts is
    embedded in a cached system prompt as exemplars (see ``load_gold_exemplars``);
  * **a calibration layer** — a :class:`~criteria_engine.calibration.GoldCalibrator`
    fit on your gold marks, passed as ``calibrator=`` and applied to the model's
    scores in :meth:`predict`.

How it's wired (per the Claude API skill):
  * model: ``claude-sonnet-4-6`` by default (``SonnetScorer``); ``HaikuScorer``
    pins Haiku 4.5. The model is configurable via ``model=``.
  * structured outputs: each criterion is constrained to its exact integer
    range via a json_schema ``enum``, so the model cannot return an out-of-range
    or non-integer score.
  * prompt caching: the rubric + exemplars are a stable prefix carried in
    ``system`` with a ``cache_control`` breakpoint; the per-script text is the
    volatile suffix (~0.1x input cost on cache reads).
  * adaptive thinking is enabled on the models that support it (Sonnet 4.6 /
    Opus 4.x) and off on Haiku 4.5, which doesn't take the parameter.

The Anthropic SDK is imported lazily, so the core engine stays dependency-free.
Install with ``pip install anthropic`` to use this backend.

Cache shape matches the handover's ``aes_precision_cache.jsonl``:
  * cache key  : ``"{version}|{model}|{sha1(text)}"``
  * evidence   : units of ``{label, start, end, text, confidence}``
"""

from __future__ import annotations

import abc
import csv
import hashlib
import json
import os
from dataclasses import dataclass, asdict

from .ranges import CRITERIA, FULL_NAMES, RANGES, clamp
from .priors import BAND_PROFILE, total_band
from .engine import predict_criteria

CACHE_VERSION = "criterion_scores_v1"
DEFAULT_MODEL = "claude-sonnet-4-6"


def _supports_thinking(model: str) -> bool:
    """Adaptive thinking is available on Sonnet 4.6 and the Opus 4.x models."""
    return "sonnet-4-6" in model or "opus-4" in model


# --- Structured-output schema --------------------------------------------

def _response_schema() -> dict:
    """json_schema that forces in-range integer scores plus evidence."""
    score_props = {
        c: {"type": "integer", "enum": list(range(RANGES[c] + 1))}
        for c in CRITERIA
    }
    evidence_props = {c: {"type": "string"} for c in CRITERIA}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["scores", "evidence", "units"],
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "required": list(CRITERIA),
                "properties": score_props,
            },
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "required": list(CRITERIA),
                "properties": evidence_props,
            },
            "units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["label", "start", "end", "text", "confidence"],
                    "properties": {
                        "label": {"type": "string"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "text": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            },
        },
    }


RESPONSE_SCHEMA = _response_schema()


# --- Evidence / prediction containers ------------------------------------

@dataclass
class EvidenceUnit:
    """A span of evidence, mirroring the handover precision-cache unit schema."""
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


# --- JSONL cache (handover-compatible key shape) -------------------------

class JsonlCache:
    """Append-only JSONL cache keyed like the handover precision cache."""

    def __init__(self, path: str, model: str = DEFAULT_MODEL, version: str = CACHE_VERSION):
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


# --- Exemplars: "training on the data" by in-context calibration ----------

def _truncate(text: str, words: int = 130) -> str:
    parts = (text or "").split()
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


def load_gold_exemplars(
    csv_path: str, per_band: int = 1, min_word_count: int = 30
) -> list[dict]:
    """Pick a spread of human-marked exemplars from the gold CSV.

    Reads the handover's ``gold_clean_365_with_year_proxy.csv`` (columns
    ``Research ID, AU, TS, ID, CS/PD, Voc, Coh, Pa, SS, Pun, Spell, Total,
    WordCount, Raw text``) and returns up to ``per_band`` prose exemplars per
    total band, so Haiku sees the full mark range. Non-prose / very short
    responses are skipped.
    """
    by_band: dict[str, list[dict]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                wc = int(float(row.get("WordCount", 0) or 0))
                total = int(float(row.get("Total", 0) or 0))
                profile = {c: int(float(row[c])) for c in CRITERIA}
            except (TypeError, ValueError, KeyError):
                continue
            text = (row.get("Raw text") or "").strip()
            if wc < min_word_count or not text:
                continue
            band = total_band(total)
            bucket = by_band.setdefault(band, [])
            if len(bucket) < per_band:
                bucket.append({
                    "id": row.get("Research ID", ""),
                    "total": total,
                    "scores": profile,
                    "text": text,
                })
    out: list[dict] = []
    for band in BAND_PROFILE:  # canonical band order, low to high
        out.extend(by_band.get(band, []))
    return out


# --- Prompt construction (system = stable/cacheable, user = volatile) -----

def build_rubric_system_prompt(exemplars: list[dict] | None = None) -> str:
    """Stable, cacheable instructions: ranges, band anchors, and exemplars."""
    ranges = "\n".join(f"  - {c} ({FULL_NAMES[c]}): 0-{RANGES[c]}" for c in CRITERIA)
    anchors = "\n".join(
        f"  total {band}: " + ", ".join(f"{c}={BAND_PROFILE[band][c]}" for c in CRITERIA)
        for band in BAND_PROFILE
    )
    parts = [
        "You are an expert NAPLAN-style narrative writing marker. Mark the "
        "student script against the rubric. Score EACH criterion as an integer "
        "STRICTLY within its range, give a one-line rationale per criterion "
        "grounded in the text, and list any short evidence spans.\n",
        f"Criteria and ranges:\n{ranges}\n",
        "Typical (not mandatory) profiles by total band, for internal "
        f"consistency:\n{anchors}\n",
        "The total mark is the sum of the ten criterion scores (0-47); do not "
        "report a separate total — it is computed from your criterion scores.\n",
    ]
    if exemplars:
        ex_lines = ["Human-marked exemplars (calibrate your marking to these):"]
        for ex in exemplars:
            sc = ", ".join(f"{c}={ex['scores'][c]}" for c in CRITERIA)
            ex_lines.append(
                f"\n[Exemplar {ex['id']} | total {ex['total']}] {sc}\n"
                f"Script: \"{_truncate(ex['text'])}\""
            )
        parts.append("\n".join(ex_lines) + "\n")
    return "\n".join(parts)


def build_user_message(text: str, year_level: int | None = None) -> str:
    yl = f"Year level: {year_level}\n" if year_level is not None else ""
    return (
        f"{yl}Mark this script. Return JSON only.\n\nScript:\n\"\"\"\n"
        + (text or "") + "\n\"\"\"\n"
    )


# --- Scorers --------------------------------------------------------------

class LLMScorer(abc.ABC):
    """Abstract scorer; implement :meth:`_call_model` for a provider."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache: JsonlCache | None = None,
        exemplars: list[dict] | None = None,
        calibrator=None,
    ):
        self.model = model
        self.cache = cache
        self.exemplars = exemplars
        self.calibrator = calibrator  # optional GoldCalibrator trained on gold
        self._system = build_rubric_system_prompt(exemplars)

    @abc.abstractmethod
    def _call_model(self, system: str, user: str) -> dict:
        """Return the parsed JSON dict (scores/evidence/units)."""
        raise NotImplementedError

    def score(self, text: str, year_level: int | None = None) -> CriterionPrediction:
        if self.cache is not None:
            cached = self.cache.get(text)
            if cached is not None and "payload" in cached:
                return _to_prediction(cached["payload"])

        raw = self._call_model(self._system, build_user_message(text, year_level))
        pred = _to_prediction(raw)

        if self.cache is not None:
            self.cache.put(text, {"payload": {
                "scores": pred.scores,
                "evidence": pred.evidence,
                "units": [u.to_dict() for u in pred.units],
            }})
        return pred

    def predict(self, text: str, year_level: int | None = None, **kwargs) -> dict:
        """Score with the LLM, then run the fit layer; returns the full result
        including ``tentative_total`` (the mark)."""
        pred = self.score(text, year_level=year_level)
        scores = self.calibrator.apply(pred.scores) if self.calibrator else pred.scores
        return predict_criteria(
            text=text,
            year_level=year_level,
            criterion_scores=scores,
            evidence=pred.evidence,
            **kwargs,
        )


class ClaudeScorer(LLMScorer):
    """Concrete scorer calling Claude (defaults to Sonnet 4.6).

    Example
    -------
        from criteria_engine.llm_backend import ClaudeScorer, JsonlCache, load_gold_exemplars
        from criteria_engine.calibration import GoldCalibrator

        exemplars = load_gold_exemplars("gold_clean_365_with_year_proxy.csv", per_band=3)
        scorer = ClaudeScorer(
            exemplars=exemplars,
            calibrator=GoldCalibrator.load("calibrator.json"),  # trained on gold
            cache=JsonlCache("aes_sonnet_cache.jsonl"),
        )
        result = scorer.predict(my_script)   # result["tentative_total"] is the mark
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        cache: JsonlCache | None = None,
        exemplars: list[dict] | None = None,
        calibrator=None,
        client=None,
        max_tokens: int = 4096,
        use_thinking: bool | None = None,
    ):
        super().__init__(model=model, cache=cache, exemplars=exemplars, calibrator=calibrator)
        self._client = client
        self.max_tokens = max_tokens
        # Adaptive thinking on the models that support it (Sonnet 4.6 / Opus 4.x),
        # off on Haiku 4.5 which doesn't take the parameter.
        self.use_thinking = _supports_thinking(model) if use_thinking is None else use_thinking
        if cache is not None:
            cache.model = model  # keep cache key model in sync

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy
            self._client = anthropic.Anthropic()
        return self._client

    def _call_model(self, system: str, user: str) -> dict:
        client = self._get_client()
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            # Stable rubric + exemplars cached; the script (in `user`) is volatile.
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
            # Structured outputs: schema-valid, in-range integer scores.
            output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        )
        if self.use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        resp = client.messages.create(**kwargs)
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)


class SonnetScorer(ClaudeScorer):
    """ClaudeScorer pinned to Sonnet 4.6 (adaptive thinking on)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("model", "claude-sonnet-4-6")
        super().__init__(**kwargs)


class HaikuScorer(ClaudeScorer):
    """ClaudeScorer pinned to Haiku 4.5 (thinking off)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("model", "claude-haiku-4-5")
        super().__init__(**kwargs)


class OpusScorer(ClaudeScorer):
    """ClaudeScorer pinned to Opus 4.8 (most capable; adaptive thinking on)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("model", "claude-opus-4-8")
        super().__init__(**kwargs)


def _to_prediction(raw: dict) -> CriterionPrediction:
    scores = {c: clamp(c, float(raw.get("scores", {}).get(c, 0))) for c in CRITERIA}
    evidence = {c: str(raw.get("evidence", {}).get(c, "")) for c in CRITERIA}
    units = [EvidenceUnit(**u) for u in raw.get("units", []) if isinstance(u, dict)]
    return CriterionPrediction(scores=scores, evidence=evidence, units=units)
