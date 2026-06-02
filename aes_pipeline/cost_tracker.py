"""
cost_tracker.py
===============
Lightweight token-usage tracker for the AES pipeline.

Usage:
    from cost_tracker import tracker
    tracker.record("correct", response.usage, model="claude-opus-4-7")

    # At the end of a run:
    tracker.print_summary()

Pricing is hardcoded to Anthropic list prices as of June 2026.
Cache read/write tokens are tracked separately where available.
"""

from dataclasses import dataclass, field
from typing import Optional
import threading


# ---------------------------------------------------------------------------
# Pricing table — USD per million tokens
# ---------------------------------------------------------------------------
_PRICING = {
    # Opus 4
    "claude-opus-4-7":           {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    "claude-opus-4-6":           {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    # Sonnet 4
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00, "cache_write":  3.75, "cache_read": 0.30},
    "claude-sonnet-4-5":         {"input":  3.00, "output": 15.00, "cache_write":  3.75, "cache_read": 0.30},
    # Haiku 4
    "claude-haiku-4-5-20251001": {"input":  0.80, "output":  4.00, "cache_write":  1.00, "cache_read": 0.08},
    "claude-haiku-4-5":          {"input":  0.80, "output":  4.00, "cache_write":  1.00, "cache_read": 0.08},
}

_DEFAULT_PRICING = {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30}


def _price(model: str) -> dict:
    # Exact match first; then prefix match (handles dated suffixes)
    if model in _PRICING:
        return _PRICING[model]
    for key, val in _PRICING.items():
        if model.startswith(key) or key.startswith(model):
            return val
    return _DEFAULT_PRICING


# ---------------------------------------------------------------------------
# Usage record
# ---------------------------------------------------------------------------
@dataclass
class _StageUsage:
    stage: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        p = _price(self.model)
        return (
            self.input_tokens       * p["input"]       / 1_000_000 +
            self.output_tokens      * p["output"]      / 1_000_000 +
            self.cache_write_tokens * p["cache_write"] / 1_000_000 +
            self.cache_read_tokens  * p["cache_read"]  / 1_000_000
        )


# ---------------------------------------------------------------------------
# Thread-safe tracker
# ---------------------------------------------------------------------------
class CostTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._stages: dict[str, _StageUsage] = {}

    def record(self, stage: str, usage, model: str):
        """
        Record usage from one API response.

        usage: anthropic Usage object (has .input_tokens, .output_tokens,
               optionally .cache_creation_input_tokens, .cache_read_input_tokens)
        """
        if usage is None:
            return
        inp  = getattr(usage, "input_tokens", 0) or 0
        out  = getattr(usage, "output_tokens", 0) or 0
        cw   = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cr   = getattr(usage, "cache_read_input_tokens", 0) or 0

        with self._lock:
            key = f"{stage}:{model}"
            if key not in self._stages:
                self._stages[key] = _StageUsage(stage=stage, model=model)
            s = self._stages[key]
            s.calls += 1
            s.input_tokens       += inp
            s.output_tokens      += out
            s.cache_write_tokens += cw
            s.cache_read_tokens  += cr

    def print_summary(self):
        with self._lock:
            if not self._stages:
                print("  [cost] No API calls recorded.")
                return

            print("\n=== API COST SUMMARY ===\n")
            header = f"  {'Stage':<22} {'Model':<28} {'Calls':>6} {'In tok':>9} {'Out tok':>9} {'Cache W':>9} {'Cache R':>9} {'Cost USD':>10}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            total_cost = 0.0
            for key in sorted(self._stages):
                s = self._stages[key]
                cost = s.cost_usd
                total_cost += cost
                print(
                    f"  {s.stage:<22} {s.model:<28} {s.calls:>6,} "
                    f"{s.input_tokens:>9,} {s.output_tokens:>9,} "
                    f"{s.cache_write_tokens:>9,} {s.cache_read_tokens:>9,} "
                    f"${cost:>9.4f}"
                )

            print("  " + "-" * (len(header) - 2))
            print(f"  {'TOTAL':<22} {'':<28} {'':>6} {'':>9} {'':>9} {'':>9} {'':>9} ${total_cost:>9.4f}")
            print()

    def total_cost(self) -> float:
        with self._lock:
            return sum(s.cost_usd for s in self._stages.values())


# Module-level singleton — import this everywhere
tracker = CostTracker()
