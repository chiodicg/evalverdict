"""Threshold gates. By default a gate needs the whole interval to clear it (D5)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from evalverdict.config import GateConfig

PASS = "pass"
FAIL = "fail"
INCONCLUSIVE = "inconclusive"
INVALID = "invalid"
EXIT_CODES = {PASS: 0, FAIL: 1, INCONCLUSIVE: 2, INVALID: 3}
_SEVERITY = {PASS: 0, INCONCLUSIVE: 1, FAIL: 2, INVALID: 3}


def _check_bound(low: float, high: float, threshold: float, is_minimum: bool) -> str:
    if is_minimum:
        return PASS if low >= threshold else FAIL if high < threshold else INCONCLUSIVE
    return PASS if high <= threshold else FAIL if low > threshold else INCONCLUSIVE


def evaluate_gate(result: Mapping[str, Any] | None, gate: GateConfig) -> str:
    """Compare one result with its thresholds. A metric with no value is inconclusive."""
    if result is None or "value" not in result:
        return INCONCLUSIVE
    if gate.gate == "point":
        low = high = float(result["value"])
    else:
        low, high = float(result["ci"]["low"]), float(result["ci"]["high"])
    checks = []
    if gate.min is not None:
        checks.append(_check_bound(low, high, gate.min, is_minimum=True))
    if gate.max is not None:
        checks.append(_check_bound(low, high, gate.max, is_minimum=False))
    return overall_outcome(checks)


def overall_outcome(outcomes: Iterable[str]) -> str:
    """Any failure fails, any doubt is inconclusive, otherwise pass. No gates is a pass."""
    return worst_outcome(list(outcomes) or [PASS])


def worst_outcome(outcomes: Iterable[str]) -> str:
    return max(outcomes, key=_SEVERITY.__getitem__)


def apply_gates(
    gates: Mapping[str, GateConfig], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {f"{result['scorer']}.{result['name']}": result for result in results}
    return [
        {
            "key": key,
            "outcome": evaluate_gate(by_key.get(key), gate),
            **gate.model_dump(),
        }
        for key, gate in gates.items()
    ]
