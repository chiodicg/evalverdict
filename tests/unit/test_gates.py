"""EV-009: threshold gates and exit codes."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from evalverdict.config import GateConfig
from evalverdict.scoring.gates import (
    EXIT_CODES,
    FAIL,
    INCONCLUSIVE,
    INVALID,
    PASS,
    evaluate_gate,
    overall_outcome,
    worst_outcome,
)


def result(value: float, low: float, high: float) -> dict[str, Any]:
    return {"value": value, "ci": {"low": low, "high": high}}


NOT_MEASURED = {"name": "ttft_p50_s", "reason": "no item has this measurement"}


@pytest.mark.ticket("EV-009-T01")
def test_whole_interval_above_the_minimum_passes() -> None:
    assert evaluate_gate(result(0.8, 0.75, 0.85), GateConfig(min=0.7)) == PASS


@pytest.mark.ticket("EV-009-T02")
def test_whole_interval_below_the_minimum_fails() -> None:
    assert evaluate_gate(result(0.6, 0.55, 0.65), GateConfig(min=0.7)) == FAIL


@pytest.mark.ticket("EV-009-T03")
def test_interval_straddling_the_minimum_is_inconclusive() -> None:
    assert evaluate_gate(result(0.72, 0.65, 0.79), GateConfig(min=0.7)) == INCONCLUSIVE


@pytest.mark.ticket("EV-009-T04")
@pytest.mark.parametrize(
    ("values", "expected"),
    [((1.5, 1.2, 1.8), PASS), ((2.5, 2.2, 2.8), FAIL), ((2.0, 1.7, 2.3), INCONCLUSIVE)],
)
def test_maximum_mirrors_the_minimum(values: tuple[float, float, float], expected: str) -> None:
    assert evaluate_gate(result(*values), GateConfig(max=2.0)) == expected


@pytest.mark.ticket("EV-009-T05")
def test_bound_exactly_on_the_threshold_counts_as_clearing_it() -> None:
    assert evaluate_gate(result(0.75, 0.7, 0.8), GateConfig(min=0.7)) == PASS
    assert evaluate_gate(result(1.8, 1.5, 2.0), GateConfig(max=2.0)) == PASS


@pytest.mark.ticket("EV-009-T06")
def test_point_mode_ignores_the_interval() -> None:
    straddling = result(0.72, 0.65, 0.79)

    assert evaluate_gate(straddling, GateConfig(min=0.7, gate="point")) == PASS
    assert evaluate_gate(result(0.68, 0.6, 0.75), GateConfig(min=0.7, gate="point")) == FAIL


@pytest.mark.ticket("EV-009-T07")
def test_min_and_max_both_apply() -> None:
    band = GateConfig(min=0.2, max=0.8)

    assert evaluate_gate(result(0.5, 0.4, 0.6), band) == PASS
    assert evaluate_gate(result(0.9, 0.85, 0.95), band) == FAIL


@pytest.mark.ticket("EV-009-T08")
@pytest.mark.parametrize("bad", [{}, {"min": 0.9, "max": 0.1}, {"min": 0.5, "gate": "maybe"}])
def test_gate_without_a_threshold_or_with_min_above_max_is_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GateConfig(**bad)


@pytest.mark.ticket("EV-009-T09")
def test_not_measured_is_inconclusive_never_a_pass() -> None:
    assert evaluate_gate(NOT_MEASURED, GateConfig(min=0.0)) == INCONCLUSIVE
    assert evaluate_gate(NOT_MEASURED, GateConfig(max=10.0, gate="point")) == INCONCLUSIVE


@pytest.mark.ticket("EV-009-T12")
@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [([PASS, PASS], PASS), ([PASS, INCONCLUSIVE], INCONCLUSIVE), ([INCONCLUSIVE, FAIL], FAIL)],
)
def test_overall_outcome_is_the_worst_gate(outcomes: list[str], expected: str) -> None:
    assert overall_outcome(outcomes) == expected


def test_no_gates_passes() -> None:
    assert overall_outcome([]) == PASS


@pytest.mark.ticket("EV-009-T12")
def test_exit_codes_match_the_spec() -> None:
    assert EXIT_CODES == {PASS: 0, FAIL: 1, INCONCLUSIVE: 2, INVALID: 3}


@pytest.mark.ticket("EV-009-T13")
def test_with_several_targets_the_worst_result_wins() -> None:
    assert worst_outcome([PASS, FAIL]) == FAIL
    assert worst_outcome([INCONCLUSIVE, FAIL, PASS]) == FAIL
    assert worst_outcome([PASS, INVALID, FAIL]) == INVALID
