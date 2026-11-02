"""EV-009: every registered scorer honours the contract. New scorers are covered
automatically, and must supply a contract example to pass."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_core import PydanticUndefined

from evalverdict.scoring.base import BootstrapConfig, NotMeasured, Scorer, ScoreResult
from evalverdict.scoring.registry import registered_scorers
from fakes.judge import StubJudge

# Built-in scorers only; test doubles registered by other test modules are not shipped.
SCORERS = {
    name: cls
    for name, cls in registered_scorers().items()
    if cls.__module__.startswith("evalverdict.")
}
FAST = BootstrapConfig(n_resamples=50)


def build(cls: type[Scorer], params: dict[str, Any], metrics: list[str] | None = None) -> Scorer:
    scorer = cls(params, bootstrap=FAST, metrics=metrics)
    if cls.needs_judge:
        scorer.bind_judge(StubJudge(scores=[0.6] * 100), "judge/model", [])  # type: ignore[attr-defined]
    return scorer


def run_example(scorer: Scorer) -> tuple[ScoreResult | NotMeasured, ...]:
    example = type(scorer).contract_example()
    return scorer.score(example.predictions, example.references, example.metadata)


@pytest.mark.ticket("EV-009-T26")
@pytest.mark.parametrize("scorer_type", sorted(SCORERS))
def test_default_params_produce_results_with_intervals(scorer_type: str) -> None:
    cls = SCORERS[scorer_type]
    results = run_example(build(cls, cls.contract_example().params))

    assert results
    for result in results:
        if isinstance(result, ScoreResult):
            assert (
                result.ci.low <= result.value <= result.ci.high or result.ci.low == result.ci.high
            )
            assert result.n > 0


@pytest.mark.ticket("EV-009-T26")
@pytest.mark.parametrize("scorer_type", sorted(SCORERS))
def test_unknown_param_is_rejected(scorer_type: str) -> None:
    cls = SCORERS[scorer_type]

    with pytest.raises(ValidationError):
        cls({**cls.contract_example().params, "definitely_not_a_param": 1})


@pytest.mark.ticket("EV-009-T26")
@pytest.mark.parametrize(
    ("scorer_type", "metric"),
    [
        (name, metric)
        for name, cls in sorted(SCORERS.items())
        for metric in cls(cls.contract_example().params).available_metrics()
    ],
)
def test_each_metric_can_be_selected_alone(scorer_type: str, metric: str) -> None:
    cls = SCORERS[scorer_type]

    results = run_example(build(cls, cls.contract_example().params, metrics=[metric]))

    assert [result.name for result in results] == [metric]


@pytest.mark.ticket("EV-009-T27")
@pytest.mark.parametrize("scorer_type", sorted(SCORERS))
def test_every_param_has_a_default_or_is_explicitly_required(scorer_type: str) -> None:
    for name, field in SCORERS[scorer_type].Params.model_fields.items():
        has_default = field.default is not PydanticUndefined or field.default_factory is not None
        assert field.is_required() or has_default, name


@pytest.mark.ticket("EV-007-T17")
@pytest.mark.ticket("EV-008-T20")
def test_contract_covers_the_milestone_scorers() -> None:
    assert {"rouge", "bertscore", "latency", "cost", "judge"} <= set(SCORERS)
