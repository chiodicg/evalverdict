"""EV-007: latency, throughput and cost."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from evalverdict.data import Prediction
from evalverdict.scorers.operational import CostScorer, LatencyScorer
from evalverdict.scoring.base import NotMeasured, Scorer, ScoreResult

LATENCIES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 3.0]


def predictions(**fields: list[float | None]) -> list[Prediction]:
    count = len(next(iter(fields.values())))
    return [
        Prediction(case_id=str(index), **{key: values[index] for key, values in fields.items()})
        for index in range(count)
    ]


def score(scorer: Scorer, items: list[Prediction]) -> dict[str, ScoreResult | NotMeasured]:
    return {
        result.name: result
        for result in scorer.score(items, [None] * len(items), [{}] * len(items))
    }


@pytest.mark.ticket("EV-007-T08")
def test_latency_percentiles_match_numpy() -> None:
    results = score(LatencyScorer(), predictions(latency_s=LATENCIES))

    assert results["latency_p50_s"].value == pytest.approx(np.percentile(LATENCIES, 50))
    assert results["latency_p95_s"].value == pytest.approx(np.percentile(LATENCIES, 95))
    assert results["latency_p50_s"].higher_is_better is False


@pytest.mark.ticket("EV-007-T08")
def test_latency_percentiles_carry_intervals() -> None:
    result = score(LatencyScorer(), predictions(latency_s=LATENCIES))["latency_p50_s"]

    assert result.ci.low < result.value < result.ci.high


@pytest.mark.ticket("EV-007-T09")
def test_configured_percentiles_decide_the_metrics() -> None:
    scorer = LatencyScorer(params={"percentiles": [50, 90, 99]})

    names = set(score(scorer, predictions(latency_s=LATENCIES)))

    assert {"latency_p50_s", "latency_p90_s", "latency_p99_s"} <= names
    assert "latency_p95_s" not in names


@pytest.mark.ticket("EV-007-T10")
@pytest.mark.parametrize("bad", [-1, 101])
def test_percentile_outside_range_is_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        LatencyScorer(params={"percentiles": [bad]})


@pytest.mark.ticket("EV-007-T11")
def test_ttft_percentiles_on_known_values() -> None:
    results = score(LatencyScorer(), predictions(ttft_s=LATENCIES))

    assert results["ttft_p95_s"].value == pytest.approx(np.percentile(LATENCIES, 95))


@pytest.mark.ticket("EV-007-T12")
def test_throughput_is_the_mean_and_higher_is_better() -> None:
    results = score(LatencyScorer(), predictions(throughput_tokens_per_s=[10.0, 20.0, 60.0]))

    assert results["throughput_tokens_per_s"].value == pytest.approx(30)
    assert results["throughput_tokens_per_s"].higher_is_better is True


@pytest.mark.ticket("EV-007-T13")
def test_cost_per_prediction_shows_its_source() -> None:
    items = [
        Prediction(case_id="a", cost_usd=0.01, cost_source="provider"),
        Prediction(case_id="b", cost_usd=0.03, cost_source="provider"),
    ]

    result = score(CostScorer(), items)["cost_per_prediction_usd"]

    assert result.value == pytest.approx(0.02)
    assert result.details["cost_sources"] == ["provider"]
    assert result.higher_is_better is False


@pytest.mark.ticket("EV-007-T14")
def test_prediction_with_no_ttft_is_not_measured_rather_than_zero() -> None:
    result = score(LatencyScorer(), predictions(latency_s=LATENCIES))["ttft_p50_s"]

    assert isinstance(result, NotMeasured)
    assert not hasattr(result, "value")


@pytest.mark.ticket("EV-007-T15")
def test_items_without_cost_are_not_treated_as_zero() -> None:
    items = predictions(cost_usd=[0.1, None, 0.3, None])

    result = score(CostScorer(), items)["cost_per_prediction_usd"]

    assert result.value == pytest.approx(0.2)
    assert result.n == 2
    assert result.details["n_items"] == 4


def test_metric_subset_selects_one_measure() -> None:
    results = LatencyScorer(metrics=["latency_p50_s"]).score(
        predictions(latency_s=LATENCIES), [None] * 10
    )

    assert [result.name for result in results] == ["latency_p50_s"]
