"""EV-002: the scorer contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.metrics import f1_score

from evalverdict.scoring.base import (
    FLAG_DEGENERATE,
    FLAG_SMALL_SAMPLE,
    FLAG_UNRELIABLE_CI,
    BootstrapConfig,
    ConfidenceInterval,
    Family,
    ItemValues,
    Scorer,
    ScoreResult,
    UndefinedMetricError,
)


class MeanScorer(Scorer):
    family = Family.GENERATION
    metrics = ("mean", "doubled")

    class Params(Scorer.Params):
        scale: float = 1.0

    def item_values(
        self, predictions: Sequence[Any], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        values = np.asarray(predictions, dtype=float) * self.params.scale
        return ItemValues(arrays={"mean": values, "doubled": values * 2})


class F1Scorer(Scorer):
    family = Family.CLASSIFICATION
    metrics = ("f1",)

    def item_values(
        self, predictions: Sequence[Any], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        return ItemValues(arrays={"pred": np.asarray(predictions), "true": np.asarray(references)})

    def aggregate(self, values: ItemValues, idx: np.ndarray, metric: str) -> float:
        return float(
            f1_score(values.arrays["true"][idx], values.arrays["pred"][idx], zero_division=0)
        )


def make_interval() -> ConfidenceInterval:
    return ConfidenceInterval(
        low=0.79, high=0.87, level=0.95, method="percentile", n_resamples=2000, n_undefined=0
    )


def make_result(**overrides: Any) -> ScoreResult:
    fields: dict[str, Any] = {
        "name": "mean",
        "family": Family.GENERATION,
        "value": 0.83,
        "ci": make_interval(),
        "n": 120,
        "higher_is_better": True,
        "per_item": (),
        "details": {},
    }
    return ScoreResult(**{**fields, **overrides})


@pytest.mark.ticket("EV-002-T01")
def test_score_result_cannot_be_built_without_an_interval() -> None:
    with pytest.raises(TypeError):
        ScoreResult(  # type: ignore[call-arg]
            name="x",
            family=Family.GENERATION,
            value=0.5,
            n=10,
            higher_is_better=True,
            per_item=(),
            details={},
        )


@pytest.mark.ticket("EV-002-T01")
def test_score_result_rejects_a_none_interval() -> None:
    with pytest.raises(TypeError):
        make_result(ci=None)


@pytest.mark.ticket("EV-002-T02")
def test_subclass_cannot_override_score() -> None:
    with pytest.raises(TypeError, match="score"):

        class Bad(MeanScorer):
            def score(self, *args: Any, **kwargs: Any) -> tuple[ScoreResult, ...]:
                return ()


@pytest.mark.ticket("EV-002-T03")
def test_subclass_must_declare_metrics() -> None:
    with pytest.raises(TypeError, match="metrics"):

        class Bad(Scorer):
            family = Family.GENERATION

            def item_values(self, predictions: Any, references: Any, metadata: Any) -> ItemValues:
                return ItemValues(arrays={})


@pytest.mark.ticket("EV-002-T03")
def test_subclass_must_implement_item_values() -> None:
    with pytest.raises(TypeError, match="item_values"):

        class Bad(Scorer):
            family = Family.GENERATION
            metrics = ("x",)


@pytest.mark.ticket("EV-002-T04")
@pytest.mark.parametrize(
    "bad",
    [
        {"n_resamples": 0},
        {"n_resamples": -5},
        {"level": 0.0},
        {"level": 1.0},
        {"level": 1.5},
        {"method": "none"},
        {"off": True},
        {"enabled": False},
    ],
)
def test_bootstrap_config_cannot_disable_or_break_the_interval(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BootstrapConfig(**bad)


@pytest.mark.ticket("EV-002-T05")
def test_unknown_param_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not_a_param"):
        MeanScorer(params={"not_a_param": 1})


def test_params_change_the_result() -> None:
    default = MeanScorer().score([0.1, 0.2, 0.3], [0, 0, 0])[0]
    scaled = MeanScorer(params={"scale": 2.0}).score([0.1, 0.2, 0.3], [0, 0, 0])[0]

    assert scaled.value == pytest.approx(default.value * 2)


def test_metric_subset_produces_only_those_metrics() -> None:
    results = MeanScorer(metrics=["doubled"]).score([0.1, 0.2], [0, 0])

    assert [result.name for result in results] == ["doubled"]


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonexistent"):
        MeanScorer(metrics=["nonexistent"])


def test_mismatched_predictions_and_references_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        MeanScorer().score([0.1, 0.2], [0])


@pytest.mark.ticket("EV-002-T06")
def test_string_form_matches_the_spec() -> None:
    assert str(make_result()) == "0.83 ±0.04 on n=120"


@pytest.mark.ticket("EV-002-T07")
def test_asymmetric_interval_keeps_exact_bounds() -> None:
    interval = ConfidenceInterval(
        low=0.70, high=0.90, level=0.95, method="percentile", n_resamples=2000, n_undefined=0
    )

    result = make_result(value=0.85, ci=interval)

    assert str(result) == "0.85 ±0.10 on n=120"
    assert (result.ci.low, result.ci.high) == (0.70, 0.90)


@pytest.mark.ticket("EV-002-T09")
def test_default_aggregate_is_the_mean_of_selected_items() -> None:
    scorer = MeanScorer()
    values = ItemValues(arrays={"mean": np.array([1.0, 2.0, 6.0])})

    assert scorer.aggregate(values, np.array([0, 2, 2]), "mean") == pytest.approx(13 / 3)


@pytest.mark.ticket("EV-002-T10")
def test_overridden_aggregate_is_recomputed_not_averaged() -> None:
    true = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0] * 4
    pred = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0] * 4

    result = F1Scorer().score(pred, true)[0]

    assert result.value == pytest.approx(f1_score(true, pred))
    assert result.ci.low < result.value < result.ci.high


@pytest.mark.ticket("EV-002-T14")
def test_empty_input_is_an_error() -> None:
    with pytest.raises(ValueError, match="no items"):
        MeanScorer().score([], [])


@pytest.mark.ticket("EV-002-T15")
def test_single_item_gives_a_flagged_degenerate_interval() -> None:
    result = MeanScorer().score([0.5], [0])[0]

    assert (result.ci.low, result.ci.high) == (0.5, 0.5)
    assert FLAG_DEGENERATE in result.flags


@pytest.mark.ticket("EV-002-T16")
def test_small_sample_flag_applies_below_thirty() -> None:
    small = MeanScorer().score([0.5] * 29, [0] * 29)[0]
    enough = MeanScorer().score([0.5] * 30, [0] * 30)[0]

    assert FLAG_SMALL_SAMPLE in small.flags
    assert FLAG_SMALL_SAMPLE not in enough.flags


@pytest.mark.ticket("EV-002-T17")
def test_nan_item_value_is_rejected_with_its_id() -> None:
    with pytest.raises(ValueError, match="item-b"):
        MeanScorer().score([0.1, float("nan")], [0, 0], ids=["item-a", "item-b"])


def _sometimes_undefined(every: int) -> type[Scorer]:
    class Sometimes(MeanScorer):
        calls: ClassVar[int] = 0

        def aggregate(self, values: ItemValues, idx: np.ndarray, metric: str) -> float:
            type(self).calls += 1
            if type(self).calls % every == 0:
                raise UndefinedMetricError("undefined on this resample")
            return super().aggregate(values, idx, metric)

    return Sometimes


@pytest.mark.ticket("EV-002-T18")
def test_undefined_resamples_are_counted_and_flagged_above_five_percent() -> None:
    config = BootstrapConfig(n_resamples=1000)
    data = [0.1, 0.5, 0.9] * 20

    over = _sometimes_undefined(16)(bootstrap=config, metrics=["mean"]).score(data, [0] * 60)[0]
    under = _sometimes_undefined(25)(bootstrap=config, metrics=["mean"]).score(data, [0] * 60)[0]

    assert over.ci.n_undefined > 50
    assert FLAG_UNRELIABLE_CI in over.flags
    assert 0 < under.ci.n_undefined < 50
    assert FLAG_UNRELIABLE_CI not in under.flags


@pytest.mark.ticket("EV-002-T19")
def test_metric_undefined_everywhere_is_an_error_not_a_fabricated_interval() -> None:
    class Never(MeanScorer):
        def aggregate(self, values: ItemValues, idx: np.ndarray, metric: str) -> float:
            return float("nan")

    with pytest.raises(UndefinedMetricError):
        Never().score([0.1, 0.2], [0, 0])


def test_resamples_undefined_but_point_defined_is_an_error() -> None:
    class OnlyWhole(MeanScorer):
        def aggregate(self, values: ItemValues, idx: np.ndarray, metric: str) -> float:
            if len(set(idx.tolist())) != len(idx):
                raise UndefinedMetricError("repeats")
            return super().aggregate(values, idx, metric)

    with pytest.raises(UndefinedMetricError):
        OnlyWhole(bootstrap=BootstrapConfig(n_resamples=20)).score(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0] * 10
        )


@pytest.mark.ticket("EV-002-T20")
def test_bootstrap_settings_are_used_and_recorded() -> None:
    scorer = MeanScorer(bootstrap=BootstrapConfig(level=0.90, n_resamples=500))

    interval = scorer.score([0.1, 0.5, 0.9] * 10, [0] * 30)[0].ci

    assert (interval.level, interval.n_resamples) == (0.90, 500)


def test_per_item_scores_carry_ids_and_values() -> None:
    result = MeanScorer(metrics=["mean"]).score([0.25, 0.75], [0, 0], ids=["a", "b"])[0]

    assert [(item.item_id, item.value) for item in result.per_item] == [("a", 0.25), ("b", 0.75)]


def test_lower_is_better_metric_is_recorded() -> None:
    class Latency(MeanScorer):
        lower_is_better = frozenset({"mean"})

    results = Latency().score([0.1, 0.2], [0, 0])

    assert {r.name: r.higher_is_better for r in results} == {"mean": False, "doubled": True}
