"""EV-003: the scorer registry and scorer entries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from evalverdict.config import ScorerEntry
from evalverdict.scoring.base import BootstrapConfig, Family, ItemValues, Scorer
from evalverdict.scoring.registry import build_scorer, register_scorer, scorer_types


class _Base(Scorer):
    family = Family.GENERATION
    metrics = ("a", "b")

    class Params(Scorer.Params):
        scale: float = 1.0

    def item_values(
        self, predictions: Sequence[Any], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        array = np.asarray(predictions, dtype=float) * self.params.scale
        return ItemValues(arrays={"a": array, "b": array})


register_scorer("registry_test")(_Base)


@pytest.mark.ticket("EV-003-T08")
def test_registering_a_type_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="registry_test"):
        register_scorer("registry_test")(_Base)


@pytest.mark.ticket("EV-003-T09")
def test_unknown_type_lists_the_valid_types() -> None:
    with pytest.raises(ValueError, match="rouge"):
        build_scorer(ScorerEntry(type="nope"), BootstrapConfig(), seed=0)


def test_builtin_scorers_are_registered() -> None:
    assert "rouge" in scorer_types()


@pytest.mark.ticket("EV-003-T12")
def test_unknown_metric_lists_the_valid_metrics() -> None:
    with pytest.raises(ValueError, match="rouge1"):
        build_scorer(ScorerEntry(type="rouge", metrics=["nope"]), BootstrapConfig(), seed=0)


@pytest.mark.ticket("EV-003-T15")
def test_per_scorer_bootstrap_override_applies_to_that_scorer_only() -> None:
    run_default = BootstrapConfig(n_resamples=300)

    overridden = build_scorer(
        ScorerEntry(type="registry_test", bootstrap={"n_resamples": 100}), run_default, seed=0
    )
    inherited = build_scorer(ScorerEntry(type="registry_test"), run_default, seed=0)

    assert overridden.bootstrap.n_resamples == 100
    assert inherited.bootstrap.n_resamples == 300
