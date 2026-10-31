"""EV-002: bootstrap resampling and intervals."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from evalverdict.scoring.base import BootstrapConfig, Family, ItemValues, Scorer
from evalverdict.scoring.bootstrap import basic_interval, percentile_interval, resample_indices


class ValueScorer(Scorer):
    family = Family.GENERATION
    metrics = ("value",)

    def item_values(
        self, predictions: Sequence[Any], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        return ItemValues(arrays={"value": np.asarray(predictions, dtype=float)})


@pytest.mark.ticket("EV-002-T08")
def test_same_seed_gives_identical_intervals_and_different_seeds_differ() -> None:
    data = list(np.linspace(0, 1, 50))

    first = ValueScorer(seed=1).score(data, [0] * 50)[0].ci
    again = ValueScorer(seed=1).score(data, [0] * 50)[0].ci
    other = ValueScorer(seed=2).score(data, [0] * 50)[0].ci

    assert first == again
    assert first != other


@pytest.mark.ticket("EV-002-T11")
def test_percentile_interval_matches_numpy_quantiles() -> None:
    draws = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    low, high = percentile_interval(draws, level=0.90)

    assert low == pytest.approx(np.quantile(draws, 0.05))
    assert high == pytest.approx(np.quantile(draws, 0.95))


def test_basic_interval_reflects_percentiles_around_the_point() -> None:
    draws = np.array([0.4, 0.5, 0.6, 0.7, 0.8])

    low, high = basic_interval(draws, level=0.50, point=0.5)

    assert low == pytest.approx(2 * 0.5 - np.quantile(draws, 0.75))
    assert high == pytest.approx(2 * 0.5 - np.quantile(draws, 0.25))


def test_method_setting_selects_the_interval() -> None:
    data = list(np.linspace(0, 1, 40) ** 3)

    percentile = ValueScorer().score(data, [0] * 40)[0].ci
    basic = ValueScorer(bootstrap=BootstrapConfig(method="basic")).score(data, [0] * 40)[0].ci

    assert basic.method == "basic"
    assert (basic.low, basic.high) != (percentile.low, percentile.high)


@pytest.mark.ticket("EV-002-T12")
def test_cluster_resampling_keeps_variants_together_and_counts_sources() -> None:
    clusters = np.repeat(np.arange(10), 4)
    data = np.repeat(np.arange(10, dtype=float), 4)

    for idx in resample_indices(40, 200, seed=3, clusters=clusters):
        counts = np.bincount(clusters[idx], minlength=10)
        assert set(counts.tolist()) <= {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40}

    result = ValueScorer().score(list(data), [0] * 40, clusters=list(clusters))[0]
    assert result.n == 10


@pytest.mark.ticket("EV-002-T13")
def test_repeats_of_an_item_are_clustered_under_that_item() -> None:
    ids = [f"item-{index}" for index in range(10) for _ in range(3)]
    data = [0.1 * index for index in range(10) for _ in range(3)]

    result = ValueScorer().score(data, [0] * 30, ids=ids, clusters=ids)[0]

    assert result.n == 10


@pytest.mark.ticket("EV-002-T12")
def test_score_uses_clusters_so_correlated_variants_give_a_wider_interval() -> None:
    data = list(np.repeat(np.linspace(0, 1, 10), 10))
    clusters = list(np.repeat(np.arange(10), 10))

    naive = ValueScorer().score(data, [0] * 100)[0].ci
    clustered = ValueScorer().score(data, [0] * 100, clusters=clusters)[0].ci

    assert clustered.high - clustered.low > 2 * (naive.high - naive.low)


def test_no_clusters_resamples_with_replacement_at_full_size() -> None:
    draws = list(resample_indices(20, 5, seed=1))

    assert all(len(idx) == 20 for idx in draws)
    assert any(len(set(idx.tolist())) < 20 for idx in draws)


def test_cluster_list_of_wrong_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="clusters"):
        ValueScorer().score([0.1, 0.2], [0, 0], clusters=["a"])
