"""EV-002: intervals must cover the truth at close to their stated rate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from evalverdict.scoring.base import BootstrapConfig, Family, ItemValues, Scorer
from evalverdict.scoring.bootstrap import percentile_interval, resample_indices

DATASETS = 2000
RESAMPLES = 400


class BernoulliMean(Scorer):
    family = Family.CLASSIFICATION
    metrics = ("accuracy",)

    def item_values(
        self, predictions: Sequence[Any], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        return ItemValues(arrays={"accuracy": np.asarray(predictions, dtype=float)})


class BinaryF1(Scorer):
    family = Family.CLASSIFICATION
    metrics = ("f1",)

    def item_values(
        self, predictions: Sequence[Any], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        return ItemValues(arrays={"pred": np.asarray(predictions), "true": np.asarray(references)})

    def aggregate(self, values: ItemValues, idx: np.ndarray, metric: str) -> float:
        pred, true = values.arrays["pred"][idx], values.arrays["true"][idx]
        true_positive = int(np.sum((pred == 1) & (true == 1)))
        wrong = int(np.sum(pred != true))
        return 2 * true_positive / (2 * true_positive + wrong) if true_positive else 0.0


@pytest.mark.statistical
@pytest.mark.ticket("EV-002-T21")
def test_interval_for_a_mean_covers_the_truth_at_about_95_percent() -> None:
    rng = np.random.default_rng(101)
    config = BootstrapConfig(n_resamples=RESAMPLES)
    truth = 0.7
    covered = 0

    for dataset in range(DATASETS):
        sample = (rng.random(120) < truth).astype(float)
        interval = BernoulliMean(bootstrap=config, seed=dataset).score(sample, sample)[0].ci
        covered += interval.low <= truth <= interval.high

    assert 0.93 <= covered / DATASETS <= 0.97


@pytest.mark.statistical
@pytest.mark.ticket("EV-002-T22")
def test_interval_for_f1_covers_the_truth_at_about_95_percent() -> None:
    rng = np.random.default_rng(202)
    config = BootstrapConfig(n_resamples=RESAMPLES)
    truth = 0.8
    covered = 0

    for dataset in range(DATASETS):
        true = (rng.random(200) < 0.5).astype(int)
        flipped = rng.random(200) < 0.2
        pred = np.where(flipped, 1 - true, true)
        interval = BinaryF1(bootstrap=config, seed=dataset).score(pred, true)[0].ci
        covered += interval.low <= truth <= interval.high

    assert 0.93 <= covered / DATASETS <= 0.97


@pytest.mark.statistical
@pytest.mark.ticket("EV-002-T23")
def test_cluster_resampling_covers_the_truth_and_naive_resampling_does_not() -> None:
    rng = np.random.default_rng(303)
    sources, variants = 60, 4
    clusters = np.repeat(np.arange(sources), variants)
    total = sources * variants
    cluster_hits = naive_hits = 0

    for dataset in range(DATASETS):
        effects = np.repeat(rng.normal(0, 1, sources), variants)
        sample = effects + rng.normal(0, 0.3, total)
        for use_clusters in (True, False):
            draws = np.array(
                [
                    sample[idx].mean()
                    for idx in resample_indices(
                        total, RESAMPLES, seed=dataset, clusters=clusters if use_clusters else None
                    )
                ]
            )
            low, high = percentile_interval(draws, level=0.95)
            if use_clusters:
                cluster_hits += low <= 0 <= high
            else:
                naive_hits += low <= 0 <= high

    assert 0.93 <= cluster_hits / DATASETS <= 0.97
    assert naive_hits / DATASETS < 0.80
