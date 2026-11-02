"""EV-010: the paired comparison must have the error rates it claims."""

from __future__ import annotations

import numpy as np
import pytest

from evalverdict.scoring.base import BootstrapConfig
from evalverdict.scoring.compare import paired_interval

DATASETS = 2000
CONFIG = BootstrapConfig(n_resamples=400)
ITEMS = 120


def simulate(rng: np.random.Generator, shift: float) -> tuple[np.ndarray, np.ndarray]:
    difficulty = rng.normal(0, 1, ITEMS)
    baseline = difficulty + rng.normal(0, 0.5, ITEMS)
    candidate = difficulty + shift + rng.normal(0, 0.5, ITEMS)
    return baseline, candidate


@pytest.mark.statistical
@pytest.mark.ticket("EV-010-T18")
def test_identical_models_rarely_differ() -> None:
    rng = np.random.default_rng(11)
    false_positives = 0
    for dataset in range(DATASETS):
        baseline, candidate = simulate(rng, 0.0)
        _, low, high, _ = paired_interval(
            lambda idx, cand=candidate, base=baseline: float(cand[idx].mean() - base[idx].mean()),
            ITEMS,
            CONFIG,
            seed=dataset,
            metric="m",
        )
        false_positives += not (low <= 0 <= high)

    assert 0.03 <= false_positives / DATASETS <= 0.07


@pytest.mark.statistical
@pytest.mark.ticket("EV-010-T19")
def test_pairing_detects_a_real_difference_more_often() -> None:
    rng = np.random.default_rng(22)
    paired_hits = unpaired_hits = 0
    for dataset in range(300):
        baseline, candidate = simulate(rng, 0.15)
        _, low, _, _ = paired_interval(
            lambda idx, cand=candidate, base=baseline: float(cand[idx].mean() - base[idx].mean()),
            ITEMS,
            CONFIG,
            seed=dataset,
            metric="m",
        )
        paired_hits += low > 0
        other = np.random.default_rng(dataset + 10_000)
        draws = [
            candidate[other.integers(0, ITEMS, ITEMS)].mean()
            - baseline[other.integers(0, ITEMS, ITEMS)].mean()
            for _ in range(CONFIG.n_resamples)
        ]
        unpaired_hits += np.quantile(draws, 0.025) > 0

    assert paired_hits > 2 * unpaired_hits


@pytest.mark.statistical
@pytest.mark.ticket("EV-010-T20")
def test_difference_interval_covers_the_true_difference() -> None:
    rng = np.random.default_rng(33)
    covered = 0
    for dataset in range(DATASETS):
        baseline, candidate = simulate(rng, 0.2)
        _, low, high, _ = paired_interval(
            lambda idx, cand=candidate, base=baseline: float(cand[idx].mean() - base[idx].mean()),
            ITEMS,
            CONFIG,
            seed=dataset,
            metric="m",
        )
        covered += low <= 0.2 <= high

    assert 0.93 <= covered / DATASETS <= 0.97
