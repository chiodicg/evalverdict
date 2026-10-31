"""Seeded bootstrap resampling and interval calculation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

IndexArray = NDArray[np.int64]


def resample_indices(
    n: int,
    n_resamples: int,
    seed: int | np.random.SeedSequence,
    clusters: NDArray[Any] | None = None,
) -> Iterator[IndexArray]:
    """Yield item indices for each resample.

    With `clusters`, whole clusters are drawn with replacement, so variants or repeats
    of one source item stay together and cannot inflate the effective sample size.
    """
    rng = np.random.default_rng(seed)
    if clusters is None:
        for _ in range(n_resamples):
            yield rng.integers(0, n, size=n, dtype=np.int64)
        return

    _, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == group) for group in range(inverse.max() + 1)]
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(members), size=len(members))
        yield np.concatenate([members[group] for group in chosen]).astype(np.int64)


def percentile_interval(draws: NDArray[np.float64], level: float) -> tuple[float, float]:
    tail = (1 - level) / 2
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


def basic_interval(draws: NDArray[np.float64], level: float, point: float) -> tuple[float, float]:
    low, high = percentile_interval(draws, level)
    return 2 * point - high, 2 * point - low
