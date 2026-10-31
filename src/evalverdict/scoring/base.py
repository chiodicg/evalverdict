"""The contract every scorer shares. The base class owns the confidence interval."""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Literal, final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from evalverdict.scoring.bootstrap import basic_interval, percentile_interval, resample_indices

FLAG_DEGENERATE = "degenerate_interval"
FLAG_SMALL_SAMPLE = "small_sample"
FLAG_UNRELIABLE_CI = "unreliable_interval"

SMALL_SAMPLE_LIMIT = 30
UNDEFINED_SHARE_LIMIT = 0.05


class Family(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    RANKING = "ranking"
    AGENTIC = "agentic"
    ROBUSTNESS = "robustness"
    FAIRNESS = "fairness"
    OPERATIONAL = "operational"
    EXPLAINABILITY = "explainability"


class UndefinedMetricError(ValueError):
    """A metric cannot be computed on this data, for example AUROC with one class."""


class BootstrapConfig(BaseModel):
    """Interval settings. Nothing here can turn an interval off (D15)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_resamples: int = Field(default=2000, gt=0)
    level: float = Field(default=0.95, gt=0, lt=1)
    method: Literal["percentile", "basic"] = "percentile"


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float
    level: float
    method: str
    n_resamples: int
    n_undefined: int


@dataclass(frozen=True)
class ItemScore:
    item_id: str
    value: float | None
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreResult:
    name: str
    family: Family
    value: float
    ci: ConfidenceInterval
    n: int
    higher_is_better: bool
    per_item: tuple[ItemScore, ...]
    details: Mapping[str, Any]
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ci, ConfidenceInterval):
            raise TypeError("a ScoreResult needs a ConfidenceInterval")

    def __str__(self) -> str:
        half = (self.ci.high - self.ci.low) / 2
        return f"{self.value:.2f} ±{half:.2f} on n={self.n}"


@dataclass(frozen=True)
class ItemValues:
    """Per-item data. `arrays` holds item scores or whatever `aggregate` needs."""

    arrays: Mapping[str, NDArray[Any]]
    details: Sequence[Mapping[str, Any]] = ()
    result_details: Mapping[str, Any] = field(default_factory=dict)


class Scorer(ABC):
    """Subclasses compute item values and an aggregate. `score` adds the interval."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

    family: ClassVar[Family]
    metrics: ClassVar[tuple[str, ...]]
    lower_is_better: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "score" in cls.__dict__:
            raise TypeError(f"{cls.__name__} must not override score(): it owns the interval")
        if not getattr(cls, "metrics", ()):
            raise TypeError(f"{cls.__name__} must declare metrics")
        if cls.item_values is Scorer.item_values:
            raise TypeError(f"{cls.__name__} must implement item_values()")

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        bootstrap: BootstrapConfig | None = None,
        seed: int = 0,
        metrics: Sequence[str] | None = None,
    ) -> None:
        self.params: Any = self.Params(**(params or {}))
        self.bootstrap = bootstrap or BootstrapConfig()
        self.seed = seed
        selected = tuple(metrics) if metrics is not None else self.metrics
        unknown = set(selected) - set(self.metrics)
        if unknown:
            raise ValueError(f"unknown metrics {sorted(unknown)}; options: {list(self.metrics)}")
        self.selected_metrics = selected

    @abstractmethod
    def item_values(
        self, predictions: Sequence[Any], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        """Compute per-item data for all metrics."""

    def aggregate(self, values: ItemValues, idx: NDArray[np.int64], metric: str) -> float:
        """Compute one metric on the items in `idx`. Override for F1, AUROC and the like."""
        return float(values.arrays[metric][idx].mean())

    @final
    def score(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        metadata: Any = None,
        *,
        ids: Sequence[str] | None = None,
        clusters: Sequence[Any] | None = None,
    ) -> tuple[ScoreResult, ...]:
        n = len(predictions)
        if n == 0:
            raise ValueError("no items to score")
        if len(references) != n:
            raise ValueError("predictions and references must be the same length")
        item_ids = tuple(ids) if ids is not None else tuple(str(index) for index in range(n))
        if len(item_ids) != n:
            raise ValueError("ids must match the number of predictions")
        groups = np.asarray(clusters) if clusters is not None else None
        if groups is not None and len(groups) != n:
            raise ValueError("clusters must match the number of predictions")

        values = self.item_values(predictions, references, metadata)
        _reject_nan(values, item_ids)
        return tuple(
            self._score_metric(metric, values, item_ids, groups) for metric in self.selected_metrics
        )

    def _score_metric(
        self,
        metric: str,
        values: ItemValues,
        item_ids: tuple[str, ...],
        groups: NDArray[Any] | None,
    ) -> ScoreResult:
        n_items = len(item_ids)
        value = self._evaluate(values, np.arange(n_items, dtype=np.int64), metric)
        if value is None:
            raise UndefinedMetricError(f"{metric} is undefined on this data")

        config = self.bootstrap
        seed = np.random.SeedSequence([self.seed, zlib.crc32(metric.encode())])
        draws = [
            self._evaluate(values, idx, metric)
            for idx in resample_indices(n_items, config.n_resamples, seed, groups)
        ]
        defined = np.array([draw for draw in draws if draw is not None])
        if defined.size == 0:
            raise UndefinedMetricError(f"{metric} is undefined on every resample")

        if config.method == "basic":
            low, high = basic_interval(defined, config.level, value)
        else:
            low, high = percentile_interval(defined, config.level)
        interval = ConfidenceInterval(
            low=low,
            high=high,
            level=config.level,
            method=config.method,
            n_resamples=config.n_resamples,
            n_undefined=config.n_resamples - int(defined.size),
        )
        n = int(np.unique(groups).size) if groups is not None else n_items
        return ScoreResult(
            name=metric,
            family=self.family,
            value=value,
            ci=interval,
            n=n,
            higher_is_better=metric not in self.lower_is_better,
            per_item=_item_scores(metric, values, item_ids),
            details=dict(values.result_details),
            flags=_flags(n, interval),
        )

    def _evaluate(self, values: ItemValues, idx: NDArray[np.int64], metric: str) -> float | None:
        try:
            result = self.aggregate(values, idx, metric)
        except UndefinedMetricError:
            return None
        return None if np.isnan(result) else result


def _reject_nan(values: ItemValues, item_ids: tuple[str, ...]) -> None:
    for name, array in values.arrays.items():
        if len(array) != len(item_ids):
            raise ValueError(f"item array {name!r} does not match the number of items")
        if np.issubdtype(array.dtype, np.floating) and np.isnan(array).any():
            raise ValueError(
                f"NaN in {name!r} for item {item_ids[int(np.argmax(np.isnan(array)))]}"
            )


def _item_scores(
    metric: str, values: ItemValues, item_ids: tuple[str, ...]
) -> tuple[ItemScore, ...]:
    array = values.arrays.get(metric)
    return tuple(
        ItemScore(
            item_id=item_id,
            value=float(array[index]) if array is not None else None,
            detail=values.details[index] if values.details else {},
        )
        for index, item_id in enumerate(item_ids)
    )


def _flags(n: int, interval: ConfidenceInterval) -> tuple[str, ...]:
    flags = []
    if n == 1:
        flags.append(FLAG_DEGENERATE)
    if n < SMALL_SAMPLE_LIMIT:
        flags.append(FLAG_SMALL_SAMPLE)
    if interval.n_undefined / interval.n_resamples > UNDEFINED_SHARE_LIMIT:
        flags.append(FLAG_UNRELIABLE_CI)
    return tuple(flags)
