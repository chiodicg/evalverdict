"""Operational metrics: latency, throughput and cost."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator

from evalverdict.data import Prediction
from evalverdict.scoring.base import ContractExample, Family, ItemValues, Scorer
from evalverdict.scoring.registry import register_scorer


def _measured(predictions: Sequence[Prediction], field_name: str) -> NDArray[np.float64]:
    values = [getattr(item, field_name) for item in predictions]
    return np.array([np.nan if value is None else value for value in values], dtype=float)


@register_scorer("latency")
class LatencyScorer(Scorer):
    family = Family.OPERATIONAL
    metrics = ("latency_p50_s", "latency_p95_s", "ttft_p50_s", "ttft_p95_s")
    wants_predictions = True

    class Params(Scorer.Params):
        percentiles: list[float] = Field(default=[50, 95], min_length=1)

        @field_validator("percentiles")
        @classmethod
        def _in_range(cls, values: list[float]) -> list[float]:
            if any(not 0 <= value <= 100 for value in values):
                raise ValueError("percentiles must be between 0 and 100")
            return values

    @classmethod
    def contract_example(cls) -> ContractExample:
        items = [
            Prediction(
                case_id=str(index),
                latency_s=0.5 + index / 10,
                ttft_s=0.1 + index / 50,
                throughput_tokens_per_s=40.0 + index,
            )
            for index in range(5)
        ]
        return ContractExample(params={}, predictions=items, references=[None] * 5)

    def available_metrics(self) -> tuple[str, ...]:
        labels = [f"{value:g}" for value in self.params.percentiles]
        return (
            *(f"latency_p{label}_s" for label in labels),
            *(f"ttft_p{label}_s" for label in labels),
            "throughput_tokens_per_s",
        )

    def is_lower_better(self, metric: str) -> bool:
        return metric != "throughput_tokens_per_s"

    def item_values(
        self, predictions: Sequence[Prediction], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        latency = _measured(predictions, "latency_s")
        ttft = _measured(predictions, "ttft_s")
        arrays = {"throughput_tokens_per_s": _measured(predictions, "throughput_tokens_per_s")}
        for value in self.params.percentiles:
            arrays[f"latency_p{value:g}_s"] = latency
            arrays[f"ttft_p{value:g}_s"] = ttft
        return ItemValues(arrays=arrays, sparse=frozenset(arrays))

    def aggregate(self, values: ItemValues, idx: NDArray[np.int64], metric: str) -> float:
        array = values.arrays[metric][idx]
        if metric == "throughput_tokens_per_s":
            return float(array.mean())
        percentile = float(metric.split("_p")[1].removesuffix("_s"))
        return float(np.percentile(array, percentile))


@register_scorer("cost")
class CostScorer(Scorer):
    family = Family.OPERATIONAL
    metrics = ("cost_per_prediction_usd",)
    lower_is_better = frozenset(metrics)
    wants_predictions = True

    @classmethod
    def contract_example(cls) -> ContractExample:
        items = [Prediction(case_id=str(index), cost_usd=0.001 * (index + 1)) for index in range(4)]
        return ContractExample(params={}, predictions=items, references=[None] * 4)

    def item_values(
        self, predictions: Sequence[Prediction], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        sources = sorted({item.cost_source for item in predictions})
        return ItemValues(
            arrays={"cost_per_prediction_usd": _measured(predictions, "cost_usd")},
            sparse=frozenset(self.metrics),
            result_details={"cost_sources": sources},
        )
