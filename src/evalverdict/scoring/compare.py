"""Paired comparison of two runs on the same items (D4, D7, D10, D15)."""

from __future__ import annotations

import json
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evalverdict.config import ScorerEntry
from evalverdict.data import Dataset, Prediction, load_dataset
from evalverdict.scoring.base import BootstrapConfig, Scorer, UndefinedMetricError
from evalverdict.scoring.bootstrap import basic_interval, percentile_interval, resample_indices
from evalverdict.scoring.registry import build_scorer, scorer_class

BETTER = "better"
WORSE = "worse"
INCONCLUSIVE = "inconclusive"
SETTING_KEYS = ("type", "import", "params", "metrics", "bootstrap")
Statistic = Callable[[NDArray[np.int64]], float | None]


def paired_interval(
    statistic: Statistic,
    n: int,
    config: BootstrapConfig,
    seed: int,
    metric: str,
    clusters: NDArray[Any] | None = None,
) -> tuple[float, float, float, int]:
    """Bootstrap a difference using the same resampled items for both models.

    Returns the point difference, the interval bounds and the number of undefined resamples.
    """
    point = statistic(np.arange(n, dtype=np.int64))
    if point is None:
        raise UndefinedMetricError(f"{metric} difference is undefined on this data")
    sequence = np.random.SeedSequence([seed, zlib.crc32(metric.encode())])
    draws = [statistic(idx) for idx in resample_indices(n, config.n_resamples, sequence, clusters)]
    defined = np.array([draw for draw in draws if draw is not None], dtype=float)
    if defined.size == 0:
        raise UndefinedMetricError(f"{metric} difference is undefined on every resample")
    if config.method == "basic":
        low, high = basic_interval(defined, config.level, point)
    else:
        low, high = percentile_interval(defined, config.level)
    return point, low, high, config.n_resamples - int(defined.size)


def verdict(low: float, high: float, *, higher_is_better: bool) -> str:
    """Candidate minus baseline. Better or worse only when the whole interval agrees."""
    if low > 0:
        return BETTER if higher_is_better else WORSE
    if high < 0:
        return WORSE if higher_is_better else BETTER
    return INCONCLUSIVE


def recommend(metrics: list[dict[str, Any]]) -> dict[str, str]:
    """Primary metrics alone decide (D7). Any doubt among them is inconclusive."""
    primaries = [item for item in metrics if item.get("primary")]
    if not primaries:
        return {"decision": "none", "reason": "no primary metrics are declared in the gates"}
    if any(not item["comparable"] for item in primaries):
        return {"decision": "inconclusive", "reason": "a primary metric cannot be compared"}
    verdicts = {item["verdict"] for item in primaries}
    if verdicts == {BETTER}:
        return {
            "decision": "candidate",
            "reason": "the candidate is better on every primary metric",
        }
    if verdicts == {WORSE}:
        return {"decision": "baseline", "reason": "the candidate is worse on every primary metric"}
    return {"decision": "inconclusive", "reason": "primary metrics disagree or are inconclusive"}


def settings_differ(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return any(left.get(key) != right.get(key) for key in SETTING_KEYS)


class _Run:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.manifest = json.loads((folder / "manifest.json").read_text())
        self.scores = json.loads((folder / "scores.json").read_text())
        rows = (folder / "predictions.jsonl").read_text().splitlines()
        self.predictions = [Prediction(**json.loads(row)) for row in rows if row]
        self.usable = [item for item in self.predictions if item.error is None]
        self.positions = {(item.case_id, item.repeat): n for n, item in enumerate(self.usable)}
        self.results = {f"{r['scorer']}.{r['name']}": r for r in self.scores["results"]}
        self.settings = {entry["name"]: entry for entry in self.manifest["scorers"]}


def _comparable_entry(key: str, reason: str, primary: bool) -> dict[str, Any]:
    return {"key": key, "comparable": False, "reason": reason, "primary": primary}


def _per_item_statistic(
    base: _Run, cand: _Run, key: str, pairs: list[tuple[str, int]]
) -> tuple[Statistic, int]:
    base_items = base.results[key]["per_item"]
    cand_items = cand.results[key]["per_item"]
    rows = [
        (base_items[base.positions[pair]]["value"], cand_items[cand.positions[pair]]["value"])
        for pair in pairs
    ]
    measured = np.array([row for row in rows if None not in row], dtype=float)
    if measured.size == 0:
        raise UndefinedMetricError("no item is measured in both runs")

    def statistic(idx: NDArray[np.int64]) -> float:
        return float(measured[idx, 1].mean() - measured[idx, 0].mean())

    return statistic, len(measured)


def _recomputed_statistic(
    base: _Run,
    cand: _Run,
    key: str,
    pairs: list[tuple[str, int]],
    scorer: Scorer,
    dataset: Dataset,
) -> tuple[Statistic, int]:
    metric = key.split(".", 1)[1]
    cases = {case.id: case for case in dataset.cases}

    def values_for(run: _Run) -> Any:
        chosen = [run.usable[run.positions[pair]] for pair in pairs]
        inputs = chosen if scorer.wants_predictions else [item.output for item in chosen]
        refs = [cases[item.case_id].reference for item in chosen]
        context: Any = [cases[item.case_id] for item in chosen]
        if not scorer.wants_cases:
            context = [case.metadata for case in context]
        return scorer.item_values(inputs, refs, context)

    base_values, cand_values = values_for(base), values_for(cand)
    keep = np.arange(len(pairs), dtype=np.int64)
    if metric in base_values.sparse:
        both = ~np.isnan(base_values.arrays[metric]) & ~np.isnan(cand_values.arrays[metric])
        keep = np.flatnonzero(both).astype(np.int64)
        if keep.size == 0:
            raise UndefinedMetricError("no item is measured in both runs")

    def statistic(idx: NDArray[np.int64]) -> float | None:
        try:
            chosen = keep[idx]
            return scorer.aggregate(cand_values, chosen, metric) - scorer.aggregate(
                base_values, chosen, metric
            )
        except UndefinedMetricError:
            return None

    return statistic, len(keep)


def _entry(settings: Mapping[str, Any]) -> ScorerEntry:
    return ScorerEntry.model_validate(
        {
            "type": settings.get("type"),
            "import": settings.get("import"),
            "name": settings["name"],
            "params": settings.get("params") or {},
            "metrics": settings.get("metrics"),
        }
    )


def compare_runs(baseline: Path, candidate: Path) -> dict[str, Any]:
    """Compare two target runs. Raises if they cannot be compared at all."""
    base, cand = _Run(baseline), _Run(candidate)
    for run in (base, cand):
        if run.manifest["status"] == "invalid":
            raise ValueError(f"cannot compare: {run.folder.name} is an invalid run")
    if base.manifest["dataset"]["sha256"] != cand.manifest["dataset"]["sha256"]:
        raise ValueError("cannot compare runs on different datasets")

    pairs = [pair for pair in base.positions if pair in cand.positions]
    primary = {key for key, gate in cand.manifest.get("gates", {}).items() if gate.get("primary")}
    repeats = max(base.manifest.get("repeats", 1), cand.manifest.get("repeats", 1))
    clusters = np.array([case_id for case_id, _ in pairs]) if repeats > 1 else None
    dataset: Dataset | None = None

    metrics = []
    for key in sorted(set(base.results) | set(cand.results)):
        scorer_name = key.split(".", 1)[0]
        is_primary = key in primary
        if key not in base.results or key not in cand.results:
            metrics.append(_comparable_entry(key, "only present in one run", is_primary))
            continue
        left, right = base.settings[scorer_name], cand.settings[scorer_name]
        if settings_differ(left, right):
            metrics.append(_comparable_entry(key, "scorer settings differ", is_primary))
            continue
        base_result, cand_result = base.results[key], cand.results[key]
        if "value" not in base_result or "value" not in cand_result:
            metrics.append(_comparable_entry(key, "not measured in both runs", is_primary))
            continue

        cls = scorer_class(_entry(right))
        bootstrap = BootstrapConfig(**right["bootstrap"])
        try:
            if cls.aggregate is Scorer.aggregate:
                method = "per_item"
                statistic, n_paired = _per_item_statistic(base, cand, key, pairs)
            else:
                method = "recomputed"
                if dataset is None:
                    dataset = load_dataset(Path(cand.manifest["dataset"]["path"]))
                    if dataset.sha256 != cand.manifest["dataset"]["sha256"]:
                        raise ValueError("the dataset file has changed since the runs")
                scorer = build_scorer(_entry(right), bootstrap, cand.manifest["seed"])
                statistic, n_paired = _recomputed_statistic(base, cand, key, pairs, scorer, dataset)
            point, low, high, undefined = paired_interval(
                statistic,
                n_paired,
                bootstrap,
                cand.manifest["seed"],
                key,
                clusters if n_paired == len(pairs) else None,
            )
        except UndefinedMetricError as error:
            metrics.append(_comparable_entry(key, str(error), is_primary))
            continue
        higher = bool(cand_result["higher_is_better"])
        metrics.append(
            {
                "key": key,
                "comparable": True,
                "primary": is_primary,
                "method": method,
                "higher_is_better": higher,
                "baseline_value": base_result["value"],
                "candidate_value": cand_result["value"],
                "difference": point,
                "ci": {
                    "low": low,
                    "high": high,
                    "level": bootstrap.level,
                    "method": bootstrap.method,
                    "n_resamples": bootstrap.n_resamples,
                    "n_undefined": undefined,
                },
                "n_paired": n_paired,
                "verdict": verdict(low, high, higher_is_better=higher),
            }
        )
    return {
        "baseline": baseline.name,
        "candidate": candidate.name,
        "baseline_status": base.manifest["status"],
        "candidate_status": cand.manifest["status"],
        "n_paired_items": len(pairs),
        "metrics": metrics,
        "recommendation": recommend(metrics),
    }
