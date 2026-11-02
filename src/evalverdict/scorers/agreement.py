"""Judge agreement with labels: weighted Cohen's kappa, reported per label provenance."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from evalverdict.labelling import SCALE, SUMMARY_FILE, load_labels
from evalverdict.scoring.base import (
    BootstrapConfig,
    ContractExample,
    Family,
    ItemValues,
    Scorer,
    UndefinedMetricError,
)
from evalverdict.scoring.registry import register_scorer

TITLES = {
    "human_independent": "Judge vs human labels",
    "ai_draft_human_reviewed": "Judge vs reviewed labels",
    "ai_draft": "AI drafts vs human labels",
}


@register_scorer("agreement")
class AgreementScorer(Scorer):
    """Predictions are one rater's scores, references the other's, on a 0-10 scale."""

    family = Family.GENERATION
    metrics = ("weighted_kappa",)

    class Params(Scorer.Params):
        weights: Literal["quadratic", "linear"] = "quadratic"
        judge_scale: float = Field(default=1.0, gt=0)

    @classmethod
    def contract_example(cls) -> ContractExample:
        return ContractExample(
            params={}, predictions=[3, 5, 7, 8, 2, 9, 6, 4], references=[4, 5, 6, 8, 3, 9, 5, 4]
        )

    def item_values(
        self, predictions: Sequence[Any], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        rater = np.rint(np.asarray(predictions, dtype=float) * self.params.judge_scale)
        return ItemValues(
            arrays={"rater": rater.astype(int), "label": np.asarray(references, dtype=int)}
        )

    def aggregate(self, values: ItemValues, idx: NDArray[np.int64], metric: str) -> float:
        rater, label = values.arrays["rater"][idx], values.arrays["label"][idx]
        if np.unique(rater).size < 2 or np.unique(label).size < 2:
            raise UndefinedMetricError("kappa is undefined when a rater uses only one score")
        return float(
            cohen_kappa_score(rater, label, labels=list(SCALE), weights=self.params.weights)
        )


def _judge_scores(run_dir: Path, judge_scorer: str) -> dict[tuple[str, str], float]:
    found: dict[tuple[str, str], float] = {}
    for scores_path in run_dir.glob("*/scores.json"):
        target = scores_path.parent.name
        for result in json.loads(scores_path.read_text()).get("results", []):
            if result["scorer"] == judge_scorer and result["name"] == "score":
                for item in result["per_item"]:
                    if item["value"] is not None:
                        found[(target, item["item_id"])] = item["value"]
    return found


def agreement_results(
    run_dir: Path,
    labels_path: Path | None,
    judge_scorer: str,
    bootstrap: BootstrapConfig | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Agreement for each provenance, kept separate (D13). Never pools label types."""
    if labels_path is None:
        reason = "no labels were supplied, so judge agreement is unmeasured"
        return [{"scorer": "agreement", "name": "weighted_kappa", "reason": reason}]
    labels = load_labels(labels_path)
    summary_path = labels_path.parent / SUMMARY_FILE
    changed = (
        json.loads(summary_path.read_text())["n_changed_in_review"]
        if summary_path.is_file()
        else None
    )
    judged = _judge_scores(run_dir, judge_scorer)

    groups: dict[str, tuple[list[float], list[int], list[str], float]] = {}
    for provenance in ("human_independent", "ai_draft_human_reviewed"):
        rows = [
            row
            for row in labels
            if row["provenance"] == provenance and (row["target"], row["case_id"]) in judged
        ]
        groups[provenance] = (
            [judged[(row["target"], row["case_id"])] for row in rows],
            [row["label"] for row in rows],
            [row["item"] for row in rows],
            10.0,
        )
    human_rows = [row for row in labels if row["provenance"] == "human_independent"]
    groups["ai_draft"] = (
        [float(row["draft"]) for row in human_rows],
        [row["label"] for row in human_rows],
        [row["item"] for row in human_rows],
        1.0,
    )

    results = []
    for provenance, (rater, label, items, scale) in groups.items():
        details: dict[str, Any] = {"provenance": provenance, "title": TITLES[provenance]}
        if provenance == "ai_draft_human_reviewed":
            details["n_changed_in_review"] = changed
        if not items:
            results.append(
                {
                    "scorer": "agreement",
                    "name": "weighted_kappa",
                    "reason": f"no {provenance} labels match judged items",
                    "details": details,
                }
            )
            continue
        scorer = AgreementScorer(params={"judge_scale": scale}, bootstrap=bootstrap, seed=seed)
        try:
            result = scorer.score(rater, label, ids=items)[0]
        except UndefinedMetricError as error:
            results.append(
                {
                    "scorer": "agreement",
                    "name": "weighted_kappa",
                    "reason": str(error),
                    "details": details,
                }
            )
            continue
        entry = {"scorer": "agreement", **asdict(result)}
        entry["details"] = {**entry["details"], **details}
        results.append(entry)
    return results
