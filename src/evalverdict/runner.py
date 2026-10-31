"""Run targets over a dataset, score the predictions and write the run directory."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from evalverdict.config import RunConfig, ScorerEntry
from evalverdict.data import Dataset, Prediction, load_dataset
from evalverdict.scoring.base import Scorer
from evalverdict.scoring.registry import build_scorer
from evalverdict.targets.base import build_target

VERSIONED_PACKAGES = (
    "evalverdict",
    "numpy",
    "pandas",
    "scikit-learn",
    "litellm",
    "rouge-score",
    "bert-score",
    "evaluate",
    "deepeval",
    "inspect-ai",
    "langfuse",
    "evidently",
    "fairlearn",
    "shap",
    "lime",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"


def _versions() -> dict[str, str]:
    found = {"python": platform.python_version()}
    for package in VERSIONED_PACKAGES:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return found


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _resolved_scorers(
    entries: list[ScorerEntry], scorers: dict[str, Scorer]
) -> list[dict[str, Any]]:
    resolved = []
    for entry in entries:
        scorer = scorers.get(entry.resolved_name)
        resolved.append(
            {
                "name": entry.resolved_name,
                "type": entry.type,
                "enabled": entry.enabled,
                "metrics": list(scorer.selected_metrics) if scorer else entry.metrics,
                "params": scorer.params.model_dump(mode="json") if scorer else entry.params,
                "bootstrap": scorer.bootstrap.model_dump() if scorer else entry.bootstrap,
            }
        )
    return resolved


def _write_target_run(
    folder: Path,
    config: RunConfig,
    dataset: Dataset,
    target_name: str,
    predictions: list[Prediction],
    scorers: dict[str, Scorer],
    created_at: str,
) -> None:
    folder.mkdir(parents=True)
    ids = [case.id for case in dataset.cases]
    outputs = [prediction.output for prediction in predictions]
    references = [case.reference for case in dataset.cases]
    item_metadata = [case.metadata for case in dataset.cases]

    results = []
    for name, scorer in scorers.items():
        for result in scorer.score(outputs, references, item_metadata, ids=ids):
            results.append({"scorer": name, **asdict(result)})

    disabled = [entry.resolved_name for entry in config.scorers if not entry.enabled]
    (folder / "scores.json").write_text(_dump({"results": results, "disabled": disabled}))
    (folder / "predictions.jsonl").write_text(
        "".join(
            json.dumps(asdict(item), sort_keys=True, default=_json_default) + "\n"
            for item in predictions
        )
    )
    manifest = {
        "name": config.name,
        "created_at": created_at,
        "target": target_name,
        "target_config": config.targets[target_name].model_dump(mode="json"),
        "config_sha256": config.sha256(),
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "n": len(dataset.cases),
        },
        "seed": config.seed,
        "versions": _versions(),
        "git_sha": _git_sha(),
        "scorers": _resolved_scorers(config.scorers, scorers),
    }
    (folder / "manifest.json").write_text(_dump(manifest))


def execute(config: RunConfig, runs_root: Path = Path("runs")) -> Path:
    """Run every target. All setup errors surface before any output is written."""
    dataset = load_dataset(config.dataset)
    scorers = {
        entry.resolved_name: build_scorer(entry, config.bootstrap, config.seed)
        for entry in config.scorers
        if entry.enabled
    }
    targets = {name: build_target(target) for name, target in config.targets.items()}

    now = datetime.now(UTC)
    run_dir = runs_root / f"{now:%Y%m%dT%H%M%S%fZ}-{config.name}"
    for name, target in targets.items():
        predictions = [target.predict(case) for case in dataset.cases]
        _write_target_run(
            run_dir / name, config, dataset, name, predictions, scorers, now.isoformat()
        )
    return run_dir
