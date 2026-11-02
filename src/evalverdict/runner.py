"""Run targets over a dataset, score the predictions and write the run directory."""

from __future__ import annotations

import json
import platform
import subprocess
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from evalverdict.budget import Budget
from evalverdict.cache import ResponseCache, cache_key
from evalverdict.config import RunConfig, ScorerEntry, TargetConfig
from evalverdict.data import Dataset, Prediction, TestCase, load_dataset
from evalverdict.engines.judge_client import JudgeClient
from evalverdict.scorers.generation import base_model
from evalverdict.scoring.base import Scorer
from evalverdict.scoring.gates import INVALID, apply_gates, overall_outcome
from evalverdict.scoring.registry import build_scorer
from evalverdict.targets.base import Target, build_target
from evalverdict.targets.litellm_target import LiteLLMTarget, apply_shared_sampling

STATUS_COMPLETE = "complete"
STATUS_INCOMPLETE = "incomplete"
STATUS_INVALID = "invalid"
STOP_BUDGET = "budget"
STOP_INTERRUPTED = "interrupted"

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


@dataclass
class Collected:
    predictions: list[Prediction] = field(default_factory=list)
    stop_reason: str | None = None
    calls: int = 0
    cache_hits: int = 0
    spend_usd: float = 0.0
    unmeasured_calls: int = 0


def _collect(
    target: Target,
    target_config: dict[str, Any],
    cases: tuple[TestCase, ...],
    config: RunConfig,
    cache: ResponseCache,
    budget: Budget,
) -> Collected:
    """Call the target for every case and repeat. Partial results survive any stop."""
    collected = Collected()
    finished: dict[int, Prediction] = {}
    jobs: deque[tuple[int, TestCase, int, str]] = deque()
    for repeat in range(config.repeats):
        for index, case in enumerate(cases):
            order = repeat * len(cases) + index
            key = cache_key(target_config, case.input, repeat)
            cached = cache.get(key)
            if cached is not None:
                finished[order] = replace(cached, case_id=case.id, repeat=repeat)
                collected.cache_hits += 1
            else:
                jobs.append((order, case, repeat, key))

    pending: dict[Future[Prediction], tuple[int, int, str]] = {}
    pool = ThreadPoolExecutor(max_workers=config.concurrency)
    try:
        while (jobs or pending) and collected.stop_reason is None:
            while jobs and len(pending) < config.concurrency:
                if not budget.reserve():
                    if not pending:
                        collected.stop_reason = STOP_BUDGET
                    break
                order, case, repeat, key = jobs.popleft()
                pending[pool.submit(target.predict, case)] = (order, repeat, key)
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                order, repeat, key = pending.pop(future)
                prediction = replace(future.result(), repeat=repeat)
                budget.settle(prediction.cost_usd)
                collected.calls += 1
                collected.spend_usd += prediction.cost_usd or 0.0
                collected.unmeasured_calls += target.paid and prediction.cost_usd is None
                cache.put(key, prediction)
                finished[order] = prediction
    except KeyboardInterrupt:
        collected.stop_reason = STOP_INTERRUPTED
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    collected.predictions = [finished[order] for order in sorted(finished)]
    return collected


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


def _score(
    collected: Collected, dataset: Dataset, config: RunConfig, scorers: dict[str, Scorer]
) -> list[dict[str, Any]]:
    """API errors are left out of metrics. Parse failures stay in and count as wrong (D10)."""
    usable = [item for item in collected.predictions if item.error is None]
    if not usable:
        return []
    cases = {case.id: case for case in dataset.cases}
    ids = [item.case_id for item in usable]
    outputs = [item.output for item in usable]
    references = [cases[item.case_id].reference for item in usable]
    item_metadata = [cases[item.case_id].metadata for item in usable]
    clusters = ids if config.repeats > 1 else None

    results = []
    for name, scorer in scorers.items():
        inputs = usable if scorer.wants_predictions else outputs
        context = [cases[item.case_id] for item in usable] if scorer.wants_cases else item_metadata
        for result in scorer.score(inputs, references, context, ids=ids, clusters=clusters):
            results.append({"scorer": name, **asdict(result)})
    return results


def _summary(
    collected: Collected, dataset: Dataset, config: RunConfig, scorers: dict[str, Scorer]
) -> dict[str, Any]:
    completed = len(collected.predictions)
    errors = sum(item.error is not None for item in collected.predictions)
    parse_failures = sum(item.parse_failed for item in collected.predictions)
    error_rate = errors / completed if completed else 0.0
    health = [scorer.health() for scorer in scorers.values()]
    judge_failures = sum(item.failed for item in health)
    too_many_failures = any(
        item.total and item.failed / item.total > config.max_error_rate for item in health
    )
    stop_reason = collected.stop_reason or next(
        (item.stop_reason for item in health if item.stop_reason), None
    )
    if error_rate > config.max_error_rate or too_many_failures:
        status = STATUS_INVALID
    elif stop_reason is not None:
        status = STATUS_INCOMPLETE
    else:
        status = STATUS_COMPLETE
    return {
        "status": status,
        "stop_reason": stop_reason,
        "n_requested": len(dataset.cases) * config.repeats,
        "n_completed": completed,
        "n_errors": errors,
        "error_rate": error_rate,
        "n_parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / completed if completed else 0.0,
        "cache_hits": collected.cache_hits,
        "judge_failures": judge_failures,
        "repeats": config.repeats,
        "concurrency": config.concurrency,
    }


def _write_target_run(
    folder: Path,
    config: RunConfig,
    dataset: Dataset,
    target_name: str,
    collected: Collected,
    scorers: dict[str, Scorer],
    created_at: str,
) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    disabled = [entry.resolved_name for entry in config.scorers if not entry.enabled]
    results = _score(collected, dataset, config, scorers)
    summary = _summary(collected, dataset, config, scorers)
    gates = apply_gates(config.gates, results)
    summary["outcome"] = (
        INVALID
        if summary["status"] != STATUS_COMPLETE
        else overall_outcome(gate["outcome"] for gate in gates)
    )
    (folder / "scores.json").write_text(
        _dump({"results": results, "disabled": disabled, "gates": gates, "run": summary})
    )
    (folder / "predictions.jsonl").write_text(
        "".join(
            json.dumps(asdict(item), sort_keys=True, default=_json_default) + "\n"
            for item in collected.predictions
        )
    )
    manifest = {
        "name": config.name,
        "created_at": created_at,
        "status": summary["status"],
        "outcome": summary["outcome"],
        "target": target_name,
        "target_config": config.targets[target_name].model_dump(mode="json"),
        "config_sha256": config.sha256(),
        "resume_sha256": config.resume_sha256(),
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "n": len(dataset.cases),
        },
        "seed": config.seed,
        "concurrency": config.concurrency,
        "repeats": config.repeats,
        "spend": {
            "usd": collected.spend_usd,
            "calls": collected.calls,
            "cache_hits": collected.cache_hits,
            "unmeasured_calls": collected.unmeasured_calls,
        },
        "versions": _versions(),
        "git_sha": _git_sha(),
        "scorers": _resolved_scorers(config.scorers, scorers),
    }
    return manifest


def _check_targets(config: RunConfig, targets: dict[str, Target]) -> None:
    for name, target in targets.items():
        if not target.paid:
            continue
        if config.budget_usd is None:
            raise ValueError(f"target {name!r} costs money, so budget_usd is required")
        if not target.cost_known:
            raise ValueError(f"target {name!r} has no cost source; add pricing to its config")


def _target_models(config: RunConfig) -> list[str]:
    models = [target.model_dump().get("model") for target in config.targets.values()]
    return [model for model in models if isinstance(model, str)]


def _judge_client(
    config: RunConfig,
    model: str,
    cache: ResponseCache,
    budget: Budget,
) -> JudgeClient:
    judge = config.judge
    if judge is None:
        raise ValueError("a judge scorer needs a judge block in the config")
    if not judge.allow_self_judging and any(
        base_model(model) == base_model(target) for target in _target_models(config)
    ):
        raise ValueError(f"judge {model} is a model under test; set allow_self_judging to allow it")
    settings: dict[str, Any] = {
        "kind": "litellm",
        "model": model,
        "max_tokens": judge.max_tokens,
        "params": judge.params,
    }
    if judge.pricing is not None:
        settings["pricing"] = judge.pricing
    target = LiteLLMTarget(TargetConfig(**settings))
    if target.paid and config.budget_usd is None:
        raise ValueError(f"judge {model} costs money, so budget_usd is required")
    if target.paid and not target.cost_known:
        raise ValueError(f"judge {model} has no cost source; add pricing to the judge config")
    identity = {"max_tokens": judge.max_tokens, "params": judge.params}
    return JudgeClient(target, model, cache, budget, identity)


def _bind_judges(
    config: RunConfig, scorers: dict[str, Scorer], cache: ResponseCache, budget: Budget
) -> list[JudgeClient]:
    clients: dict[str, JudgeClient] = {}
    for scorer in scorers.values():
        if not scorer.needs_judge:
            continue
        if config.judge is None:
            raise ValueError("a judge scorer needs a judge block in the config")
        model = scorer.params.judge_model or config.judge.model
        clients.setdefault(model, _judge_client(config, model, cache, budget))
        scorer.bind_judge(clients[model], model, _target_models(config))  # type: ignore[attr-defined]
    return list(clients.values())


def _check_resume(run_dir: Path, config: RunConfig, dataset: Dataset) -> None:
    manifests = sorted(run_dir.glob("*/manifest.json"))
    if not manifests:
        raise ValueError(f"cannot resume: no run found at {run_dir}")
    for path in manifests:
        manifest = json.loads(path.read_text())
        if manifest["resume_sha256"] != config.resume_sha256():
            raise ValueError("cannot resume: the config has changed")
        if manifest["dataset"]["sha256"] != dataset.sha256:
            raise ValueError("cannot resume: the dataset has changed")


def run_statuses(run_dir: Path) -> dict[str, str]:
    """The status of each target in a run."""
    return {
        path.parent.name: json.loads(path.read_text())["status"]
        for path in sorted(run_dir.glob("*/manifest.json"))
    }


@dataclass
class Prepared:
    """Everything a run needs, checked and built, with no model call made."""

    config: RunConfig
    dataset: Dataset
    scorers: dict[str, Scorer]
    targets: dict[str, Target]
    judges: list[JudgeClient]
    cache: ResponseCache
    budget: Budget
    dropped: list[str]


def _check_gates(config: RunConfig, scorers: dict[str, Scorer]) -> None:
    problems = []
    for key in config.gates:
        scorer_name, _, metric = key.partition(".")
        scorer = scorers.get(scorer_name)
        if scorer is None:
            enabled = sorted(scorers)
            problems.append(f"gate {key}: no enabled scorer {scorer_name!r} (enabled: {enabled})")
        elif metric not in scorer.selected_metrics:
            options = list(scorer.selected_metrics)
            problems.append(f"gate {key}: metric {metric!r} is not produced; options: {options}")
    if problems:
        raise ValueError("; ".join(problems))


def prepare(config: RunConfig, resume: Path | None = None) -> Prepared:
    """Load, build and check everything. Raises before any model call or output."""
    dataset = load_dataset(config.dataset)
    if resume is not None:
        _check_resume(resume, config, dataset)
    scorers = {
        entry.resolved_name: build_scorer(entry, config.bootstrap, config.seed)
        for entry in config.scorers
        if entry.enabled
    }
    _check_gates(config, scorers)
    targets = {name: build_target(target) for name, target in config.targets.items()}
    _check_targets(config, targets)
    dropped = apply_shared_sampling(targets, config.sampling)
    cache = ResponseCache(config.cache_dir)
    budget = Budget(config.budget_usd)
    judges = _bind_judges(config, scorers, cache, budget)
    return Prepared(config, dataset, scorers, targets, judges, cache, budget, dropped)


def run_outcomes(run_dir: Path) -> dict[str, str]:
    """The gate outcome of each target in a run."""
    return {
        path.parent.name: json.loads(path.read_text())["outcome"]
        for path in sorted(run_dir.glob("*/manifest.json"))
    }


def execute(config: RunConfig, runs_root: Path = Path("runs"), resume: Path | None = None) -> Path:
    """Run every target. All setup errors surface before any model call or output."""
    ready = prepare(config, resume)
    dataset, scorers, judges = ready.dataset, ready.scorers, ready.judges

    now = datetime.now(UTC)
    run_dir = resume or runs_root / f"{now:%Y%m%dT%H%M%S%fZ}-{config.name}"
    for name, target in ready.targets.items():
        target_config = config.targets[name].model_dump(mode="json")
        identity = {**target_config, "parameters_sent": target.parameters_sent()}
        collected = _collect(target, identity, dataset.cases, config, ready.cache, ready.budget)
        written = _write_target_run(
            run_dir / name, config, dataset, name, collected, scorers, now.isoformat()
        )
        written["parameters_sent"] = target.parameters_sent()
        written["sampling_dropped"] = ready.dropped
        if config.judge is not None:
            written["judge"] = {
                "model": config.judge.model,
                "allow_self_judging": config.judge.allow_self_judging,
                "spend_usd": sum(client.spend_usd for client in judges),
                "calls": sum(client.calls for client in judges),
                "cache_hits": sum(client.cache_hits for client in judges),
            }
        (run_dir / name / "manifest.json").write_text(_dump(written))
    return run_dir
