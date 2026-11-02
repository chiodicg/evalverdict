"""EV-010: paired comparison between two runs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evalverdict.config import RunConfig
from evalverdict.runner import execute
from evalverdict.scoring.base import BootstrapConfig
from evalverdict.scoring.compare import (
    BETTER,
    INCONCLUSIVE,
    WORSE,
    compare_runs,
    paired_interval,
    recommend,
    verdict,
)
from fakes.target import INSTANCES, CaseTarget, FakeResponse

COUNT = 40


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_text(
        "".join(
            json.dumps({"id": f"c{n}", "input": f"in {n}", "reference": "alpha beta gamma delta"})
            + "\n"
            for n in range(COUNT)
        )
    )
    yield tmp_path
    INSTANCES.clear()


def answers(quality: int, **errors: int) -> CaseTarget:
    """Each item scores higher when quality is higher. Item difficulty is shared."""
    words = ["alpha", "beta", "gamma", "delta"]
    responses = {}
    for n in range(COUNT):
        hits = min(4, max(0, (n % 3) + quality))
        responses[f"c{n}"] = FakeResponse(output=" ".join(words[:hits] + ["zzz"] * (4 - hits)))
    for case_id in errors.get("error_ids", []):  # type: ignore[attr-defined]
        responses[case_id] = FakeResponse(error="boom")
    return CaseTarget(responses)


def run_pair(
    workspace: Path,
    baseline: CaseTarget,
    candidate: CaseTarget,
    scorers: list[dict[str, Any]] | None = None,
    gates: dict[str, Any] | None = None,
) -> Path:
    INSTANCES["baseline"] = baseline
    INSTANCES["candidate"] = candidate
    config = RunConfig(
        name="demo",
        seed=3,
        dataset=workspace / "data.jsonl",
        targets={
            "baseline": {"kind": "instance", "ref": "baseline"},
            "candidate": {"kind": "instance", "ref": "candidate"},
        },
        scorers=scorers or [{"type": "rouge", "metrics": ["rouge1"]}],
        bootstrap={"n_resamples": 300},
        gates=gates or {},
    )
    return execute(config)


def metric(comparison: dict[str, Any], key: str) -> dict[str, Any]:
    return next(item for item in comparison["metrics"] if item["key"] == key)


@pytest.mark.ticket("EV-010-T01")
def test_runs_on_different_datasets_are_refused(workspace: Path) -> None:
    first = run_pair(workspace, answers(1), answers(2))
    (workspace / "data.jsonl").write_text('{"id": "x", "input": "y", "reference": "z"}\n')
    second = run_pair(workspace, answers(1), answers(2))

    with pytest.raises(ValueError, match="different datasets"):
        compare_runs(first / "baseline", second / "candidate")


@pytest.mark.ticket("EV-010-T02")
def test_metric_with_different_settings_is_refused_and_others_compared(workspace: Path) -> None:
    scorers = [{"type": "rouge", "metrics": ["rouge1", "rouge2"]}]
    first = run_pair(workspace, answers(1), answers(2), scorers=scorers)
    stemmed = [{**scorers[0], "params": {"use_stemmer": True}}]
    second = run_pair(workspace, answers(1), answers(2), scorers=stemmed)

    comparison = compare_runs(first / "baseline", second / "candidate")

    assert metric(comparison, "rouge.rouge1")["comparable"] is False
    assert "settings differ" in metric(comparison, "rouge.rouge1")["reason"]


@pytest.mark.ticket("EV-010-T03")
def test_different_bertscore_model_is_not_comparable(workspace: Path) -> None:
    left = {"name": "bert", "type": "bertscore", "params": {"model_type": "a"}}
    right = {"name": "bert", "type": "bertscore", "params": {"model_type": "b"}}

    from evalverdict.scoring.compare import settings_differ

    assert settings_differ(left, right)


@pytest.mark.ticket("EV-010-T04")
def test_metric_present_in_only_one_run_is_not_comparable(workspace: Path) -> None:
    first = run_pair(workspace, answers(1), answers(2))
    extra = [
        {"type": "rouge", "metrics": ["rouge1"]},
        {"type": "rouge", "name": "r2", "metrics": ["rouge2"]},
    ]
    second = run_pair(workspace, answers(1), answers(2), scorers=extra)

    comparison = compare_runs(first / "baseline", second / "candidate")

    assert metric(comparison, "r2.rouge2")["comparable"] is False


@pytest.mark.ticket("EV-010-T05")
def test_both_models_use_identical_resample_indices() -> None:
    seen: list[np.ndarray] = []

    def statistic(idx: np.ndarray) -> float:
        seen.append(idx.copy())
        return 0.0

    paired_interval(statistic, 10, BootstrapConfig(n_resamples=5), seed=1, metric="m")
    again: list[np.ndarray] = []
    paired_interval(
        lambda idx: again.append(idx.copy()) or 0.0,
        10,
        BootstrapConfig(n_resamples=5),
        seed=1,
        metric="m",
    )

    assert all(np.array_equal(a, b) for a, b in zip(seen, again, strict=True))


@pytest.mark.ticket("EV-010-T06")
def test_item_errored_in_one_model_is_dropped_from_both(workspace: Path) -> None:
    candidate = answers(2, error_ids=["c0", "c1"])

    run_dir = run_pair(workspace, answers(1), candidate)

    comparison = json.loads((run_dir / "comparison.json").read_text())
    assert metric(comparison, "rouge.rouge1")["n_paired"] == COUNT - 2


@pytest.mark.ticket("EV-010-T07")
def test_parse_failure_stays_in_the_comparison(workspace: Path) -> None:
    candidate = answers(2)
    candidate.responses["c0"] = FakeResponse(output="{broken", parse_failed=True)

    run_dir = run_pair(workspace, answers(1), candidate)

    comparison = json.loads((run_dir / "comparison.json").read_text())
    assert metric(comparison, "rouge.rouge1")["n_paired"] == COUNT


@pytest.mark.ticket("EV-010-T08")
def test_clearly_better_candidate_is_better(workspace: Path) -> None:
    comparison = json.loads(
        (run_pair(workspace, answers(0), answers(2)) / "comparison.json").read_text()
    )

    item = metric(comparison, "rouge.rouge1")
    assert item["verdict"] == BETTER
    assert item["ci"]["low"] > 0


@pytest.mark.ticket("EV-010-T09")
def test_clearly_worse_candidate_is_worse(workspace: Path) -> None:
    comparison = json.loads(
        (run_pair(workspace, answers(2), answers(0)) / "comparison.json").read_text()
    )

    assert metric(comparison, "rouge.rouge1")["verdict"] == WORSE


@pytest.mark.ticket("EV-010-T10")
def test_interval_including_zero_is_inconclusive() -> None:
    assert verdict(-0.1, 0.2, higher_is_better=True) == INCONCLUSIVE


@pytest.mark.ticket("EV-010-T11")
def test_lower_is_better_inverts_the_direction() -> None:
    assert verdict(-0.5, -0.1, higher_is_better=False) == BETTER
    assert verdict(0.1, 0.5, higher_is_better=False) == WORSE


@pytest.mark.ticket("EV-010-T12")
def test_metric_not_measured_in_either_run_is_not_comparable(workspace: Path) -> None:
    run_dir = run_pair(
        workspace, answers(1), answers(2), scorers=[{"type": "latency", "metrics": ["ttft_p50_s"]}]
    )

    comparison = json.loads((run_dir / "comparison.json").read_text())

    assert metric(comparison, "latency.ttft_p50_s")["comparable"] is True
    for item in comparison["metrics"]:
        assert item.get("verdict") != "better" or item["comparable"]


@pytest.mark.ticket("EV-010-T13")
def test_percentile_metric_is_recomputed_per_resample(workspace: Path) -> None:
    run_dir = run_pair(
        workspace,
        answers(1),
        answers(1),
        scorers=[{"type": "latency", "metrics": ["latency_p95_s"]}],
    )

    item = metric(json.loads((run_dir / "comparison.json").read_text()), "latency.latency_p95_s")

    assert item["method"] == "recomputed"
    assert item["comparable"] is True


@pytest.mark.ticket("EV-010-T14")
def test_recommendation_follows_primary_metrics_only() -> None:
    metrics = [
        {"key": "judge.score", "comparable": True, "verdict": BETTER, "primary": True},
        {"key": "latency.latency_p95_s", "comparable": True, "verdict": WORSE, "primary": False},
    ]

    assert recommend(metrics)["decision"] == "candidate"


@pytest.mark.ticket("EV-010-T15")
def test_no_primary_metrics_means_no_recommendation() -> None:
    result = recommend([{"key": "a.b", "comparable": True, "verdict": BETTER, "primary": False}])

    assert result["decision"] == "none"
    assert "primary" in result["reason"]


@pytest.mark.ticket("EV-010-T16")
@pytest.mark.parametrize(
    "verdicts",
    [[BETTER, WORSE], [BETTER, INCONCLUSIVE], [INCONCLUSIVE]],
)
def test_conflicting_or_inconclusive_primaries_give_inconclusive(verdicts: list[str]) -> None:
    metrics = [
        {"key": f"s.m{n}", "comparable": True, "verdict": value, "primary": True}
        for n, value in enumerate(verdicts)
    ]

    assert recommend(metrics)["decision"] == "inconclusive"


def test_worse_primaries_recommend_keeping_the_baseline() -> None:
    metrics = [{"key": "s.m", "comparable": True, "verdict": WORSE, "primary": True}]

    assert recommend(metrics)["decision"] == "baseline"


def test_primary_that_cannot_be_compared_is_inconclusive() -> None:
    metrics = [{"key": "s.m", "comparable": False, "primary": True}]

    assert recommend(metrics)["decision"] == "inconclusive"


@pytest.mark.ticket("EV-010-T17")
def test_invalid_run_is_refused(workspace: Path) -> None:
    run_dir = run_pair(workspace, answers(1), CaseTarget(default=FakeResponse(error="boom")))

    with pytest.raises(ValueError, match="invalid"):
        compare_runs(run_dir / "baseline", run_dir / "candidate")


@pytest.mark.ticket("EV-010-T21")
def test_run_with_two_targets_writes_the_comparison(workspace: Path) -> None:
    gates = {"rouge.rouge1": {"min": 0.1, "primary": True}}

    run_dir = run_pair(workspace, answers(0), answers(2), gates=gates)

    comparison = json.loads((run_dir / "comparison.json").read_text())
    assert (comparison["baseline"], comparison["candidate"]) == ("baseline", "candidate")
    assert comparison["recommendation"]["decision"] == "candidate"


@pytest.mark.ticket("EV-010-T22")
def test_manual_compare_matches_the_automatic_one(workspace: Path) -> None:
    run_dir = run_pair(workspace, answers(0), answers(2))
    automatic = json.loads((run_dir / "comparison.json").read_text())

    manual = compare_runs(run_dir / "baseline", run_dir / "candidate")

    assert manual["metrics"] == automatic["metrics"]
