"""EV-004: concurrency, failure policy, budget, resume and cached reruns."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from evalverdict.config import RunConfig
from evalverdict.runner import execute
from fakes.target import INSTANCES, CaseTarget, FakeResponse


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    INSTANCES.clear()


def dataset(folder: Path, count: int) -> Path:
    path = folder / "data.jsonl"
    rows = [
        json.dumps({"id": f"c{index}", "input": f"in {index}", "reference": "alpha beta"})
        for index in range(count)
    ]
    path.write_text("\n".join(rows) + "\n")
    return path


def make_config(folder: Path, target: CaseTarget, count: int = 20, **extra: Any) -> RunConfig:
    INSTANCES["t"] = target
    settings: dict[str, Any] = {
        "name": "demo",
        "seed": 1,
        "dataset": dataset(folder, count),
        "targets": {"a": {"kind": "instance", "ref": "t"}},
        "scorers": [{"type": "rouge", "metrics": ["rouge1"]}],
        "bootstrap": {"n_resamples": 50},
    }
    return RunConfig(**{**settings, **extra})


def read_scores(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "a" / "scores.json").read_text())


def read_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "a" / "manifest.json").read_text())


def errors_for(count: int) -> dict[str, FakeResponse]:
    return {f"c{index}": FakeResponse(error="boom") for index in range(count)}


@pytest.mark.ticket("EV-004-T06")
def test_concurrency_limit_is_respected_and_recorded(workspace: Path) -> None:
    target = CaseTarget(delay_s=0.01)

    run_dir = execute(make_config(workspace, target, concurrency=2))

    assert target.max_in_flight == 2
    assert read_manifest(run_dir)["concurrency"] == 2


@pytest.mark.ticket("EV-004-T01")
def test_second_run_uses_the_cache_and_makes_no_calls(workspace: Path) -> None:
    target = CaseTarget()
    config = make_config(workspace, target)

    execute(config)
    target.calls.clear()
    run_dir = execute(config)

    assert target.calls == []
    assert read_scores(run_dir)["run"]["cache_hits"] == 20


@pytest.mark.ticket("EV-004-T07")
def test_api_error_is_excluded_from_metrics_and_counted(workspace: Path) -> None:
    run_dir = execute(make_config(workspace, CaseTarget(errors_for(1))))

    scores = read_scores(run_dir)

    assert scores["results"][0]["n"] == 19
    assert scores["run"]["n_errors"] == 1
    assert scores["run"]["error_rate"] == pytest.approx(0.05)


@pytest.mark.ticket("EV-004-T08")
def test_parse_failure_stays_in_metrics_and_is_counted(workspace: Path) -> None:
    bad = {"c0": FakeResponse(output="{broken", parse_failed=True)}

    run_dir = execute(make_config(workspace, CaseTarget(bad)))

    scores = read_scores(run_dir)
    assert scores["results"][0]["n"] == 20
    assert scores["run"]["parse_failure_rate"] == pytest.approx(0.05)


@pytest.mark.ticket("EV-004-T09")
@pytest.mark.parametrize(("errors", "status"), [(1, "complete"), (2, "invalid")])
def test_run_is_invalid_only_above_the_error_limit(
    workspace: Path, errors: int, status: str
) -> None:
    run_dir = execute(make_config(workspace, CaseTarget(errors_for(errors))))

    assert read_scores(run_dir)["run"]["status"] == status
    assert read_manifest(run_dir)["status"] == status


def test_max_error_rate_setting_is_used(workspace: Path) -> None:
    config = make_config(workspace, CaseTarget(errors_for(2)), max_error_rate=0.2)

    assert read_scores(execute(config))["run"]["status"] == "complete"


def test_all_items_failing_gives_an_invalid_run_with_no_results(workspace: Path) -> None:
    run_dir = execute(make_config(workspace, CaseTarget(default=FakeResponse(error="boom"))))

    scores = read_scores(run_dir)
    assert scores["run"]["status"] == "invalid"
    assert scores["results"] == []


@pytest.mark.ticket("EV-004-T10")
def test_budget_stops_new_calls_and_keeps_partial_results(workspace: Path) -> None:
    target = CaseTarget(paid=True, default=FakeResponse(output="alpha", cost_usd=0.1))

    run_dir = execute(make_config(workspace, target, budget_usd=0.55, concurrency=1))

    scores = read_scores(run_dir)
    assert len(target.calls) == 5
    assert scores["run"]["status"] == "incomplete"
    assert scores["run"]["stop_reason"] == "budget"
    assert scores["results"][0]["n"] == 5


@pytest.mark.ticket("EV-004-T11")
def test_budget_allows_for_calls_in_flight(workspace: Path) -> None:
    target = CaseTarget(paid=True, default=FakeResponse(output="alpha", cost_usd=0.1), delay_s=0.01)

    run_dir = execute(make_config(workspace, target, budget_usd=1.05, concurrency=4))

    assert read_manifest(run_dir)["spend"]["usd"] <= 1.05


@pytest.mark.ticket("EV-004-T11")
def test_first_call_runs_alone_because_its_cost_is_unknown(workspace: Path) -> None:
    target = CaseTarget(paid=True, default=FakeResponse(output="alpha", cost_usd=0.5), delay_s=0.01)

    execute(make_config(workspace, target, budget_usd=0.6, concurrency=4))

    assert len(target.calls) == 1


@pytest.mark.ticket("EV-004-T12")
def test_paid_target_without_a_cost_source_is_refused_before_any_call(workspace: Path) -> None:
    target = CaseTarget(paid=True, cost_known=False)

    with pytest.raises(ValueError, match="cost"):
        execute(make_config(workspace, target, budget_usd=5))

    assert target.calls == []


@pytest.mark.ticket("EV-004-T13")
def test_paid_run_needs_a_budget_and_a_free_run_does_not(workspace: Path) -> None:
    paid = CaseTarget(paid=True)

    with pytest.raises(ValueError, match="budget_usd"):
        execute(make_config(workspace, paid))

    assert paid.calls == []
    execute(make_config(workspace, CaseTarget()))


@pytest.mark.ticket("EV-004-T14")
def test_interrupt_flushes_partial_results_as_incomplete(workspace: Path) -> None:
    target = CaseTarget(interrupt_on="c5")

    run_dir = execute(make_config(workspace, target, concurrency=1))

    scores = read_scores(run_dir)
    assert scores["run"]["status"] == "incomplete"
    assert scores["run"]["stop_reason"] == "interrupted"
    assert scores["results"][0]["n"] == 5


@pytest.mark.ticket("EV-004-T15")
def test_resume_calls_only_the_remaining_items(workspace: Path) -> None:
    target = CaseTarget(paid=True, default=FakeResponse(output="alpha", cost_usd=0.1))
    first = make_config(workspace, target, budget_usd=0.55, concurrency=1)
    run_dir = execute(first)
    target.calls.clear()

    raised = first.model_copy(update={"budget_usd": 5.0})
    resumed = execute(raised, resume=run_dir)

    assert resumed == run_dir
    assert len(target.calls) == 15
    assert read_scores(run_dir)["run"]["status"] == "complete"
    assert read_scores(run_dir)["results"][0]["n"] == 20


@pytest.mark.ticket("EV-004-T16")
@pytest.mark.parametrize(
    "change",
    [
        lambda config, folder: config.model_copy(update={"seed": 99}),
        lambda config, folder: config.model_copy(update={"dataset": dataset(folder, 25)}),
    ],
)
def test_resume_with_a_changed_config_or_dataset_is_refused(
    workspace: Path, change: Callable[[RunConfig, Path], RunConfig]
) -> None:
    config = make_config(workspace, CaseTarget())
    run_dir = execute(config)

    with pytest.raises(ValueError, match="resume"):
        execute(change(config, workspace), resume=run_dir)


def test_resume_refuses_a_directory_that_is_not_a_run(workspace: Path) -> None:
    with pytest.raises(ValueError, match="resume"):
        execute(make_config(workspace, CaseTarget()), resume=workspace / "missing")


@pytest.mark.ticket("EV-004-T17")
def test_rerun_from_cache_is_identical_without_model_calls(workspace: Path) -> None:
    target = CaseTarget()
    config = make_config(workspace, target)
    first = read_scores(execute(config))
    target.calls.clear()
    target.interrupt_on = "c0"

    second = read_scores(execute(config))

    assert target.calls == []
    assert second["results"] == first["results"]


@pytest.mark.ticket("EV-004-T18")
def test_repeats_count_items_and_keep_each_repeat(workspace: Path) -> None:
    target = CaseTarget()

    run_dir = execute(make_config(workspace, target, count=10, repeats=3))

    scores = read_scores(run_dir)
    assert len(target.calls) == 30
    assert scores["results"][0]["n"] == 10
    assert scores["run"]["repeats"] == 3
    rows = (run_dir / "a" / "predictions.jsonl").read_text().splitlines()
    assert sorted({json.loads(row)["repeat"] for row in rows}) == [0, 1, 2]


@pytest.mark.ticket("EV-004-T19")
def test_final_spend_is_the_sum_of_call_costs(workspace: Path) -> None:
    costs = {f"c{index}": FakeResponse(output="alpha", cost_usd=0.01 * index) for index in range(4)}
    target = CaseTarget(costs, paid=True)

    run_dir = execute(make_config(workspace, target, count=4, budget_usd=5))

    assert read_manifest(run_dir)["spend"] == {
        "usd": pytest.approx(0.06),
        "calls": 4,
        "cache_hits": 0,
        "unmeasured_calls": 0,
    }


def test_spend_covers_all_targets_in_a_run(workspace: Path) -> None:
    INSTANCES["t2"] = CaseTarget(paid=True, default=FakeResponse(output="b", cost_usd=0.1))
    first = CaseTarget(paid=True, default=FakeResponse(output="a", cost_usd=0.1))
    targets = {"a": {"kind": "instance", "ref": "t"}, "b": {"kind": "instance", "ref": "t2"}}

    execute(make_config(workspace, first, count=10, budget_usd=1.5, targets=targets, concurrency=1))

    assert len(first.calls) + len(INSTANCES["t2"].calls) == 15
