"""EV-009: validate, estimates and config checks before any spend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from evalverdict.cli import app
from evalverdict.config import RunConfig
from evalverdict.estimate import estimate_run
from fakes.target import INSTANCES, CaseTarget

ROOT = Path(__file__).resolve().parents[2]
RUBRIC = str(ROOT / "rubrics" / "summary_quality.yaml")
runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text(
        "".join(f'{{"id": "c{n}", "input": "x", "reference": "y"}}\n' for n in range(10))
    )
    return tmp_path


def settings(workspace: Path, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "demo",
        "dataset": str(workspace / "data.jsonl"),
        "targets": {"a": {"kind": "fake"}},
        "scorers": [{"type": "rouge", "metrics": ["rouge1"]}],
    }
    return {**base, **overrides}


def write(workspace: Path, values: dict[str, Any]) -> Path:
    path = workspace / "run.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def validate(workspace: Path, values: dict[str, Any]) -> Any:
    return runner.invoke(app, ["validate", "-c", str(write(workspace, values))])


@pytest.mark.ticket("EV-009-T15")
def test_valid_config_passes_and_prints_an_estimate(workspace: Path) -> None:
    result = validate(workspace, settings(workspace))

    assert result.exit_code == 0, result.output
    assert "10 target calls" in result.output
    assert not (workspace / "runs").exists()


@pytest.mark.ticket("EV-009-T16")
def test_estimate_counts_targets_repeats_and_judge_calls() -> None:
    values = settings(
        Path("."),
        targets={"a": {"kind": "fake"}, "b": {"kind": "fake"}},
        repeats=3,
        judge={"model": "anthropic/claude-sonnet-5-5"},
        scorers=[{"type": "judge", "params": {"rubric": RUBRIC}}, {"type": "rouge"}],
    )
    values["dataset"] = None

    estimate = estimate_run(RunConfig.model_validate({**values, "dataset": "x"}), n_cases=10)

    assert estimate.target_calls == 2 * 10 * 3
    assert estimate.judge_calls == 2 * 10 * 3
    assert estimate.judge_cost_usd == pytest.approx(60 * (2000 * 2e-6 + 500 * 1e-5))


def test_estimate_prices_paid_targets_and_leaves_free_ones_at_zero() -> None:
    config = RunConfig.model_validate(
        settings(
            Path("."),
            dataset="x",
            targets={"a": {"kind": "litellm", "model": "openai/gpt-6-luna"}, "b": {"kind": "fake"}},
        )
    )

    estimate = estimate_run(config, n_cases=100)

    assert estimate.target_cost_usd == pytest.approx(100 * (1500 * 1e-7 + 300 * 5e-7))


@pytest.mark.ticket("EV-009-T17")
def test_paid_model_without_a_cost_source_is_refused(workspace: Path) -> None:
    target = {"kind": "litellm", "model": "unmapped/model"}

    result = validate(workspace, settings(workspace, targets={"a": target}, budget_usd=5))

    assert result.exit_code == 3
    assert "cost source" in result.output


@pytest.mark.ticket("EV-009-T18")
def test_estimate_above_budget_warns(workspace: Path) -> None:
    target = {"kind": "litellm", "model": "openai/gpt-6-luna"}

    result = validate(workspace, settings(workspace, targets={"a": target}, budget_usd=0.00001))

    assert "exceeds budget_usd" in result.output


@pytest.mark.ticket("EV-009-T19")
def test_unknown_param_names_scorer_and_key(workspace: Path) -> None:
    scorers = [{"type": "rouge", "params": {"stemmer": True}}]

    result = validate(workspace, settings(workspace, scorers=scorers))

    assert result.exit_code == 3
    assert "stemmer" in result.output


@pytest.mark.ticket("EV-009-T10")
@pytest.mark.parametrize(
    ("scorers", "gate"),
    [
        ([{"type": "rouge", "enabled": False}], "rouge.rouge1"),
        ([{"type": "rouge", "metrics": ["rouge2"]}], "rouge.rouge1"),
    ],
)
def test_gate_on_disabled_scorer_or_unselected_metric_is_rejected(
    workspace: Path, scorers: list[dict[str, Any]], gate: str
) -> None:
    values = settings(workspace, scorers=scorers, gates={gate: {"min": 0.5}})

    result = validate(workspace, values)

    assert result.exit_code == 3
    assert gate in result.output


@pytest.mark.ticket("EV-009-T11")
def test_gate_on_an_unknown_metric_lists_the_options(workspace: Path) -> None:
    values = settings(workspace, gates={"rouge.nope": {"min": 0.5}})

    result = validate(workspace, values)

    assert result.exit_code == 3
    assert "rouge1" in result.output


def test_validate_makes_no_target_calls(workspace: Path) -> None:
    INSTANCES["v"] = target = CaseTarget()
    values = settings(workspace, targets={"a": {"kind": "instance", "ref": "v"}})

    validate(workspace, values)

    assert target.calls == []
    INSTANCES.clear()
