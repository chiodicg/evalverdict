"""EV-009: the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from evalverdict.cli import app
from fakes.target import INSTANCES, CaseTarget, FakeResponse

runner = CliRunner()


@pytest.mark.ticket("EV-009-T20")
def test_scorers_list_shows_types_and_metrics() -> None:
    result = runner.invoke(app, ["scorers", "list"])

    assert result.exit_code == 0
    assert "rouge" in result.output
    assert "rouge1" in result.output
    assert "judge" in result.output


@pytest.mark.ticket("EV-009-T21")
def test_scorers_describe_shows_params_types_and_defaults() -> None:
    result = runner.invoke(app, ["scorers", "describe", "bertscore", "--json"])

    assert result.exit_code == 0
    described = json.loads(result.output)
    assert described["params"]["model_type"] == {"type": "string", "default": "roberta-large"}
    assert "f1" in described["metrics"]


def test_required_params_are_marked_required() -> None:
    described = json.loads(runner.invoke(app, ["scorers", "describe", "judge", "--json"]).output)

    assert described["params"]["rubric"]["required"] is True


@pytest.mark.ticket("EV-009-T22")
def test_describe_unknown_scorer_exits_three_with_options() -> None:
    result = runner.invoke(app, ["scorers", "describe", "nope"])

    assert result.exit_code == 3
    assert "rouge" in result.output


@pytest.mark.ticket("EV-009-T28")
@pytest.mark.parametrize(
    "command",
    [[], ["run"], ["validate"], ["scorers"], ["scorers", "list"], ["scorers", "describe"]],
)
def test_every_command_has_help(command: list[str]) -> None:
    result = runner.invoke(app, [*command, "--help"])

    assert result.exit_code == 0
    assert "Usage" in result.output


def run_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: CaseTarget, gates: dict[str, Any]
) -> Any:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text(
        "".join(f'{{"id": "c{n}", "input": "x", "reference": "alpha"}}\n' for n in range(30))
    )
    INSTANCES["cli"] = target
    config = {
        "name": "demo",
        "dataset": str(data),
        "targets": {"a": {"kind": "instance", "ref": "cli"}},
        "scorers": [{"type": "rouge", "metrics": ["rouge1"]}],
        "bootstrap": {"n_resamples": 50},
        "gates": gates,
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    try:
        return runner.invoke(app, ["run", "-c", str(path)])
    finally:
        INSTANCES.clear()


@pytest.mark.ticket("EV-009-T12")
@pytest.mark.parametrize(
    ("output", "threshold", "code"),
    [("alpha", 0.5, 0), ("beta", 0.5, 1)],
)
def test_run_exit_code_follows_the_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str, threshold: float, code: int
) -> None:
    target = CaseTarget(default=FakeResponse(output=output))

    result = run_with(tmp_path, monkeypatch, target, {"rouge.rouge1": {"min": threshold}})

    assert result.exit_code == code, result.output


@pytest.mark.ticket("EV-009-T12")
def test_run_with_a_straddling_gate_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = {f"c{n}": FakeResponse(output="alpha" if n % 2 else "beta") for n in range(30)}

    result = run_with(tmp_path, monkeypatch, CaseTarget(responses), {"rouge.rouge1": {"min": 0.5}})

    assert result.exit_code == 2, result.output


@pytest.mark.ticket("EV-009-T14")
def test_invalid_run_exits_three(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = CaseTarget(default=FakeResponse(error="boom"))

    result = run_with(tmp_path, monkeypatch, target, {"rouge.rouge1": {"min": 0.1}})

    assert result.exit_code == 3
    assert "invalid" in result.output
