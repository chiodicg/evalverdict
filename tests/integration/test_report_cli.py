"""EV-011: reports are written by run and rebuilt by report, with no model calls."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from evalverdict.cli import app
from fakes.target import INSTANCES, CaseTarget, FakeResponse

runner = CliRunner()


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text(
        "".join(
            json.dumps({"id": f"c{n}", "input": "x", "reference": "alpha"}) + "\n"
            for n in range(10)
        )
    )
    INSTANCES["b"] = CaseTarget(default=FakeResponse(output="alpha"))
    INSTANCES["c"] = CaseTarget(default=FakeResponse(output="alpha beta"))
    config = {
        "name": "demo",
        "dataset": str(data),
        "targets": {
            "baseline": {"kind": "instance", "ref": "b"},
            "candidate": {"kind": "instance", "ref": "c"},
        },
        "scorers": [{"type": "rouge", "metrics": ["rouge1"]}, {"type": "latency"}],
        "bootstrap": {"n_resamples": 50},
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config))
    runner.invoke(app, ["run", "-c", "run.yaml"])
    yield next((tmp_path / "runs").glob("*-demo"))
    INSTANCES.clear()


def test_run_writes_target_and_comparison_reports(run_dir: Path) -> None:
    assert (run_dir / "baseline" / "report.html").is_file()
    assert (run_dir / "candidate" / "report.html").is_file()
    assert (run_dir / "comparison.html").is_file()


@pytest.mark.ticket("EV-011-T23")
def test_report_command_rebuilds_identically_with_no_model_calls(run_dir: Path) -> None:
    before = (run_dir / "baseline" / "report.html").read_text()
    (run_dir / "baseline" / "report.html").unlink()
    INSTANCES["b"].interrupt_on = "c0"
    INSTANCES["b"].calls.clear()

    result = runner.invoke(app, ["report", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert (run_dir / "baseline" / "report.html").read_text() == before
    assert INSTANCES["b"].calls == []


@pytest.mark.ticket("EV-011-T25")
def test_report_on_a_folder_that_is_not_a_run_exits_three(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", str(tmp_path)])

    assert result.exit_code == 3
