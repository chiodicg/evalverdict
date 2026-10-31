"""EV-003: an end-to-end run with a fake target."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import fakes.target  # noqa: F401  registers kind: fake
from evalverdict.cli import app

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "generation.jsonl"
runner = CliRunner()


def write_config(
    folder: Path, *, targets: str = "a: {kind: fake, seed: 1}", extra: str = ""
) -> Path:
    path = folder / "run.yaml"
    path.write_text(
        f"name: demo\nseed: 5\ndataset: {FIXTURE}\n"
        f"bootstrap: {{n_resamples: 200}}\n"
        f"targets:\n  {targets}\n"
        f"scorers:\n  - {{type: rouge, metrics: [rouge1]}}\n{extra}"
    )
    return path


def target_dir(root: Path, target: str = "a") -> Path:
    return next((root / "runs").glob(f"*-demo/{target}"))


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.ticket("EV-003-T19")
def test_run_writes_all_outputs_with_intervals(workspace: Path) -> None:
    result = runner.invoke(app, ["run", "-c", str(write_config(workspace))])

    assert result.exit_code == 0, result.output
    folder = target_dir(workspace)
    assert {path.name for path in folder.iterdir()} == {
        "manifest.json",
        "predictions.jsonl",
        "scores.json",
    }
    score = json.loads((folder / "scores.json").read_text())["results"][0]
    assert (score["scorer"], score["name"], score["n"]) == ("rouge", "rouge1", 4)
    assert score["ci"]["low"] <= score["value"] <= score["ci"]["high"]
    assert len((folder / "predictions.jsonl").read_text().splitlines()) == 4


@pytest.mark.ticket("EV-003-T20")
def test_each_target_gets_its_own_directory(workspace: Path) -> None:
    config = write_config(workspace, targets="a: {kind: fake, seed: 1}\n  b: {kind: fake, seed: 2}")

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code == 0, result.output
    assert target_dir(workspace, "a").is_dir()
    assert target_dir(workspace, "b").is_dir()


@pytest.mark.ticket("EV-003-T21")
def test_same_config_and_seed_give_identical_scores(workspace: Path) -> None:
    config = write_config(workspace)

    runner.invoke(app, ["run", "-c", str(config)])
    first = sorted((workspace / "runs").glob("*-demo/a/scores.json"))[0].read_text()
    for folder in (workspace / "runs").iterdir():
        folder.rename(workspace / f"old-{folder.name}")
    runner.invoke(app, ["run", "-c", str(config)])
    second = sorted((workspace / "runs").glob("*-demo/a/scores.json"))[0].read_text()

    assert first == second


@pytest.mark.ticket("EV-003-T22")
def test_manifest_records_hashes_versions_and_resolved_settings(workspace: Path) -> None:
    extra = "  - {type: rouge, name: unused, enabled: false}\n"
    runner.invoke(app, ["run", "-c", str(write_config(workspace, extra=extra))])

    manifest = json.loads((target_dir(workspace) / "manifest.json").read_text())

    assert len(manifest["config_sha256"]) == 64
    assert len(manifest["dataset"]["sha256"]) == 64
    assert manifest["dataset"]["n"] == 4
    assert manifest["seed"] == 5
    assert "numpy" in manifest["versions"]
    assert "git_sha" in manifest
    resolved = {entry["name"]: entry for entry in manifest["scorers"]}
    assert resolved["rouge"]["params"]["use_stemmer"] is False
    assert resolved["rouge"]["bootstrap"]["n_resamples"] == 200
    assert resolved["unused"]["enabled"] is False


@pytest.mark.ticket("EV-003-T10")
def test_disabled_scorer_does_not_run_and_is_recorded(workspace: Path) -> None:
    extra = "  - {type: rouge, name: unused, enabled: false}\n"
    runner.invoke(app, ["run", "-c", str(write_config(workspace, extra=extra))])

    scores = json.loads((target_dir(workspace) / "scores.json").read_text())

    assert {score["scorer"] for score in scores["results"]} == {"rouge"}
    assert scores["disabled"] == ["unused"]


@pytest.mark.ticket("EV-003-T11")
def test_metric_subset_produces_only_that_metric(workspace: Path) -> None:
    runner.invoke(app, ["run", "-c", str(write_config(workspace))])

    scores = json.loads((target_dir(workspace) / "scores.json").read_text())

    assert [score["name"] for score in scores["results"]] == ["rouge1"]


@pytest.mark.ticket("EV-003-T13")
def test_same_type_under_two_names_runs_with_separate_results(workspace: Path) -> None:
    extra = "  - {type: rouge, name: stemmed, metrics: [rouge1], params: {use_stemmer: true}}\n"
    runner.invoke(app, ["run", "-c", str(write_config(workspace, extra=extra))])

    scores = json.loads((target_dir(workspace) / "scores.json").read_text())

    assert {score["scorer"] for score in scores["results"]} == {"rouge", "stemmed"}


@pytest.mark.ticket("EV-003-T23")
def test_rouge_runs_under_the_network_guard(workspace: Path) -> None:
    result = runner.invoke(app, ["run", "-c", str(write_config(workspace))])

    assert result.exit_code == 0, result.output


@pytest.mark.ticket("EV-003-T24")
def test_missing_extra_gives_an_install_hint_and_no_partial_run(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "rouge_score", None)

    result = runner.invoke(app, ["run", "-c", str(write_config(workspace))])

    assert result.exit_code == 3
    assert 'pip install "evalverdict[nlp]"' in result.output
    assert not (workspace / "runs").exists()


def test_invalid_config_exits_with_code_three(workspace: Path) -> None:
    path = workspace / "bad.yaml"
    path.write_text("name: demo\n")

    result = runner.invoke(app, ["run", "-c", str(path)])

    assert result.exit_code == 3
    assert "dataset" in result.output


def test_unknown_target_kind_exits_with_code_three(workspace: Path) -> None:
    config = write_config(workspace, targets="a: {kind: nonexistent}")

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code == 3
    assert "nonexistent" in result.output
