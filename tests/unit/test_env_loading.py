"""EV-035: load API keys from a local .env file."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evalverdict import cli
from evalverdict.env import load_env

pytest_plugins = ("pytester",)

ROOT = Path(__file__).resolve().parents[2]
GUARD_CONFTEST = f"""
import sys

sys.path.insert(0, {str(ROOT / "tests")!r})
from fakes.guards import *
"""


@pytest.fixture
def clean_variable(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a variable with monkeypatch, so anything loaded is removed afterwards."""
    name = "EVALVERDICT_TEST_VALUE"
    monkeypatch.setenv(name, "placeholder")
    monkeypatch.delenv(name)
    return name


@pytest.mark.ticket("EV-035-T01")
def test_loads_values_from_env_file_in_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_variable: str
) -> None:
    (tmp_path / ".env").write_text(f"{clean_variable}=from-file\n")
    monkeypatch.chdir(tmp_path)

    load_env()

    assert os.environ[clean_variable] == "from-file"


@pytest.mark.ticket("EV-035-T02")
def test_exported_value_takes_priority_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_variable: str
) -> None:
    (tmp_path / ".env").write_text(f"{clean_variable}=from-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(clean_variable, "exported")

    load_env()

    assert os.environ[clean_variable] == "exported"


@pytest.mark.ticket("EV-035-T03")
def test_missing_file_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_variable: str
) -> None:
    monkeypatch.chdir(tmp_path)

    load_env()

    assert clean_variable not in os.environ


@pytest.mark.ticket("EV-035-T04")
def test_does_not_pick_up_an_env_file_from_a_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_variable: str
) -> None:
    (tmp_path / ".env").write_text(f"{clean_variable}=from-parent\n")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)

    load_env()

    assert clean_variable not in os.environ


@pytest.mark.ticket("EV-035-T05")
def test_handles_quotes_spaces_export_and_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_variable: str
) -> None:
    (tmp_path / ".env").write_text(f'# comment\n\nexport {clean_variable}="two words"\n')
    monkeypatch.chdir(tmp_path)

    load_env()

    assert os.environ[clean_variable] == "two words"


@pytest.mark.ticket("EV-035-T06")
def test_never_prints_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_variable: str,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / ".env").write_text(f"{clean_variable}=canary-secret\nnot a valid line\n")
    monkeypatch.chdir(tmp_path)

    load_env()

    captured = capfd.readouterr()
    assert "canary-secret" not in captured.out + captured.err + caplog.text


@pytest.mark.ticket("EV-035-T07")
def test_cli_startup_loads_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_variable: str
) -> None:
    (tmp_path / ".env").write_text(f"{clean_variable}=from-file\n")
    monkeypatch.chdir(tmp_path)

    cli.main()

    assert os.environ[clean_variable] == "from-file"


@pytest.mark.ticket("EV-035-T08")
def test_live_test_gets_keys_from_env_file(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(GUARD_CONFTEST)
    pytester.path.joinpath(".env").write_text("ANTHROPIC_API_KEY=from-file\n")
    pytester.makepyfile(
        """
        import os

        import pytest

        @pytest.mark.live("ANTHROPIC_API_KEY")
        def test_paid():
            assert os.environ["ANTHROPIC_API_KEY"] == "from-file"
        """
    )

    result = pytester.runpytest_subprocess(
        "-c", str(ROOT / "pyproject.toml"), "--rootdir", str(pytester.path), "-m", "live"
    )

    result.assert_outcomes(passed=1)


@pytest.mark.ticket("EV-035-T09")
def test_default_test_does_not_get_keys_from_env_file(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(GUARD_CONFTEST)
    pytester.path.joinpath(".env").write_text("ANTHROPIC_API_KEY=from-file\n")
    pytester.makepyfile(
        """
        import os

        def test_free():
            assert "ANTHROPIC_API_KEY" not in os.environ
        """
    )

    result = pytester.runpytest_subprocess(
        "-c", str(ROOT / "pyproject.toml"), "--rootdir", str(pytester.path)
    )

    result.assert_outcomes(passed=1)
