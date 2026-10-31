"""EV-001: packaging, optional dependencies and secret-file rules."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC_EXTRAS = {"nlp", "judge", "redteam", "tracing", "drift", "fairness", "explain"}
HEAVY_PACKAGES = (
    "torch",
    "transformers",
    "deepeval",
    "inspect_ai",
    "langfuse",
    "evidently",
    "fairlearn",
    "shap",
    "lime",
)


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9._-]+", requirement)
    assert match is not None
    return match.group(0).lower().replace("_", "-")


@pytest.mark.ticket("EV-001-T01")
def test_package_imports_with_every_heavy_dependency_blocked() -> None:
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        BLOCKED = {HEAVY_PACKAGES!r}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked: {{name}}")
                return None

        sys.meta_path.insert(0, Blocker())
        import evalverdict
        import evalverdict.cli
        print(evalverdict.__version__)
        """
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


@pytest.mark.ticket("EV-001-T02")
@pytest.mark.parametrize(
    ("module", "extra"),
    [
        ("evaluate", "nlp"),
        ("bert_score", "nlp"),
        ("rouge_score", "nlp"),
        ("deepeval", "judge"),
        ("inspect_ai", "redteam"),
        ("inspect_evals", "redteam"),
        ("langfuse", "tracing"),
        ("evidently", "drift"),
        ("fairlearn", "fairness"),
        ("shap", "explain"),
        ("lime", "explain"),
    ],
)
def test_missing_extra_names_the_install_command(
    module: str, extra: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evalverdict._optional import MissingExtraError, optional_import

    monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(MissingExtraError) as excinfo:
        optional_import(module)

    assert f'pip install "evalverdict[{extra}]"' in str(excinfo.value)


def test_broken_dependency_inside_an_installed_extra_is_not_reported_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evalverdict import _optional

    def fail_inside_package(module: str) -> None:
        raise ModuleNotFoundError("No module named 'some_inner_dep'", name="some_inner_dep")

    monkeypatch.setattr(_optional.importlib, "import_module", fail_inside_package)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        _optional.optional_import("evidently")

    assert not isinstance(excinfo.value, _optional.MissingExtraError)


@pytest.mark.ticket("EV-001-T03")
def test_core_dependencies_exclude_heavy_packages() -> None:
    core = {_requirement_name(r) for r in _pyproject()["project"]["dependencies"]}

    assert core.isdisjoint({"torch", "transformers", "deepeval", "inspect-ai", "shap"})


@pytest.mark.ticket("EV-001-T04")
def test_python_and_pandas_constraints_match_the_spec() -> None:
    project = _pyproject()["project"]
    pandas = next(r for r in project["dependencies"] if _requirement_name(r) == "pandas")

    assert "<3" in pandas
    assert project["requires-python"].startswith(">=3.12")


@pytest.mark.ticket("EV-001-T05")
def test_every_spec_extra_exists_and_all_includes_them() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]

    assert set(extras) >= SPEC_EXTRAS
    combined = " ".join(extras["all"])
    assert all(extra in combined for extra in SPEC_EXTRAS)


@pytest.mark.ticket("EV-001-T06")
@pytest.mark.parametrize("pattern", [".env", "runs/", "deploy/langfuse/.env", ".venv/"])
def test_secret_and_output_paths_are_gitignored(pattern: str) -> None:
    lines = (ROOT / ".gitignore").read_text().splitlines()

    assert pattern in lines


@pytest.mark.ticket("EV-001-T07")
def test_env_example_names_all_keys_with_empty_values() -> None:
    entries = dict(
        line.split("=", 1)
        for line in (ROOT / ".env.example").read_text().splitlines()
        if line and not line.startswith("#")
    )

    assert {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    } <= set(entries)
    assert not any(entries.values())
