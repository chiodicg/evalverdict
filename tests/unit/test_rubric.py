"""EV-008: rubric files."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalverdict.rubric import RubricError, load_rubric

VALID = "name: quality\nversion: 1\nfields: [input, actual_output]\nsteps:\n  - Check it.\n"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "rubric.yaml"
    path.write_text(text)
    return path


@pytest.mark.ticket("EV-008-T01")
def test_valid_rubric_loads_with_a_stable_hash(tmp_path: Path) -> None:
    first = load_rubric(write(tmp_path, VALID))
    second = load_rubric(write(tmp_path, VALID))

    assert first.name == "quality"
    assert first.steps == ("Check it.",)
    assert first.sha256 == second.sha256


@pytest.mark.ticket("EV-008-T02")
def test_one_character_change_changes_the_hash(tmp_path: Path) -> None:
    first = load_rubric(write(tmp_path, VALID))
    second = load_rubric(write(tmp_path, VALID.replace("Check it.", "Check it!")))

    assert first.sha256 != second.sha256


@pytest.mark.ticket("EV-008-T03")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("name: quality\nversion: 1\nfields: [input]\n", "steps"),
        ("version: 1\nfields: [input]\nsteps: [a]\n", "name"),
        ("name: q\nversion: 1\nfields: [wrong]\nsteps: [a]\n", "fields"),
        ("name: q\nversion: 1\nfields: [input]\nsteps: []\n", "steps"),
        ("{not yaml: [", "rubric.yaml"),
    ],
)
def test_bad_rubric_error_names_the_file_and_field(
    tmp_path: Path, text: str, expected: str
) -> None:
    with pytest.raises(RubricError, match=expected) as excinfo:
        load_rubric(write(tmp_path, text))

    assert "rubric.yaml" in str(excinfo.value)


def test_missing_rubric_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RubricError, match="not found"):
        load_rubric(tmp_path / "missing.yaml")


def test_shipped_rubric_is_valid() -> None:
    rubric = load_rubric(Path(__file__).resolve().parents[2] / "rubrics" / "summary_quality.yaml")

    assert rubric.name == "summary_quality"
    assert len(rubric.steps) >= 3
