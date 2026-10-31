"""EV-003: dataset loading and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalverdict.data import DatasetError, load_dataset


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


@pytest.mark.ticket("EV-003-T04")
def test_valid_rows_load_with_metadata_preserved(tmp_path: Path) -> None:
    file = write(
        tmp_path / "d.jsonl",
        '{"id": "a", "input": "in", "reference": "ref", "metadata": {"group": "x"}, "graded": 1}\n',
    )

    case = load_dataset(file).cases[0]

    assert (case.id, case.input, case.reference) == ("a", "in", "ref")
    assert case.metadata == {"group": "x"}
    assert case.extra == {"graded": 1}


@pytest.mark.ticket("EV-003-T05")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"id": "a", "input": "x"}\n{broken\n', "line 2"),
        ('{"input": "x"}\n', "line 1"),
        ('{"id": "a", "input": "x"}\n{"id": "a", "input": "y"}\n', "duplicate id 'a'"),
    ],
)
def test_bad_rows_are_rejected_with_their_location(
    tmp_path: Path, text: str, expected: str
) -> None:
    with pytest.raises(DatasetError, match=expected):
        load_dataset(write(tmp_path / "d.jsonl", text))


@pytest.mark.ticket("EV-003-T06")
def test_empty_dataset_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(write(tmp_path / "d.jsonl", "\n"))


def test_missing_file_is_a_dataset_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "missing.jsonl")


@pytest.mark.ticket("EV-003-T07")
def test_hash_is_stable_and_changes_with_one_byte(tmp_path: Path) -> None:
    text = '{"id": "a", "input": "x"}\n'
    first = load_dataset(write(tmp_path / "a.jsonl", text)).sha256
    same = load_dataset(write(tmp_path / "b.jsonl", text)).sha256
    changed = load_dataset(write(tmp_path / "c.jsonl", text.replace("x", "y"))).sha256

    assert first == same
    assert first != changed
