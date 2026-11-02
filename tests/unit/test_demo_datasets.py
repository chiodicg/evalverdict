"""EV-012: committed datasets match their recorded provenance."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalverdict.cli import app
from evalverdict.data import load_dataset

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "datasets" / "README.md").read_text()
TABLE = re.findall(r"^\| (\S+\.jsonl) \| (\d+) \| ([0-9a-f]{64}) \| (.+?) \|$", README, re.M)
FILES = sorted(path.name for path in (ROOT / "datasets").glob("*.jsonl"))


def test_every_committed_dataset_is_in_the_readme_table() -> None:
    assert sorted(name for name, *_ in TABLE) == FILES


@pytest.mark.ticket("EV-012-T01")
@pytest.mark.parametrize("name", FILES)
def test_rows_have_id_input_and_reference(name: str) -> None:
    for case in load_dataset(ROOT / "datasets" / name).cases:
        assert case.id and case.input and case.reference


@pytest.mark.ticket("EV-012-T02")
@pytest.mark.parametrize("name", FILES)
def test_ids_and_inputs_are_unique(name: str) -> None:
    cases = load_dataset(ROOT / "datasets" / name).cases

    assert len({case.input for case in cases}) == len(cases)


@pytest.mark.ticket("EV-012-T03")
@pytest.mark.parametrize(("name", "rows", "sha256", "licence"), TABLE)
def test_hash_and_row_count_match_the_readme(
    name: str, rows: str, sha256: str, licence: str
) -> None:
    raw = (ROOT / "datasets" / name).read_bytes()

    assert hashlib.sha256(raw).hexdigest() == sha256
    assert len(raw.decode().splitlines()) == int(rows)


@pytest.mark.ticket("EV-012-T04")
@pytest.mark.parametrize("name", FILES)
def test_readme_records_source_licence_and_build(name: str) -> None:
    section = README.split(f"## {name}", 1)[1]

    for field in ("**Source:**", "**Licence:**", "**Build:**", "**Approval:**"):
        assert field in section


@pytest.mark.ticket("EV-012-T06")
def test_no_contact_details_are_committed() -> None:
    for line in (ROOT / "datasets" / "summaries.jsonl").read_text().splitlines():
        row = json.dumps(json.loads(line))
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", row)
        assert not re.search(r"\b0\d{3,4}\s?\d{3}\s?\d{3,4}\b", row)


@pytest.mark.ticket("EV-012-T05")
def test_demo_config_validates_without_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)

    result = CliRunner().invoke(app, ["validate", "-c", "configs/demo_generation.yaml"])

    assert result.exit_code == 0, result.output
    assert "400 target calls, 400 judge calls" in result.output
