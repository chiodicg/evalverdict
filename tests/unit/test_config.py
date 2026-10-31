"""EV-003: run configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from evalverdict.config import RunConfig, load_config

VALID = {
    "name": "demo",
    "dataset": "data.jsonl",
    "targets": {"a": {"kind": "fake"}},
    "scorers": [{"type": "rouge"}],
}


@pytest.mark.ticket("EV-003-T01")
def test_valid_config_parses_to_typed_objects(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        "name: demo\nseed: 7\ndataset: data.jsonl\n"
        "targets:\n  a: {kind: fake, count: 3}\n"
        "scorers:\n  - {type: rouge, metrics: [rouge1]}\n"
    )

    config = load_config(path)

    assert config.seed == 7
    assert config.targets["a"].kind == "fake"
    assert config.scorers[0].metrics == ["rouge1"]
    assert config.scorers[0].resolved_name == "rouge"


@pytest.mark.ticket("EV-003-T02")
@pytest.mark.parametrize("field", ["dataset", "targets", "scorers"])
def test_missing_required_field_is_named(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        RunConfig(**{key: value for key, value in VALID.items() if key != field})


@pytest.mark.ticket("EV-003-T03")
def test_unknown_top_level_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="surprise"):
        RunConfig(**VALID, surprise=True)


@pytest.mark.ticket("EV-003-T14")
def test_two_scorers_with_the_same_name_are_rejected() -> None:
    entries = [{"type": "rouge"}, {"type": "rouge"}]

    with pytest.raises(ValidationError, match="rouge"):
        RunConfig(**{**VALID, "scorers": entries})


def test_empty_targets_or_scorers_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RunConfig(**{**VALID, "targets": {}})
    with pytest.raises(ValidationError):
        RunConfig(**{**VALID, "scorers": []})


def test_config_hash_is_stable_and_sensitive() -> None:
    assert RunConfig(**VALID).sha256() == RunConfig(**VALID).sha256()
    assert RunConfig(**VALID).sha256() != RunConfig(**{**VALID, "seed": 9}).sha256()
