"""EV-009: custom scorers loaded by import path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from evalverdict.config import ScorerEntry
from evalverdict.scoring.base import BootstrapConfig
from evalverdict.scoring.registry import build_scorer

MODULE = """
import numpy as np
from evalverdict.scoring.base import Family, ItemValues, Scorer


class LengthScorer(Scorer):
    family = Family.GENERATION
    metrics = ("length",)

    def item_values(self, predictions, references, metadata):
        return ItemValues(arrays={"length": np.array([len(str(p)) for p in predictions], float)})


class NotAScorer:
    pass
"""


@pytest.fixture
def custom_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    (tmp_path / "my_scorers.py").write_text(MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "my_scorers"
    sys.modules.pop("my_scorers", None)


@pytest.mark.ticket("EV-009-T23")
def test_custom_scorer_runs_with_an_interval(custom_module: str) -> None:
    entry = ScorerEntry.model_validate({"import": f"{custom_module}:LengthScorer"})

    scorer = build_scorer(entry, BootstrapConfig(n_resamples=50), seed=0)
    result = scorer.score(["ab", "abcd"], ["", ""])[0]

    assert entry.resolved_name == "LengthScorer"
    assert result.value == pytest.approx(3)
    assert result.ci.low <= 3 <= result.ci.high


@pytest.mark.ticket("EV-009-T24")
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("my_scorers:NotAScorer", "does not subclass Scorer"),
        ("my_scorers:Missing", "Missing"),
        ("no_such_module:Thing", "no_such_module"),
        ("my_scorers", "module:Class"),
    ],
)
def test_bad_import_path_is_refused_clearly(custom_module: str, path: str, expected: str) -> None:
    entry = ScorerEntry.model_validate({"import": path})

    with pytest.raises(ValueError, match=expected):
        build_scorer(entry, BootstrapConfig(), seed=0)


def test_entry_needs_exactly_one_of_type_and_import() -> None:
    with pytest.raises(ValueError):
        ScorerEntry.model_validate({})
    with pytest.raises(ValueError):
        ScorerEntry.model_validate({"type": "rouge", "import": "a:B"})


@pytest.mark.ticket("EV-009-T25")
def test_custom_scorer_cannot_skip_the_interval(custom_module: str) -> None:
    from evalverdict.scoring.base import Scorer

    with pytest.raises(TypeError, match="score"):

        class Sneaky(Scorer):
            metrics = ("x",)

            def item_values(self, predictions, references, metadata):  # type: ignore[no-untyped-def]
                raise AssertionError

            def score(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return ()
