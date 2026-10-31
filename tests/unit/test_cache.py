"""EV-004: the response cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalverdict.cache import ResponseCache, cache_key
from evalverdict.data import Prediction

CONFIG = {"kind": "litellm", "model": "m", "temperature": 0}


def stored(tmp_path: Path) -> tuple[ResponseCache, str]:
    cache = ResponseCache(tmp_path / "cache")
    key = cache_key(CONFIG, "input text", repeat=0)
    cache.put(key, Prediction(case_id="a", output="saved", cost_usd=0.5))
    return cache, key


@pytest.mark.ticket("EV-004-T01")
def test_stored_prediction_is_returned_on_the_same_request(tmp_path: Path) -> None:
    cache, key = stored(tmp_path)

    assert cache.get(key) == Prediction(case_id="a", output="saved", cost_usd=0.5)


@pytest.mark.ticket("EV-004-T02")
@pytest.mark.parametrize(
    ("config", "text"),
    [
        ({**CONFIG, "model": "other"}, "input text"),
        (CONFIG, "different input"),
        ({**CONFIG, "temperature": 1}, "input text"),
    ],
)
def test_changed_model_prompt_or_parameter_is_a_new_key(
    config: dict[str, object], text: str
) -> None:
    assert cache_key(config, text, repeat=0) != cache_key(CONFIG, "input text", repeat=0)


@pytest.mark.ticket("EV-004-T03")
def test_each_repeat_has_its_own_key() -> None:
    keys = {cache_key(CONFIG, "input text", repeat=index) for index in range(3)}

    assert len(keys) == 3


@pytest.mark.ticket("EV-004-T04")
def test_corrupt_entry_is_ignored_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cache, key = stored(tmp_path)
    next((tmp_path / "cache").rglob("*.json")).write_text("{broken")

    assert cache.get(key) is None
    assert "corrupt" in caplog.text


@pytest.mark.ticket("EV-004-T05")
def test_error_responses_are_not_cached(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    key = cache_key(CONFIG, "input text", repeat=0)

    cache.put(key, Prediction(case_id="a", error="rate limited"))

    assert cache.get(key) is None
