"""EV-003: ROUGE."""

from __future__ import annotations

import pytest

from evalverdict.scorers.generation import RougeScorer

PREDICTIONS = ["the cat sat on the mat", "hello there world", "running quickly home"]
REFERENCES = ["a cat sat on a mat", "hello world", "he ran home fast"]
# Values recorded from evaluate.load("rouge") with use_aggregator=False.
EVALUATE_VALUES = {
    "rouge1": [0.666667, 0.8, 0.285714],
    "rouge2": [0.4, 0.0, 0.0],
    "rougeL": [0.666667, 0.8, 0.285714],
    "rougeLsum": [0.666667, 0.8, 0.285714],
}


def items(result: object) -> list[float]:
    return [item.value for item in result.per_item]  # type: ignore[attr-defined]


@pytest.mark.ticket("EV-003-T16")
@pytest.mark.parametrize("metric", list(EVALUATE_VALUES))
def test_each_metric_matches_evaluate(metric: str) -> None:
    result = RougeScorer(metrics=[metric]).score(PREDICTIONS, REFERENCES)[0]

    assert items(result) == pytest.approx(EVALUATE_VALUES[metric], abs=1e-6)


@pytest.mark.live
def test_evaluate_backend_gives_the_same_values() -> None:
    direct = RougeScorer(metrics=["rouge1"]).score(PREDICTIONS, REFERENCES)[0]
    via_hub = RougeScorer(params={"backend": "evaluate"}, metrics=["rouge1"]).score(
        PREDICTIONS, REFERENCES
    )[0]

    assert items(via_hub) == pytest.approx(items(direct))


@pytest.mark.ticket("EV-003-T17")
def test_stemming_changes_the_score() -> None:
    plain = RougeScorer(metrics=["rouge1"]).score(["cats running"], ["cat run"])[0]
    stemmed = RougeScorer(params={"use_stemmer": True}, metrics=["rouge1"]).score(
        ["cats running"], ["cat run"]
    )[0]

    assert plain.value == 0
    assert stemmed.value == pytest.approx(1)


@pytest.mark.ticket("EV-003-T18")
def test_empty_prediction_scores_zero() -> None:
    assert RougeScorer(metrics=["rouge1"]).score([""], ["some text"])[0].value == 0


def test_mean_of_items_is_the_headline_value() -> None:
    result = RougeScorer(metrics=["rouge2"]).score(PREDICTIONS, REFERENCES)[0]

    assert result.value == pytest.approx(0.4 / 3)
    assert result.ci.low <= result.value <= result.ci.high
