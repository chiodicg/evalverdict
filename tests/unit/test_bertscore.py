"""EV-007: BERTScore. Live tests download a model but cost nothing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evalverdict._optional import MissingExtraError
from evalverdict.engines import bertscore
from evalverdict.scorers.generation import BertScoreScorer

PREDICTIONS = ["the cat sat", "dogs bark"]
REFERENCES = ["a cat was sitting", "the stock market fell"]
# Values recorded from evaluate.load("bertscore") with distilbert-base-uncased.
RECORDED_F1 = [0.8720048069953918, 0.6460782289505005]
SMALL_MODEL = {"model_type": "distilbert-base-uncased"}


@pytest.fixture
def stub_engine(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def compute(predictions: list[str], references: list[str], **options: object) -> tuple:
        calls.append({"predictions": predictions, **options})
        scale = 1.0 if options["model_type"] == "roberta-large" else 0.5
        values = np.array([0.8, 0.4]) * scale
        return values, values + 0.1, values + 0.05

    monkeypatch.setattr(bertscore, "compute_bertscore", compute)
    return calls


@pytest.mark.ticket("EV-007-T04")
def test_metric_subset_produces_only_f1(stub_engine: list[dict[str, object]]) -> None:
    results = BertScoreScorer(metrics=["f1"]).score(PREDICTIONS, REFERENCES)

    assert [result.name for result in results] == ["f1"]
    assert results[0].value == pytest.approx(0.65)


@pytest.mark.ticket("EV-007-T05")
def test_model_type_reaches_the_engine_and_changes_the_result(
    stub_engine: list[dict[str, object]],
) -> None:
    default = BertScoreScorer(metrics=["f1"]).score(PREDICTIONS, REFERENCES)[0]
    other = BertScoreScorer(params=SMALL_MODEL, metrics=["f1"]).score(PREDICTIONS, REFERENCES)[0]

    assert stub_engine[0]["model_type"] == "roberta-large"
    assert stub_engine[1]["model_type"] == "distilbert-base-uncased"
    assert other.value != default.value


@pytest.mark.ticket("EV-007-T07")
def test_missing_extra_gives_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "bert_score", None)

    with pytest.raises(MissingExtraError, match="evalverdict\\[nlp\\]"):
        BertScoreScorer().check_dependencies()


@pytest.mark.ticket("EV-007-T06")
def test_empty_text_scores_zero_and_never_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def backend(predictions: list[str], *rest: object) -> tuple:
        seen.append(predictions)
        filled = np.full(len(predictions), 0.5)
        return filled, filled, filled

    monkeypatch.setattr(bertscore, "_run_backend", backend)

    precision, recall, f1 = bertscore.compute_bertscore(
        ["", "real answer", "   "],
        ["reference", "reference", "reference"],
        model_type="m",
        batch_size=2,
        backend="bert_score",
    )

    assert seen == [["real answer"]]
    assert f1.tolist() == [0.0, 0.5, 0.0]
    assert precision.tolist() == recall.tolist() == [0.0, 0.5, 0.0]


@pytest.mark.ticket("EV-015-T05")
def test_pairs_are_sent_in_chunks_that_fit_one_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], int]] = []

    def backend(predictions: list[str], references: list[str], model: str, size: int, *rest):
        calls.append((predictions, size))
        values = np.array([float(text[1:]) for text in predictions])
        return values, values, values

    monkeypatch.setattr(bertscore, "_run_backend", backend)

    _, _, f1 = bertscore.compute_bertscore(
        ["p1", "p2", "", "p4", "p5", "p6"],
        ["r"] * 6,
        model_type="m",
        batch_size=4,
        backend="bert_score",
    )

    # bert_score orders sentences through a set, so a batch boundary would move with
    # Python's hash seed. Each call holds at most batch_size sentences: one batch.
    assert calls == [(["p1", "p2"], 4), (["p4", "p5"], 4), (["p6"], 4)]
    assert f1.tolist() == [1.0, 2.0, 0.0, 4.0, 5.0, 6.0]


def test_batch_size_one_still_sends_whole_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes: list[tuple[int, int]] = []

    def backend(predictions: list[str], references: list[str], model: str, size: int, *rest):
        sizes.append((len(predictions), size))
        values = np.full(len(predictions), 0.5)
        return values, values, values

    monkeypatch.setattr(bertscore, "_run_backend", backend)

    bertscore.compute_bertscore(
        ["a", "b"], ["c", "d"], model_type="m", batch_size=1, backend="bert_score"
    )

    assert sizes == [(1, 2), (1, 2)]


@pytest.mark.live
@pytest.mark.ticket("EV-015-T05")
def test_scores_are_identical_under_different_hash_seeds(tmp_path: Path) -> None:
    script = tmp_path / "score.py"
    script.write_text(
        "import json, random\n"
        "from evalverdict.engines.bertscore import compute_bertscore\n"
        "rng = random.Random(0)\n"
        "words = 'the a cat dog sat ran on under mat rug red blue big small fell rose'.split()\n"
        "def text():\n"
        "    return ' '.join(rng.choice(words) for _ in range(rng.randint(4, 12)))\n"
        "preds = [text() for _ in range(60)]\n"
        "refs = [text() for _ in range(60)]\n"
        "_, _, f1 = compute_bertscore(preds, refs, model_type='distilbert-base-uncased',\n"
        "                             batch_size=8, backend='bert_score')\n"
        "print(json.dumps([float(value) for value in f1]))\n"
    )

    def run(seed: str) -> list[float]:
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, str(script)], env=env, capture_output=True, text=True, check=True
        )
        return json.loads(out.stdout.strip().splitlines()[-1])

    assert run("1") == run("2")


def test_all_empty_predictions_make_no_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def backend(*args: object) -> tuple:
        raise AssertionError("the model must not run")

    monkeypatch.setattr(bertscore, "_run_backend", backend)

    _, _, f1 = bertscore.compute_bertscore(
        ["", ""], ["a", "b"], model_type="m", batch_size=2, backend="bert_score"
    )

    assert f1.tolist() == [0.0, 0.0]


@pytest.mark.live
@pytest.mark.ticket("EV-007-T01")
def test_real_scores_match_recorded_values() -> None:
    result = BertScoreScorer(params=SMALL_MODEL, metrics=["f1"]).score(PREDICTIONS, REFERENCES)[0]

    assert [item.value for item in result.per_item] == pytest.approx(RECORDED_F1, abs=1e-4)


@pytest.mark.live
@pytest.mark.ticket("EV-007-T02")
def test_both_backends_give_identical_values() -> None:
    direct = BertScoreScorer(params=SMALL_MODEL, metrics=["f1"]).score(PREDICTIONS, REFERENCES)[0]
    via_hub = BertScoreScorer(params={**SMALL_MODEL, "backend": "evaluate"}, metrics=["f1"]).score(
        PREDICTIONS, REFERENCES
    )[0]

    assert [item.value for item in via_hub.per_item] == pytest.approx(
        [item.value for item in direct.per_item], abs=1e-6
    )


@pytest.mark.live
@pytest.mark.ticket("EV-007-T03")
def test_identical_text_scores_higher_than_unrelated_text() -> None:
    results = BertScoreScorer(params=SMALL_MODEL, metrics=["f1"]).score(
        ["the cat sat on the mat", "the stock market fell"],
        ["the cat sat on the mat", "the cat sat on the mat"],
    )[0]

    assert results.per_item[0].value > results.per_item[1].value


@pytest.mark.live
@pytest.mark.ticket("EV-007-T06")
def test_empty_prediction_through_the_real_model_scores_lowest() -> None:
    results = BertScoreScorer(params=SMALL_MODEL, metrics=["f1"]).score(
        ["", "the cat sat"], ["the cat sat", "the cat sat"]
    )[0]

    assert results.per_item[0].value < results.per_item[1].value
