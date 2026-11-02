"""EV-014: judge agreement with labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from evalverdict.scorers.agreement import AgreementScorer, agreement_results
from evalverdict.scoring.base import FLAG_SMALL_SAMPLE, ScoreResult, UndefinedMetricError

JUDGE = [3, 5, 7, 8, 2, 9, 6, 4, 7, 5, 6, 8, 3, 7, 9, 5, 4, 6]
LABELS = [4, 5, 6, 8, 3, 9, 5, 4, 8, 6, 6, 7, 2, 7, 8, 5, 5, 6]


@pytest.mark.ticket("EV-014-T11")
def test_weighted_kappa_matches_scikit_learn() -> None:
    result = AgreementScorer().score(JUDGE, LABELS)[0]

    assert result.value == pytest.approx(cohen_kappa_score(JUDGE, LABELS, weights="quadratic"))


@pytest.mark.ticket("EV-014-T12")
def test_linear_weights_change_the_result_and_match_scikit_learn() -> None:
    quadratic = AgreementScorer().score(JUDGE, LABELS)[0]
    linear = AgreementScorer(params={"weights": "linear"}).score(JUDGE, LABELS)[0]

    assert linear.value == pytest.approx(cohen_kappa_score(JUDGE, LABELS, weights="linear"))
    assert linear.value != quadratic.value


@pytest.mark.ticket("EV-014-T13")
def test_perfect_agreement_is_one_and_a_constant_rater_is_undefined() -> None:
    assert AgreementScorer().score(LABELS, LABELS)[0].value == pytest.approx(1.0)
    with pytest.raises(UndefinedMetricError):
        AgreementScorer().score([5] * 10, [5] * 10)


@pytest.mark.ticket("EV-014-T14")
def test_fifteen_items_get_an_interval_and_the_small_sample_flag() -> None:
    result = AgreementScorer().score(JUDGE[:15], LABELS[:15])[0]

    assert result.ci.low < result.value < result.ci.high
    assert FLAG_SMALL_SAMPLE in result.flags


def run_with_judge(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    rng = np.random.default_rng(1)
    for target in ("baseline", "candidate"):
        items = [{"item_id": f"c{n}", "value": float(rng.integers(2, 10)) / 10} for n in range(30)]
        scores = {
            "results": [
                {"scorer": "summary_quality", "name": "score", "value": 0.6, "per_item": items}
            ]
        }
        (run_dir / target).mkdir(parents=True)
        (run_dir / target / "scores.json").write_text(json.dumps(scores))
    return run_dir


def labels_file(tmp_path: Path) -> Path:
    rows = []
    for n in range(30):
        target = "baseline" if n % 2 else "candidate"
        provenance = "human_independent" if n < 12 else "ai_draft_human_reviewed"
        rows.append(
            {
                "item": f"L{n:02d}",
                "target": target,
                "case_id": f"c{n}",
                "label": (n * 3) % 9 + 1,
                "draft": (n * 3) % 9 + 2,
                "provenance": provenance,
            }
        )
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "labels_summary.json").write_text(json.dumps({"n_changed_in_review": 4}))
    return path


@pytest.mark.ticket("EV-014-T10")
@pytest.mark.ticket("EV-014-T15")
def test_three_results_are_reported_separately_never_pooled(tmp_path: Path) -> None:
    results = agreement_results(run_with_judge(tmp_path), labels_file(tmp_path), "summary_quality")

    by_provenance = {result["details"]["provenance"]: result for result in results}
    assert set(by_provenance) == {"human_independent", "ai_draft_human_reviewed", "ai_draft"}
    assert by_provenance["human_independent"]["n"] == 12
    assert by_provenance["ai_draft_human_reviewed"]["n"] == 18
    assert by_provenance["ai_draft"]["n"] == 12
    assert by_provenance["ai_draft_human_reviewed"]["details"]["n_changed_in_review"] == 4
    assert (
        by_provenance["ai_draft_human_reviewed"]["details"]["title"] == "Judge vs reviewed labels"
    )


@pytest.mark.ticket("EV-014-T17")
def test_without_labels_agreement_is_reported_as_unmeasured(tmp_path: Path) -> None:
    results = agreement_results(run_with_judge(tmp_path), None, "summary_quality")

    assert results == [
        {
            "scorer": "agreement",
            "name": "weighted_kappa",
            "reason": "no labels were supplied, so judge agreement is unmeasured",
        }
    ]


def test_judge_scores_are_put_on_the_label_scale() -> None:
    result: ScoreResult = AgreementScorer(params={"judge_scale": 10}).score(
        [0.3, 0.5, 0.7, 0.9, 0.2], [3, 5, 7, 9, 2]
    )[0]

    assert result.value == pytest.approx(1.0)
