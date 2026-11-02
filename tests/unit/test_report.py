"""EV-011: the HTML report."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from evalverdict.report.render import (
    ReportError,
    format_score,
    render_comparison_report,
    render_target_report,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "target_report.html"
RUBRIC_TEXT = "name: summary_quality\nsteps:\n  - Check the summary.\n"


def ci(low: float, high: float) -> dict[str, Any]:
    return {
        "low": low,
        "high": high,
        "level": 0.95,
        "method": "percentile",
        "n_resamples": 2000,
        "n_undefined": 0,
    }


def result(
    scorer: str, name: str, value: float, low: float, high: float, **extra: Any
) -> dict[str, Any]:
    return {
        "scorer": scorer,
        "name": name,
        "family": extra.pop("family", "generation"),
        "value": value,
        "ci": ci(low, high),
        "n": extra.pop("n", 120),
        "higher_is_better": extra.pop("higher_is_better", True),
        "per_item": extra.pop("per_item", []),
        "details": extra.pop("details", {}),
        "flags": extra.pop("flags", []),
    }


BASE_SCORES: dict[str, Any] = {
    "results": [
        result(
            "judge",
            "score",
            0.83,
            0.79,
            0.87,
            details={
                "rubric_name": "summary_quality",
                "rubric_text": RUBRIC_TEXT,
                "rubric_sha256": "a" * 64,
                "judge_model": "anthropic/claude-sonnet-5-5",
                "self_judging": False,
                "shared_vendor_with": [],
                "judge_failures": 0,
            },
            per_item=[
                {"item_id": f"c{n}", "value": 0.8, "detail": {"reason": "ok"}} for n in range(30)
            ],
        ),
        result(
            "latency", "latency_p95_s", 1.2, 1.0, 1.5, family="operational", higher_is_better=False
        ),
        {
            "scorer": "latency",
            "name": "ttft_p50_s",
            "family": "operational",
            "reason": "no item has this measurement",
        },
        result(
            "cost",
            "cost_per_prediction_usd",
            0.0003,
            0.00025,
            0.00035,
            family="operational",
            higher_is_better=False,
            details={"cost_sources": ["litellm_price"]},
        ),
    ],
    "disabled": [],
    "gates": [
        {
            "key": "judge.score",
            "outcome": "pass",
            "min": 0.75,
            "max": None,
            "gate": "ci",
            "primary": True,
        },
        {
            "key": "latency.latency_p95_s",
            "outcome": "inconclusive",
            "min": None,
            "max": 1.3,
            "gate": "ci",
            "primary": False,
        },
    ],
    "run": {
        "status": "complete",
        "stop_reason": None,
        "n_requested": 120,
        "n_completed": 120,
        "n_errors": 0,
        "error_rate": 0.0,
        "n_parse_failures": 2,
        "parse_failure_rate": 2 / 120,
        "cache_hits": 0,
        "judge_failures": 0,
        "repeats": 1,
        "concurrency": 4,
        "outcome": "inconclusive",
    },
}
BASE_MANIFEST: dict[str, Any] = {
    "name": "haiku-vs-luna",
    "created_at": "2026-11-02T10:00:00+00:00",
    "status": "complete",
    "outcome": "inconclusive",
    "target": "baseline",
    "target_config": {"kind": "litellm", "model": "anthropic/claude-haiku-5-5"},
    "dataset": {"path": "datasets/summaries.jsonl", "sha256": "b" * 64, "n": 120},
    "seed": 42,
    "concurrency": 4,
    "repeats": 1,
    "spend": {"usd": 0.036, "calls": 120, "cache_hits": 0, "unmeasured_calls": 0},
    "judge": {
        "model": "anthropic/claude-sonnet-5-5",
        "spend_usd": 1.08,
        "calls": 120,
        "cache_hits": 0,
        "allow_self_judging": False,
    },
    "parameters_sent": {"max_tokens": 1024},
    "sampling_dropped": ["temperature"],
    "scorers": [
        {
            "name": "judge",
            "type": "judge",
            "import": None,
            "enabled": True,
            "metrics": ["score"],
            "params": {
                "rubric": "rubrics/summary_quality.yaml",
                "threshold": 0.5,
                "strict_mode": False,
                "judge_model": None,
                "concurrency": 4,
            },
            "bootstrap": {"n_resamples": 2000, "level": 0.95, "method": "percentile"},
        },
        {
            "name": "latency",
            "type": "latency",
            "import": None,
            "enabled": True,
            "metrics": ["latency_p95_s", "ttft_p50_s"],
            "params": {"percentiles": [50, 95]},
            "bootstrap": {"n_resamples": 2000, "level": 0.95, "method": "percentile"},
        },
        {
            "name": "cost",
            "type": "cost",
            "import": None,
            "enabled": True,
            "metrics": ["cost_per_prediction_usd"],
            "params": {},
            "bootstrap": {"n_resamples": 5000, "level": 0.95, "method": "percentile"},
        },
    ],
    "versions": {"python": "3.12", "evalverdict": "0.1.0"},
    "git_sha": "c" * 40,
    "report": {"max_items_shown": 20},
}


def make_run(
    tmp_path: Path, scores: dict[str, Any] | None = None, manifest: dict[str, Any] | None = None
) -> Path:
    folder = tmp_path / "baseline"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scores.json").write_text(json.dumps(scores or BASE_SCORES))
    (folder / "manifest.json").write_text(json.dumps(manifest or BASE_MANIFEST))
    return folder


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def changed(update: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(base)
    value.update(update)
    return value


@pytest.mark.ticket("EV-011-T01")
def test_score_shows_value_half_width_n_and_exact_interval(tmp_path: Path) -> None:
    text = text_of(render_target_report(make_run(tmp_path)))

    assert "0.83 ±0.04 on n=120" in text
    assert "95% CI 0.79 to 0.87" in text


@pytest.mark.ticket("EV-011-T01")
@pytest.mark.parametrize(
    ("value", "low", "high", "expected"),
    [
        (0.83, 0.79, 0.87, "0.83 ±0.04 on n=120"),
        (0.0003, 0.00025, 0.00035, "0.0003 ±5e-05 on n=120"),
    ],
)
def test_small_values_keep_their_significant_digits(
    value: float, low: float, high: float, expected: str
) -> None:
    assert format_score(value, low, high, 120) == expected


@pytest.mark.ticket("EV-011-T02")
def test_every_headline_number_has_its_interval_and_n(tmp_path: Path) -> None:
    html = render_target_report(make_run(tmp_path))

    headlines = re.findall(r'<span class="headline">(.*?)</span>', html)
    assert len(headlines) == 3
    assert all(re.search(r"±.* on n=\d+", headline) for headline in headlines)


@pytest.mark.ticket("EV-011-T03")
def test_not_measured_says_so_rather_than_zero(tmp_path: Path) -> None:
    text = text_of(render_target_report(make_run(tmp_path)))

    assert "ttft_p50_s Not measured: no item has this measurement" in text


@pytest.mark.ticket("EV-011-T04")
@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("small_sample", "fewer than 30"),
        ("unreliable_interval", "unreliable"),
        ("degenerate_interval", "one item"),
    ],
)
def test_result_flags_are_visible(tmp_path: Path, flag: str, expected: str) -> None:
    scores = copy.deepcopy(BASE_SCORES)
    scores["results"][0]["flags"] = [flag]

    assert expected in render_target_report(make_run(tmp_path, scores))


@pytest.mark.ticket("EV-011-T04")
@pytest.mark.parametrize(
    ("status", "expected"), [("invalid", "Invalid run"), ("incomplete", "Incomplete run")]
)
def test_run_status_warning_is_prominent(tmp_path: Path, status: str, expected: str) -> None:
    scores = copy.deepcopy(BASE_SCORES)
    scores["run"]["status"] = status
    scores["run"]["stop_reason"] = "budget" if status == "incomplete" else None

    html = render_target_report(make_run(tmp_path, scores))

    assert f'<div class="banner banner-{status}"' in html
    assert expected in html


@pytest.mark.ticket("EV-011-T05")
def test_judge_rubric_text_and_hash_are_shown(tmp_path: Path) -> None:
    html = render_target_report(make_run(tmp_path))

    assert "Check the summary." in html
    assert "a" * 64 in html
    assert "anthropic/claude-sonnet-5-5" in html


@pytest.mark.ticket("EV-011-T06")
def test_gate_outcomes_are_distinct_with_thresholds_and_mode(tmp_path: Path) -> None:
    html = render_target_report(make_run(tmp_path))

    assert 'class="outcome outcome-pass"' in html
    assert 'class="outcome outcome-inconclusive"' in html
    assert "min 0.75" in text_of(html)
    assert "whole interval" in text_of(html)


@pytest.mark.ticket("EV-011-T09")
def test_only_non_default_scorer_settings_are_listed(tmp_path: Path) -> None:
    text = text_of(render_target_report(make_run(tmp_path)))

    assert "cost: bootstrap.n_resamples = 5000" in text
    assert "threshold" not in text.split("Non-default settings")[1].split("</section>")[0]


@pytest.mark.ticket("EV-011-T10")
def test_dropped_sampling_is_disclosed_with_what_was_sent(tmp_path: Path) -> None:
    text = text_of(render_target_report(make_run(tmp_path)))

    assert "ran at provider defaults for: temperature" in text
    assert "max_tokens = 1024" in text


@pytest.mark.ticket("EV-011-T11")
@pytest.mark.parametrize(
    ("repeats", "expected"), [(1, "test-item sampling only"), (3, "repeated 3 times")]
)
def test_report_states_what_the_interval_covers(
    tmp_path: Path, repeats: int, expected: str
) -> None:
    manifest = changed({"repeats": repeats}, BASE_MANIFEST)

    assert expected in text_of(render_target_report(make_run(tmp_path, manifest=manifest)))


@pytest.mark.ticket("EV-011-T12")
def test_self_judging_is_prominently_disclosed(tmp_path: Path) -> None:
    scores = copy.deepcopy(BASE_SCORES)
    scores["results"][0]["details"]["self_judging"] = True

    html = render_target_report(make_run(tmp_path, scores))

    assert 'class="disclosure disclosure-warning"' in html
    assert "judged by a model under test" in html


@pytest.mark.ticket("EV-011-T13")
def test_judge_sharing_a_vendor_is_disclosed(tmp_path: Path) -> None:
    scores = copy.deepcopy(BASE_SCORES)
    scores["results"][0]["details"]["shared_vendor_with"] = ["anthropic/claude-haiku-5-5"]

    text = text_of(render_target_report(make_run(tmp_path, scores)))

    assert "shares a vendor with anthropic/claude-haiku-5-5" in text


@pytest.mark.ticket("EV-011-T14")
def test_label_agreement_results_are_shown_separately(tmp_path: Path) -> None:
    scores = copy.deepcopy(BASE_SCORES)
    for provenance, value in (("human_independent", 0.61), ("ai_draft_human_reviewed", 0.74)):
        scores["results"].append(
            result(
                "agreement",
                "weighted_kappa",
                value,
                value - 0.2,
                value + 0.15,
                n=15 if provenance == "human_independent" else 35,
                details={"provenance": provenance, "n_changed_in_review": 4},
            )
        )

    text = text_of(render_target_report(make_run(tmp_path, scores)))

    assert "Judge vs human labels" in text
    assert "Judge vs reviewed labels" in text
    assert "4 changed in review" in text


@pytest.mark.ticket("EV-011-T15")
def test_error_and_parse_failure_rates_are_shown(tmp_path: Path) -> None:
    text = text_of(render_target_report(make_run(tmp_path)))

    assert "API errors 0 of 120 (0.0%)" in text
    assert "Parse failures 2 of 120 (1.7%)" in text


@pytest.mark.ticket("EV-011-T16")
def test_spend_shows_cost_sources_and_judge_separately(tmp_path: Path) -> None:
    text = text_of(render_target_report(make_run(tmp_path)))

    assert "Target spend $0.0360" in text
    assert "Judge spend $1.0800" in text
    assert "litellm_price" in text


@pytest.mark.ticket("EV-011-T17")
def test_concurrency_and_timing_window_are_next_to_latency(tmp_path: Path) -> None:
    text = text_of(render_target_report(make_run(tmp_path)))
    operational = text.split("Operational")[1]

    assert "concurrency 4" in operational
    assert "2026-11-02" in operational


@pytest.mark.ticket("EV-011-T18")
def test_confusion_matrix_and_reliability_bins_are_rendered(tmp_path: Path) -> None:
    scores = copy.deepcopy(BASE_SCORES)
    scores["results"].append(
        result(
            "classification",
            "accuracy",
            0.9,
            0.85,
            0.94,
            family="classification",
            details={
                "labels": ["neg", "pos"],
                "confusion_matrix": [[50, 5], [7, 58]],
                "reliability_bins": [
                    {"low": 0.0, "high": 0.5, "accuracy": 0.3, "confidence": 0.35, "count": 40}
                ],
            },
        )
    )

    html = render_target_report(make_run(tmp_path, scores))

    assert "<td>58</td>" in html
    assert "<svg" in html


@pytest.mark.ticket("EV-011-T19")
def test_families_render_together_under_one_contract(tmp_path: Path) -> None:
    text = text_of(render_target_report(make_run(tmp_path)))

    assert "Generation" in text
    assert "Operational" in text


@pytest.mark.ticket("EV-011-T20")
def test_report_is_self_contained(tmp_path: Path) -> None:
    html = render_target_report(make_run(tmp_path))

    assert "<script src" not in html
    assert "<link" not in html
    assert not re.search(r"url\(\s*['\"]?https?:", html)


@pytest.mark.ticket("EV-011-T21")
def test_model_output_is_escaped(tmp_path: Path) -> None:
    scores = copy.deepcopy(BASE_SCORES)
    scores["results"][0]["per_item"][0]["detail"] = {"reason": "<script>alert(1)</script>"}

    html = render_target_report(make_run(tmp_path, scores))

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.ticket("EV-011-T22")
def test_item_list_is_capped_and_states_the_total(tmp_path: Path) -> None:
    manifest = changed({"report": {"max_items_shown": 5}}, BASE_MANIFEST)

    html = render_target_report(make_run(tmp_path, manifest=manifest))

    assert html.count('<tr class="item">') == 5
    assert "Showing 5 of 30 items" in text_of(html)


@pytest.mark.ticket("EV-011-T07")
@pytest.mark.ticket("EV-011-T08")
def test_comparison_shows_differences_verdicts_and_recommendation(tmp_path: Path) -> None:
    comparison = {
        "baseline": "baseline",
        "candidate": "candidate",
        "baseline_status": "complete",
        "candidate_status": "complete",
        "n_paired_items": 118,
        "metrics": [
            {
                "key": "judge.score",
                "comparable": True,
                "primary": True,
                "method": "per_item",
                "higher_is_better": True,
                "baseline_value": 0.78,
                "candidate_value": 0.83,
                "difference": 0.05,
                "ci": ci(0.02, 0.08),
                "n_paired": 118,
                "verdict": "better",
            },
            {
                "key": "latency.latency_p95_s",
                "comparable": True,
                "primary": False,
                "method": "recomputed",
                "higher_is_better": False,
                "baseline_value": 1.0,
                "candidate_value": 1.2,
                "difference": 0.2,
                "ci": ci(0.1, 0.3),
                "n_paired": 118,
                "verdict": "worse",
            },
            {
                "key": "bert.f1",
                "comparable": False,
                "primary": False,
                "reason": "scorer settings differ",
            },
        ],
        "recommendation": {
            "decision": "candidate",
            "reason": "the candidate is better on every primary metric",
        },
    }
    (tmp_path / "comparison.json").write_text(json.dumps(comparison))

    html = render_comparison_report(tmp_path)
    text = text_of(html)

    assert "+0.05 [+0.02, +0.08] on 118 paired items" in text
    assert 'class="verdict verdict-better"' in html
    assert "Secondary" in text
    assert "Recommendation: candidate" in text
    assert "not comparable: scorer settings differ" in text


def test_refused_comparison_is_explained(tmp_path: Path) -> None:
    (tmp_path / "comparison.json").write_text(
        json.dumps(
            {
                "baseline": "a",
                "candidate": "b",
                "refused": "cannot compare runs on different datasets",
            }
        )
    )

    assert "cannot compare runs on different datasets" in render_comparison_report(tmp_path)


@pytest.mark.ticket("EV-011-T24")
def test_report_matches_the_reviewed_golden_file(tmp_path: Path) -> None:
    html = render_target_report(make_run(tmp_path))

    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN.write_text(html)
    assert html == GOLDEN.read_text()


@pytest.mark.ticket("EV-011-T25")
def test_missing_scores_file_is_a_clear_error(tmp_path: Path) -> None:
    folder = tmp_path / "baseline"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps(BASE_MANIFEST))

    with pytest.raises(ReportError, match=r"scores\.json"):
        render_target_report(folder)
