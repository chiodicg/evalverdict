"""EV-014: the blind labelling workflow."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pytest

from evalverdict.labelling import (
    HUMAN_FILE,
    REVIEW_FILE,
    TO_DRAFT_FILE,
    LabellingError,
    export_review,
    import_labels,
    sample_items,
)

TARGETS = {"baseline": "anthropic/claude-haiku-5-5", "candidate": "openai/gpt-6-luna"}


def make_run(tmp_path: Path, count: int = 80) -> Path:
    run_dir = tmp_path / "runs" / "20261102-demo"
    data = tmp_path / "data.jsonl"
    data.write_text(
        "".join(
            json.dumps({"id": f"c{n:02d}", "input": f"page {n}", "reference": f"ref {n}"}) + "\n"
            for n in range(count)
        )
    )
    for name, model in TARGETS.items():
        folder = run_dir / name
        folder.mkdir(parents=True)
        rows = [
            {
                "case_id": f"c{n:02d}",
                "output": f"summary {n} version {len(name)}",
                "error": None,
                "repeat": 0,
            }
            for n in range(count)
        ]
        (folder / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = {
            "target": name,
            "target_config": {"model": model},
            "dataset": {"path": str(data)},
        }
        (folder / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_scores(path: Path, scores: dict[str, str], column: str = "score") -> None:
    rows = read_csv(path)
    for row in rows:
        row[column] = scores.get(row["item"], row.get(column, ""))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def labels_dir(tmp_path: Path) -> Path:
    sample_items(make_run(tmp_path), tmp_path / "labels", n=50, human_n=15, seed=7, exclude=set())
    return tmp_path / "labels"


def key(labels_dir: Path) -> dict[str, Any]:
    return json.loads((labels_dir / "private" / "key.json").read_text())


def fill_drafts(labels_dir: Path, score: str = "6") -> None:
    rows = read_csv(labels_dir / TO_DRAFT_FILE)
    write_scores(labels_dir / TO_DRAFT_FILE, {row["item"]: score for row in rows})
    write_scores(
        labels_dir / TO_DRAFT_FILE, {row["item"]: "draft reason" for row in rows}, "reason"
    )


def fill_human(labels_dir: Path, score: str = "7") -> None:
    rows = read_csv(labels_dir / HUMAN_FILE)
    write_scores(labels_dir / HUMAN_FILE, {row["item"]: score for row in rows})


@pytest.mark.ticket("EV-014-T01")
def test_sample_is_balanced_by_model_and_repeatable(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    sample_items(run_dir, tmp_path / "a", n=50, human_n=15, seed=7, exclude=set())
    sample_items(run_dir, tmp_path / "b", n=50, human_n=15, seed=7, exclude=set())

    first = key(tmp_path / "a")
    assert first == key(tmp_path / "b")
    targets = [entry["target"] for entry in first["items"].values()]
    assert targets.count("baseline") == targets.count("candidate") == 25


def test_excluded_cases_are_never_sampled(tmp_path: Path) -> None:
    excluded = {f"c{n:02d}" for n in range(30)}

    sample_items(
        make_run(tmp_path), tmp_path / "labels", n=50, human_n=15, seed=7, exclude=excluded
    )

    assert not {entry["case_id"] for entry in key(tmp_path / "labels")["items"].values()} & excluded


def test_sample_refuses_when_too_few_cases_remain(tmp_path: Path) -> None:
    with pytest.raises(LabellingError, match="enough"):
        sample_items(
            make_run(tmp_path, count=20),
            tmp_path / "labels",
            n=50,
            human_n=15,
            seed=7,
            exclude=set(),
        )


@pytest.mark.ticket("EV-014-T02")
def test_labelling_files_do_not_reveal_the_model(labels_dir: Path) -> None:
    for name in (HUMAN_FILE, TO_DRAFT_FILE):
        text = (labels_dir / name).read_text().lower()
        for canary in (
            "baseline",
            "candidate",
            "haiku",
            "luna",
            "anthropic",
            "openai",
            "claude",
            "gpt",
        ):
            assert canary not in text, (name, canary)


@pytest.mark.ticket("EV-014-T03")
def test_item_ids_are_anonymous_and_shuffled(labels_dir: Path) -> None:
    items = key(labels_dir)["items"]

    assert all(re.fullmatch(r"L\d{2}", item) for item in items)
    order = [entry["target"] for _, entry in sorted(items.items())]
    assert order != sorted(order)


@pytest.mark.ticket("EV-014-T04")
def test_human_and_draft_sets_split_fifteen_and_thirty_five(labels_dir: Path) -> None:
    human = {row["item"] for row in read_csv(labels_dir / HUMAN_FILE)}
    drafts = {row["item"] for row in read_csv(labels_dir / TO_DRAFT_FILE)}
    targets = [key(labels_dir)["items"][item]["target"] for item in human]

    assert len(human) == 15
    assert len(drafts) == 50
    assert human <= drafts
    assert {"baseline", "candidate"} == set(targets)


@pytest.mark.ticket("EV-014-T06")
def test_review_export_is_refused_until_the_human_labels_are_complete(labels_dir: Path) -> None:
    fill_drafts(labels_dir)

    with pytest.raises(LabellingError, match="human"):
        export_review(labels_dir)


def test_review_export_is_refused_until_drafts_are_complete(labels_dir: Path) -> None:
    fill_human(labels_dir)

    with pytest.raises(LabellingError, match="draft"):
        export_review(labels_dir)


@pytest.mark.ticket("EV-014-T05")
def test_review_file_holds_only_the_thirty_five_non_human_items(labels_dir: Path) -> None:
    fill_human(labels_dir)
    fill_drafts(labels_dir)

    export_review(labels_dir)

    review = read_csv(labels_dir / REVIEW_FILE)
    human = {row["item"] for row in read_csv(labels_dir / HUMAN_FILE)}
    assert len(review) == 35
    assert not {row["item"] for row in review} & human
    assert all(row["draft_score"] == "6" and row["reviewed_score"] == "6" for row in review)


@pytest.mark.ticket("EV-014-T07")
@pytest.mark.parametrize("bad", ["11", "-1", "seven", ""])
def test_out_of_range_or_missing_labels_are_rejected(labels_dir: Path, bad: str) -> None:
    fill_human(labels_dir)
    first = read_csv(labels_dir / HUMAN_FILE)[0]["item"]
    write_scores(labels_dir / HUMAN_FILE, {first: bad})
    fill_drafts(labels_dir)

    with pytest.raises(LabellingError, match=first):
        export_review(labels_dir)


def reviewed(labels_dir: Path, changes: dict[str, str] | None = None) -> None:
    fill_human(labels_dir)
    fill_drafts(labels_dir)
    export_review(labels_dir)
    if changes:
        write_scores(labels_dir / REVIEW_FILE, changes, "reviewed_score")


@pytest.mark.ticket("EV-014-T08")
def test_import_records_provenance_and_counts_review_changes(labels_dir: Path) -> None:
    review_items = [row["item"] for row in read_csv(labels_dir / TO_DRAFT_FILE)]
    reviewed(labels_dir)
    review = [row["item"] for row in read_csv(labels_dir / REVIEW_FILE)]
    write_scores(labels_dir / REVIEW_FILE, {review[0]: "2", review[1]: "9"}, "reviewed_score")

    summary = import_labels(labels_dir)

    rows = [json.loads(line) for line in (labels_dir / "labels.jsonl").read_text().splitlines()]
    assert summary["n_changed_in_review"] == 2
    assert {row["provenance"] for row in rows} == {"human_independent", "ai_draft_human_reviewed"}
    assert len(rows) == len(review_items)
    human_row = next(row for row in rows if row["provenance"] == "human_independent")
    assert human_row["label"] == 7 and human_row["draft"] == 6


@pytest.mark.ticket("EV-014-T09")
def test_labels_without_provenance_are_rejected(tmp_path: Path) -> None:
    from evalverdict.labelling import load_labels

    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps({"target": "baseline", "case_id": "c1", "label": 5}) + "\n")

    with pytest.raises(LabellingError, match="provenance"):
        load_labels(path)


@pytest.mark.ticket("EV-014-T16")
def test_drafting_is_refused_once_judge_scores_exist_for_an_item(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    scores = {
        "results": [
            {
                "scorer": "summary_quality",
                "name": "score",
                "value": 0.7,
                "details": {"rubric_text": "steps"},
                "per_item": [{"item_id": "c00", "value": 0.7}],
            }
        ]
    }
    (run_dir / "baseline" / "scores.json").write_text(json.dumps(scores))

    with pytest.raises(LabellingError, match="judge scores"):
        sample_items(run_dir, tmp_path / "labels", n=50, human_n=15, seed=7, exclude=set())


@pytest.mark.ticket("EV-014-T02")
def test_summary_that_names_a_model_stops_sampling(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, count=50)
    path = run_dir / "candidate" / "predictions.jsonl"
    path.write_text(path.read_text().replace("summary", "As GPT, my summary"))

    with pytest.raises(LabellingError, match="unblind"):
        sample_items(run_dir, tmp_path / "labels", n=50, human_n=15, seed=7, exclude=set())
