"""Blind labelling, so judge agreement can be measured honestly (D13).

Flow: sample 50 outputs with model identity hidden; Carla labels 15 independently;
Claude drafts all 50 without seeing her labels or any judge score; Carla reviews the
other 35; import records each label's provenance and every change made in review.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

HUMAN_FILE = "human_labels.csv"
TO_DRAFT_FILE = "claude_drafts.csv"
REVIEW_FILE = "review.csv"
LABELS_FILE = "labels.jsonl"
SUMMARY_FILE = "labels_summary.json"
PROVENANCE = {"human_independent", "ai_draft_human_reviewed"}
SCALE = range(0, 11)
IDENTITY_WORDS = re.compile(
    r"\b(claude|anthropic|haiku|sonnet|opus|gpt|openai|chatgpt|luna)\b", re.I
)


class LabellingError(ValueError):
    """The labelling files are incomplete or out of order."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise LabellingError(f"{path} is missing")
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_dirs(run_dir: Path) -> list[Path]:
    folders = sorted(path.parent for path in run_dir.glob("*/manifest.json"))
    if len(folders) != 2:
        raise LabellingError(
            f"labelling needs a run with exactly two targets, found {len(folders)}"
        )
    return folders


def _judged_cases(folder: Path) -> set[str]:
    scores_path = folder / "scores.json"
    if not scores_path.is_file():
        return set()
    judged = set()
    for result in json.loads(scores_path.read_text()).get("results", []):
        if "rubric_text" in (result.get("details") or {}):
            judged |= {item["item_id"] for item in result["per_item"] if item["value"] is not None}
    return judged


def sample_items(
    run_dir: Path, out_dir: Path, *, n: int, human_n: int, seed: int, exclude: set[str]
) -> None:
    """Pick n distinct cases, half per model, and write blind labelling files."""
    folders = _target_dirs(run_dir)
    if any(_judged_cases(folder) for folder in folders):
        raise LabellingError("this run already has judge scores; label a run without a judge")
    outputs: dict[str, dict[str, str]] = {}
    for folder in folders:
        usable = {
            row["case_id"]: row["output"]
            for row in _read_jsonl(folder / "predictions.jsonl")
            if row["error"] is None and row.get("repeat", 0) == 0 and (row["output"] or "").strip()
        }
        outputs[folder.name] = usable
    first, second = (folder.name for folder in folders)
    candidates = sorted((set(outputs[first]) & set(outputs[second])) - exclude)
    if len(candidates) < n:
        raise LabellingError(f"not enough cases to sample: {len(candidates)} available, {n} needed")

    rng = random.Random(seed)
    chosen = rng.sample(candidates, n)
    assignment = [
        (case_id, first if index < n // 2 else second) for index, case_id in enumerate(chosen)
    ]
    rng.shuffle(assignment)
    items = {
        f"L{index + 1:02d}": {"case_id": case, "target": target}
        for index, (case, target) in enumerate(assignment)
    }

    by_target = {
        name: [item for item, entry in items.items() if entry["target"] == name]
        for name in (first, second)
    }
    human = sorted(
        rng.sample(by_target[first], human_n // 2)
        + rng.sample(by_target[second], human_n - human_n // 2)
    )

    dataset_path = Path(json.loads((folders[0] / "manifest.json").read_text())["dataset"]["path"])
    cases = {row["id"]: row for row in _read_jsonl(dataset_path)}

    revealing = sorted(
        item
        for item, entry in items.items()
        if IDENTITY_WORDS.search(outputs[entry["target"]][entry["case_id"]])
    )
    if revealing:
        raise LabellingError(
            f"summaries name a model or vendor, which would unblind them: {revealing}"
        )

    def row(item: str) -> dict[str, Any]:
        entry = items[item]
        case = cases[entry["case_id"]]
        return {
            "item": item,
            "page": case["input"],
            "summary": outputs[entry["target"]][entry["case_id"]],
            "reference": case.get("reference", ""),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "private").mkdir(exist_ok=True)
    _write_csv(
        out_dir / HUMAN_FILE,
        ["item", "page", "summary", "reference", "score", "notes"],
        [{**row(item), "score": "", "notes": ""} for item in human],
    )
    _write_csv(
        out_dir / TO_DRAFT_FILE,
        ["item", "page", "summary", "reference", "score", "reason"],
        [{**row(item), "score": "", "reason": ""} for item in sorted(items)],
    )
    key = {"run_dir": str(run_dir), "seed": seed, "items": items, "human": human}
    (out_dir / "private" / "key.json").write_text(json.dumps(key, indent=2, sort_keys=True))


def _scores(rows: list[dict[str, str]], column: str, what: str) -> dict[str, int]:
    found: dict[str, int] = {}
    problems = []
    for row in rows:
        value = (row.get(column) or "").strip()
        if not value.isdigit() or int(value) not in SCALE:
            problems.append(f"{row['item']} ({value or 'blank'})")
        else:
            found[row["item"]] = int(value)
    if problems:
        raise LabellingError(f"{what} incomplete or outside 0-10: {', '.join(problems)}")
    return found


def export_review(labels_dir: Path) -> None:
    """Write the 35 drafts for Carla's review, once her 15 and all drafts are complete."""
    human_rows = _read_csv(labels_dir / HUMAN_FILE)
    _scores(human_rows, "score", "human labels")
    draft_rows = _read_csv(labels_dir / TO_DRAFT_FILE)
    drafts = _scores(draft_rows, "score", "draft labels")
    human = {row["item"] for row in human_rows}
    review = [
        {
            "item": row["item"],
            "page": row["page"],
            "summary": row["summary"],
            "reference": row["reference"],
            "draft_score": drafts[row["item"]],
            "draft_reason": row["reason"],
            "reviewed_score": drafts[row["item"]],
            "notes": "",
        }
        for row in draft_rows
        if row["item"] not in human
    ]
    _write_csv(labels_dir / REVIEW_FILE, list(review[0]), review)
    state = {
        "human_sha256": _sha256(labels_dir / HUMAN_FILE),
        "drafts_sha256": _sha256(labels_dir / TO_DRAFT_FILE),
    }
    (labels_dir / "private" / "state.json").write_text(json.dumps(state, indent=2))


def import_labels(labels_dir: Path) -> dict[str, Any]:
    """Combine everything into labels.jsonl, with provenance and the review change log."""
    key = json.loads((labels_dir / "private" / "key.json").read_text())
    human = _scores(_read_csv(labels_dir / HUMAN_FILE), "score", "human labels")
    drafts = _scores(_read_csv(labels_dir / TO_DRAFT_FILE), "score", "draft labels")
    review_rows = _read_csv(labels_dir / REVIEW_FILE)
    reviewed = _scores(review_rows, "reviewed_score", "reviewed labels")

    rows = []
    for item, entry in sorted(key["items"].items()):
        provenance = "human_independent" if item in human else "ai_draft_human_reviewed"
        label = human.get(item, reviewed.get(item))
        if label is None:
            raise LabellingError(f"{item} has no label")
        rows.append(
            {"item": item, **entry, "label": label, "draft": drafts[item], "provenance": provenance}
        )
    changes = [
        {"item": item, "draft": drafts[item], "reviewed": reviewed[item]}
        for item in sorted(reviewed)
        if reviewed[item] != drafts[item]
    ]
    (labels_dir / LABELS_FILE).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (labels_dir / "review_changes.jsonl").write_text(
        "".join(json.dumps(change, sort_keys=True) + "\n" for change in changes)
    )
    summary = {
        "n_human": len(human),
        "n_reviewed": len(reviewed),
        "n_changed_in_review": len(changes),
    }
    (labels_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def load_labels(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    for row in rows:
        if row.get("provenance") not in PROVENANCE:
            raise LabellingError(f"label row {row.get('item', row)} has no valid provenance")
    return rows
