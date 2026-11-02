"""Render run and comparison reports from the files a run writes. No model calls."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader

from evalverdict.config import ScorerEntry
from evalverdict.scoring.base import BootstrapConfig
from evalverdict.scoring.registry import scorer_class

FLAG_TEXT = {
    "small_sample": "fewer than 30 items, so treat this interval with caution",
    "unreliable_interval": (
        "the metric was undefined on over 5% of resamples, so the interval may be unreliable"
    ),
    "degenerate_interval": "only one item, so there is no real interval",
}
PROVENANCE_TITLES = {
    "human_independent": "Judge vs human labels",
    "ai_draft_human_reviewed": "Judge vs reviewed labels",
    "ai_draft": "AI drafts vs human labels",
}
SMALL = 0.01


class ReportError(ValueError):
    """A run directory is missing the files a report needs."""


def _number(value: float, signed: bool = False) -> str:
    if value == 0:
        return "+0" if signed else "0"
    pattern = ("+" if signed else "") + (".2f" if abs(value) >= SMALL else ".2g")
    return format(value, pattern)


def format_score(value: float, low: float, high: float, n: int) -> str:
    return f"{_number(value)} ±{_number((high - low) / 2)} on n={n}"


def _interval_text(ci: dict[str, Any]) -> str:
    return f"{round(ci['level'] * 100):g}% CI {_number(ci['low'])} to {_number(ci['high'])}"


def _environment() -> Environment:
    env = Environment(loader=PackageLoader("evalverdict.report"), autoescape=True)
    env.filters["number"] = _number
    env.filters["signed"] = lambda value: _number(value, signed=True)
    env.filters["percent"] = lambda value: f"{value * 100:.1f}%"
    env.globals.update(format_score=format_score, interval_text=_interval_text)
    return env


def _read(folder: Path, name: str) -> Any:
    path = folder / name
    if not path.is_file():
        raise ReportError(f"cannot build a report: {path} is missing")
    return json.loads(path.read_text())


def _non_default_settings(scorers: list[dict[str, Any]]) -> list[str]:
    defaults_bootstrap = BootstrapConfig().model_dump()
    lines = []
    for settings in scorers:
        if not settings.get("enabled", True):
            continue
        entry = ScorerEntry.model_validate(
            {"type": settings.get("type"), "import": settings.get("import")}
        )
        try:
            fields = scorer_class(entry).Params.model_fields
        except ValueError:
            lines.append(f"{settings['name']}: settings unavailable for this scorer")
            continue
        for name, value in (settings.get("params") or {}).items():
            field = fields.get(name)
            if field is None or field.is_required() or field.default != value:
                lines.append(f"{settings['name']}: {name} = {value}")
        for name, value in (settings.get("bootstrap") or {}).items():
            if defaults_bootstrap.get(name) != value:
                lines.append(f"{settings['name']}: bootstrap.{name} = {value}")
    return lines


def render_target_report(folder: Path) -> str:
    manifest = _read(folder, "manifest.json")
    scores = _read(folder, "scores.json")
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    agreement = []
    judges = []
    for result in scores["results"]:
        details = result.get("details") or {}
        if "provenance" in details:
            agreement.append(result)
            continue
        families[result.get("family", "other")].append(result)
        if "rubric_text" in details:
            judges.append(result)
    cost_sources = sorted(
        {
            source
            for result in scores["results"]
            for source in (result.get("details") or {}).get("cost_sources", [])
        }
    )
    return (
        _environment()
        .get_template("report.html.j2")
        .render(
            manifest=manifest,
            run=scores["run"],
            gates=scores.get("gates", []),
            families=dict(families),
            agreement=agreement,
            judges=judges,
            disabled=scores.get("disabled", []),
            cost_sources=cost_sources,
            flag_text=FLAG_TEXT,
            provenance_titles=PROVENANCE_TITLES,
            max_items=(manifest.get("report") or {}).get("max_items_shown", 20),
            non_default=_non_default_settings(manifest.get("scorers", [])),
        )
    )


def render_comparison_report(run_dir: Path) -> str:
    comparison = _read(run_dir, "comparison.json")
    metrics = comparison.get("metrics", [])
    return (
        _environment()
        .get_template("comparison.html.j2")
        .render(
            comparison=comparison,
            primary=[item for item in metrics if item.get("primary")],
            secondary=[item for item in metrics if not item.get("primary")],
        )
    )


def write_reports(run_dir: Path) -> list[Path]:
    """Write report.html for each target, and comparison.html if there is one."""
    folders = sorted(path.parent for path in run_dir.glob("*/manifest.json"))
    if not folders:
        raise ReportError(f"no run found at {run_dir}")
    written = []
    for folder in folders:
        path = folder / "report.html"
        path.write_text(render_target_report(folder))
        written.append(path)
    if (run_dir / "comparison.json").is_file():
        path = run_dir / "comparison.html"
        path.write_text(render_comparison_report(run_dir))
        written.append(path)
    return written
