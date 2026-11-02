"""Command-line entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from evalverdict._optional import MissingExtraError
from evalverdict.config import load_config
from evalverdict.env import load_env
from evalverdict.estimate import estimate_run
from evalverdict.runner import execute, prepare, run_outcomes
from evalverdict.scoring.compare import compare_runs
from evalverdict.scoring.gates import EXIT_CODES, INVALID, worst_outcome
from evalverdict.scoring.registry import registered_scorers

EXIT_INVALID = EXIT_CODES[INVALID]
SETUP_ERRORS = (ValidationError, ValueError, OSError, MissingExtraError)
ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Path to a run config.")]

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Evaluate models and decide whether to change them.",
)
scorers_app = typer.Typer(no_args_is_help=True, help="List scorers and their settings.")
app.add_typer(scorers_app, name="scorers")


def _fail(error: Exception) -> typer.Exit:
    typer.echo(f"Error: {error}", err=True)
    return typer.Exit(EXIT_INVALID)


@app.callback()
def main() -> None:
    """Evaluate models and decide whether to change them."""
    load_env()


@app.command()
def run(
    config: ConfigOption,
    resume: Annotated[
        Path | None, typer.Option(help="Continue this run directory. Finished calls are reused.")
    ] = None,
) -> None:
    """Run every target in a config, score the results and write a run directory."""
    try:
        run_dir = execute(load_config(config), resume=resume)
    except SETUP_ERRORS as error:
        raise _fail(error) from error
    typer.echo(f"Run written to {run_dir}")
    outcomes = run_outcomes(run_dir)
    for target, outcome in outcomes.items():
        typer.echo(f"{target}: {outcome}")
    raise typer.Exit(EXIT_CODES[worst_outcome(outcomes.values())])


@app.command()
def validate(config: ConfigOption) -> None:
    """Check a config and estimate its calls and cost, with no model calls."""
    try:
        loaded = load_config(config)
        ready = prepare(loaded)
        estimate = estimate_run(loaded, len(ready.dataset.cases))
    except SETUP_ERRORS as error:
        raise _fail(error) from error
    typer.echo(f"Config is valid. {estimate.describe()}")
    if loaded.budget_usd is not None and estimate.total_cost_usd > loaded.budget_usd:
        typer.echo(
            f"Warning: estimated cost exceeds budget_usd (${loaded.budget_usd:.2f}). "
            "The run will stop when the budget is reached."
        )


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="Baseline target run directory.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate target run directory.")],
    output: Annotated[Path, typer.Option(help="Where to write the comparison.")] = Path(
        "comparison.json"
    ),
) -> None:
    """Compare two runs on the same items, with an interval on each difference."""
    try:
        result = compare_runs(baseline, candidate)
    except SETUP_ERRORS as error:
        raise _fail(error) from error
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for item in result["metrics"]:
        if item["comparable"]:
            ci = item["ci"]
            typer.echo(
                f"{item['key']}: {item['difference']:+.3f} "
                f"[{ci['low']:+.3f}, {ci['high']:+.3f}] {item['verdict']}"
            )
        else:
            typer.echo(f"{item['key']}: not comparable ({item['reason']})")
    recommendation = result["recommendation"]
    typer.echo(f"Recommendation: {recommendation['decision']} ({recommendation['reason']})")
    typer.echo(f"Written to {output}")


@scorers_app.command("list")
def list_scorers() -> None:
    """Show every scorer and the metrics it offers."""
    for name, cls in sorted(registered_scorers().items()):
        typer.echo(f"{name:<12} {cls.family.value:<14} {', '.join(_metrics_of(cls))}")


def _metrics_of(cls: type[Any]) -> tuple[str, ...]:
    """Metrics with default params, falling back to the declared list."""
    try:
        return tuple(cls(cls.contract_example().params).available_metrics())
    except Exception:  # A custom scorer may lack an example or need required params.
        return tuple(cls.metrics)


def _describe_params(cls: type[Any]) -> dict[str, dict[str, Any]]:
    schema = cls.Params.model_json_schema().get("properties", {})
    described: dict[str, dict[str, Any]] = {}
    for name, field in cls.Params.model_fields.items():
        prop = schema.get(name, {})
        kinds = [option.get("type") for option in prop.get("anyOf", [prop])]
        entry: dict[str, Any] = {"type": "|".join(str(kind) for kind in kinds if kind)}
        if field.is_required():
            entry["required"] = True
        else:
            entry["default"] = prop.get("default", field.default)
        described[name] = entry
    return described


@scorers_app.command("describe")
def describe_scorer(
    scorer_type: Annotated[str, typer.Argument(help="Scorer type, such as bertscore.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print as JSON.")] = False,
) -> None:
    """Show a scorer's parameters, defaults and metrics."""
    scorers = registered_scorers()
    if scorer_type not in scorers:
        raise _fail(ValueError(f"unknown scorer {scorer_type!r}; options: {sorted(scorers)}"))
    cls = scorers[scorer_type]
    described: dict[str, Any] = {
        "type": scorer_type,
        "family": cls.family.value,
        "metrics": list(_metrics_of(cls)),
        "params": _describe_params(cls),
    }
    if as_json:
        typer.echo(json.dumps(described, indent=2, default=str))
        return
    typer.echo(f"{scorer_type} ({described['family']})")
    typer.echo(f"Metrics: {', '.join(described['metrics'])}")
    params: dict[str, dict[str, Any]] = described["params"]
    for name, entry in params.items():
        detail = "required" if entry.get("required") else f"default {entry['default']!r}"
        typer.echo(f"  {name}: {entry['type']}, {detail}")
