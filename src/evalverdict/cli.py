"""Command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from evalverdict._optional import MissingExtraError
from evalverdict.config import load_config
from evalverdict.env import load_env
from evalverdict.runner import execute

EXIT_INVALID = 3

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Evaluate models and decide whether to change them.",
)


@app.callback()
def main() -> None:
    """Evaluate models and decide whether to change them."""
    load_env()


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to a run config.")],
) -> None:
    """Run every target in a config, score the results and write a run directory."""
    try:
        run_dir = execute(load_config(config))
    except (ValidationError, ValueError, OSError, MissingExtraError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(EXIT_INVALID) from error
    typer.echo(f"Run written to {run_dir}")
