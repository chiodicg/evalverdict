"""Command-line entry point."""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Evaluate models and decide whether to change them.",
)


@app.callback()
def main() -> None:
    """Evaluate models and decide whether to change them."""
