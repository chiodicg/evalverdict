"""Load API keys from a local .env file."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    """Load `.env` from the working directory. Exported variables take priority."""
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
