"""Loaders for the small synthetic datasets in tests/fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE_NAMES = ("agentic", "classification", "extraction", "generation", "ranking", "subgroups")


def load_fixture(name: str) -> list[dict[str, Any]]:
    lines = (FIXTURES / f"{name}.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line]


def fixture_sha256(name: str) -> str:
    return hashlib.sha256((FIXTURES / f"{name}.jsonl").read_bytes()).hexdigest()
