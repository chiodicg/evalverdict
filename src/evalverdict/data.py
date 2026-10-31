"""Test cases, predictions and dataset loading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

CORE_FIELDS = {"id", "input", "reference", "metadata"}


class DatasetError(ValueError):
    """A dataset file is missing or malformed."""


@dataclass(frozen=True)
class TestCase:
    __test__: ClassVar[bool] = False  # stop pytest collecting this as a test class

    id: str
    input: Any
    reference: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    case_id: str
    output: Any = None
    error: str | None = None
    latency_s: float | None = None
    ttft_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class Dataset:
    path: Path
    cases: tuple[TestCase, ...]
    sha256: str


def _parse_case(line: str, number: int) -> TestCase:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise DatasetError(f"line {number}: invalid JSON: {error}") from error
    if not isinstance(row, dict) or "id" not in row:
        raise DatasetError(f"line {number}: each row needs an 'id'")
    return TestCase(
        id=str(row["id"]),
        input=row.get("input"),
        reference=row.get("reference"),
        metadata=row.get("metadata") or {},
        extra={key: value for key, value in row.items() if key not in CORE_FIELDS},
    )


def load_dataset(path: Path) -> Dataset:
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")
    raw = path.read_bytes()
    cases: list[TestCase] = []
    seen: set[str] = set()
    for number, line in enumerate(raw.decode().splitlines(), start=1):
        if not line.strip():
            continue
        case = _parse_case(line, number)
        if case.id in seen:
            raise DatasetError(f"line {number}: duplicate id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetError(f"dataset is empty: {path}")
    return Dataset(path=path, cases=tuple(cases), sha256=hashlib.sha256(raw).hexdigest())
