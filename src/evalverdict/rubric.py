"""Judge rubrics: versioned YAML whose full text is shown in every report."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

RubricField = Literal["input", "actual_output", "expected_output", "retrieval_context"]


class RubricError(ValueError):
    """A rubric file is missing or malformed."""


class _RubricFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: int
    description: str = ""
    fields: list[RubricField] = Field(min_length=1)
    steps: list[str] = Field(min_length=1)


@dataclass(frozen=True)
class Rubric:
    name: str
    version: int
    fields: tuple[str, ...]
    steps: tuple[str, ...]
    text: str
    sha256: str


def load_rubric(path: Path) -> Rubric:
    if not path.is_file():
        raise RubricError(f"rubric not found: {path}")
    text = path.read_text()
    try:
        parsed = _RubricFile.model_validate(yaml.safe_load(text))
    except (yaml.YAMLError, ValidationError) as error:
        raise RubricError(f"invalid rubric {path}: {error}") from error
    return Rubric(
        name=parsed.name,
        version=parsed.version,
        fields=tuple(parsed.fields),
        steps=tuple(parsed.steps),
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
