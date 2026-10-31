"""Run configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalverdict.scoring.base import BootstrapConfig


class ScorerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    name: str | None = None
    enabled: bool = True
    metrics: list[str] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    bootstrap: dict[str, Any] | None = None

    @property
    def resolved_name(self) -> str:
        return self.name or self.type


class TargetConfig(BaseModel):
    """Target settings. Each target kind validates its own extra keys."""

    model_config = ConfigDict(extra="allow")

    kind: str


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = 0
    dataset: Path
    targets: dict[str, TargetConfig] = Field(min_length=1)
    scorers: list[ScorerEntry] = Field(min_length=1)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    sampling: dict[str, Any] = Field(default_factory=dict)
    budget_usd: float | None = Field(default=None, gt=0)
    max_error_rate: float = Field(default=0.05, ge=0, le=1)
    concurrency: int = Field(default=4, gt=0)
    repeats: int = Field(default=1, gt=0)
    cache_dir: Path = Path(".cache/evalverdict")

    @model_validator(mode="after")
    def _names_are_unique(self) -> RunConfig:
        names = [entry.resolved_name for entry in self.scorers]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"scorer names must be unique, repeated: {repeated}")
        return self

    def sha256(self) -> str:
        return _hash(self.model_dump(mode="json"))

    def resume_sha256(self) -> str:
        """Hash the settings that change results. Budget and pacing can change on resume."""
        return _hash(self.model_dump(mode="json", exclude={"budget_usd", "concurrency"}))


def _hash(settings: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(path.read_text()) or {})
