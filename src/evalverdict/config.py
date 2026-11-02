"""Run configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evalverdict.scoring.base import BootstrapConfig


class ScorerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: str | None = None
    import_path: str | None = Field(default=None, alias="import")
    name: str | None = None
    enabled: bool = True
    metrics: list[str] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    bootstrap: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _one_source(self) -> ScorerEntry:
        if (self.type is None) == (self.import_path is None):
            raise ValueError("a scorer entry needs exactly one of 'type' and 'import'")
        return self

    @property
    def resolved_name(self) -> str:
        if self.name:
            return self.name
        if self.type:
            return self.type
        return str(self.import_path).rsplit(":", 1)[-1]


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None
    gate: Literal["ci", "point"] = "ci"
    primary: bool = False

    @model_validator(mode="after")
    def _has_threshold(self) -> GateConfig:
        if self.min is None and self.max is None:
            raise ValueError("a gate needs min, max or both")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("a gate's min cannot be above its max")
        return self


class TargetConfig(BaseModel):
    """Target settings. Each target kind validates its own extra keys."""

    model_config = ConfigDict(extra="allow")

    kind: str


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    pricing: dict[str, float] | None = None
    max_tokens: int = Field(default=1024, gt=0)
    params: dict[str, Any] = Field(default_factory=dict)
    allow_self_judging: bool = False

    @field_validator("model")
    @classmethod
    def _exact_model(cls, model: str) -> str:
        if model.startswith("~") or "latest" in model:
            raise ValueError(f"the judge needs an exact model ID, not an alias: {model}")
        return model


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_items_shown: int = Field(default=20, ge=0)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = 0
    dataset: Path
    targets: dict[str, TargetConfig] = Field(min_length=1)
    scorers: list[ScorerEntry] = Field(min_length=1)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    judge: JudgeConfig | None = None
    gates: dict[str, GateConfig] = Field(default_factory=dict)
    report: ReportConfig = Field(default_factory=ReportConfig)
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
