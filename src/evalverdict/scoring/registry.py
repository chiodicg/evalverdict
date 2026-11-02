"""Scorers register themselves here, so adding one needs no central edit."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

from pydantic import ValidationError

from evalverdict.config import ScorerEntry
from evalverdict.scoring.base import BootstrapConfig, Scorer

_SCORERS: dict[str, type[Scorer]] = {}


def register_scorer(scorer_type: str) -> Callable[[type[Scorer]], type[Scorer]]:
    def decorate(cls: type[Scorer]) -> type[Scorer]:
        if scorer_type in _SCORERS:
            raise ValueError(f"scorer type {scorer_type!r} is already registered")
        _SCORERS[scorer_type] = cls
        return cls

    return decorate


def _load_builtin_scorers() -> None:
    import evalverdict.scorers

    for module in pkgutil.iter_modules(evalverdict.scorers.__path__):
        importlib.import_module(f"evalverdict.scorers.{module.name}")


def scorer_types() -> list[str]:
    _load_builtin_scorers()
    return sorted(_SCORERS)


def registered_scorers() -> dict[str, type[Scorer]]:
    _load_builtin_scorers()
    return dict(_SCORERS)


def _import_scorer(path: str) -> type[Scorer]:
    """Load a custom scorer. This runs local code, so use trusted sources only."""
    if ":" not in path:
        raise ValueError(f"custom scorer path must look like module:Class, got {path!r}")
    module_name, class_name = path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"cannot import custom scorer module {module_name!r}: {error}") from error
    found = getattr(module, class_name, None)
    if found is None:
        raise ValueError(f"module {module_name!r} has no {class_name!r}")
    if not isinstance(found, type) or not issubclass(found, Scorer):
        raise ValueError(f"{path} does not subclass Scorer")
    return found


def scorer_class(entry: ScorerEntry) -> type[Scorer]:
    _load_builtin_scorers()
    if entry.import_path is not None:
        return _import_scorer(entry.import_path)
    if entry.type not in _SCORERS:
        raise ValueError(f"unknown scorer type {entry.type!r}; options: {sorted(_SCORERS)}")
    return _SCORERS[entry.type]


def build_scorer(entry: ScorerEntry, run_bootstrap: BootstrapConfig, seed: int) -> Scorer:
    cls = scorer_class(entry)
    bootstrap = run_bootstrap
    if entry.bootstrap:
        bootstrap = BootstrapConfig(**{**run_bootstrap.model_dump(), **entry.bootstrap})
    try:
        scorer = cls(entry.params, bootstrap=bootstrap, seed=seed, metrics=entry.metrics)
    except ValidationError as error:
        raise ValueError(f"scorer {entry.resolved_name!r}: {error}") from error
    scorer.check_dependencies()
    return scorer
