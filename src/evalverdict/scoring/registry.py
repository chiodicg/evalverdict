"""Scorers register themselves here, so adding one needs no central edit."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

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


def build_scorer(entry: ScorerEntry, run_bootstrap: BootstrapConfig, seed: int) -> Scorer:
    _load_builtin_scorers()
    if entry.type not in _SCORERS:
        raise ValueError(f"unknown scorer type {entry.type!r}; options: {sorted(_SCORERS)}")
    bootstrap = run_bootstrap
    if entry.bootstrap:
        bootstrap = BootstrapConfig(**{**run_bootstrap.model_dump(), **entry.bootstrap})
    scorer = _SCORERS[entry.type](
        entry.params, bootstrap=bootstrap, seed=seed, metrics=entry.metrics
    )
    scorer.check_dependencies()
    return scorer
