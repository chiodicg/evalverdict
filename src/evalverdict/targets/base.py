"""The target contract and registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any, Protocol

from evalverdict.config import TargetConfig
from evalverdict.data import Prediction, TestCase


class Target(Protocol):
    @property
    def paid(self) -> bool: ...

    @property
    def cost_known(self) -> bool: ...

    def predict(self, case: TestCase) -> Prediction: ...

    def parameters_sent(self) -> dict[str, Any]: ...


TargetFactory = Callable[[TargetConfig], Target]
_TARGETS: dict[str, TargetFactory] = {}


def register_target(kind: str) -> Callable[[TargetFactory], TargetFactory]:
    def decorate(factory: TargetFactory) -> TargetFactory:
        if kind in _TARGETS:
            raise ValueError(f"target kind {kind!r} is already registered")
        _TARGETS[kind] = factory
        return factory

    return decorate


def _load_builtin_targets() -> None:
    import evalverdict.targets

    for module in pkgutil.iter_modules(evalverdict.targets.__path__):
        if module.name != "base":
            importlib.import_module(f"evalverdict.targets.{module.name}")


def build_target(config: TargetConfig) -> Target:
    _load_builtin_targets()
    if config.kind not in _TARGETS:
        raise ValueError(f"unknown target kind {config.kind!r}; options: {sorted(_TARGETS)}")
    return _TARGETS[config.kind](config)
