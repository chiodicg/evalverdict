"""Judge calls, under the same cache and budget as target calls."""

from __future__ import annotations

import threading
from typing import Any, Protocol

from evalverdict.budget import Budget
from evalverdict.cache import ResponseCache, cache_key
from evalverdict.data import TestCase
from evalverdict.targets.base import Target


class JudgeCallError(RuntimeError):
    """The judge model returned an error."""


class JudgeBudgetStop(RuntimeError):
    """The budget has no room for another judge call."""


class JudgeBackend(Protocol):
    def complete(self, prompt: str) -> str: ...


class JudgeClient:
    def __init__(
        self,
        target: Target,
        model: str,
        cache: ResponseCache,
        budget: Budget,
        identity: dict[str, Any],
    ) -> None:
        self.target = target
        self.model = model
        self.cache = cache
        self.budget = budget
        self.identity = {"judge": model, **identity}
        self.spend_usd = 0.0
        self.calls = 0
        self.cache_hits = 0
        self._lock = threading.Lock()

    def _reserve(self) -> None:
        with self.budget.changed:
            while not self.budget.reserve():
                if self.budget.in_flight == 0:
                    raise JudgeBudgetStop("the budget has no room for another judge call")
                self.budget.changed.wait(timeout=1)

    def complete(self, prompt: str) -> str:
        key = cache_key(self.identity, prompt, repeat=0)
        cached = self.cache.get(key)
        if cached is not None:
            with self._lock:
                self.cache_hits += 1
            return str(cached.output)

        self._reserve()
        prediction = self.target.predict(TestCase(id=key[:16], input=prompt))
        with self.budget.changed:
            self.budget.settle(prediction.cost_usd)
            self.budget.changed.notify_all()
        with self._lock:
            self.calls += 1
            self.spend_usd += prediction.cost_usd or 0.0
        if prediction.error is not None:
            raise JudgeCallError(prediction.error)
        self.cache.put(key, prediction)
        return str(prediction.output)
