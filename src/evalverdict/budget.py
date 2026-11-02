"""Spend tracking across a run."""

from __future__ import annotations

import threading

ROUNDING_TOLERANCE = 1e-9


class Budget:
    """Tracks spend across a run, allowing for calls that are still in flight.

    The first call runs alone, because nothing is known about cost until it returns.
    After that, each call in flight is assumed to cost as much as the dearest seen so far.
    """

    def __init__(self, limit_usd: float | None) -> None:
        self.changed = threading.Condition()
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self.in_flight = 0
        self.dearest_call_usd: float | None = None

    def reserve(self) -> bool:
        if self.limit_usd is not None:
            if self.dearest_call_usd is None:
                if self.in_flight > 0:
                    return False
            else:
                projected = self.spent_usd + (self.in_flight + 1) * self.dearest_call_usd
                if projected > self.limit_usd + ROUNDING_TOLERANCE:
                    return False
        self.in_flight += 1
        return True

    def settle(self, cost_usd: float | None) -> None:
        self.in_flight -= 1
        charged = cost_usd if cost_usd is not None else (self.dearest_call_usd or 0.0)
        self.spent_usd += charged
        self.dearest_call_usd = max(self.dearest_call_usd or 0.0, charged)
