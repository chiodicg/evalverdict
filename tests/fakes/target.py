"""A scripted model target for offline tests."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass

from evalverdict.config import TargetConfig
from evalverdict.data import Prediction, TestCase
from evalverdict.targets.base import register_target


@dataclass(frozen=True)
class FakeResponse:
    output: str | None = None
    error: str | None = None
    latency_s: float = 0.1
    ttft_s: float | None = 0.05
    input_tokens: int | None = 10
    output_tokens: int | None = 5
    cost_usd: float | None = 0.0001
    parse_failed: bool = False


class FakeTarget:
    """Returns one scripted response per call, and records what it was asked."""

    paid = False
    cost_known = True

    def __init__(self, script: Iterable[FakeResponse]) -> None:
        self._script = list(script)
        self.calls: list[str] = []

    @classmethod
    def from_seed(
        cls, count: int, *, seed: int, error_rate: float = 0.0, malformed_rate: float = 0.0
    ) -> FakeTarget:
        rng = random.Random(seed)
        script = []
        for index in range(count):
            roll = rng.random()
            latency_s = round(rng.uniform(0.05, 2.0), 4)
            if roll < error_rate:
                script.append(FakeResponse(error="scripted error", latency_s=latency_s))
            elif roll < error_rate + malformed_rate:
                script.append(FakeResponse(output="{malformed", latency_s=latency_s))
            else:
                script.append(FakeResponse(output=f"answer {index}", latency_s=latency_s))
        return cls(script)

    def parameters_sent(self) -> dict[str, object]:
        return {}

    def predict(self, case: object) -> Prediction:
        case_id = str(getattr(case, "id", case))
        position = len(self.calls)
        if position >= len(self._script):
            raise AssertionError(f"fake target script ran out at call {position + 1}")
        self.calls.append(case_id)
        response = self._script[position]
        return Prediction(
            case_id=case_id,
            output=response.output,
            error=response.error,
            latency_s=response.latency_s,
            ttft_s=response.ttft_s,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            parse_failed=response.parse_failed,
        )


class CaseTarget:
    """Answers by case ID, so results don't depend on call order. Safe across threads."""

    def __init__(
        self,
        responses: dict[str, FakeResponse] | None = None,
        *,
        default: FakeResponse | None = None,
        paid: bool = False,
        cost_known: bool = True,
        delay_s: float = 0.0,
        interrupt_on: str | None = None,
        block_on: str | None = None,
    ) -> None:
        self.responses = responses or {}
        self.default = default or FakeResponse(output="answer")
        self.paid = paid
        self.cost_known = cost_known
        self.delay_s = delay_s
        self.interrupt_on = interrupt_on
        self.block_on = block_on
        self.release = threading.Event()
        self.calls: list[str] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def parameters_sent(self) -> dict[str, object]:
        return {}

    def predict(self, case: TestCase) -> Prediction:
        if case.id == self.interrupt_on:
            raise KeyboardInterrupt
        with self._lock:
            self.calls.append(case.id)
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        time.sleep(self.delay_s)
        if case.id == self.block_on:
            self.release.wait()
        with self._lock:
            self._in_flight -= 1
        response = self.responses.get(case.id, self.default)
        return Prediction(
            case_id=case.id,
            output=response.output,
            error=response.error,
            latency_s=response.latency_s,
            ttft_s=response.ttft_s,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            parse_failed=response.parse_failed,
        )


INSTANCES: dict[str, CaseTarget] = {}


@register_target("instance")
def build_instance_target(config: TargetConfig) -> CaseTarget:
    return INSTANCES[config.model_dump()["ref"]]


@register_target("fake")
def build_fake_target(config: TargetConfig) -> CaseTarget:
    """Gives each case a stable output derived from the seed."""
    options = config.model_dump()
    seed = options.get("seed", 0)

    class Seeded(CaseTarget):
        def predict(self, case: TestCase) -> Prediction:
            rng = random.Random(f"{seed}:{case.id}")
            words = " ".join(rng.choice(["cat", "rain", "bread", "train"]) for _ in range(5))
            self.default = FakeResponse(output=words, latency_s=round(rng.uniform(0.05, 2), 4))
            return super().predict(case)

    return Seeded()
