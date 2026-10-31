"""A scripted model target for offline tests."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from evalverdict.config import TargetConfig
from evalverdict.data import Prediction
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


class FakeTarget:
    """Returns one scripted response per call, and records what it was asked."""

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
        )


@register_target("fake")
def build_fake_target(config: TargetConfig) -> FakeTarget:
    options = config.model_dump()
    return FakeTarget.from_seed(
        options.get("count", 1000),
        seed=options.get("seed", 0),
        error_rate=options.get("error_rate", 0.0),
        malformed_rate=options.get("malformed_rate", 0.0),
    )
