"""Call and cost estimates, so a run can be priced before any money is spent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalverdict.config import RunConfig, TargetConfig
from evalverdict.scoring.registry import build_scorer
from evalverdict.targets.base import build_target
from evalverdict.targets.litellm_target import LiteLLMTarget

# Planning assumptions from the spec's Risks table. Real usage is measured by EV-013.
TARGET_TOKENS = (1500, 300)
JUDGE_TOKENS = (2000, 500)


@dataclass(frozen=True)
class Estimate:
    target_calls: int
    judge_calls: int
    target_cost_usd: float
    judge_cost_usd: float

    @property
    def total_cost_usd(self) -> float:
        return self.target_cost_usd + self.judge_cost_usd

    def describe(self) -> str:
        return (
            f"{self.target_calls} target calls, {self.judge_calls} judge calls, "
            f"estimated cost ${self.total_cost_usd:.4f} "
            f"(targets ${self.target_cost_usd:.4f}, judge ${self.judge_cost_usd:.4f})"
        )


def _call_cost(target: object, tokens: tuple[int, int]) -> float:
    if not isinstance(target, LiteLLMTarget) or not target.paid:
        return 0.0
    return target.estimate_call_usd(*tokens) or 0.0


def estimate_run(config: RunConfig, n_cases: int) -> Estimate:
    calls_per_target = n_cases * config.repeats
    target_cost = sum(
        calls_per_target * _call_cost(build_target(target), TARGET_TOKENS)
        for target in config.targets.values()
    )
    target_calls = calls_per_target * len(config.targets)

    judge_calls = 0
    judge_cost = 0.0
    for entry in config.scorers:
        if not entry.enabled:
            continue
        scorer = build_scorer(entry, config.bootstrap, config.seed)
        if not scorer.needs_judge or config.judge is None:
            continue
        calls = target_calls * scorer.judge_calls_per_item
        model = scorer.params.judge_model or config.judge.model
        settings: dict[str, Any] = {
            "kind": "litellm",
            "model": model,
            "params": config.judge.params,
        }
        if config.judge.pricing is not None:
            settings["pricing"] = config.judge.pricing
        judge = LiteLLMTarget(TargetConfig(**settings))
        judge_calls += calls
        judge_cost += calls * _call_cost(judge, JUDGE_TOKENS)
    return Estimate(target_calls, judge_calls, target_cost, judge_cost)
