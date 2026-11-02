"""LLM calls through LiteLLM, with honest timing and cost measurement."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evalverdict.config import TargetConfig
from evalverdict.data import Prediction, TestCase
from evalverdict.targets.base import Target, register_target

SAMPLING_PARAMS = {"temperature", "top_p", "seed"}
MAX_BACKOFF_S = 30.0
MILLION = 1_000_000


def _litellm() -> ModuleType:
    """Import LiteLLM using its bundled price list, so the import itself makes no request."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    return import_module("litellm")


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class LiteLLMOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    model: str
    system_prompt: str | None = None
    max_tokens: int = Field(default=1024, gt=0)
    timeout_s: float = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)
    params: dict[str, Any] = Field(default_factory=dict)
    pricing: Pricing | None = None


@dataclass(frozen=True)
class _Reply:
    text: str
    usage: Any
    latency_s: float
    ttft_s: float | None
    first_to_last_s: float | None


def _supported_sampling(model: str) -> set[str]:
    litellm = _litellm()
    info = litellm.model_cost.get(model.split("/", 1)[-1], {})
    if info.get("supports_sampling_params") is False:
        return set()
    try:
        supported = litellm.get_supported_openai_params(model=model) or []
    except Exception:  # LiteLLM raises assorted errors for models it doesn't know.
        return set()
    return SAMPLING_PARAMS & set(supported)


class LiteLLMTarget:
    def __init__(self, config: TargetConfig) -> None:
        self.options = LiteLLMOptions.model_validate(config.model_dump())
        self.mocked = "mock_response" in self.options.params
        self.paid = not self.mocked
        self.sampling: dict[str, Any] = {}
        self.clock: Callable[[], float] = time.monotonic
        self.sleep: Callable[[float], None] = time.sleep
        self.completion: Callable[..., Any] = _litellm().completion

        requested = SAMPLING_PARAMS & set(self.options.params)
        rejected = requested - self.supported_sampling()
        if rejected:
            raise ValueError(f"{self.options.model} does not accept {sorted(rejected)}")

    def supported_sampling(self) -> set[str]:
        return SAMPLING_PARAMS if self.mocked else _supported_sampling(self.options.model)

    def parameters_sent(self) -> dict[str, Any]:
        sent = {"max_tokens": self.options.max_tokens, **self.options.params, **self.sampling}
        sent.pop("mock_response", None)
        return sent

    @property
    def cost_known(self) -> bool:
        """Whether a cost can be worked out before any call is made."""
        return (
            self.mocked
            or self.options.pricing is not None
            or self._mapped_cost(1000, 1000) is not None
        )

    def _mapped_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        try:
            prompt, completion = _litellm().cost_per_token(
                model=self.options.model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
        except Exception:  # Unmapped models raise several different exception types.
            return None
        total = float(prompt) + float(completion)
        return total if total > 0 else None

    def estimate_call_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        mapped = self._mapped_cost(input_tokens, output_tokens)
        if mapped is not None:
            return mapped
        pricing = self.options.pricing
        if pricing is None:
            return None
        total = input_tokens * pricing.input_per_mtok + output_tokens * pricing.output_per_mtok
        return total / MILLION

    def _cost(self, usage: Any) -> tuple[float | None, str]:
        """Provider figure, then mapped price, then config price, otherwise not measured."""
        if usage is None:
            return None, "not_measured"
        reported = getattr(usage, "cost", None)
        if reported:
            return float(reported), "provider"
        mapped = self._mapped_cost(usage.prompt_tokens, usage.completion_tokens)
        if mapped is not None:
            return mapped, "litellm_price"
        pricing = self.options.pricing
        if pricing is not None:
            total = usage.prompt_tokens * pricing.input_per_mtok
            total += usage.completion_tokens * pricing.output_per_mtok
            return total / MILLION, "config_price"
        return None, "not_measured"

    def _messages(self, case: TestCase) -> list[dict[str, str]]:
        messages = [{"role": "user", "content": str(case.input)}]
        if self.options.system_prompt:
            messages.insert(0, {"role": "system", "content": self.options.system_prompt})
        return messages

    def _attempt(self, case: TestCase) -> _Reply:
        started = self.clock()
        stream = self.completion(
            model=self.options.model,
            messages=self._messages(case),
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=self.options.max_tokens,
            timeout=self.options.timeout_s,
            **self.options.params,
            **self.sampling,
        )
        parts: list[str] = []
        first_at: float | None = None
        last_at: float | None = None
        usage = None
        for chunk in stream:
            usage = getattr(chunk, "usage", None) or usage
            choices = getattr(chunk, "choices", None)
            text = choices[0].delta.content if choices else None
            if text:
                last_at = self.clock()
                first_at = first_at if first_at is not None else last_at
                parts.append(text)
        finished = self.clock()
        spread = last_at - first_at if first_at is not None and last_at is not None else None
        return _Reply(
            text="".join(parts),
            usage=usage,
            latency_s=finished - started,
            ttft_s=first_at - started if first_at is not None else None,
            first_to_last_s=spread if spread else None,
        )

    def _backoff(self, attempt: int, error: Exception) -> float:
        headers = getattr(getattr(error, "response", None), "headers", None) or {}
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return min(2.0**attempt, MAX_BACKOFF_S)

    def predict(self, case: TestCase) -> Prediction:
        litellm = _litellm()
        retryable = (
            litellm.RateLimitError,
            litellm.Timeout,
            litellm.APIConnectionError,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
        )
        retries = 0
        while True:
            try:
                reply = self._attempt(case)
                break
            except retryable as error:
                if retries >= self.options.max_retries:
                    return Prediction(case_id=case.id, error=type(error).__name__, retries=retries)
                self.sleep(self._backoff(retries, error))
                retries += 1
            except litellm.AuthenticationError as error:
                return Prediction(case_id=case.id, error=type(error).__name__, retries=retries)
            except (
                Exception
            ) as error:  # Provider SDK messages can echo keys, so keep the type only.
                return Prediction(case_id=case.id, error=type(error).__name__, retries=retries)

        usage = reply.usage
        cost, source = self._cost(usage)
        output_tokens = usage.completion_tokens if usage is not None else None
        throughput = None
        if output_tokens and reply.first_to_last_s:
            throughput = output_tokens / reply.first_to_last_s
        return Prediction(
            case_id=case.id,
            output=reply.text,
            latency_s=reply.latency_s,
            ttft_s=reply.ttft_s,
            throughput_tokens_per_s=throughput,
            input_tokens=usage.prompt_tokens if usage is not None else None,
            output_tokens=output_tokens,
            cost_usd=cost,
            cost_source=source,
            retries=retries,
        )


@register_target("litellm")
def build_litellm_target(config: TargetConfig) -> LiteLLMTarget:
    return LiteLLMTarget(config)


def apply_shared_sampling(targets: Mapping[str, Target], sampling: Mapping[str, Any]) -> list[str]:
    """Send a sampling setting only if every LLM target accepts it (D11).

    Returns the settings that were dropped.
    """
    llm_targets = [target for target in targets.values() if isinstance(target, LiteLLMTarget)]
    if not llm_targets:
        return []
    shared = set.intersection(*(target.supported_sampling() for target in llm_targets))
    accepted = {name: value for name, value in sampling.items() if name in shared}
    for target in llm_targets:
        target.sampling = accepted
    return sorted(set(sampling) - shared)
