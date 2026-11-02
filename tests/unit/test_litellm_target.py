"""EV-005: the LiteLLM target."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import litellm
import pytest

from evalverdict.config import RunConfig, TargetConfig
from evalverdict.data import TestCase
from evalverdict.runner import execute
from evalverdict.targets.litellm_target import LiteLLMTarget, apply_shared_sampling

CASE = TestCase(id="a", input="hello")


def chunk(
    content: str | None = None, reasoning: str | None = None, usage: Any = None
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)


def usage(prompt: int = 100, completion: int = 40, cost: float | None = None) -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, cost=cost)


class Clock:
    """A clock advanced by the fake stream, so timings are exact."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make_target(model: str = "openai/gpt-6-luna", **options: Any) -> LiteLLMTarget:
    return LiteLLMTarget(TargetConfig(kind="litellm", model=model, **options))


def timed_stream(clock: Clock, steps: list[tuple[float, SimpleNamespace]]) -> Iterator[Any]:
    for delay, item in steps:
        clock.now += delay
        yield item


def replying(*streams: Any) -> Callable[..., Any]:
    """Each call returns the next stream, or raises it if it is an exception."""
    queue = list(streams)
    calls: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    completion.calls = calls  # type: ignore[attr-defined]
    return completion


def rate_limited(headers: dict[str, str] | None = None) -> Exception:
    request = httpx.Request("POST", "https://provider.invalid")
    response = httpx.Response(429, headers=headers or {}, request=request)
    return litellm.RateLimitError("failed", llm_provider="openai", model="m", response=response)


def timed_out() -> Exception:
    return litellm.Timeout("slow", model="m", llm_provider="openai")


def test_target_is_paid_and_rejects_unknown_options() -> None:
    assert make_target().paid is True
    with pytest.raises(ValueError, match="nope"):
        make_target(nope=1)


def test_mock_response_is_not_a_paid_call() -> None:
    assert make_target(params={"mock_response": "hi"}).paid is False


@pytest.mark.ticket("EV-005-T01")
def test_stream_is_reassembled_in_order() -> None:
    target = make_target()
    target.completion = replying(iter([chunk("hel"), chunk("lo "), chunk("there"), chunk()]))

    assert target.predict(CASE).output == "hello there"


def test_input_and_system_prompt_are_sent() -> None:
    target = make_target(system_prompt="Be brief.", max_tokens=50)
    target.completion = replying(iter([chunk("ok")]))

    target.predict(CASE)

    sent = target.completion.calls[0]  # type: ignore[attr-defined]
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hello"},
    ]
    assert sent["max_tokens"] == 50
    assert sent["stream"] is True


@pytest.mark.ticket("EV-005-T02")
def test_ttft_ignores_reasoning_output() -> None:
    clock = Clock()
    target = make_target()
    target.clock = clock
    steps = [(1.0, chunk(reasoning="thinking")), (2.0, chunk("answer")), (1.0, chunk(" done"))]
    target.completion = replying(timed_stream(clock, steps))

    assert target.predict(CASE).ttft_s == pytest.approx(3.0)


@pytest.mark.ticket("EV-005-T03")
def test_timings_match_known_values() -> None:
    clock = Clock()
    target = make_target()
    target.clock = clock
    steps = [
        (0.5, chunk("a")),
        (1.0, chunk("b")),
        (1.0, chunk("c")),
        (0.0, chunk(usage=usage(completion=50))),
    ]
    target.completion = replying(timed_stream(clock, steps))

    prediction = target.predict(CASE)

    assert prediction.ttft_s == pytest.approx(0.5)
    assert prediction.latency_s == pytest.approx(2.5)
    assert prediction.throughput_tokens_per_s == pytest.approx(25.0)


@pytest.mark.ticket("EV-005-T04")
def test_single_chunk_has_no_throughput_and_no_crash() -> None:
    target = make_target()
    target.completion = replying(iter([chunk("all at once", usage=usage())]))

    assert target.predict(CASE).throughput_tokens_per_s is None


@pytest.mark.ticket("EV-005-T05")
def test_missing_usage_is_not_measured_rather_than_zero() -> None:
    target = make_target()
    target.completion = replying(iter([chunk("text")]))

    prediction = target.predict(CASE)

    assert prediction.input_tokens is None
    assert prediction.output_tokens is None
    assert prediction.cost_usd is None
    assert prediction.cost_source == "not_measured"


@pytest.mark.ticket("EV-005-T06")
def test_cost_on_the_response_wins() -> None:
    target = make_target(pricing={"input_per_mtok": 99, "output_per_mtok": 99})
    target.completion = replying(iter([chunk("x", usage=usage(cost=0.0123))]))

    prediction = target.predict(CASE)

    assert (prediction.cost_usd, prediction.cost_source) == (0.0123, "response_usage")


@pytest.mark.ticket("EV-005-T07")
def test_mapped_price_is_used_when_no_cost_is_reported() -> None:
    target = make_target("openai/gpt-6-luna")
    target.completion = replying(iter([chunk("x", usage=usage(1000, 1000))]))

    prediction = target.predict(CASE)

    assert prediction.cost_source == "litellm_price"
    assert prediction.cost_usd == pytest.approx(1000 * 1e-7 + 1000 * 5e-7)


@pytest.mark.ticket("EV-005-T08")
def test_config_price_is_used_for_an_unmapped_model() -> None:
    target = make_target("unmapped/model", pricing={"input_per_mtok": 2, "output_per_mtok": 10})
    target.completion = replying(iter([chunk("x", usage=usage(1_000_000, 1_000_000))]))

    prediction = target.predict(CASE)

    assert (prediction.cost_usd, prediction.cost_source) == (12.0, "config_price")


@pytest.mark.ticket("EV-005-T09")
def test_no_cost_source_is_not_measured_and_never_zero() -> None:
    target = make_target("unmapped/model")
    target.completion = replying(iter([chunk("x", usage=usage())]))

    prediction = target.predict(CASE)

    assert prediction.cost_usd is None
    assert prediction.cost_source == "not_measured"
    assert target.cost_known is False


@pytest.mark.ticket("EV-005-T10")
def test_zero_reported_cost_on_real_usage_falls_through() -> None:
    target = make_target("unmapped/model", pricing={"input_per_mtok": 2, "output_per_mtok": 10})
    target.completion = replying(iter([chunk("x", usage=usage(1_000_000, 0, cost=0.0))]))

    prediction = target.predict(CASE)

    assert (prediction.cost_usd, prediction.cost_source) == (2.0, "config_price")


@pytest.mark.ticket("EV-005-T11")
def test_unmapped_model_lookup_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def unmapped(**kwargs: Any) -> None:
        raise litellm.exceptions.NotFoundError("unmapped", llm_provider="x", model="m")

    monkeypatch.setattr(litellm, "cost_per_token", unmapped)
    target = make_target("unmapped/model")
    target.completion = replying(iter([chunk("x", usage=usage())]))

    assert target.predict(CASE).cost_source == "not_measured"


@pytest.mark.ticket("EV-005-T12")
def test_shared_sampling_is_sent_when_every_target_supports_it() -> None:
    targets = {"a": make_target("openai/gpt-6-luna"), "b": make_target("openai/gpt-6-luna")}

    apply_shared_sampling(targets, {"temperature": 0, "seed": 5})

    assert targets["a"].parameters_sent()["temperature"] == 0
    assert targets["b"].parameters_sent()["seed"] == 5


@pytest.mark.ticket("EV-005-T13")
def test_shared_sampling_is_dropped_for_all_if_any_target_lacks_it() -> None:
    targets = {
        "a": make_target("openai/gpt-6-luna"),
        "b": make_target("anthropic/claude-haiku-5-5"),
    }

    dropped = apply_shared_sampling(targets, {"temperature": 0, "seed": 5})

    assert set(dropped) == {"temperature", "seed"}
    assert "temperature" not in targets["a"].parameters_sent()
    assert "seed" not in targets["b"].parameters_sent()


@pytest.mark.ticket("EV-005-T14")
def test_explicit_param_the_model_rejects_is_refused() -> None:
    with pytest.raises(ValueError, match="temperature"):
        make_target("anthropic/claude-haiku-5-5", params={"temperature": 0})


def test_setting_rejected_only_at_call_time_becomes_a_recorded_error() -> None:
    error = litellm.UnsupportedParamsError(message="needs reasoning off", model="m")
    target = make_target(max_retries=3)
    target.completion = replying(error)

    prediction = target.predict(CASE)

    assert prediction.error == "UnsupportedParamsError"
    assert prediction.retries == 0


@pytest.mark.ticket("EV-005-T15")
def test_transient_error_is_retried_with_latency_from_the_final_attempt() -> None:
    clock = Clock()
    target = make_target()
    target.clock = clock
    target.sleep = lambda seconds: setattr(clock, "now", clock.now + seconds)
    target.completion = replying(rate_limited(), iter([chunk("ok", usage=usage())]))

    prediction = target.predict(CASE)

    assert prediction.output == "ok"
    assert prediction.retries == 1
    assert prediction.latency_s == pytest.approx(0)


@pytest.mark.ticket("EV-005-T16")
def test_retry_after_header_sets_the_wait() -> None:
    waits: list[float] = []
    target = make_target()
    target.sleep = waits.append
    target.completion = replying(
        rate_limited({"retry-after": "7"}), iter([chunk("ok", usage=usage())])
    )

    target.predict(CASE)

    assert waits == [7.0]


@pytest.mark.ticket("EV-005-T17")
def test_exhausted_retries_return_the_error() -> None:
    target = make_target(max_retries=2)
    target.sleep = lambda seconds: None
    target.completion = replying(rate_limited(), rate_limited(), rate_limited())

    prediction = target.predict(CASE)

    assert prediction.error == "RateLimitError"
    assert prediction.output is None
    assert prediction.retries == 2


@pytest.mark.ticket("EV-005-T18")
def test_timeout_mid_stream_records_an_error_and_no_partial_text() -> None:
    def broken() -> Iterator[Any]:
        yield chunk("partial")
        raise timed_out()

    target = make_target(max_retries=0)
    target.completion = replying(broken())

    prediction = target.predict(CASE)

    assert prediction.error == "Timeout"
    assert prediction.output is None


@pytest.mark.ticket("EV-005-T19")
def test_authentication_error_fails_fast_without_the_key() -> None:
    secret = "sk-canary-value"
    error = litellm.AuthenticationError(f"bad key {secret}", llm_provider="openai", model="m")
    target = make_target()
    target.completion = replying(error)

    prediction = target.predict(CASE)

    assert prediction.retries == 0
    assert secret not in (prediction.error or "")
    assert prediction.error == "AuthenticationError"


@pytest.mark.ticket("EV-005-T20")
def test_run_through_litellm_mock_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text('{"id": "a", "input": "hi", "reference": "alpha beta"}\n')
    target = {"kind": "litellm", "model": "openai/gpt-6-luna", "params": {"mock_response": "alpha"}}
    config = RunConfig(
        name="demo",
        dataset=data,
        targets={"a": target},
        scorers=[{"type": "rouge", "metrics": ["rouge1"]}],
        bootstrap={"n_resamples": 20},
        sampling={"seed": 5},
    )

    run_dir = execute(config)

    row = json.loads((run_dir / "a" / "predictions.jsonl").read_text())
    manifest = json.loads((run_dir / "a" / "manifest.json").read_text())
    assert row["output"] == "alpha"
    assert row["ttft_s"] is not None
    assert manifest["parameters_sent"]["seed"] == 5


def test_changing_sampling_changes_the_cache_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text('{"id": "a", "input": "hi", "reference": "x"}\n')
    target = {"kind": "litellm", "model": "openai/gpt-6-luna", "params": {"mock_response": "x"}}
    settings: dict[str, Any] = {
        "name": "demo",
        "dataset": data,
        "targets": {"a": target},
        "scorers": [{"type": "rouge", "metrics": ["rouge1"]}],
        "bootstrap": {"n_resamples": 20},
    }

    execute(RunConfig(**settings, sampling={"seed": 1}))
    second = execute(RunConfig(**settings, sampling={"seed": 2}))

    assert json.loads((second / "a" / "scores.json").read_text())["run"]["cache_hits"] == 0
