"""EV-006: real calls to the demo models. Each call is capped at a few tokens.

Run with: pytest -m live tests/live/test_routes.py -s
Every test prints a ROUTE line, so the findings can be recorded in SPEC.md.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from evalverdict.config import TargetConfig
from evalverdict.data import Prediction, TestCase
from evalverdict.targets.litellm_target import LiteLLMTarget, _litellm

HAIKU = "anthropic/claude-haiku-5-5"
LUNA = "openai/gpt-6-luna"
SONNET = "anthropic/claude-sonnet-5-5"
OPENROUTER_LUNA = "openrouter/openai/gpt-6-luna"
MODELS = [
    pytest.param(HAIKU, marks=pytest.mark.live("ANTHROPIC_API_KEY")),
    pytest.param(LUNA, marks=pytest.mark.live("OPENAI_API_KEY")),
    pytest.param(SONNET, marks=pytest.mark.live("ANTHROPIC_API_KEY")),
]
PROMPT = TestCase(id="route", input="Reply with exactly: ready")
KEY_FOR = {HAIKU: "ANTHROPIC_API_KEY", LUNA: "OPENAI_API_KEY", SONNET: "ANTHROPIC_API_KEY"}


def report(test: str, model: str, **facts: Any) -> None:
    print("ROUTE " + json.dumps({"test": test, "model": model, **facts}, default=str))


def call(model: str, **options: Any) -> Prediction:
    target = LiteLLMTarget(TargetConfig(kind="litellm", model=model, max_tokens=50, **options))
    return target.predict(PROMPT)


def facts(prediction: Prediction) -> dict[str, Any]:
    return {
        "output": prediction.output,
        "error": prediction.error,
        "ttft_s": prediction.ttft_s,
        "latency_s": prediction.latency_s,
        "input_tokens": prediction.input_tokens,
        "output_tokens": prediction.output_tokens,
        "cost_usd": prediction.cost_usd,
        "cost_source": prediction.cost_source,
    }


@pytest.mark.ticket("EV-006-T01")
@pytest.mark.ticket("EV-006-T02")
@pytest.mark.ticket("EV-006-T03")
@pytest.mark.parametrize("model", MODELS)
def test_streaming_call_returns_text_timing_and_usage(model: str) -> None:
    prediction = call(model)
    report("stream", model, **facts(prediction))

    assert prediction.error is None
    assert prediction.output
    assert prediction.ttft_s is not None and prediction.ttft_s > 0
    assert prediction.output_tokens


@pytest.mark.ticket("EV-006-T06")
@pytest.mark.parametrize("model", MODELS)
def test_cost_is_measured_for_every_demo_model(model: str) -> None:
    prediction = call(model)
    report("cost", model, cost_usd=prediction.cost_usd, cost_source=prediction.cost_source)

    assert prediction.cost_source != "not_measured"


@pytest.mark.ticket("EV-006-T04")
@pytest.mark.parametrize("model", MODELS)
def test_record_whether_temperature_is_accepted(model: str) -> None:
    litellm = _litellm()
    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": PROMPT.input}],
            max_tokens=20,
            temperature=0,
        )
        outcome = "accepted"
    except Exception as error:  # Recording the provider's answer is the point of this test.
        outcome = f"rejected: {type(error).__name__}"
    report("temperature", model, outcome=outcome)


@pytest.mark.ticket("EV-006-T05")
@pytest.mark.live("OPENAI_API_KEY")
def test_record_whether_seed_is_accepted_and_repeatable() -> None:
    litellm = _litellm()
    replies = []
    try:
        for _ in range(2):
            response = litellm.completion(
                model=LUNA,
                messages=[{"role": "user", "content": "Name one colour."}],
                max_tokens=20,
                seed=5,
            )
            replies.append(response.choices[0].message.content)
        outcome = "accepted"
    except Exception as error:  # Recording the provider's answer is the point of this test.
        outcome = f"rejected: {type(error).__name__}"
    report("seed", LUNA, outcome=outcome, replies=replies, identical=len(set(replies)) == 1)


@pytest.mark.ticket("EV-006-T07")
@pytest.mark.live("OPENROUTER_API_KEY")
def test_record_whether_openrouter_reports_cost_on_streams() -> None:
    prediction = call(OPENROUTER_LUNA, pricing={"input_per_mtok": 0.1, "output_per_mtok": 0.5})
    report("openrouter", OPENROUTER_LUNA, **facts(prediction))

    assert prediction.error is None


@pytest.mark.ticket("EV-006-T08")
@pytest.mark.live("ANTHROPIC_API_KEY")
def test_sonnet_returns_valid_judge_json() -> None:
    target = LiteLLMTarget(TargetConfig(kind="litellm", model=SONNET, max_tokens=120))
    prompt = (
        "Score this summary from 0 to 10 for accuracy. Source: 'The shop opens at nine.' "
        "Summary: 'The shop opens at nine.' "
        'Reply with only JSON: {"reason": "<one sentence>", "score": <integer>}'
    )
    prediction = target.predict(TestCase(id="json", input=prompt))
    report("judge_json", SONNET, output=prediction.output, error=prediction.error)

    parsed = json.loads(prediction.output[prediction.output.index("{") :].rsplit("}", 1)[0] + "}")
    assert set(parsed) == {"reason", "score"}
