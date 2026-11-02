"""EV-008: the LLM judge."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from evalverdict._optional import MissingExtraError
from evalverdict.budget import Budget
from evalverdict.cache import ResponseCache
from evalverdict.config import JudgeConfig, RunConfig, TargetConfig
from evalverdict.data import TestCase
from evalverdict.engines.judge_client import JudgeBudgetStop, JudgeCallError, JudgeClient
from evalverdict.runner import execute
from evalverdict.scorers.generation import JudgeScorer
from evalverdict.targets.litellm_target import LiteLLMTarget
from fakes.judge import StubJudge
from fakes.target import INSTANCES, CaseTarget, FakeResponse

ROOT = Path(__file__).resolve().parents[2]
RUBRIC = str(ROOT / "rubrics" / "summary_quality.yaml")
CASES = [
    TestCase(id="a", input="Source text one.", reference="Reference one."),
    TestCase(id="b", input="Source text two.", reference="Reference two."),
]
OUTPUTS = ["Summary one.", "Summary two."]
REFERENCES = ["Reference one.", "Reference two."]
REPLY = json.dumps({"reason": "good", "score": 8})


def scorer_with(client: Any, **params: Any) -> JudgeScorer:
    scorer = JudgeScorer(params={"rubric": RUBRIC, "concurrency": 1, **params})
    scorer.bind_judge(client, "anthropic/claude-sonnet-5-5", ["openai/gpt-6-luna"])
    return scorer


def run_config(
    tmp_path: Path, judge: dict[str, Any] | None, *, target: str = "model/a", **extra: Any
):
    data = tmp_path / "data.jsonl"
    data.write_text('{"id": "a", "input": "src", "reference": "ref"}\n')
    INSTANCES["t"] = CaseTarget(default=FakeResponse(output="summary"))
    settings: dict[str, Any] = {
        "name": "demo",
        "dataset": data,
        "targets": {"a": {"kind": "instance", "ref": "t", "model": target}},
        "judge": judge,
        "scorers": [{"type": "judge", "params": {"rubric": RUBRIC}}],
        "bootstrap": {"n_resamples": 20},
    }
    return RunConfig(**{**settings, **extra})


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.ticket("EV-008-T04")
def test_score_of_seven_of_ten_becomes_point_seven_with_a_reason() -> None:
    judge = StubJudge(scores=[0.7, 0.3], reasons=["clear", "vague"])

    result = scorer_with(judge).score(OUTPUTS, REFERENCES, CASES, ids=["a", "b"])[0]

    assert [item.value for item in result.per_item] == pytest.approx([0.7, 0.3])
    assert result.per_item[0].detail == {"reason": "clear"}
    assert result.value == pytest.approx(0.5)


@pytest.mark.ticket("EV-008-T05")
def test_result_details_hold_the_rubric_and_judge() -> None:
    result = scorer_with(StubJudge(scores=[0.5, 0.5])).score(OUTPUTS, REFERENCES, CASES)[0]

    rubric_text = Path(RUBRIC).read_text()
    assert result.details["rubric_text"] == rubric_text
    assert len(result.details["rubric_sha256"]) == 64
    assert result.details["judge_model"] == "anthropic/claude-sonnet-5-5"


@pytest.mark.ticket("EV-008-T06")
def test_judge_equal_to_a_target_is_refused(workspace: Path) -> None:
    config = run_config(workspace, {"model": "model/a"})

    with pytest.raises(ValueError, match="under test"):
        execute(config)


@pytest.mark.ticket("EV-008-T06")
def test_same_model_through_another_route_is_still_self_judging(workspace: Path) -> None:
    config = run_config(workspace, {"model": "openrouter/vendor/model-x"}, target="vendor/model-x")

    with pytest.raises(ValueError, match="under test"):
        execute(config)


@pytest.mark.ticket("EV-008-T07")
def test_override_allows_self_judging_and_flags_it(workspace: Path) -> None:
    mock = {"mock_response": REPLY}
    judge = {"model": "openai/gpt-6-luna", "allow_self_judging": True, "params": mock}
    config = run_config(workspace, judge, target="openai/gpt-6-luna")

    run_dir = execute(config)

    scores = json.loads((run_dir / "a" / "scores.json").read_text())
    assert scores["results"][0]["details"]["self_judging"] is True


@pytest.mark.ticket("EV-008-T08")
@pytest.mark.parametrize("alias", ["~google/gemini-flash-latest", "openai/gpt-latest"])
def test_alias_is_always_refused(alias: str) -> None:
    with pytest.raises(ValidationError, match="exact"):
        JudgeConfig(model=alias, allow_self_judging=True)


@pytest.mark.ticket("EV-008-T09")
def test_shared_vendor_is_flagged_without_blocking() -> None:
    scorer = JudgeScorer(params={"rubric": RUBRIC, "concurrency": 1})
    scorer.bind_judge(
        StubJudge(scores=[0.5, 0.5]),
        "anthropic/claude-sonnet-5-5",
        ["anthropic/claude-haiku-5-5", "openai/gpt-6-luna"],
    )

    result = scorer.score(OUTPUTS, REFERENCES, CASES)[0]

    assert result.details["shared_vendor_with"] == ["anthropic/claude-haiku-5-5"]


@pytest.mark.ticket("EV-008-T10")
def test_prompts_never_name_the_scored_model() -> None:
    judge = StubJudge(scores=[0.5, 0.5])

    scorer_with(judge).score(OUTPUTS, REFERENCES, CASES)

    assert len(judge.prompts) == 2
    joined = " ".join(judge.prompts)
    assert "Summary one." in joined
    for canary in ("gpt-6-luna", "haiku", "openai", "anthropic", "baseline", "candidate"):
        assert canary not in joined.lower()


def client_with(tmp_path: Path, target: CaseTarget, limit: float | None = None) -> JudgeClient:
    return JudgeClient(target, "judge/model", ResponseCache(tmp_path / "c"), Budget(limit), {})


@pytest.mark.ticket("EV-008-T11")
def test_same_judge_request_is_served_from_the_cache(tmp_path: Path) -> None:
    target = CaseTarget(default=FakeResponse(output=REPLY))
    client = client_with(tmp_path, target)

    first = client.complete("the prompt")
    second = client.complete("the prompt")

    assert first == second == REPLY
    assert len(target.calls) == 1
    assert client.cache_hits == 1


@pytest.mark.ticket("EV-008-T12")
def test_judge_cost_counts_against_the_budget(tmp_path: Path) -> None:
    target = CaseTarget(default=FakeResponse(output=REPLY, cost_usd=0.5))
    budget = Budget(10)
    client = JudgeClient(target, "judge/model", ResponseCache(tmp_path / "c"), budget, {})

    client.complete("one")
    client.complete("two")

    assert budget.spent_usd == pytest.approx(1.0)
    assert client.spend_usd == pytest.approx(1.0)


@pytest.mark.ticket("EV-008-T12")
def test_budget_stop_during_judging_keeps_scored_items(tmp_path: Path) -> None:
    target = CaseTarget(default=FakeResponse(output=REPLY, cost_usd=0.5))
    client = client_with(tmp_path, target, limit=0.75)
    scorer = scorer_with(client)

    result = scorer.score(OUTPUTS, REFERENCES, CASES)[0]

    assert result.n == 1
    assert scorer.health().stop_reason == "budget"


def test_client_raises_when_the_budget_is_spent(tmp_path: Path) -> None:
    target = CaseTarget(default=FakeResponse(output=REPLY, cost_usd=0.5))
    client = client_with(tmp_path, target, limit=0.75)
    client.complete("one")

    with pytest.raises(JudgeBudgetStop):
        client.complete("two")


def test_waiting_call_proceeds_once_the_first_cost_is_known(tmp_path: Path) -> None:
    target = CaseTarget(default=FakeResponse(output=REPLY, cost_usd=0.01), delay_s=0.02)
    client = client_with(tmp_path, target, limit=5)
    errors: list[Exception] = []

    def call(prompt: str) -> None:
        try:
            client.complete(prompt)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=call, args=(f"p{index}",)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert len(target.calls) == 4


@pytest.mark.ticket("EV-008-T13")
def test_judge_route_reporting_zero_cost_uses_pricing() -> None:
    options = {"pricing": {"input_per_mtok": 2, "output_per_mtok": 10}}
    target = LiteLLMTarget(TargetConfig(kind="litellm", model="unmapped/judge", **options))

    usage = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0, cost=0.0)

    assert target._cost(usage) == (2.0, "config_price")


@pytest.mark.ticket("EV-008-T14")
def test_scorer_level_judge_override_is_used_and_recorded(workspace: Path) -> None:
    mock = {"mock_response": REPLY}
    params = {"rubric": RUBRIC, "judge_model": "openai/gpt-6-luna"}
    config = run_config(
        workspace,
        {"model": "anthropic/claude-sonnet-5-5", "params": mock},
        scorers=[{"type": "judge", "params": params}],
    )

    run_dir = execute(config)

    scores = json.loads((run_dir / "a" / "scores.json").read_text())
    manifest = json.loads((run_dir / "a" / "manifest.json").read_text())
    assert scores["results"][0]["details"]["judge_model"] == "openai/gpt-6-luna"
    assert manifest["judge"]["model"] == "anthropic/claude-sonnet-5-5"
    assert manifest["scorers"][0]["params"]["judge_model"] == "openai/gpt-6-luna"


@pytest.mark.ticket("EV-008-T15")
def test_strict_mode_turns_scores_into_pass_or_fail_at_the_threshold() -> None:
    scorer = scorer_with(StubJudge(scores=[0.7, 0.4]), strict_mode=True, threshold=0.6)

    result = scorer.score(OUTPUTS, REFERENCES, CASES)[0]

    assert [item.value for item in result.per_item] == [1.0, 0.0]


def test_without_strict_mode_scores_stay_on_the_zero_to_one_scale() -> None:
    result = scorer_with(StubJudge(scores=[0.7, 0.4])).score(OUTPUTS, REFERENCES, CASES)[0]

    assert [item.value for item in result.per_item] == pytest.approx([0.7, 0.4])


class Garbled:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    def complete(self, prompt: str) -> str:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


@pytest.mark.ticket("EV-008-T16")
def test_unparseable_reply_is_retried_then_counted_as_a_judge_failure() -> None:
    scorer = scorer_with(Garbled(["not json at all"]))

    result = scorer.score(OUTPUTS[:1] + OUTPUTS[1:], REFERENCES, CASES)[0]

    assert result.name == "score"
    assert not hasattr(result, "value")
    assert scorer.health().failed == 2


def test_reply_wrapped_in_a_code_fence_is_accepted() -> None:
    scorer = scorer_with(Garbled([f"```json\n{REPLY}\n```"]))

    assert scorer.score(OUTPUTS, REFERENCES, CASES)[0].value == pytest.approx(0.8)


def test_reparsed_after_one_bad_reply_succeeds() -> None:
    client = Garbled(["junk", REPLY, REPLY])
    scorer = scorer_with(client)

    assert scorer.score(OUTPUTS, REFERENCES, CASES)[0].n == 2


@pytest.mark.ticket("EV-008-T17")
def test_judge_failures_above_the_limit_make_the_run_invalid(workspace: Path) -> None:
    config = run_config(
        workspace, {"model": "judge/model", "params": {"mock_response": "not json"}}
    )

    run_dir = execute(config)

    scores = json.loads((run_dir / "a" / "scores.json").read_text())
    assert scores["run"]["status"] == "invalid"
    assert scores["run"]["judge_failures"] == 1


def test_judge_scorer_without_a_judge_block_is_refused(workspace: Path) -> None:
    config = run_config(workspace, None)

    with pytest.raises(ValueError, match="judge"):
        execute(config)


def test_judge_call_error_is_raised_to_the_scorer(tmp_path: Path) -> None:
    client = client_with(tmp_path, CaseTarget(default=FakeResponse(error="RateLimitError")))

    with pytest.raises(JudgeCallError):
        client.complete("prompt")


def test_paid_judge_needs_a_budget(workspace: Path) -> None:
    config = run_config(workspace, {"model": "anthropic/claude-sonnet-5-5"})

    with pytest.raises(ValueError, match="budget_usd"):
        execute(config)


@pytest.mark.ticket("EV-008-T18")
def test_missing_extra_gives_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "deepeval", None)

    with pytest.raises(MissingExtraError, match="judge"):
        JudgeScorer(params={"rubric": RUBRIC}).check_dependencies()


@pytest.mark.ticket("EV-008-T19")
def test_telemetry_opt_out_is_set_and_nothing_reaches_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)

    scorer_with(StubJudge(scores=[0.5, 0.5])).score(OUTPUTS, REFERENCES, CASES)

    assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"


@pytest.mark.ticket("EV-008-T22")
def test_each_target_manifest_shows_only_its_own_judge_spend(workspace: Path) -> None:
    INSTANCES["t2"] = CaseTarget(default=FakeResponse(output="other summary"))
    targets = {
        "a": {"kind": "instance", "ref": "t", "model": "model/a"},
        "b": {"kind": "instance", "ref": "t2", "model": "model/b"},
    }
    config = run_config(
        workspace, {"model": "judge/model", "params": {"mock_response": REPLY}}, targets=targets
    )

    run_dir = execute(config)

    for name in ("a", "b"):
        manifest = json.loads((run_dir / name / "manifest.json").read_text())
        assert manifest["judge"]["calls"] + manifest["judge"]["cache_hits"] == 1
