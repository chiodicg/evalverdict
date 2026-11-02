"""A call that never returns must not freeze a run (found live in EV-013)."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from evalverdict.config import RunConfig
from evalverdict.data import TestCase
from evalverdict.runner import execute
from evalverdict.scorers.generation import JudgeScorer
from fakes.judge import StubJudge
from fakes.target import INSTANCES, CaseTarget, FakeResponse

ROOT = Path(__file__).resolve().parents[2]
LIMIT_S = 15


def finishes(work: Callable[[], Any]) -> Any:
    """Run `work` in a daemon thread, so a hang fails the test instead of the suite."""
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = work()
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(LIMIT_S)
    assert not thread.is_alive(), f"did not finish within {LIMIT_S} seconds"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_text(
        "".join(
            json.dumps({"id": f"c{n}", "input": "x", "reference": "alpha"}) + "\n" for n in range(8)
        )
    )
    yield tmp_path
    for target in INSTANCES.values():
        target.release.set()
    INSTANCES.clear()


def config(workspace: Path, target: CaseTarget, **extra: Any) -> RunConfig:
    INSTANCES["t"] = target
    return RunConfig(
        name="demo",
        dataset=workspace / "data.jsonl",
        targets={"a": {"kind": "instance", "ref": "t"}},
        scorers=[{"type": "rouge", "metrics": ["rouge1"]}],
        bootstrap={"n_resamples": 20},
        **extra,
    )


@pytest.mark.ticket("EV-004-T20")
def test_stalled_call_becomes_an_error_and_the_run_finishes(workspace: Path) -> None:
    target = CaseTarget(default=FakeResponse(output="alpha"), block_on="c3")

    run_dir = finishes(
        lambda: execute(config(workspace, target, call_deadline_s=0.5, max_error_rate=0.2))
    )

    rows = [
        json.loads(line) for line in (run_dir / "a" / "predictions.jsonl").read_text().splitlines()
    ]
    stalled = [row for row in rows if row["error"]]
    assert [(row["case_id"], row["error"]) for row in stalled] == [("c3", "Stalled")]
    assert len(rows) == 8


@pytest.mark.ticket("EV-004-T21")
def test_interrupt_does_not_wait_for_a_stalled_call(workspace: Path) -> None:
    target = CaseTarget(default=FakeResponse(output="alpha"), block_on="c0", interrupt_on="c2")

    run_dir = finishes(lambda: execute(config(workspace, target, concurrency=2)))

    scores = json.loads((run_dir / "a" / "scores.json").read_text())
    assert scores["run"]["stop_reason"] == "interrupted"


class BlockingJudge(StubJudge):
    def __init__(self) -> None:
        super().__init__(scores=[0.5] * 10)
        self.release = threading.Event()

    def complete(self, prompt: str) -> str:
        if "second" in prompt:
            self.release.wait()
        return super().complete(prompt)


@pytest.mark.ticket("EV-008-T21")
def test_stalled_judge_call_is_counted_as_a_judge_failure() -> None:
    judge = BlockingJudge()
    rubric = str(ROOT / "rubrics" / "summary_quality.yaml")
    scorer = JudgeScorer(params={"rubric": rubric, "concurrency": 2, "call_deadline_s": 0.5})
    scorer.bind_judge(judge, "judge/model", [])
    cases = [TestCase(id="a", input="first source"), TestCase(id="b", input="second source")]

    try:
        result = finishes(lambda: scorer.score(["one", "two"], ["r1", "r2"], cases)[0])
    finally:
        judge.release.set()

    assert result.n == 1
    assert scorer.health().failed == 1
