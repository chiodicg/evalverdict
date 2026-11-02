"""A scripted LLM judge that records every prompt it receives."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeDecision:
    score: float
    reason: str


class StubJudge:
    def __init__(self, scores: Sequence[float], reasons: Sequence[str] | None = None) -> None:
        self._scores = list(scores)
        self._reasons = list(reasons) if reasons is not None else ["scripted"] * len(self._scores)
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        """Reply as a judge model would, with a 0-10 score in JSON."""
        with self._lock:
            decision = self.score(prompt)
        return json.dumps({"reason": decision.reason, "score": decision.score * 10})

    def score(self, prompt: str) -> JudgeDecision:
        position = len(self.prompts)
        if position >= len(self._scores):
            raise AssertionError(f"stub judge script ran out at call {position + 1}")
        self.prompts.append(prompt)
        return JudgeDecision(score=self._scores[position], reason=self._reasons[position])
