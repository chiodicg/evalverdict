"""DeepEval G-Eval, driven by our own client so cache and budget always apply."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from evalverdict._optional import optional_import
from evalverdict.engines.judge_client import JudgeBackend
from evalverdict.rubric import Rubric

PARSE_RETRIES = 2
_FIELD_NAMES = {
    "input": "INPUT",
    "actual_output": "ACTUAL_OUTPUT",
    "expected_output": "EXPECTED_OUTPUT",
    "retrieval_context": "RETRIEVAL_CONTEXT",
}


class JudgeParseError(RuntimeError):
    """The judge's reply could not be read, even after retrying."""


def _deepeval(module: str) -> Any:
    """Import DeepEval with its telemetry and update checks switched off."""
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
    os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "NO")
    return optional_import(module)


def _json_text(reply: str) -> str:
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    return match.group(0) if match else reply


@lru_cache(maxsize=1)
def _judge_model_class() -> type:
    base: Any = _deepeval("deepeval.models").DeepEvalBaseLLM

    class JudgeModel(base):  # type: ignore[misc]
        def __init__(self, backend: JudgeBackend) -> None:
            self.backend = backend
            super().__init__(model="evalverdict-judge")

        def load_model(self) -> JudgeBackend:
            return self.backend

        def get_model_name(self) -> str:
            return "evalverdict-judge"

        def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            if schema is None:
                return self.backend.complete(prompt)
            for _ in range(PARSE_RETRIES + 1):
                reply = self.backend.complete(prompt)
                try:
                    return schema(**json.loads(_json_text(reply)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    # A cached bad reply would repeat, so vary the prompt on retry.
                    prompt = f"{prompt}\n\nReturn only valid JSON."
            raise JudgeParseError("the judge's reply was not valid JSON")

        async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            return self.generate(prompt, schema, **kwargs)

    return JudgeModel


def judge_item(
    backend: JudgeBackend,
    rubric: Rubric,
    *,
    case_input: Any,
    output: str,
    reference: Any,
    threshold: float,
) -> tuple[float, str]:
    """Score one item against a rubric. Returns the 0-1 score and the judge's reason.

    DeepEval's own strict_mode returns the raw 0-10 score, so it is not used here.
    """
    test_case_module = _deepeval("deepeval.test_case")
    metrics_module = _deepeval("deepeval.metrics")
    params = test_case_module.SingleTurnParams
    metric = metrics_module.GEval(
        name=rubric.name,
        evaluation_steps=list(rubric.steps),
        evaluation_params=[getattr(params, _FIELD_NAMES[name]) for name in rubric.fields],
        model=_judge_model_class()(backend),
        threshold=threshold,
        async_mode=False,
        verbose_mode=False,
    )
    case = test_case_module.LLMTestCase(
        input=str(case_input),
        actual_output=output,
        expected_output=None if reference is None else str(reference),
    )
    metric.measure(case, _show_indicator=False)
    return float(metric.score), str(metric.reason)
