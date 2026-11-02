"""Generation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from evalverdict._optional import optional_import
from evalverdict.concurrency import StalledError, run_all
from evalverdict.data import TestCase
from evalverdict.engines import bertscore
from evalverdict.engines.bertscore import Backend
from evalverdict.engines.deepeval_judge import JudgeParseError, judge_item
from evalverdict.engines.judge_client import (
    JudgeBackend,
    JudgeBudgetStop,
    JudgeCallError,
)
from evalverdict.rubric import load_rubric
from evalverdict.scoring.base import ContractExample, Family, ItemValues, Scorer, ScorerHealth
from evalverdict.scoring.registry import register_scorer


@register_scorer("rouge")
class RougeScorer(Scorer):
    family = Family.GENERATION
    metrics = ("rouge1", "rouge2", "rougeL", "rougeLsum")

    class Params(Scorer.Params):
        use_stemmer: bool = False
        backend: Literal["rouge_score", "evaluate"] = "rouge_score"

    @classmethod
    def contract_example(cls) -> ContractExample:
        return ContractExample(
            params={},
            predictions=["the cat sat", "dogs bark", "rain again"],
            references=["a cat sat", "the dog barks", "sunny today"],
        )

    def check_dependencies(self) -> None:
        optional_import(self.params.backend)

    def item_values(
        self, predictions: Sequence[Any], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        texts = [str(prediction) for prediction in predictions]
        targets = [str(reference) for reference in references]
        if self.params.backend == "evaluate":
            scores = (
                optional_import("evaluate")
                .load("rouge")
                .compute(
                    predictions=texts,
                    references=targets,
                    use_aggregator=False,
                    use_stemmer=self.params.use_stemmer,
                )
            )
            return ItemValues(arrays={key: np.asarray(scores[key]) for key in self.metrics})

        module = optional_import("rouge_score.rouge_scorer")
        scorer = module.RougeScorer(list(self.metrics), use_stemmer=self.params.use_stemmer)
        rows = [scorer.score(target, text) for text, target in zip(texts, targets, strict=True)]
        return ItemValues(
            arrays={key: np.array([row[key].fmeasure for row in rows]) for key in self.metrics}
        )


@register_scorer("bertscore")
class BertScoreScorer(Scorer):
    family = Family.GENERATION
    metrics = ("precision", "recall", "f1")

    class Params(Scorer.Params):
        model_type: str = "roberta-large"
        batch_size: int = Field(default=32, gt=0)
        backend: Backend = "bert_score"

    @classmethod
    def contract_example(cls) -> ContractExample:
        """Empty text scores 0 without loading a model, which keeps this test fast."""
        return ContractExample(params={}, predictions=["", ""], references=["a", "b"])

    def check_dependencies(self) -> None:
        optional_import(self.params.backend)

    def item_values(
        self, predictions: Sequence[Any], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        precision, recall, f1 = bertscore.compute_bertscore(
            [str(prediction) for prediction in predictions],
            [str(reference) for reference in references],
            model_type=self.params.model_type,
            batch_size=self.params.batch_size,
            backend=self.params.backend,
        )
        return ItemValues(arrays={"precision": precision, "recall": recall, "f1": f1})


def vendor_of(model: str) -> str:
    parts = model.split("/")
    return parts[1] if parts[0] == "openrouter" and len(parts) > 2 else parts[0]


def base_model(model: str) -> str:
    return model.rsplit("/", 1)[-1]


@register_scorer("judge")
class JudgeScorer(Scorer):
    """LLM-as-judge against a rubric. One judge call per item."""

    family = Family.GENERATION
    metrics = ("score",)
    wants_cases = True
    needs_judge = True
    judge_calls_per_item = 1

    class Params(Scorer.Params):
        rubric: Path
        threshold: float = Field(default=0.5, ge=0, le=1)
        strict_mode: bool = False
        judge_model: str | None = None
        concurrency: int = Field(default=4, gt=0)
        call_deadline_s: float = Field(default=600, gt=0)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rubric = load_rubric(self.params.rubric)
        self.backend: JudgeBackend | None = None
        self.judge_model = ""
        self.target_models: list[str] = []
        self._health = ScorerHealth()

    @classmethod
    def contract_example(cls) -> ContractExample:
        rubric = Path(__file__).resolve().parents[3] / "rubrics" / "summary_quality.yaml"
        cases = [TestCase(id=f"c{index}", input=f"source {index}") for index in range(3)]
        return ContractExample(
            params={"rubric": str(rubric), "concurrency": 1},
            predictions=["summary a", "summary b", "summary c"],
            references=["ref a", "ref b", "ref c"],
            metadata=cases,
        )

    def check_dependencies(self) -> None:
        optional_import("deepeval")

    def bind_judge(
        self, backend: JudgeBackend, judge_model: str, target_models: Sequence[str]
    ) -> None:
        self.backend = backend
        self.judge_model = judge_model
        self.target_models = list(target_models)

    def health(self) -> ScorerHealth:
        return self._health

    def item_values(
        self, predictions: Sequence[Any], references: Sequence[Any], metadata: Any
    ) -> ItemValues:
        if self.backend is None:
            raise ValueError("the judge scorer needs a judge")
        backend = self.backend

        def judge(index: int) -> tuple[float, str]:
            score, reason = judge_item(
                backend,
                self.rubric,
                case_input=metadata[index].input,
                output=str(predictions[index]),
                reference=references[index],
                threshold=self.params.threshold,
            )
            if self.params.strict_mode:
                score = float(score >= self.params.threshold)
            return score, reason

        scores = np.full(len(predictions), np.nan)
        details: list[dict[str, Any]] = [{} for _ in predictions]
        failed = 0
        stop_reason = None
        outcomes = run_all(
            judge, range(len(predictions)), self.params.concurrency, self.params.call_deadline_s
        )
        for index, (outcome, error) in enumerate(outcomes):
            if error is None and outcome is not None:
                scores[index], reason = outcome
                details[index] = {"reason": reason}
            elif isinstance(error, JudgeBudgetStop):
                stop_reason = "budget"
            elif isinstance(error, (JudgeCallError, JudgeParseError, StalledError)):
                failed += 1
                details[index] = {"judge_failure": type(error).__name__}
            elif error is not None:
                raise error
        self._health = ScorerHealth(failed, len(predictions), stop_reason)

        judge_vendor = vendor_of(self.judge_model)
        return ItemValues(
            arrays={"score": scores},
            details=details,
            sparse=frozenset(self.metrics),
            result_details={
                "rubric_name": self.rubric.name,
                "rubric_version": self.rubric.version,
                "rubric_text": self.rubric.text,
                "rubric_sha256": self.rubric.sha256,
                "judge_model": self.judge_model,
                "self_judging": any(
                    base_model(model) == base_model(self.judge_model)
                    for model in self.target_models
                ),
                "shared_vendor_with": [
                    model for model in self.target_models if vendor_of(model) == judge_vendor
                ],
                "judge_failures": failed,
            },
        )
