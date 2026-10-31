"""Generation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import Field

from evalverdict._optional import optional_import
from evalverdict.engines import bertscore
from evalverdict.engines.bertscore import Backend
from evalverdict.scoring.base import Family, ItemValues, Scorer
from evalverdict.scoring.registry import register_scorer


@register_scorer("rouge")
class RougeScorer(Scorer):
    family = Family.GENERATION
    metrics = ("rouge1", "rouge2", "rougeL", "rougeLsum")

    class Params(Scorer.Params):
        use_stemmer: bool = False
        backend: Literal["rouge_score", "evaluate"] = "rouge_score"

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
