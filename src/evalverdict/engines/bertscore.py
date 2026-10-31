"""BERTScore, kept in one module so its backend can change."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from evalverdict._optional import optional_import

Backend = Literal["bert_score", "evaluate"]
FloatArray = NDArray[np.float64]


def _run_backend(
    predictions: list[str],
    references: list[str],
    model_type: str,
    batch_size: int,
    backend: Backend,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if backend == "evaluate":
        metric = optional_import("evaluate").load("bertscore")
        result = metric.compute(
            predictions=predictions,
            references=references,
            model_type=model_type,
            batch_size=batch_size,
        )
        precision = np.asarray(result["precision"], dtype=float)
        recall = np.asarray(result["recall"], dtype=float)
        f1 = np.asarray(result["f1"], dtype=float)
        return precision, recall, f1
    score = optional_import("bert_score").score
    precision, recall, f1 = score(
        predictions, references, model_type=model_type, batch_size=batch_size
    )
    return precision.numpy().astype(float), recall.numpy().astype(float), f1.numpy().astype(float)


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str,
    batch_size: int,
    backend: Backend,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return per-item precision, recall and F1.

    Empty text scores 0 and never reaches the model: an empty answer has no content, and
    bert_score 0.3.13 crashes on it under transformers 5.
    """
    total = len(predictions)
    filled = [
        index for index in range(total) if predictions[index].strip() and references[index].strip()
    ]
    scores = np.zeros((3, total))
    if filled:
        computed = _run_backend(
            [predictions[index] for index in filled],
            [references[index] for index in filled],
            model_type,
            batch_size,
            backend,
        )
        for row, values in enumerate(computed):
            scores[row, filled] = values
    return scores[0], scores[1], scores[2]
