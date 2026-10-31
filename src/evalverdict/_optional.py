"""Lazy imports for dependencies that live in optional extras."""

from __future__ import annotations

import importlib
from types import ModuleType

_EXTRA_BY_MODULE = {
    "bert_score": "nlp",
    "deepeval": "judge",
    "evaluate": "nlp",
    "evidently": "drift",
    "fairlearn": "fairness",
    "inspect_ai": "redteam",
    "inspect_evals": "redteam",
    "langfuse": "tracing",
    "lime": "explain",
    "rouge_score": "nlp",
    "shap": "explain",
}


class MissingExtraError(ImportError):
    """An optional dependency is not installed."""


def optional_import(module: str) -> ModuleType:
    """Import an optional module, or explain which extra provides it.

    Only a missing top-level package becomes `MissingExtraError`. Any other import
    failure, such as a broken dependency inside the package, is raised unchanged.
    """
    top_level = module.split(".")[0]
    extra = _EXTRA_BY_MODULE[top_level]
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] != top_level:
            raise
        raise MissingExtraError(
            f'{top_level} is not installed. Run: pip install "evalverdict[{extra}]"'
        ) from error
