"""A content-addressed cache of model responses, so reruns cost nothing."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evalverdict.data import Prediction

logger = logging.getLogger(__name__)


def cache_key(target_config: dict[str, Any], case_input: Any, repeat: int) -> str:
    """Hash everything that can change a response. Each repeat gets its own entry."""
    payload = {"target": target_config, "input": case_input, "repeat": repeat}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ResponseCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Prediction | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return Prediction(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):
            logger.warning("ignoring corrupt cache entry %s", path)
            return None

    def put(self, key: str, prediction: Prediction) -> None:
        """Store a successful response. Errors stay uncached so they get retried."""
        if prediction.error is not None:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(prediction), sort_keys=True))
