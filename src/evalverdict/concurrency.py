"""Concurrency that a stuck call cannot freeze.

Each task runs on its own daemon thread. A task past its deadline is reported as stalled
and abandoned, so neither the run nor the interpreter's exit waits for it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from typing import Any

POLL_S = 0.2


class StalledError(RuntimeError):
    """A call ran past its deadline and was abandoned."""


def spawn[R](function: Callable[..., R], *args: Any) -> Future[R]:
    future: Future[R] = Future()

    def run() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)

    threading.Thread(target=run, daemon=True).start()
    return future


def run_all[T, R](
    function: Callable[[T], R],
    items: Sequence[T],
    concurrency: int,
    deadline_s: float,
) -> list[tuple[R | None, BaseException | None]]:
    """Run `function` on every item, at most `concurrency` at once.

    Returns (result, None) or (None, error) per item, in item order. A call past its
    deadline gets a StalledError.
    """
    outcomes: list[tuple[R | None, BaseException | None]] = [(None, None)] * len(items)
    queue = list(range(len(items)))
    pending: dict[Future[R], tuple[int, float]] = {}
    while queue or pending:
        while queue and len(pending) < concurrency:
            index = queue.pop(0)
            pending[spawn(function, items[index])] = (index, time.monotonic())
        done, _ = wait(pending, timeout=POLL_S, return_when=FIRST_COMPLETED)
        for future in done:
            index, _ = pending.pop(future)
            error = future.exception()
            outcomes[index] = (None, error) if error else (future.result(), None)
        now = time.monotonic()
        for future, (index, started) in list(pending.items()):
            if now - started > deadline_s:
                pending.pop(future)
                outcomes[index] = (None, StalledError(f"no reply within {deadline_s}s"))
    return outcomes
