"""Pytest guards: default tests get no network, no API keys and no spend."""

from __future__ import annotations

import os
import re
import socket
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import pytest

from evalverdict.env import load_env

__all__ = [
    "allow_localhost",
    "network_guard",
    "pytest_collection_modifyitems",
    "pytest_runtest_makereport",
    "pytest_runtest_setup",
]

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
TICKET_ID = re.compile(r"EV-\d{3}-T\d{2}")
GUARD_KEY = pytest.StashKey["NetworkGuard"]()


class NetworkBlockedError(RuntimeError):
    """A default test tried to use the network."""


@dataclass
class NetworkGuard:
    allow_localhost: bool = False
    attempts: list[str] = field(default_factory=list)

    def check(self, host: object) -> None:
        name = host.decode() if isinstance(host, bytes) else str(host)
        if self.allow_localhost and name in LOCAL_HOSTS:
            return
        self.attempts.append(name)
        raise NetworkBlockedError(f"network access is blocked in default tests: {name}")


def _is_secret_variable(name: str) -> bool:
    return name.endswith("_API_KEY") or name.startswith("LANGFUSE_")


@pytest.fixture(autouse=True)
def network_guard(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Generator[NetworkGuard]:
    guard = NetworkGuard()
    if request.node.get_closest_marker("live") is not None:
        yield guard
        return

    for name in [variable for variable in os.environ if _is_secret_variable(variable)]:
        monkeypatch.delenv(name)

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock: socket.socket, address: Any) -> None:
        if sock.family != socket.AF_UNIX:
            guard.check(address[0])
        real_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any) -> int:
        if sock.family != socket.AF_UNIX:
            guard.check(address[0])
        return real_connect_ex(sock, address)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None:
            guard.check(host)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    request.node.stash[GUARD_KEY] = guard
    yield guard


@pytest.fixture
def allow_localhost(network_guard: NetworkGuard) -> None:
    """Let this test reach servers on this machine, and nowhere else."""
    network_guard.allow_localhost = True


def pytest_runtest_setup(item: pytest.Item) -> None:
    marker = item.get_closest_marker("live")
    if marker is None:
        return
    load_env()
    missing = [name for name in marker.args if not os.environ.get(name)]
    if missing:
        pytest.skip(f"live test needs {', '.join(missing)}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item) -> Generator[None, Any, Any]:
    report = yield
    guard = item.stash.get(GUARD_KEY, None)
    if report.when == "call" and report.passed and guard is not None and guard.attempts:
        report.outcome = "failed"
        report.longrepr = f"test attempted network access: {', '.join(guard.attempts)}"
    return report


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        for marker in item.iter_markers("ticket"):
            ticket_id = marker.args[0] if len(marker.args) == 1 else marker.args
            if not TICKET_ID.fullmatch(str(ticket_id)):
                raise pytest.UsageError(f"bad ticket ID {ticket_id!r} on {item.nodeid}")
