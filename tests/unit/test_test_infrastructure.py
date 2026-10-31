"""EV-032: test isolation guards, fakes and fixture datasets."""

from __future__ import annotations

import http.server
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from fakes.datasets import FIXTURE_NAMES, fixture_sha256, load_fixture
from fakes.judge import StubJudge
from fakes.target import FakeResponse, FakeTarget

pytest_plugins = ("pytester",)

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = str(ROOT / "pyproject.toml")
GUARD_CONFTEST = f"""
import sys

sys.path.insert(0, {str(ROOT / "tests")!r})
from fakes.guards import *
"""


def run_nested(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    """Run pytest in a subprocess using the project's real configuration."""
    pytester.makeconftest(GUARD_CONFTEST)
    return pytester.runpytest_subprocess(
        "-c", PYPROJECT, "--rootdir", str(pytester.path), "-p", "no:cacheprovider", *args
    )


@pytest.mark.ticket("EV-032-T01")
def test_default_test_cannot_reach_a_public_host(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import socket

        def test_connect():
            socket.create_connection(("example.com", 80), timeout=1)
        """
    )

    result = run_nested(pytester)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*blocked in default tests: example.com*"])


@pytest.mark.ticket("EV-032-T01")
def test_connecting_by_ip_address_is_blocked_and_names_the_address(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        """
        import socket

        def test_connect():
            sock = socket.socket()
            sock.connect(("93.184.216.34", 80))
        """
    )

    result = run_nested(pytester)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*blocked in default tests: 93.184.216.34*"])


def test_network_attempt_fails_the_test_even_if_the_code_swallows_the_error(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        """
        import socket

        def test_telemetry_style_swallow():
            try:
                socket.create_connection(("telemetry.example.com", 443), timeout=1)
            except Exception:
                pass
        """
    )

    result = run_nested(pytester)

    assert result.ret != 0
    result.stdout.fnmatch_lines(["*telemetry.example.com*"])


def test_localhost_is_blocked_without_the_fixture(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import socket

        def test_connect():
            socket.create_connection(("127.0.0.1", 9), timeout=1)
        """
    )

    result = run_nested(pytester)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*blocked in default tests: 127.0.0.1*"])


@pytest.mark.ticket("EV-032-T02")
def test_localhost_fixture_allows_a_local_server(allow_localhost: None) -> None:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5) as reply:
            assert reply.read() == b"ok"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.ticket("EV-032-T03")
def test_live_tests_are_deselected_by_default(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.live
        def test_paid():
            raise AssertionError("must not run")

        def test_free():
            pass
        """
    )

    result = run_nested(pytester)

    result.assert_outcomes(passed=1, deselected=1)


@pytest.mark.ticket("EV-032-T04")
def test_live_test_without_keys_is_skipped_with_a_reason(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.live("ANTHROPIC_API_KEY")
        def test_paid():
            raise AssertionError("must not run")
        """
    )

    result = run_nested(pytester, "-m", "live", "-rs")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*ANTHROPIC_API_KEY*"])


def test_live_test_with_keys_runs_with_network_and_keys_available(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-value")
    pytester.makepyfile(
        """
        import os
        import socket

        import pytest

        @pytest.mark.live("ANTHROPIC_API_KEY")
        def test_paid():
            assert os.environ["ANTHROPIC_API_KEY"] == "test-value"
            assert "guarded_connect" not in socket.socket.connect.__qualname__
        """
    )

    result = run_nested(pytester, "-m", "live")

    result.assert_outcomes(passed=1)


@pytest.mark.ticket("EV-032-T05")
def test_fake_target_returns_scripted_responses_in_order() -> None:
    target = FakeTarget(
        [
            FakeResponse(output="good", latency_s=0.4, ttft_s=0.1, output_tokens=7, cost_usd=0.01),
            FakeResponse(error="rate limited"),
            FakeResponse(output="{not valid json"),
        ]
    )

    first, second, third = (target.predict(item) for item in ("a", "b", "c"))

    assert (first.case_id, first.output, first.latency_s, first.ttft_s) == ("a", "good", 0.4, 0.1)
    assert (first.output_tokens, first.cost_usd) == (7, 0.01)
    assert (second.output, second.error) == (None, "rate limited")
    assert third.output == "{not valid json"
    assert target.calls == ["a", "b", "c"]


def test_fake_target_fails_loudly_when_the_script_runs_out() -> None:
    target = FakeTarget([FakeResponse(output="only one")])
    target.predict("a")

    with pytest.raises(AssertionError, match="script"):
        target.predict("b")


@pytest.mark.ticket("EV-032-T06")
def test_seeded_fake_target_is_repeatable_and_seed_sensitive() -> None:
    def run(seed: int) -> list[object]:
        target = FakeTarget.from_seed(50, seed=seed, error_rate=0.2, malformed_rate=0.2)
        return [target.predict(str(index)) for index in range(50)]

    assert run(7) == run(7)
    assert run(7) != run(8)


@pytest.mark.ticket("EV-032-T07")
def test_stub_judge_returns_scripted_scores_and_records_prompts() -> None:
    judge = StubJudge(scores=[0.7, 0.2], reasons=["clear", "off topic"])

    first = judge.score("prompt one")
    second = judge.score("prompt two")

    assert (first.score, first.reason) == (0.7, "clear")
    assert (second.score, second.reason) == (0.2, "off topic")
    assert judge.prompts == ["prompt one", "prompt two"]


@pytest.mark.ticket("EV-032-T08")
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_dataset_parses_with_its_known_hash(name: str) -> None:
    rows = load_fixture(name)
    manifest = json.loads((ROOT / "tests" / "fixtures" / "hashes.json").read_text())

    assert rows
    assert len({row["id"] for row in rows}) == len(rows)
    assert fixture_sha256(name) == manifest[name]


def test_fixture_list_covers_every_family_the_ticket_names() -> None:
    assert set(FIXTURE_NAMES) == {
        "agentic",
        "classification",
        "extraction",
        "generation",
        "ranking",
        "subgroups",
    }


@pytest.mark.ticket("EV-032-T09")
def test_api_keys_are_cleared_during_default_tests(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-be-cleared")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "must-be-cleared")
    monkeypatch.setenv("SOME_NEW_PROVIDER_API_KEY", "must-be-cleared")
    pytester.makepyfile(
        """
        import os

        def test_keys_absent():
            assert "ANTHROPIC_API_KEY" not in os.environ
            assert "LANGFUSE_SECRET_KEY" not in os.environ
            assert "SOME_NEW_PROVIDER_API_KEY" not in os.environ
        """
    )

    result = run_nested(pytester)

    result.assert_outcomes(passed=1)


@pytest.mark.ticket("EV-032-T10")
@pytest.mark.parametrize("bad_id", ["EV-1-T1", "EV-001", "ticket one"])
def test_badly_formatted_ticket_marker_fails_collection(
    pytester: pytest.Pytester, bad_id: str
) -> None:
    pytester.makepyfile(
        f"""
        import pytest

        @pytest.mark.ticket({bad_id!r})
        def test_marked():
            pass
        """
    )

    result = run_nested(pytester)

    assert result.ret != 0
    result.stderr.fnmatch_lines([f"*{bad_id}*"])


def test_deepeval_pytest_plugin_is_disabled_in_the_project_config(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile("def test_nothing(): pass")

    result = run_nested(pytester)

    result.assert_outcomes(passed=1)
    assert not (pytester.path / ".deepeval").exists()
    assert "sessionfinish" not in result.stdout.str()
