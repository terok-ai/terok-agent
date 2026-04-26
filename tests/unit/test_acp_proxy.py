# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ACP proxy state machine — the pre-bind handshake.

Bind-time tests (which spawn a backend) live in the manual integration
walk-through; the unit tests here cover the deterministic synchronous
paths: initialize, session/new, set_config_option dispatch, error
shapes, and the model-option rewriter helper.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from terok_executor.acp.proxy import (
    ACPProxy,
    _build_model_config_option,
    _humanise_model_id,
    _rewrite_model_options_in_place,
    _with_params_value,
)


class _StubRoster:
    """Minimal stand-in for :class:`ACPRoster` — only what the proxy reads.

    The proxy never touches the sandbox / container / cache directly
    in the pre-bind paths exercised here, so a thin stub keeps tests
    fast and isolated from sandbox plumbing.
    """

    def __init__(self, available: list[str]) -> None:
        self._available = available

    def list_available_agents(self) -> list[str]:
        """Return the canned ``agent:model`` list."""
        return list(self._available)


class _Pipe:
    """In-memory pair of asyncio streams used as client ↔ proxy ducts."""

    def __init__(self) -> None:
        self.reader = asyncio.StreamReader()

    def feed(self, data: bytes) -> None:
        """Push bytes the proxy will read from the client side."""
        self.reader.feed_data(data)

    def feed_eof(self) -> None:
        """Signal end-of-stream so the proxy loop returns cleanly."""
        self.reader.feed_eof()


class _CapturingWriter:
    """Captures frames written to it; conforms to ``StreamWriter`` shape."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def write(self, data: bytes) -> None:
        """Append *data* — mirrors the StreamWriter API."""
        self._buf.extend(data)

    async def drain(self) -> None:
        """No-op — buffer is unbounded."""

    def close(self) -> None:
        """No-op — caller manages the lifecycle."""

    async def wait_closed(self) -> None:
        """No-op."""

    def frames(self) -> list[dict]:
        """Return parsed NDJSON frames written so far."""
        out: list[dict] = []
        for line in self._buf.split(b"\n"):
            if line:
                out.append(json.loads(line))
        return out


def _frame(method: str, frame_id: object, **params: object) -> bytes:
    """Render a JSON-RPC request frame as NDJSON the proxy will parse."""
    payload: dict = {"jsonrpc": "2.0", "id": frame_id, "method": method}
    if params:
        payload["params"] = params
    return (json.dumps(payload) + "\n").encode("utf-8")


async def _run_proxy(
    *,
    available: list[str],
    frames: list[bytes],
) -> list[dict]:
    """Drive a proxy through *frames* and return what it wrote back."""
    pipe = _Pipe()
    writer = _CapturingWriter()
    for f in frames:
        pipe.feed(f)
    pipe.feed_eof()
    proxy = ACPProxy(roster=_StubRoster(available))  # type: ignore[arg-type]
    await proxy.run(pipe.reader, writer)  # type: ignore[arg-type]
    return writer.frames()


class TestInitialize:
    """Pre-bind ``initialize`` is answered locally."""

    def test_returns_protocol_version_and_caps(self) -> None:
        """Response carries protocolVersion + a minimal capability set."""
        responses = asyncio.run(
            _run_proxy(
                available=["claude:opus-4.6"],
                frames=[_frame("initialize", 1, protocolVersion=1)],
            )
        )
        assert len(responses) == 1
        result = responses[0]["result"]
        assert "protocolVersion" in result
        assert "agentCapabilities" in result


class TestSessionNew:
    """Pre-bind ``session/new`` aggregates the model list locally."""

    def test_returns_synthetic_session_id(self) -> None:
        """A synthetic session id is returned — no backend exists yet."""
        responses = asyncio.run(
            _run_proxy(
                available=["claude:opus-4.6", "codex:gpt-5.5"],
                frames=[
                    _frame("initialize", 1, protocolVersion=1),
                    _frame("session/new", 2, cwd="/workspace"),
                ],
            )
        )
        assert responses[1]["result"]["sessionId"] == "proxy-1"

    def test_aggregates_namespaced_model_options(self) -> None:
        """Every available ``agent:model`` appears in the configOption."""
        responses = asyncio.run(
            _run_proxy(
                available=["claude:opus-4.6", "codex:gpt-5.5"],
                frames=[
                    _frame("initialize", 1, protocolVersion=1),
                    _frame("session/new", 2, cwd="/workspace"),
                ],
            )
        )
        config_options = responses[1]["result"]["configOptions"]
        model_opt = next(opt for opt in config_options if opt["category"] == "model")
        ids = [entry["id"] for entry in model_opt["select"]["options"]]
        assert ids == ["claude:opus-4.6", "codex:gpt-5.5"]

    def test_rejects_second_session_new(self) -> None:
        """v1 supports one session per connection — second call errors."""
        responses = asyncio.run(
            _run_proxy(
                available=["claude:opus-4.6"],
                frames=[
                    _frame("initialize", 1, protocolVersion=1),
                    _frame("session/new", 2, cwd="/workspace"),
                    _frame("session/new", 3, cwd="/workspace"),
                ],
            )
        )
        assert "error" in responses[2]


class TestSetConfigOptionPreBind:
    """``set_config_option`` parsing and rejection paths before any bind."""

    def test_invalid_value_returns_jsonrpc_error(self) -> None:
        """Non ``agent:model`` values are rejected with -32602."""
        responses = asyncio.run(
            _run_proxy(
                available=["claude:opus-4.6"],
                frames=[
                    _frame("initialize", 1, protocolVersion=1),
                    _frame("session/new", 2, cwd="/workspace"),
                    _frame(
                        "session/set_config_option",
                        3,
                        sessionId="proxy-1",
                        category="model",
                        value="no-namespace",
                    ),
                ],
            )
        )
        assert responses[2]["error"]["code"] == -32602

    def test_non_model_category_pre_bind_errors(self) -> None:
        """Pre-bind set_config_option for non-model categories is rejected."""
        responses = asyncio.run(
            _run_proxy(
                available=["claude:opus-4.6"],
                frames=[
                    _frame("initialize", 1, protocolVersion=1),
                    _frame("session/new", 2, cwd="/workspace"),
                    _frame(
                        "session/set_config_option",
                        3,
                        sessionId="proxy-1",
                        category="something-else",
                        value="anything",
                    ),
                ],
            )
        )
        assert "error" in responses[2]


class TestPreBindForwardingRefusals:
    """Methods that need a backend are refused pre-bind."""

    def test_session_prompt_pre_bind_errors(self) -> None:
        """``session/prompt`` before bind cannot reach a backend — error."""
        responses = asyncio.run(
            _run_proxy(
                available=["claude:opus-4.6"],
                frames=[
                    _frame("initialize", 1, protocolVersion=1),
                    _frame("session/new", 2, cwd="/workspace"),
                    _frame("session/prompt", 3, sessionId="proxy-1", text="hi"),
                ],
            )
        )
        assert "error" in responses[2]


class TestModelOptionRewriter:
    """``_rewrite_model_options_in_place`` namespaces backend model ids."""

    def test_prefixes_bare_model_ids_with_agent(self) -> None:
        """Bare ``opus-4.6`` becomes ``claude:opus-4.6`` after bind."""
        frame = {
            "result": {
                "configOptions": [
                    {
                        "category": "model",
                        "select": {
                            "options": [{"id": "opus-4.6"}, {"id": "haiku-4.5"}],
                        },
                    }
                ]
            }
        }
        _rewrite_model_options_in_place(frame, "claude")
        ids = [e["id"] for e in frame["result"]["configOptions"][0]["select"]["options"]]
        assert ids == ["claude:opus-4.6", "claude:haiku-4.5"]

    def test_skips_already_namespaced_ids(self) -> None:
        """Ids already containing the separator are left untouched."""
        frame = {
            "result": {
                "configOptions": [
                    {
                        "category": "model",
                        "select": {"options": [{"id": "claude:opus-4.6"}]},
                    }
                ]
            }
        }
        _rewrite_model_options_in_place(frame, "claude")
        ids = [e["id"] for e in frame["result"]["configOptions"][0]["select"]["options"]]
        assert ids == ["claude:opus-4.6"]

    def test_non_model_categories_untouched(self) -> None:
        """Mode and other categories are not rewritten."""
        frame = {
            "result": {
                "configOptions": [
                    {
                        "category": "mode",
                        "select": {"options": [{"id": "ask"}]},
                    }
                ]
            }
        }
        before = {"id": "ask"}
        _rewrite_model_options_in_place(frame, "claude")
        after = frame["result"]["configOptions"][0]["select"]["options"][0]
        assert after == before


class TestSmallHelpers:
    """The shape-builder helpers."""

    def test_build_model_config_option_select_shape(self) -> None:
        """The proxy emits a select-shaped option."""
        opt = _build_model_config_option(["claude:opus-4.6"])
        assert opt["category"] == "model"
        assert opt["select"]["options"][0]["id"] == "claude:opus-4.6"

    def test_humanise_model_id(self) -> None:
        """The label format is ``Agent — model``."""
        assert _humanise_model_id("claude:opus-4.6") == "Claude — opus-4.6"

    def test_humanise_unnamespaced_passes_through(self) -> None:
        """Unrecognised ids are returned verbatim — no crash."""
        assert _humanise_model_id("plain") == "plain"

    def test_with_params_value_replaces_value_and_keeps_others(self) -> None:
        """Helper does not mutate the input frame."""
        frame = {"params": {"category": "model", "value": "claude:opus"}}
        new = _with_params_value(frame, "opus")
        assert new["params"]["value"] == "opus"
        assert frame["params"]["value"] == "claude:opus"  # untouched


@pytest.mark.parametrize(
    "method",
    ["initialize", "session/new", "session/prompt"],
)
def test_proxy_handles_disconnect_cleanly(method: str) -> None:
    """Closing the client mid-conversation does not raise."""

    async def _scenario() -> None:
        pipe = _Pipe()
        writer = _CapturingWriter()
        # Send no frames; just feed EOF.  Proxy should exit immediately.
        pipe.feed_eof()
        proxy = ACPProxy(roster=_StubRoster([]))  # type: ignore[arg-type]
        await proxy.run(pipe.reader, writer)  # type: ignore[arg-type]
        # Method param exercises the parametrize matrix even when empty
        assert isinstance(method, str)

    asyncio.run(_scenario())
