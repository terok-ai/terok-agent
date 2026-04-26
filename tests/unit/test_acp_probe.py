# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ACP model-roster probe.

The probe drives the in-container ACP wrapper through a minimal handshake
to extract the ``configOptions[category=model]`` entry from a
``session/new`` response.  Tests use the :class:`NullRuntime`'s
``set_exec_stdio_script`` to replay deterministic byte exchanges — no
container required.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from terok_sandbox import NullRuntime, Sandbox, SandboxConfig

from terok_executor.acp.probe import ProbeError, _extract_model_ids, probe_agent_models


def _frame(method: str, **fields: object) -> bytes:
    """Render an NDJSON frame the probe expects from the backend."""
    payload: dict = {"jsonrpc": "2.0", **fields}
    if method:
        payload["method"] = method
    return (json.dumps(payload) + "\n").encode("utf-8")


def _read_request_id(frame: bytes) -> str:
    """Pluck the ``id`` from a JSON-RPC request frame for response shaping."""
    return json.loads(frame)["id"]


class TestExtractModelIds:
    """The configOptions parser handles the schema variants we've seen."""

    def test_select_options_with_id_keys(self) -> None:
        """Newer shape: ``configOptions[…].select.options[].id``."""
        result = {
            "configOptions": [
                {
                    "category": "model",
                    "select": {
                        "options": [
                            {"id": "opus-4.6", "name": "Opus"},
                            {"id": "haiku-4.5", "name": "Haiku"},
                        ]
                    },
                }
            ]
        }
        assert _extract_model_ids(result) == ("opus-4.6", "haiku-4.5")

    def test_legacy_available_models_array(self) -> None:
        """Legacy shape: top-level ``availableModels`` of strings."""
        result = {"availableModels": ["opus-4.6", "haiku-4.5"]}
        assert _extract_model_ids(result) == ("opus-4.6", "haiku-4.5")

    def test_legacy_available_models_dicts(self) -> None:
        """Legacy shape: top-level ``availableModels`` of dicts."""
        result = {"availableModels": [{"id": "opus-4.6"}]}
        assert _extract_model_ids(result) == ("opus-4.6",)

    def test_string_choices(self) -> None:
        """Choices may be plain strings."""
        result = {
            "configOptions": [
                {"category": "model", "options": ["m1", "m2"]},
            ]
        }
        assert _extract_model_ids(result) == ("m1", "m2")

    def test_unknown_shape_returns_empty(self) -> None:
        """Unrecognised shapes yield an empty tuple — caller caches that."""
        assert _extract_model_ids({"foo": "bar"}) == ()

    def test_non_model_category_is_ignored(self) -> None:
        """Other configOption categories are skipped."""
        result = {
            "configOptions": [
                {"category": "mode", "select": {"options": [{"id": "ask"}]}},
            ]
        }
        assert _extract_model_ids(result) == ()


class TestProbeAgentModels:
    """End-to-end probe with a script-driven NullRuntime backend."""

    def test_happy_path_returns_models(self) -> None:
        """A normal handshake yields the model tuple from configOptions."""
        rt = NullRuntime()
        # The probe's request-id strings are stable: px-init then
        # px-session-new.  We script responses matching those ids.
        # Easier: register a callback-less script that echoes prepared responses
        # by writing the responses on each probe write.  But NullRuntime's
        # script model is read/write steps in order; we have to consume the
        # request frame *some* way.  Approach: we know the probe writes one
        # initialize then one session/new.  We use a script that reads the
        # whole initialize frame (any bytes), writes the response, then the
        # whole session/new frame, writes the response.

        init_response = _frame(
            "",
            id="px-init",
            result={"protocolVersion": 1, "agentCapabilities": {}, "authMethods": []},
        )
        new_response = _frame(
            "",
            id="px-session-new",
            result={
                "sessionId": "be-1",
                "configOptions": [
                    {
                        "category": "model",
                        "select": {"options": [{"id": "opus-4.6"}, {"id": "haiku-4.5"}]},
                    }
                ],
                "availableModes": [],
            },
        )

        # The probe writes one frame, waits for the response, then writes
        # another, and waits again.  Use a script that drains stdin
        # entirely between writes.  We approximate by reading large
        # opaque chunks and writing the response.

        rt.set_exec_stdio_script(
            "task-c",
            ("terok-claude-acp",),
            (
                # Match exactly what the probe writes for initialize.
                (
                    "read",
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": "px-init",
                            "method": "initialize",
                            "params": {"protocolVersion": 1, "clientCapabilities": {}},
                        }
                    ).encode()
                    + b"\n",
                ),
                ("write", init_response),
                (
                    "read",
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": "px-session-new",
                            "method": "session/new",
                            "params": {"cwd": "/workspace", "mcpServers": []},
                        }
                    ).encode()
                    + b"\n",
                ),
                ("write", new_response),
            ),
        )
        sandbox = Sandbox(config=SandboxConfig(), runtime=rt)
        models = asyncio.run(
            probe_agent_models(
                agent_id="claude",
                container=rt.container("task-c"),
                sandbox=sandbox,
                timeout=4.0,
            )
        )
        assert models == ("opus-4.6", "haiku-4.5")

    def test_initialize_error_is_probe_error(self) -> None:
        """If the backend rejects ``initialize``, we raise ``ProbeError``."""
        rt = NullRuntime()
        rt.set_exec_stdio_script(
            "task-c",
            ("terok-codex-acp",),
            (
                (
                    "read",
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": "px-init",
                            "method": "initialize",
                            "params": {"protocolVersion": 1, "clientCapabilities": {}},
                        }
                    ).encode()
                    + b"\n",
                ),
                ("write", _frame("", id="px-init", error={"code": -1, "message": "nope"})),
            ),
        )
        sandbox = Sandbox(config=SandboxConfig(), runtime=rt)
        with pytest.raises(ProbeError):
            asyncio.run(
                probe_agent_models(
                    agent_id="codex",
                    container=rt.container("task-c"),
                    sandbox=sandbox,
                    timeout=2.0,
                )
            )

    def test_timeout_is_probe_error(self) -> None:
        """A backend that never responds raises ``ProbeError``."""
        rt = NullRuntime()
        # Register a script that reads forever (never writes a response).
        rt.set_exec_stdio_script(
            "task-c",
            ("terok-codex-acp",),
            (
                # Try to read 1 MiB — the probe's first frame is much
                # smaller, so this read effectively hangs waiting for
                # more bytes that never come.
                ("read", b"X" * (1024 * 1024)),
            ),
        )
        sandbox = Sandbox(config=SandboxConfig(), runtime=rt)
        with pytest.raises(ProbeError):
            asyncio.run(
                probe_agent_models(
                    agent_id="codex",
                    container=rt.container("task-c"),
                    sandbox=sandbox,
                    timeout=0.5,
                )
            )
