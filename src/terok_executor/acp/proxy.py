# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""ACP proxy state machine — the JSON-RPC mediator behind :class:`ACPRoster`.

The proxy speaks ACP to the connected client and ACP to one chosen
in-container backend, namespacing the model selector so a multi-agent
container looks like a single endpoint.  It parses every JSON-RPC frame
(NDJSON over stdio) — there is no byte-level passthrough; uninteresting
frames are re-serialised after parsing.

Two phases drive the lifecycle:

- **Pre-bind**: the proxy answers ``initialize`` and ``session/new``
  locally, advertising the aggregated ``agent:model`` list as a
  ``category: "model"`` configOption.  No backend process exists yet.
- **Bound**: on the first ``session/set_config_option`` for the model
  category, the proxy spawns the agent's wrapper script via
  :meth:`Sandbox.runtime.exec_stdio`, replays
  ``initialize`` + ``session/new`` + ``set_config_option`` to it, and
  from then on bridges frames in both directions.  The option list is
  rewritten on the way out so cross-agent values disappear from the
  client's view.

V1 takes shortcuts where the design is still settling: one client per
connection, one session per binding, no push notifications when the
authed-agent set changes mid-connection.  All of these are tracked
for v2.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from .model_options import (
    MODEL_NAMESPACE_SEP,
    MODEL_OPTION_CATEGORY,
    iter_model_choice_dicts,
)

if TYPE_CHECKING:
    from .roster import ACPRoster

_logger = logging.getLogger(__name__)

ACP_PROTOCOL_VERSION = 1
"""Version this proxy advertises on ``initialize``.  Mirrors what the
probe sends; in practice both backends and clients negotiate down."""

PROXY_REQUEST_ID_PREFIX = "px-"
"""Prefix for JSON-RPC request ids the proxy injects (replay of
``initialize``/``session/new`` to the backend).  Strings can't collide
with the int ids ACP clients typically use."""

JSONRPC_INVALID_REQUEST = -32600
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603


class AgentBindError(RuntimeError):
    """Surface error raised when the proxy fails to bind a backend agent.

    Always converted to a JSON-RPC error response on the wire — never
    bubbles to the caller of :meth:`ACPProxy.run`.
    """


class ACPProxy:
    """One client connection's worth of proxy state.

    Constructed by :meth:`ACPRoster.attach`; lives for the duration of
    a single client connection.  Not reusable — discard after :meth:`run`
    returns.
    """

    def __init__(self, *, roster: ACPRoster) -> None:
        self._roster = roster
        self._client_writer: asyncio.StreamWriter | None = None
        self._bound_agent: str | None = None
        self._client_session_id: str | None = None
        self._backend_session_id: str | None = None
        self._client_session_new_params: dict[str, Any] = {}
        self._backend_writer: asyncio.StreamWriter | None = None
        self._backend_reader: asyncio.StreamReader | None = None
        self._backend_pump_task: asyncio.Task | None = None
        self._backend_exec_future: asyncio.Future | None = None
        self._proxy_request_counter = 0
        self._pending_proxy_responses: dict[str, asyncio.Future] = {}
        self._closed = False

    async def run(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Run the proxy loop until the client disconnects.

        Consumes one NDJSON frame at a time, dispatches by ``method`` /
        ``id`` shape, and writes responses back.  Always cleans up
        backend resources on exit, even on cancellation or unexpected
        errors.
        """
        self._client_writer = writer
        try:
            while not self._closed:
                line = await reader.readline()
                if not line:
                    return
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    _logger.warning("ACP proxy: client sent non-JSON frame, ignoring")
                    continue
                if not isinstance(frame, dict):
                    _logger.warning("ACP proxy: client sent non-object JSON-RPC frame, ignoring")
                    continue
                await self._handle_client_frame(frame)
        finally:
            await self._teardown_backend()

    # ── Frame dispatch ────────────────────────────────────────────────

    async def _handle_client_frame(self, frame: dict[str, Any]) -> None:
        """Route one client → proxy frame to the right handler."""
        method = frame.get("method")
        if method is None:
            # Response or notification with no method — no proxy concern,
            # forward to backend if bound, drop otherwise.
            if self._is_bound:
                await self._forward_to_backend(frame)
            return

        if method == "initialize":
            await self._handle_initialize(frame)
        elif method == "session/new":
            await self._handle_session_new(frame)
        elif method == "session/set_config_option":
            await self._handle_set_config_option(frame)
        else:
            await self._passthrough_to_backend(frame)

    async def _handle_initialize(self, frame: dict[str, Any]) -> None:
        """Answer ``initialize`` locally with aggregated capabilities.

        v1 advertises a minimal capability set — the proxy does not
        modify the protocol version negotiation downstream.  When a
        backend is later bound, its capabilities may differ; clients
        should treat the proxy's caps as a conservative intersection.
        """
        await self._send_to_client(
            {
                "jsonrpc": "2.0",
                "id": frame.get("id"),
                "result": {
                    "protocolVersion": ACP_PROTOCOL_VERSION,
                    "agentCapabilities": {
                        "promptCapabilities": {},
                        "loadSession": False,
                    },
                    "authMethods": [],
                },
            }
        )

    async def _handle_session_new(self, frame: dict[str, Any]) -> None:
        """Answer ``session/new`` with the aggregated model list.

        Generates a synthetic session id (``proxy-N``) so the client can
        proceed to picking a model before any backend exists.  When a
        backend is later spawned on bind, the backend's real session id
        is captured in :attr:`_backend_session_id` and translated on
        every forwarded frame.
        """
        if self._client_session_id is not None:
            await self._reply_error(
                frame.get("id"),
                code=JSONRPC_INVALID_REQUEST,
                message="proxy supports one session per connection (v1)",
            )
            return

        # Capture the client's params (cwd, mcpServers, …) so the
        # backend's session/new at bind time receives the same context
        # the client asked for, not a hard-coded default.
        client_params = frame.get("params")
        if isinstance(client_params, dict):
            self._client_session_new_params = dict(client_params)

        self._client_session_id = "proxy-1"
        models = await self._roster.list_available_agents()
        await self._send_to_client(
            {
                "jsonrpc": "2.0",
                "id": frame.get("id"),
                "result": {
                    "sessionId": self._client_session_id,
                    "configOptions": [_build_model_config_option(models)],
                    "availableModes": [],
                },
            }
        )

    async def _handle_set_config_option(self, frame: dict[str, Any]) -> None:
        """Bind on first call; forward (with translation) on subsequent calls."""
        raw_params = frame.get("params")
        if raw_params is None:
            params: dict[str, Any] = {}
        elif isinstance(raw_params, dict):
            params = raw_params
        else:
            await self._reply_error(
                frame.get("id"),
                code=JSONRPC_INVALID_PARAMS,
                message="params must be an object",
            )
            return
        category = params.get("category")
        value = params.get("value")
        if category != MODEL_OPTION_CATEGORY or not isinstance(value, str):
            # Non-model config option: forward to backend if bound, else
            # reject — pre-bind we have no idea what valid config is.
            if self._is_bound:
                await self._forward_to_backend(frame)
            else:
                await self._reply_error(
                    frame.get("id"),
                    code=JSONRPC_INVALID_REQUEST,
                    message="set_config_option pre-bind: only model selection is allowed",
                )
            return

        agent_id, _, model_id = value.partition(MODEL_NAMESPACE_SEP)
        if not agent_id or not model_id:
            await self._reply_error(
                frame.get("id"),
                code=JSONRPC_INVALID_PARAMS,
                message=f"model id must be 'agent:model', got {value!r}",
            )
            return

        if self._bound_agent is None:
            await self._bind_and_acknowledge(frame, agent_id=agent_id, model_id=model_id)
        elif agent_id != self._bound_agent:
            await self._reply_error(
                frame.get("id"),
                code=JSONRPC_INVALID_PARAMS,
                message=(
                    f"session is already bound to agent {self._bound_agent!r}; "
                    f"v1 does not support cross-agent switches"
                ),
            )
        else:
            # Same agent, just changing model — strip the ``agent:`` prefix
            # so the backend sees its own bare model id.  Session-id
            # translation (``proxy-1`` → backend's id) is handled separately
            # inside :meth:`_forward_to_backend`, so this call only rewrites
            # ``params.value``.
            await self._forward_to_backend(
                _with_params_value(frame, model_id),
            )

    async def _passthrough_to_backend(self, frame: dict[str, Any]) -> None:
        """Forward unrecognised methods to the bound backend, or reject pre-bind.

        ``session/prompt`` and friends arrive here.  Pre-bind they have
        no destination, so the client is told to pick a model first;
        post-bind the proxy stays out of the way.
        """
        if not self._is_bound:
            await self._reply_error(
                frame.get("id"),
                code=JSONRPC_INVALID_REQUEST,
                message=(
                    "no agent bound — pick a model via "
                    "session/set_config_option before issuing this method"
                ),
            )
            return
        await self._forward_to_backend(frame)

    # ── Bind: spawn backend + replay handshake ────────────────────────

    async def _bind_and_acknowledge(
        self,
        client_frame: dict[str, Any],
        *,
        agent_id: str,
        model_id: str,
    ) -> None:
        """Spawn the backend wrapper and reply to the client's set_config_option.

        On failure, sends a JSON-RPC error back to the client and leaves
        the proxy unbound (the client may try again with a different
        agent).
        """
        try:
            await self._spawn_backend(agent_id)
            await self._replay_backend_handshake(model_id=model_id)
        except AgentBindError as exc:
            _logger.warning("ACP proxy: bind failed: %s", exc)
            await self._teardown_backend()
            await self._reply_error(
                client_frame.get("id"),
                code=JSONRPC_INTERNAL_ERROR,
                message=f"failed to bind agent {agent_id!r}: {exc}",
            )
            return

        self._bound_agent = agent_id
        # Build the post-bind option list: namespaced ids, but only for
        # the bound agent's models so the client can no longer see
        # cross-agent options.
        bound_models = await self._roster.list_available_agents()
        collapsed = [m for m in bound_models if m.startswith(f"{agent_id}{MODEL_NAMESPACE_SEP}")]
        await self._send_to_client(
            {
                "jsonrpc": "2.0",
                "id": client_frame.get("id"),
                "result": {
                    "configOptions": [
                        _build_model_config_option(
                            collapsed, current=f"{agent_id}{MODEL_NAMESPACE_SEP}{model_id}"
                        )
                    ],
                    "availableModes": [],
                },
            }
        )

    async def _spawn_backend(self, agent_id: str) -> None:
        """Start ``terok-{agent_id}-acp`` and connect asyncio pipes to it.

        Uses :meth:`Sandbox.runtime.exec_stdio` in an executor thread —
        the runtime primitive is sync and threading-based by design.
        We hand the child two anonymous pipes; the host-side ends are
        wrapped in asyncio :class:`StreamReader`/:class:`StreamWriter`
        so the proxy loop can drive them naturally.
        """
        loop = asyncio.get_running_loop()

        host_to_child_r, host_to_child_w = os.pipe()
        child_to_host_r, child_to_host_w = os.pipe()

        # Wrap the host-side ends as asyncio streams BEFORE handing the
        # other ends to the executor — connect_*_pipe attaches readers
        # to the loop, so registration must happen on the loop thread.
        write_pipe = os.fdopen(host_to_child_w, "wb", buffering=0)
        write_transport, write_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin,
            write_pipe,
        )
        self._backend_writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)

        read_pipe = os.fdopen(child_to_host_r, "rb", buffering=0)
        reader = asyncio.StreamReader(loop=loop)
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader, loop=loop),
            read_pipe,
        )
        self._backend_reader = reader

        # Hand the *other* ends to the runtime via the roster's
        # ``exec_wrapper`` — keeps the sandbox + container details
        # behind the roster's public API instead of leaking them here.
        child_in = os.fdopen(host_to_child_r, "rb", buffering=0)
        child_out = os.fdopen(child_to_host_w, "wb", buffering=0)
        self._backend_exec_future = loop.run_in_executor(
            None,
            lambda: self._roster.exec_wrapper(agent_id, stdin=child_in, stdout=child_out),
        )

        # Start the backend → client pump.
        self._backend_pump_task = loop.create_task(self._backend_pump_loop())

    async def _replay_backend_handshake(self, *, model_id: str) -> None:
        """Send ``initialize`` + ``session/new`` + ``set_config_option`` to the backend.

        Captures the backend's session id so subsequent client frames
        can be re-targeted on forwarding.  Errors propagate as
        :class:`AgentBindError`.
        """
        await self._proxy_request(
            "initialize",
            {"protocolVersion": ACP_PROTOCOL_VERSION, "clientCapabilities": {}},
        )

        # Replay the client's original ``session/new`` params so the
        # backend's session context matches what the client asked for.
        # Falls back to a safe default for synthetic test clients that
        # never sent params.
        backend_session_new_params = self._client_session_new_params or {
            "cwd": "/workspace",
            "mcpServers": [],
        }
        new_resp = await self._proxy_request(
            "session/new",
            backend_session_new_params,
        )
        backend_session_id = ((new_resp or {}).get("result") or {}).get("sessionId")
        if not isinstance(backend_session_id, str):
            raise AgentBindError("backend session/new returned no sessionId")
        self._backend_session_id = backend_session_id

        await self._proxy_request(
            "session/set_config_option",
            {
                "sessionId": backend_session_id,
                "category": MODEL_OPTION_CATEGORY,
                "value": model_id,
            },
        )

    # ── Forwarding ────────────────────────────────────────────────────

    async def _forward_to_backend(self, frame: dict[str, Any]) -> None:
        """Translate session id then write *frame* to the backend."""
        params = frame.get("params")
        if isinstance(params, dict):
            sid = params.get("sessionId")
            if sid == self._client_session_id and self._backend_session_id is not None:
                params = {**params, "sessionId": self._backend_session_id}
                frame = {**frame, "params": params}
        await self._send_to_backend(frame)

    async def _backend_pump_loop(self) -> None:
        """Read NDJSON frames from the backend and forward to the client.

        Performs the inverse session-id translation and rewrites the
        model configOption (so the client always sees the namespaced
        ``agent:model`` ids it expects).  Exits cleanly on EOF.
        """
        assert self._backend_reader is not None
        while True:
            try:
                line = await self._backend_reader.readline()
            except (asyncio.CancelledError, asyncio.IncompleteReadError):
                return
            if not line:
                return
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                _logger.warning("ACP proxy: backend sent non-JSON frame, dropping")
                continue
            if not isinstance(frame, dict):
                _logger.warning("ACP proxy: backend sent non-object JSON-RPC frame, dropping")
                continue

            # Drop responses to the proxy's own probe/replay frames;
            # they're consumed by ``_await_proxy_response``.
            frame_id = frame.get("id")
            if isinstance(frame_id, str) and frame_id.startswith(PROXY_REQUEST_ID_PREFIX):
                self._deliver_proxy_response(frame_id, frame)
                continue

            # Read ``_bound_agent`` at point of use — the pump task
            # starts in ``_spawn_backend`` *before* bind completes, so
            # snapshotting at loop entry would miss the assignment.
            if self._bound_agent is not None:
                _rewrite_model_options_in_place(frame, self._bound_agent)
            self._translate_session_id_outbound(frame)
            await self._send_to_client(frame)

    # ── Wire helpers ──────────────────────────────────────────────────

    async def _send_to_client(self, frame: dict[str, Any]) -> None:
        """Serialise *frame* as NDJSON and flush to the client writer."""
        if self._client_writer is None:
            return
        data = (json.dumps(frame) + "\n").encode("utf-8")
        self._client_writer.write(data)
        await self._client_writer.drain()

    async def _send_to_backend(self, frame: dict[str, Any]) -> None:
        """Serialise *frame* as NDJSON and write to the backend writer."""
        if self._backend_writer is None:
            raise AgentBindError("backend not running")
        data = (json.dumps(frame) + "\n").encode("utf-8")
        self._backend_writer.write(data)
        await self._backend_writer.drain()

    async def _reply_error(self, request_id: Any, *, code: int, message: str) -> None:
        """Send a JSON-RPC error response to the client."""
        await self._send_to_client(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )

    # ── Proxy-originated request bookkeeping ─────────────────────────

    async def _proxy_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Send a proxy-originated request to the backend and await its response.

        Each call gets a fresh ``px-N`` id and parks an ``asyncio.Future``
        in :attr:`_pending_proxy_responses` keyed by *that* id — so the
        backend's responses correlate by id, not by send-order, and
        out-of-order replies (legal in JSON-RPC 2.0) resolve the right
        future.

        Errors come back as :class:`AgentBindError`: timeouts, JSON-RPC
        ``error`` payloads, or backend disconnect during the wait.
        """
        self._proxy_request_counter += 1
        frame_id = f"{PROXY_REQUEST_ID_PREFIX}{self._proxy_request_counter}"
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_proxy_responses[frame_id] = future
        try:
            await self._send_to_backend(
                {"jsonrpc": "2.0", "id": frame_id, "method": method, "params": params}
            )
            response = await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as exc:
            self._pending_proxy_responses.pop(frame_id, None)
            raise AgentBindError(
                f"backend did not respond to proxy {method!r} within {timeout}s"
            ) from exc
        except Exception:
            self._pending_proxy_responses.pop(frame_id, None)
            raise
        if "error" in response:
            raise AgentBindError(f"backend rejected proxy {method!r}: {response['error']}")
        return response

    def _deliver_proxy_response(self, frame_id: str, frame: dict[str, Any]) -> None:
        """Resolve the future awaiting the response with id *frame_id*."""
        pending = self._pending_proxy_responses.pop(frame_id, None)
        if pending is not None and not pending.done():
            pending.set_result(frame)

    # ── Outbound frame rewrites ──────────────────────────────────────

    def _translate_session_id_outbound(self, frame: dict[str, Any]) -> None:
        """Rewrite backend's session id to the proxy's synthetic one in-place."""
        if self._backend_session_id is None or self._client_session_id is None:
            return
        result = frame.get("result")
        if isinstance(result, dict) and result.get("sessionId") == self._backend_session_id:
            result["sessionId"] = self._client_session_id
        params = frame.get("params")
        if isinstance(params, dict) and params.get("sessionId") == self._backend_session_id:
            params["sessionId"] = self._client_session_id

    # ── Lifecycle ────────────────────────────────────────────────────

    @property
    def _is_bound(self) -> bool:
        return self._bound_agent is not None and self._backend_writer is not None

    async def _teardown_backend(self) -> None:
        """Close pipes, cancel the pump, wait for the exec to drain."""
        self._closed = True
        # Cancel any still-pending proxy requests so awaiters unblock
        # cleanly rather than hanging until their wait_for timeout.
        for future in self._pending_proxy_responses.values():
            if not future.done():
                future.cancel()
        self._pending_proxy_responses.clear()
        if self._backend_writer is not None:
            try:
                self._backend_writer.close()
                await self._backend_writer.wait_closed()
            except Exception as exc:  # noqa: BLE001
                _logger.debug("ACP proxy: backend writer close: %s", exc)
            self._backend_writer = None
        if self._backend_pump_task is not None:
            self._backend_pump_task.cancel()
            try:
                await self._backend_pump_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._backend_pump_task = None
        if self._backend_exec_future is not None:
            try:
                await asyncio.wait_for(self._backend_exec_future, timeout=2.0)
            except Exception as exc:  # noqa: BLE001
                _logger.debug("ACP proxy: backend exec drain: %s", exc)
            self._backend_exec_future = None


# ── Module-private helpers ────────────────────────────────────────────


def _build_model_config_option(
    namespaced_models: list[str],
    *,
    current: str | None = None,
) -> dict[str, Any]:
    """Build a ``category: "model"`` configOption advertising *namespaced_models*.

    The shape mirrors what we observed in the ACP schema during design:
    a select-shaped option with one entry per choice.  Tests and
    real-world clients can both read it without round-tripping through
    the actual ACP schema definition.
    """
    return {
        "id": "model",
        "category": MODEL_OPTION_CATEGORY,
        "currentValue": current,
        "select": {
            "options": [
                {"id": ident, "name": _humanise_model_id(ident)} for ident in namespaced_models
            ],
        },
    }


def _humanise_model_id(namespaced: str) -> str:
    """Render ``claude:opus-4.6`` as ``Claude — opus-4.6`` for the picker."""
    agent, _, model = namespaced.partition(MODEL_NAMESPACE_SEP)
    if not agent or not model:
        return namespaced
    return f"{agent.capitalize()} — {model}"


def _with_params_value(frame: dict[str, Any], new_value: Any) -> dict[str, Any]:
    """Return a shallow-copied *frame* with ``params.value`` replaced."""
    params = dict(frame.get("params") or {})
    params["value"] = new_value
    out = dict(frame)
    out["params"] = params
    return out


def _rewrite_model_options_in_place(frame: dict[str, Any], bound_agent: str) -> None:
    """Mutate *frame* so any ``configOptions[category=model]`` is namespaced.

    Backends emit bare model ids (``opus-4.6``); clients expect
    namespaced ids (``claude:opus-4.6``).  After bind, only the bound
    agent's models should appear; the proxy adds the prefix here, and
    namespaces ``currentValue`` too so the client's view of the
    selected model stays consistent with the choices it sees.
    """
    result = frame.get("result")
    if not isinstance(result, dict):
        return
    for opt in result.get("configOptions") or []:
        if not isinstance(opt, dict) or opt.get("category") != MODEL_OPTION_CATEGORY:
            continue
        current = opt.get("currentValue")
        if isinstance(current, str) and MODEL_NAMESPACE_SEP not in current:
            opt["currentValue"] = f"{bound_agent}{MODEL_NAMESPACE_SEP}{current}"
    for entry in iter_model_choice_dicts(result):
        ident = entry.get("id") or entry.get("value")
        if not isinstance(ident, str) or MODEL_NAMESPACE_SEP in ident:
            continue
        prefixed = f"{bound_agent}{MODEL_NAMESPACE_SEP}{ident}"
        if "id" in entry:
            entry["id"] = prefixed
        if "value" in entry:
            entry["value"] = prefixed
