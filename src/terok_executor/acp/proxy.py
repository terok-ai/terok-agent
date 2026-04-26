# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""ACP proxy state machine — the JSON-RPC mediator behind :class:`ACPRoster`.

The proxy speaks ACP to the connected client and ACP to one chosen
in-container backend, namespacing the model selector so a multi-agent
container looks like a single endpoint.  It parses every JSON-RPC frame
(NDJSON over stdio) — there is no byte-level passthrough; uninteresting
frames are re-serialised after parsing.

Two phases:

- **Pre-bind**: the proxy answers ``initialize`` and ``session/new``
  locally, advertising the aggregated ``agent:model`` list as a
  ``category: "model"`` configOption.  No backend process exists yet.
- **Bound**: on the first ``session/set_config_option`` for the model
  category, the proxy parses the ``agent:model`` value, spawns the
  agent's wrapper script via :meth:`Sandbox.runtime.exec_stdio`, replays
  ``initialize`` + ``session/new`` to it, and from then on bridges
  frames in both directions.  The option list is rewritten on the way
  out so cross-agent values disappear from the client's view.

V1 limitations (deferred):

- One client per server connection.  A second concurrent client is
  rejected during ``initialize`` with a JSON-RPC error.
- One session per binding.  ``session/load`` and multi-session is not
  exercised.
- No live re-emission of configOption changes; clients see the up-to-
  date roster at ``session/new`` time only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .roster import ACPRoster

_logger = logging.getLogger(__name__)

ACP_PROTOCOL_VERSION = 1
"""Version this proxy advertises on ``initialize``.  Mirrors what the
probe sends; in practice both backends and clients negotiate down."""

MODEL_OPTION_CATEGORY = "model"
"""ACP semantic category for the model selector configOption."""

MODEL_NAMESPACE_SEP = ":"
"""Separator between agent and model in the namespaced id (e.g.
``claude:opus-4.6``).  Chosen over ``/`` to avoid collisions with
OpenRouter-style ids like ``anthropic/claude-opus-4``."""

PROXY_REQUEST_ID_PREFIX = "px-"
"""Prefix for JSON-RPC request ids the proxy injects (replay of
``initialize``/``session/new`` to the backend).  Strings can't collide
with the int ids ACP clients typically use."""


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
        self._backend_writer: asyncio.StreamWriter | None = None
        self._backend_reader: asyncio.StreamReader | None = None
        self._backend_pump_task: asyncio.Task | None = None
        self._backend_exec_future: asyncio.Future | None = None
        self._backend_pipe_fds: tuple[int, int, int, int] | None = None
        self._proxy_request_counter = 0
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
            await self._handle_other_client_method(frame)

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
                code=-32600,
                message="proxy supports one session per connection (v1)",
            )
            return

        self._client_session_id = "proxy-1"
        models = self._roster.list_available_agents()
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
        params = frame.get("params") or {}
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
                    code=-32600,
                    message="set_config_option pre-bind: only model selection is allowed",
                )
            return

        agent_id, _, model_id = value.partition(MODEL_NAMESPACE_SEP)
        if not agent_id or not model_id:
            await self._reply_error(
                frame.get("id"),
                code=-32602,
                message=f"model id must be 'agent:model', got {value!r}",
            )
            return

        if self._bound_agent is None:
            await self._bind_and_acknowledge(frame, agent_id=agent_id, model_id=model_id)
        elif agent_id != self._bound_agent:
            await self._reply_error(
                frame.get("id"),
                code=-32602,
                message=(
                    f"session is already bound to agent {self._bound_agent!r}; "
                    f"v1 does not support cross-agent switches"
                ),
            )
        else:
            # Same agent, just changing model — strip namespace before
            # forwarding so the backend sees its own model id.
            await self._forward_to_backend(
                _with_params_value(frame, model_id),
            )

    async def _handle_other_client_method(self, frame: dict[str, Any]) -> None:
        """Catch-all: forward to backend, or reject pre-bind."""
        if not self._is_bound:
            await self._reply_error(
                frame.get("id"),
                code=-32600,
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
                code=-32603,
                message=f"failed to bind agent {agent_id!r}: {exc}",
            )
            return

        self._bound_agent = agent_id
        # Build the post-bind option list: namespaced ids, but only for
        # the bound agent's models so the client can no longer see
        # cross-agent options.
        bound_models = self._roster.list_available_agents()
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
        self._backend_pipe_fds = (
            host_to_child_r,
            host_to_child_w,
            child_to_host_r,
            child_to_host_w,
        )

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

        # Hand the *other* ends to the runtime.  The wrapper script lives
        # on the container's ``$PATH`` (staged at L1 build time).
        wrapper = [f"terok-{agent_id}-acp"]
        sandbox = self._roster.sandbox
        container = sandbox.runtime.container(self._roster.container_name)
        child_in = os.fdopen(host_to_child_r, "rb", buffering=0)
        child_out = os.fdopen(child_to_host_w, "wb", buffering=0)

        def _run() -> int:
            return sandbox.runtime.exec_stdio(
                container,
                wrapper,
                stdin=child_in,
                stdout=child_out,
            )

        self._backend_exec_future = loop.run_in_executor(None, _run)

        # Start the backend → client pump.
        self._backend_pump_task = loop.create_task(self._backend_pump_loop())

    async def _replay_backend_handshake(self, *, model_id: str) -> None:
        """Send ``initialize`` + ``session/new`` + ``set_config_option`` to the backend.

        Captures the backend's session id so subsequent client frames
        can be re-targeted on forwarding.  Errors propagate as
        :class:`AgentBindError`.
        """
        # initialize ---------------------------------------------------
        await self._send_to_backend(
            {
                "jsonrpc": "2.0",
                "id": self._next_proxy_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": ACP_PROTOCOL_VERSION,
                    "clientCapabilities": {},
                },
            }
        )
        await self._await_proxy_response("initialize")

        # session/new --------------------------------------------------
        await self._send_to_backend(
            {
                "jsonrpc": "2.0",
                "id": self._next_proxy_id(),
                "method": "session/new",
                "params": {"cwd": "/workspace", "mcpServers": []},
            }
        )
        new_resp = await self._await_proxy_response("session/new")
        backend_session_id = ((new_resp or {}).get("result") or {}).get("sessionId")
        if not isinstance(backend_session_id, str):
            raise AgentBindError("backend session/new returned no sessionId")
        self._backend_session_id = backend_session_id

        # session/set_config_option (model) ---------------------------
        await self._send_to_backend(
            {
                "jsonrpc": "2.0",
                "id": self._next_proxy_id(),
                "method": "session/set_config_option",
                "params": {
                    "sessionId": backend_session_id,
                    "category": MODEL_OPTION_CATEGORY,
                    "value": model_id,
                },
            }
        )
        await self._await_proxy_response("session/set_config_option")

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
        agent = self._bound_agent
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

            # Drop responses to the proxy's own probe/replay frames;
            # they're consumed by ``_await_proxy_response``.
            frame_id = frame.get("id")
            if isinstance(frame_id, str) and frame_id.startswith(PROXY_REQUEST_ID_PREFIX):
                self._deliver_proxy_response(frame_id, frame)
                continue

            if agent is not None:
                _rewrite_model_options_in_place(frame, agent)
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

    _pending_proxy_responses: dict[str, asyncio.Future]

    def _next_proxy_id(self) -> str:
        """Return the next id to use for a proxy-originated request."""
        self._proxy_request_counter += 1
        return f"{PROXY_REQUEST_ID_PREFIX}{self._proxy_request_counter}"

    async def _await_proxy_response(self, label: str) -> dict[str, Any]:
        """Block until the next ``proxy:`` reply arrives from the backend.

        v1 simplification: we send proxy-originated requests one at a
        time during bind, so we can use a single pending-future slot
        rather than a full id→future map.  Future revisions that want
        concurrent proxy requests (e.g. probing while a session is in
        flight) should switch to a dict keyed by id.
        """
        if not hasattr(self, "_pending_proxy_responses"):
            self._pending_proxy_responses = {}
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        # Park the future under the most recently emitted id.
        last_id = f"{PROXY_REQUEST_ID_PREFIX}{self._proxy_request_counter}"
        self._pending_proxy_responses[last_id] = future
        try:
            response = await asyncio.wait_for(future, timeout=15.0)
        except TimeoutError as exc:
            raise AgentBindError(f"backend did not respond to proxy {label!r} within 15s") from exc
        if "error" in response:
            raise AgentBindError(f"backend rejected proxy {label!r}: {response['error']}")
        return response

    def _deliver_proxy_response(self, frame_id: str, frame: dict[str, Any]) -> None:
        """Resolve the future awaiting the response with id *frame_id*."""
        pending = getattr(self, "_pending_proxy_responses", {}).pop(frame_id, None)
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
            except (TimeoutError, Exception) as exc:  # noqa: BLE001
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
    agent's models should appear; the proxy adds the prefix here.
    """
    result = frame.get("result")
    if not isinstance(result, dict):
        return
    options = result.get("configOptions")
    if not isinstance(options, list):
        return
    for opt in options:
        if not isinstance(opt, dict) or opt.get("category") != MODEL_OPTION_CATEGORY:
            continue
        select = opt.get("select")
        if not isinstance(select, dict):
            continue
        for key in ("options", "values", "choices"):
            entries = select.get(key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    ident = entry.get("id") or entry.get("value")
                    if isinstance(ident, str) and MODEL_NAMESPACE_SEP not in ident:
                        prefixed = f"{bound_agent}{MODEL_NAMESPACE_SEP}{ident}"
                        if "id" in entry:
                            entry["id"] = prefixed
                        if "value" in entry:
                            entry["value"] = prefixed
