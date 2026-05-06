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
  locally, advertising the aggregated ``agent:model`` list in
  :class:`acp.schema.SessionModelState` plus a mirroring
  ``configOptions[category=model]``.  No backend process exists yet.
- **Bound**: on the first model-picking client request — modern
  ACP's ``session/set_model`` or older Zed's
  ``session/set_config_option(configId="model")`` — the proxy
  uses the ``agent:`` namespace prefix to pick which in-container
  wrapper to spawn via :func:`asyncio.create_subprocess_exec`
  (argv from :meth:`ACPRoster.wrapper_argv`), replays
  ``initialize`` + ``session/new`` + ``session/set_model`` (with
  the bare sub-id) to it, and from then on bridges frames in both
  directions.  Backend frames are mutated on the way out so model
  ids visible to the client are namespaced again.

V1 takes shortcuts where the design is still settling: one session
per connection (Zed reconnects on every chat — fix on the roadmap),
one bound agent per session (no cross-agent switches without
reconnect), no push notifications when the authed-agent set changes
mid-connection.  All of these are tracked for v2.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from acp import (
    PROTOCOL_VERSION as ACP_PROTOCOL_VERSION,
    InitializeResponse,
    NewSessionResponse,
)
from acp.schema import (
    Implementation,
    ModelInfo,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionModelState,
    SetSessionConfigOptionResponse,
)

from .model_options import (
    MODEL_NAMESPACE_SEP,
    MODEL_OPTION_CATEGORY,
    iter_model_choice_dicts,
)

if TYPE_CHECKING:
    from .roster import ACPRoster

_logger = logging.getLogger(__name__)

PROXY_REQUEST_ID_BASE = 1_000_000_000
"""Numeric offset for JSON-RPC ids the proxy injects during the bind
handshake replay (``initialize`` / ``session/new`` /
``session/set_model`` to the backend wrapper).  Sits well clear of
any client-side counter we've observed in practice while staying
inside JS's safe-integer range so JSON parsers handle it as a number.
Discrimination is by direct equality in the inline read loop —
there's only one outstanding request at a time during the
handshake."""

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
        self._backend_proc: asyncio.subprocess.Process | None = None
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
                if not isinstance(frame, dict):
                    _logger.warning("ACP proxy: client sent non-object JSON-RPC frame, ignoring")
                    continue
                _logger.debug("← client: %s", _summarise_frame(frame))
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
        elif method == "session/set_model":
            await self._handle_set_model(frame)
        elif method == "session/set_config_option":
            await self._handle_set_config_option(frame)
        else:
            await self._passthrough_to_backend(frame)

    async def _handle_initialize(self, frame: dict[str, Any]) -> None:
        """Answer ``initialize`` locally with aggregated capabilities.

        Built via the ACP SDK's pydantic model so missing-or-misnamed
        fields fail at construction with a precise traceback rather
        than at the client end with "failed to deserialize".  The
        capability defaults from :class:`acp.schema.AgentCapabilities`
        are conservative — clients should treat the proxy's caps as a
        floor; the bound backend's are the real ceiling.
        """
        result = InitializeResponse(
            protocol_version=ACP_PROTOCOL_VERSION,
            agent_info=Implementation(
                name="terok-acp",
                title="Terok ACP host-proxy",
                version="1",
            ),
        )
        await self._send_to_client(
            {
                "jsonrpc": "2.0",
                "id": frame.get("id"),
                "result": result.model_dump(by_alias=True, exclude_none=True, mode="json"),
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
        result = _build_session_new_response(self._client_session_id, models)
        await self._send_to_client(
            {
                "jsonrpc": "2.0",
                "id": frame.get("id"),
                "result": result.model_dump(by_alias=True, exclude_none=True, mode="json"),
            }
        )

    async def _handle_set_model(self, frame: dict[str, Any]) -> None:
        """Bind on first call; forward namespace-stripped on subsequent calls.

        Modern ACP's dedicated method for model selection.  Reads
        ``modelId`` from the request, hands the namespaced id to the
        shared :meth:`_select_model` driver.
        """
        raw_params = frame.get("params")
        if not isinstance(raw_params, dict):
            await self._reply_error(
                frame.get("id"),
                code=JSONRPC_INVALID_PARAMS,
                message="params must be an object",
            )
            return
        namespaced = raw_params.get("modelId") or raw_params.get("model_id")
        if not isinstance(namespaced, str):
            await self._reply_error(
                frame.get("id"),
                code=JSONRPC_INVALID_PARAMS,
                message="modelId must be a string",
            )
            return
        await self._select_model(
            frame,
            namespaced=namespaced,
            forward_field="modelId",
            ack_kind="set_model",
        )

    async def _handle_set_config_option(self, frame: dict[str, Any]) -> None:
        """Bind on category=model; otherwise forward to the bound backend.

        Older ACP clients (Zed v1.0.x at the time of writing) pick the
        model through :data:`session/set_config_option` with the
        category set to ``"model"`` — modern clients use the dedicated
        :data:`session/set_model` instead.  We accept both: a model
        category here triggers the same bind flow as ``set_model``,
        any other category passes through to the bound backend, and
        a non-model category pre-bind is rejected as a no-op.

        Field names: modern ACP names the discriminator ``configId``,
        older clients sent ``category``; both are accepted.
        """
        raw_params = frame.get("params")
        if not isinstance(raw_params, dict):
            await self._reply_error(
                frame.get("id"),
                code=JSONRPC_INVALID_PARAMS,
                message="params must be an object",
            )
            return
        config_id = (
            raw_params.get("configId") or raw_params.get("config_id") or raw_params.get("category")
        )
        value = raw_params.get("value")
        if config_id == MODEL_OPTION_CATEGORY and isinstance(value, str):
            await self._select_model(
                frame,
                namespaced=value,
                forward_field="value",
                ack_kind="set_config_option",
            )
            return

        # Non-model config option: forward to backend if bound, reject otherwise.
        if not self._is_bound:
            await self._reply_error(
                frame.get("id"),
                code=JSONRPC_INVALID_REQUEST,
                message="no agent bound — pick a model first",
            )
            return
        await self._forward_to_backend(frame)

    async def _select_model(
        self,
        frame: dict[str, Any],
        *,
        namespaced: str,
        forward_field: str,
        ack_kind: str,
    ) -> None:
        """Shared driver for ``set_model`` / ``set_config_option(category=model)``.

        - ``namespaced`` is the ``agent:model`` id from the client.
        - ``forward_field`` is the request param key whose value the
          backend should see stripped of the ``agent:`` prefix
          (``modelId`` for ``set_model``, ``value`` for
          ``set_config_option``).
        - ``ack_kind`` selects which response shape we send back on
          first-bind: ``set_model`` → empty ``{}``, ``set_config_option``
          → ``{config_options: [...]}``.
        """
        agent_id, _, model_id = namespaced.partition(MODEL_NAMESPACE_SEP)
        if not agent_id or not model_id:
            await self._reply_error(
                frame.get("id"),
                code=JSONRPC_INVALID_PARAMS,
                message=f"model id must be 'agent:model', got {namespaced!r}",
            )
            return

        if self._bound_agent is None:
            await self._bind_and_acknowledge(
                frame, agent_id=agent_id, model_id=model_id, ack_kind=ack_kind
            )
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
            await self._forward_to_backend(_with_params_field(frame, forward_field, model_id))

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
                message="no agent bound — pick a model first",
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
        ack_kind: str,
    ) -> None:
        """Spawn the backend wrapper and acknowledge the client.

        ``ack_kind`` selects the response shape:

        - ``"set_model"`` → :class:`SetSessionModelResponse` (empty
          ``result: {}`` body).
        - ``"set_config_option"`` → :class:`SetSessionConfigOptionResponse`
          carrying the post-bind ``configOptions`` snapshot — namespaced
          ids, but only for the bound agent's models so the client's
          option list collapses to a single agent.

        On failure, replies with a JSON-RPC error and leaves the proxy
        unbound — the client may try again with a different agent.
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
        ack_result = await self._build_bind_ack(agent_id, model_id, ack_kind=ack_kind)
        await self._send_to_client(
            {"jsonrpc": "2.0", "id": client_frame.get("id"), "result": ack_result}
        )

    async def _build_bind_ack(
        self, agent_id: str, model_id: str, *, ack_kind: str
    ) -> dict[str, Any]:
        """Build the response body for the bind-triggering request.

        ``set_model`` ack is empty.  ``set_config_option`` ack carries
        the *full* aggregated option list with ``currentValue`` set to
        the user's choice; do not filter to just the bound agent's
        models — Zed rebuilds its picker from this response and a
        filtered list silently hides the other agents.  Cross-agent
        switches are still rejected, but at the request level
        (:meth:`_select_model`), not by erasing the options.
        """
        if ack_kind == "set_model":
            return {}
        all_models = await self._roster.list_available_agents()
        current = f"{agent_id}{MODEL_NAMESPACE_SEP}{model_id}"
        opt = _build_model_config_option(all_models, current=current)
        return SetSessionConfigOptionResponse(config_options=[opt]).model_dump(
            by_alias=True, exclude_none=True, mode="json"
        )

    async def _spawn_backend(self, agent_id: str) -> None:
        """Spawn ``terok-{agent_id}-acp`` and attach asyncio pipes for the bind handshake.

        Uses :func:`asyncio.create_subprocess_exec` — proc.stdin /
        proc.stdout become the proxy's writer / reader directly.  The
        probe path goes through sandbox's ``exec_stdio`` instead
        (single-shot, sync-friendly); bind's connection-shaped
        lifecycle (long-lived, multi-request, cancel-on-disconnect)
        fits asyncio's subprocess transport more naturally and avoids
        the extra kernel pipe pair plus pump threads ``exec_stdio``
        layers in.

        :meth:`ACPRoster.wrapper_argv` hides the runtime detail
        (currently podman-specific) so a future krun runtime can
        plug in its own argv without touching the proxy.
        """
        argv = self._roster.wrapper_argv(agent_id)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if proc.stdin is None or proc.stdout is None:
            raise AgentBindError("create_subprocess_exec returned no stdio pipes")
        self._backend_writer = proc.stdin
        self._backend_reader = proc.stdout
        self._backend_proc = proc
        _logger.debug("backend[%s] subprocess pid=%s started", agent_id, proc.pid)

    async def _replay_backend_handshake(self, *, model_id: str) -> None:
        """Send ``initialize`` + ``session/new`` + ``set_model`` to the backend.

        Drives the three frames inline — write, read the matching
        response, repeat — instead of standing up the long-lived pump
        task that handles post-bind traffic.  Inline keeps the
        handshake's state local: no parked-future bookkeeping, no
        id-discrimination at the pump, and a failed handshake tears
        down without leaving an orphan pump.  :meth:`_start_pump_loop`
        spins the pump up only after all three frames acknowledge,
        from which point the client owns the conversation.

        Captures the backend's session id so subsequent client frames
        can be re-targeted on forwarding.  Errors raise
        :class:`AgentBindError`.
        """
        await self._inline_request(
            "initialize",
            {"protocolVersion": ACP_PROTOCOL_VERSION, "clientCapabilities": {}},
        )

        # Pin cwd to the container's workspace mount: ACP clients send
        # their host filesystem path here (Zed: ``/var/home/user/prog/X``)
        # which doesn't exist inside the container.  claude-agent-acp's
        # bootstrap chdirs into cwd before exec; an ENOENT there
        # surfaces as the famously misleading "Claude Code native
        # binary not found …" — see the host↔sandbox path strategy
        # note for why this is a stopgap.
        backend_session_new_params = dict(self._client_session_new_params)
        backend_session_new_params["cwd"] = "/workspace"
        backend_session_new_params.setdefault("mcpServers", [])
        new_resp = await self._inline_request(
            "session/new",
            backend_session_new_params,
        )
        backend_session_id = ((new_resp or {}).get("result") or {}).get("sessionId")
        if not isinstance(backend_session_id, str):
            raise AgentBindError("backend session/new returned no sessionId")
        self._backend_session_id = backend_session_id

        await self._inline_request(
            "session/set_model",
            {"sessionId": backend_session_id, "modelId": model_id},
        )

        # Handshake done — switch to ongoing pump for client routing.
        self._start_pump_loop()

    def _start_pump_loop(self) -> None:
        """Start the long-lived backend→client pump task.

        Called after the handshake completes; before this point, the
        bind code drives ``readline`` directly.  Idempotent — safe to
        call once even if the connection has many bind retries.
        """
        if self._backend_pump_task is not None:
            return
        loop = asyncio.get_running_loop()
        self._backend_pump_task = loop.create_task(self._backend_pump_loop())

    async def _inline_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Write a request frame and inline-read its matching response.

        Used during the bind handshake, before the pump task starts.
        Only one request is outstanding at a time on this path, so
        ``id == expected`` is enough to match the reply — no parked-
        future bookkeeping.  Ids still come from the proxy's reserved
        ``PROXY_REQUEST_ID_BASE+N`` range so responses are recognisable
        as proxy-originated if the pump later sees one in transit.

        Raises :class:`AgentBindError` on timeout, malformed JSON,
        unexpected ids, or backend disconnect during the read.
        """
        self._proxy_request_counter += 1
        frame_id = PROXY_REQUEST_ID_BASE + self._proxy_request_counter
        await self._send_to_backend(
            {"jsonrpc": "2.0", "id": frame_id, "method": method, "params": params}
        )
        try:
            return await asyncio.wait_for(
                self._read_one_inline_response(frame_id, method),
                timeout=timeout,
            )
        except TimeoutError as exc:
            raise AgentBindError(
                f"backend did not respond to proxy {method!r} within {timeout}s"
            ) from exc

    async def _read_one_inline_response(self, expected_id: int, method: str) -> dict[str, Any]:
        """Read frames from the backend until the response with *expected_id* arrives.

        Skips notifications (no ``id``) so a chatty wrapper's progress
        events don't get mistaken for the reply.  Out-of-order ids on
        a freshly-spawned single-track wrapper signal protocol confusion
        — bail with :class:`AgentBindError` rather than queue.
        """
        assert self._backend_reader is not None
        while True:
            line = await self._backend_reader.readline()
            if not line:
                raise AgentBindError(f"backend closed stdout before responding to {method!r}")
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AgentBindError(f"backend sent malformed JSON during {method!r}") from exc
            if not isinstance(frame, dict):
                raise AgentBindError(f"backend sent a non-object frame during {method!r}")
            _logger.debug("← backend (handshake): %s", _summarise_frame(frame))
            frame_id = frame.get("id")
            if frame_id == expected_id:
                if "error" in frame:
                    raise AgentBindError(f"backend rejected {method!r}: {frame['error']}")
                return frame
            if "method" in frame and frame_id is None:
                # Notification — log and keep reading.
                continue
            raise AgentBindError(
                f"backend sent unexpected frame during {method!r}: id={frame_id!r}"
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
            except asyncio.CancelledError:
                return
            except asyncio.IncompleteReadError:
                _logger.debug("backend pump: reader saw incomplete read, exiting")
                return
            if not line:
                _logger.debug("backend pump: reader returned EOF, exiting")
                return
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                _logger.warning("ACP proxy: backend sent non-JSON frame, dropping")
                continue
            if not isinstance(frame, dict):
                _logger.warning("ACP proxy: backend sent non-object JSON-RPC frame, dropping")
                continue
            _logger.debug("← backend: %s", _summarise_frame(frame))

            # The pump task starts only *after* the handshake replay
            # completes, so any frame that arrives here belongs to
            # the bound client, not to a proxy-internal request.  No
            # id discrimination needed.
            if self._bound_agent is not None:
                _rewrite_model_options_in_place(frame, self._bound_agent)
            self._translate_session_id_outbound(frame)
            await self._send_to_client(frame)

    # ── Wire helpers ──────────────────────────────────────────────────

    async def _send_to_client(self, frame: dict[str, Any]) -> None:
        """Serialise *frame* as NDJSON and flush to the client writer."""
        if self._client_writer is None:
            return
        _logger.debug("→ client: %s", _summarise_frame(frame))
        data = (json.dumps(frame) + "\n").encode("utf-8")
        self._client_writer.write(data)
        await self._client_writer.drain()

    async def _send_to_backend(self, frame: dict[str, Any]) -> None:
        """Serialise *frame* as NDJSON and write to the backend writer.

        Logs the byte count after ``drain`` returns so a future bind
        hang can be split: if the count is logged but no response
        comes back in the corresponding ``← backend`` log, the bug is
        between asyncio's writer and the wrapper subprocess; if the
        count never logs, the writer itself didn't flush.
        """
        if self._backend_writer is None:
            raise AgentBindError("backend not running")
        _logger.debug("→ backend: %s", _summarise_frame(frame))
        data = (json.dumps(frame) + "\n").encode("utf-8")
        self._backend_writer.write(data)
        await self._backend_writer.drain()
        _logger.debug("backend writer: drained %d bytes", len(data))

    async def _reply_error(self, request_id: Any, *, code: int, message: str) -> None:
        """Send a JSON-RPC error response to the client."""
        await self._send_to_client(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )

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
        """Close stdin to the wrapper subprocess, cancel the pump, reap the proc."""
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
        if self._backend_proc is not None:
            # Stdin's been closed above; healthy ACP wrappers exit on
            # EOF.  Wait briefly, then SIGTERM, then SIGKILL — the
            # daemon shouldn't keep zombie podman-exec processes
            # around if a wrapper goes rogue.
            proc = self._backend_proc
            self._backend_proc = None
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=2.0)
                _logger.debug("backend[%s] proc exited rc=%s", self._bound_agent, rc)
            except TimeoutError:
                _logger.warning(
                    "backend[%s] proc didn't exit on stdin EOF; sending SIGTERM",
                    self._bound_agent,
                )
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except TimeoutError:
                    _logger.warning(
                        "backend[%s] proc still alive after SIGTERM; SIGKILL",
                        self._bound_agent,
                    )
                    proc.kill()
                    await proc.wait()


# ── Module-private helpers ────────────────────────────────────────────


def _build_model_config_option(
    namespaced_models: list[str],
    *,
    current: str,
) -> SessionConfigOptionSelect:
    """Build a ``category: "model"`` ``select`` configOption.

    Returns the SDK pydantic model (not a dict) so the caller can drop
    it straight into a :class:`NewSessionResponse` /
    :class:`SetSessionConfigOptionResponse`.  ``current`` is required
    by the schema (``current_value`` is a non-nullable ``str``); the
    caller is expected to handle the empty-models case by skipping
    the option entirely rather than passing a placeholder.
    """
    return SessionConfigOptionSelect(
        id="model",
        name="Model",
        type="select",
        description="AI model to use",
        category=MODEL_OPTION_CATEGORY,
        current_value=current,
        options=[
            SessionConfigSelectOption(value=ident, name=_humanise_model_id(ident))
            for ident in namespaced_models
        ],
    )


def _build_session_new_response(session_id: str, models: list[str]) -> NewSessionResponse:
    """Construct the pre-bind ``session/new`` reply for *models*.

    Splits the empty-models case so the response stays schema-valid:
    when no agents probed successfully, the ``models`` block and the
    model ``configOption`` are omitted entirely (both have non-nullable
    required fields the proxy can't fill in for an empty list).  The
    ``modes`` block is also omitted — :class:`SessionModeState` requires
    a non-null ``current_mode_id`` and the proxy doesn't manage modes.
    """
    if not models:
        return NewSessionResponse(session_id=session_id)
    current = models[0]
    return NewSessionResponse(
        session_id=session_id,
        models=SessionModelState(
            available_models=[
                ModelInfo(model_id=ident, name=_humanise_model_id(ident)) for ident in models
            ],
            current_model_id=current,
        ),
        config_options=[_build_model_config_option(models, current=current)],
    )


def _humanise_model_id(namespaced: str) -> str:
    """Render ``claude:opus-4.6`` as ``Claude: opus-4.6`` for the picker.

    Colon (matching the wire-level :data:`MODEL_NAMESPACE_SEP`) keeps
    slashes free for model ids that legitimately contain them, e.g.
    OpenCode's ``opencode:opencode/big-pickle`` — humanising that to
    ``OpenCode: opencode/big-pickle`` reads as one provider plus one
    model id, while using ``/`` as the separator would chop the model
    id in the middle.  We only mutate names *we* synthesise;
    descriptions / labels coming back from real backend agents are
    forwarded verbatim so upstream formatting decisions stand.
    """
    agent, _, model = namespaced.partition(MODEL_NAMESPACE_SEP)
    if not agent or not model:
        return namespaced
    return f"{agent.capitalize()}: {model}"


def _summarise_frame(frame: dict[str, Any]) -> str:
    """One-line summary for the wire-trace debug log: id, method, errors.

    The full frame can be hundreds of bytes (model lists, prompt text)
    and would drown the log on every send.  This trims to the routing
    fields a debug session actually needs: which method, which session,
    and the result/error head — enough to follow the conversation
    without re-running with ``RUST_LOG=trace`` on the client.
    """
    parts: list[str] = []
    if "method" in frame:
        parts.append(f"method={frame['method']!r}")
    if "id" in frame:
        parts.append(f"id={frame['id']!r}")
    params = frame.get("params")
    if isinstance(params, dict):
        sid = params.get("sessionId") or params.get("session_id")
        if sid:
            parts.append(f"session={sid!r}")
        mid = params.get("modelId") or params.get("model_id")
        if mid:
            parts.append(f"model={mid!r}")
        cid = params.get("configId") or params.get("config_id") or params.get("category")
        if cid:
            parts.append(f"config={cid!r}")
        val = params.get("value")
        if val is not None:
            parts.append(f"value={val!r}")
        cwd = params.get("cwd")
        if cwd:
            parts.append(f"cwd={cwd!r}")
    if "error" in frame:
        err = frame["error"]
        if isinstance(err, dict):
            parts.append(f"error={err.get('code')!r} {err.get('message')!r}")
    elif "result" in frame:
        res = frame["result"]
        if isinstance(res, dict):
            keys = ",".join(sorted(res.keys()))
            parts.append(f"result-keys=[{keys}]")
        else:
            parts.append("result=…")
    return " ".join(parts) or "<empty>"


def _with_params_field(frame: dict[str, Any], field_name: str, new_value: Any) -> dict[str, Any]:
    """Return a shallow-copied *frame* with ``params[field_name]`` replaced.

    Used to rewrite specific request params on the way to the backend
    (e.g. strip the ``agent:`` namespace from ``modelId`` so the
    backend sees its own bare ids).  Shallow copy is enough — the
    proxy never mutates the inner ``params`` after writing.
    """
    params = dict(frame.get("params") or {})
    params[field_name] = new_value
    out = dict(frame)
    out["params"] = params
    return out


def _rewrite_model_options_in_place(frame: dict[str, Any], bound_agent: str) -> None:
    """Mutate *frame* so any model ids are namespaced as ``agent:model``.

    Backends emit bare model ids (``opus-4.6``); clients expect
    namespaced ids (``claude:opus-4.6``).  After bind, only the bound
    agent's models should appear; the proxy adds the prefix on every
    place a model id can show up:

    - ``configOptions[category=model].currentValue``
    - ``configOptions[category=model].options[*].id`` / ``.value``
    - ``models.currentModelId``
    - ``models.availableModels[*].modelId``

    The two ``models.*`` paths matter for the post-bind ``session/new``
    reply: claude-agent-acp v0.32+ carries the picker primarily in
    ``result.models``, and Zed reads from there — leaving those bare
    would surface ``opus-4.6`` in the picker even though the client
    has to send ``claude:opus-4.6`` back.
    """
    result = frame.get("result")
    if not isinstance(result, dict):
        return

    def _prefix(value: str) -> str:
        return f"{bound_agent}{MODEL_NAMESPACE_SEP}{value}"

    for opt in result.get("configOptions") or []:
        if not isinstance(opt, dict) or opt.get("category") != MODEL_OPTION_CATEGORY:
            continue
        current = opt.get("currentValue")
        if isinstance(current, str) and MODEL_NAMESPACE_SEP not in current:
            opt["currentValue"] = _prefix(current)
    for entry in iter_model_choice_dicts(result):
        ident = entry.get("id") or entry.get("value")
        if not isinstance(ident, str) or MODEL_NAMESPACE_SEP in ident:
            continue
        prefixed = _prefix(ident)
        if "id" in entry:
            entry["id"] = prefixed
        if "value" in entry:
            entry["value"] = prefixed

    models = result.get("models")
    if isinstance(models, dict):
        current = models.get("currentModelId")
        if isinstance(current, str) and MODEL_NAMESPACE_SEP not in current:
            models["currentModelId"] = _prefix(current)
        for entry in models.get("availableModels") or []:
            if not isinstance(entry, dict):
                continue
            mid = entry.get("modelId")
            if isinstance(mid, str) and MODEL_NAMESPACE_SEP not in mid:
                entry["modelId"] = _prefix(mid)
