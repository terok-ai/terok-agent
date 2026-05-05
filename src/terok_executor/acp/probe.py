# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Model-roster probe for in-container ACP agents.

Each in-container agent ships an ACP wrapper script (``terok-{agent}-acp``)
that exposes the agent over JSON-RPC on stdio.  To learn which models an
agent currently advertises, we drive a minimal handshake:

1. ``initialize`` — version negotiation
2. ``session/new`` — receive ``configOptions`` (the model list lives here)
3. close stdin — agent exits cleanly

The handshake is cheap (a few round-trips) but non-trivial to repeat:
the result is cached by :class:`~terok_executor.acp.cache.AgentRosterCache`
and reused for the lifetime of the authenticated session.

The probe is transport-agnostic on top of
:meth:`terok_sandbox.ContainerRuntime.exec_stdio` — it owns no FDs of its
own; it spawns the agent in an executor thread and bridges the two ends
of a pipe pair as asyncio :class:`StreamReader`/:class:`StreamWriter`
so the proxy loop can drive them naturally and cancel cleanly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from .model_options import iter_model_choice_dicts

if TYPE_CHECKING:
    from terok_sandbox import Sandbox

_logger = logging.getLogger(__name__)

DEFAULT_PROBE_TIMEOUT_SEC = 8.0
"""Per-call timeout for the full probe handshake.

Generous margin so a cold container exec doesn't trip the timeout.
Override per-call with the ``timeout`` parameter or globally via
``TEROK_ACP_PROBE_TIMEOUT_SECS``.
"""

ACP_PROTOCOL_VERSION = 1
"""ACP protocol version this proxy implements (matches the schema we
verified during design).  Bumped when ACP makes a breaking change."""


class ProbeError(RuntimeError):
    """Raised when an agent fails to respond to the probe handshake.

    The cache stores empty rosters for failed probes (so we don't hammer
    a misconfigured agent on every session) — callers should treat
    ``ProbeError`` as "this agent is currently unusable" rather than
    bubble it to the user.
    """


async def probe_agent_models(
    *,
    agent_id: str,
    container: Any,
    sandbox: Sandbox,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SEC,
    cwd: str = "/workspace",
) -> tuple[str, ...]:
    """Drive the minimal ACP handshake against ``terok-{agent_id}-acp``.

    Spawns the in-container wrapper via
    :meth:`terok_sandbox.ContainerRuntime.exec_stdio` (running in an
    executor thread because the primitive is sync), sends
    ``initialize`` and ``session/new``, parses the response for the
    ``category: "model"`` configOption, and returns the model ids.

    Raises :class:`ProbeError` on timeout, malformed JSON, or any
    other handshake failure.  Callers (the roster cache) typically
    catch it, cache an empty roster, and skip the agent until the
    container is restarted.
    """
    wrapper_cmd = [f"terok-{agent_id}-acp"]
    loop = asyncio.get_running_loop()

    # Two pipes: probe → child (host_in_w → host_in_r → child stdin)
    # and child → probe (host_out_w → host_out_r → child stdout).
    # Host-side ends are wrapped as asyncio streams so the readline
    # in ``_drive_handshake`` is cancellable (run_in_executor reads
    # are not).  Child-side ends go to the synchronous ``exec_stdio``
    # primitive which copies bytes via its own pump threads.
    host_in_r, host_in_w = os.pipe()
    host_out_r, host_out_w = os.pipe()

    write_pipe = os.fdopen(host_in_w, "wb", buffering=0)
    write_transport, write_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin,
        write_pipe,
    )
    writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)

    read_pipe = os.fdopen(host_out_r, "rb", buffering=0)
    reader = asyncio.StreamReader(loop=loop)
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader, loop=loop),
        read_pipe,
    )

    child_in = os.fdopen(host_in_r, "rb", buffering=0)
    child_out = os.fdopen(host_out_w, "wb", buffering=0)
    runtime = sandbox.runtime
    exec_future = loop.run_in_executor(
        None,
        lambda: runtime.exec_stdio(container, wrapper_cmd, stdin=child_in, stdout=child_out),
    )

    try:
        return await asyncio.wait_for(
            _drive_handshake(reader, writer, cwd=cwd, agent_id=agent_id),
            timeout=timeout,
        )
    except TimeoutError as exc:
        _logger.warning("ACP probe for agent %r timed out after %.1fs", agent_id, timeout)
        raise ProbeError(f"probe timed out for agent {agent_id!r}") from exc
    finally:
        # Close host-side streams; this closes the host-side pipe ends,
        # which signals EOF to the child.  The child's reader thread
        # ends, NullRuntime's script raises (mismatch) or returns, and
        # the exec future settles.
        try:
            writer.close()
        except Exception as exc:  # noqa: BLE001
            _logger.debug("ACP probe writer close: %s", exc)
        # Reader is closed implicitly when the read_pipe goes out of
        # scope; explicitly cancel any pending readline by closing.
        try:
            read_pipe.close()
        except OSError:
            pass
        # Don't hold up the caller indefinitely — the underlying
        # threads might be blocked in unkillable syscalls in pathological
        # cases.  The asyncio.run() shutdown will wait for them, but
        # caller-visible latency is bounded.
        try:
            await asyncio.wait_for(exec_future, timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            _logger.debug("ACP probe exec_future cleanup: %s", exc)


async def _drive_handshake(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    cwd: str,
    agent_id: str,
) -> tuple[str, ...]:
    """Send ``initialize`` + ``session/new`` and parse the model list."""

    async def _write_frame(payload: dict) -> None:
        writer.write((json.dumps(payload) + "\n").encode("utf-8"))
        await writer.drain()

    async def _read_response(*, expected_id: int) -> dict[str, Any]:
        # Skip notifications (no ``id``) so a chatty agent's progress
        # events don't get mistaken for the reply.  Out-of-order
        # response ids error out — queueing isn't worth it for a
        # handshake this short-lived.
        while True:
            line = await reader.readline()
            if not line:
                raise ProbeError(f"agent {agent_id!r} closed stdout before handshake completed")
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProbeError(f"agent {agent_id!r} sent malformed JSON during probe") from exc
            if not isinstance(frame, dict):
                raise ProbeError(f"agent {agent_id!r} sent a non-object JSON-RPC frame")
            frame_id = frame.get("id")
            if frame_id == expected_id:
                return frame
            if "method" in frame and frame_id is None:
                continue
            raise ProbeError(
                f"agent {agent_id!r} sent unexpected probe frame id {frame_id!r}, "
                f"expected {expected_id!r}"
            )

    # initialize ---------------------------------------------------------
    # Probe ids are local to this short-lived handshake — no shared id
    # space with the proxy, so plain integers are fine.
    await _write_frame(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "clientCapabilities": {},
            },
        }
    )
    init_response = await _read_response(expected_id=1)
    if "error" in init_response:
        raise ProbeError(f"agent {agent_id!r} rejected initialize: {init_response['error']}")

    # session/new --------------------------------------------------------
    await _write_frame(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": {"cwd": cwd, "mcpServers": []},
        }
    )
    new_response = await _read_response(expected_id=2)
    if "error" in new_response:
        raise ProbeError(f"agent {agent_id!r} rejected session/new: {new_response['error']}")

    return _extract_model_ids(new_response.get("result") or {})


def _extract_model_ids(session_new_result: dict) -> tuple[str, ...]:
    """Return the model ids from a ``session/new`` response.

    Walks the ``configOptions[category=model]`` entry via the shared
    :func:`iter_model_choice_dicts` iterator (so the schema-tolerance
    logic lives in one place — see :mod:`.proxy`).  Unknown shapes
    yield an empty tuple, which the caller caches to avoid hammering
    a misbehaving agent on every session.
    """
    out: list[str] = []
    for entry in iter_model_choice_dicts(session_new_result):
        ident = entry.get("id") or entry.get("value")
        if isinstance(ident, str):
            out.append(ident)
    return tuple(out)
