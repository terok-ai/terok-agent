# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Model-roster probe for in-container ACP agents.

Each in-container agent ships an ACP wrapper script (``terok-{agent}-acp``)
that exposes the agent over JSON-RPC on stdio.  To learn which models an
agent currently advertises, we drive a minimal handshake:

1. ``initialize`` — version negotiation
2. ``session/new`` — receive ``configOptions`` (the model list lives here)
3. close stdin — agent exits cleanly

The handshake is cheap (a few round-trips, hundreds of milliseconds at
most) but non-trivial to repeat: the result is cached by
:class:`~terok_executor.acp.cache.AgentRosterCache` and reused for the
lifetime of the authenticated session.

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

if TYPE_CHECKING:
    from terok_sandbox import Sandbox

_logger = logging.getLogger(__name__)

DEFAULT_PROBE_TIMEOUT_SEC = 8.0
"""Per-call timeout for the full probe handshake.

Empirically claude-agent-acp answers ``initialize`` in well under a
second; eight seconds gives a generous margin for cold container exec
without making startups feel sticky.  Override per-call with the
``timeout`` parameter or globally via ``TEROK_ACP_PROBE_TIMEOUT_SECS``.
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

    Returns an empty tuple on timeout or malformed responses — the
    caller is expected to cache that and skip the agent until restart.
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
        _run_exec_stdio,
        runtime,
        container,
        wrapper_cmd,
        child_in,
        child_out,
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
        except (TimeoutError, Exception) as exc:  # noqa: BLE001
            _logger.debug("ACP probe exec_future cleanup: %s", exc)


def _run_exec_stdio(runtime: Any, container: Any, cmd: list[str], stdin: Any, stdout: Any) -> int:
    """Bridge the runtime's sync ``exec_stdio`` into an executor thread."""
    return runtime.exec_stdio(container, cmd, stdin=stdin, stdout=stdout)


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

    async def _read_frame() -> dict:
        line = await reader.readline()
        if not line:
            raise ProbeError(f"agent {agent_id!r} closed stdout before handshake completed")
        return json.loads(line)

    # initialize ---------------------------------------------------------
    await _write_frame(
        {
            "jsonrpc": "2.0",
            "id": "px-init",
            "method": "initialize",
            "params": {
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "clientCapabilities": {},
            },
        }
    )
    init_response = await _read_frame()
    if "error" in init_response:
        raise ProbeError(f"agent {agent_id!r} rejected initialize: {init_response['error']}")

    # session/new --------------------------------------------------------
    await _write_frame(
        {
            "jsonrpc": "2.0",
            "id": "px-session-new",
            "method": "session/new",
            "params": {"cwd": cwd, "mcpServers": []},
        }
    )
    new_response = await _read_frame()
    if "error" in new_response:
        raise ProbeError(f"agent {agent_id!r} rejected session/new: {new_response['error']}")

    return _extract_model_ids(new_response.get("result") or {})


def _extract_model_ids(session_new_result: dict) -> tuple[str, ...]:
    """Return the model ids from a ``session/new`` response.

    Tolerates both currently-known shapes for the ACP model selector:
    a top-level ``availableModels`` array (older drafts) and a
    ``configOptions[*]`` entry with ``category == "model"`` (current
    draft).  Unknown shapes yield an empty tuple — callers cache that
    and skip the agent.
    """
    # Newer "configOptions" shape
    options = session_new_result.get("configOptions") or []
    for opt in options:
        if not isinstance(opt, dict):
            continue
        if opt.get("category") != "model":
            continue
        # Try several reasonable nestings — ACP's exact schema is still
        # in flux; we want to keep working as it stabilises.
        models: list[str] = []
        for choices_key in ("options", "values", "choices"):
            choices = (
                (opt.get("select") or {}).get(choices_key)
                if "select" in opt
                else opt.get(choices_key)
            )
            if isinstance(choices, list):
                for entry in choices:
                    if isinstance(entry, str):
                        models.append(entry)
                    elif isinstance(entry, dict):
                        ident = entry.get("id") or entry.get("value")
                        if isinstance(ident, str):
                            models.append(ident)
                if models:
                    return tuple(models)

    # Legacy "availableModels" shape (older drafts; keep for a release
    # or two until we're sure no agent emits it).
    legacy = session_new_result.get("availableModels") or []
    if isinstance(legacy, list):
        out: list[str] = []
        for entry in legacy:
            if isinstance(entry, str):
                out.append(entry)
            elif isinstance(entry, dict):
                ident = entry.get("id") or entry.get("value")
                if isinstance(ident, str):
                    out.append(ident)
        if out:
            return tuple(out)

    return ()
