# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-container ACP host-proxy daemon.

Binds a Unix socket on the host that aggregates a container's
in-image ACP agents (``terok-{agent}-acp`` wrappers) behind a single
endpoint.  Lifetime is tied to the container: the daemon polls
``runtime.container(name).state`` and exits cleanly once the
container is gone.

Inputs are container-shaped — no notion of projects, tasks, or
project IDs lives here.  Higher orchestrators (terok) translate
their own concepts to ``(container_name, socket_path)`` and spawn
this daemon as a subprocess (or call :func:`serve_acp` directly).

Standalone use::

    terok-executor acp <container_name> <socket_path>

The argv form mirrors the library entry point so consumers can pick
either depending on whether they want a subprocess or an in-process
daemon.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import sys
from pathlib import Path

from terok_sandbox import Sandbox, SandboxConfig

from .roster import ACPRoster

_logger = logging.getLogger(__name__)

CONTAINER_POLL_INTERVAL_SEC = 2.0
"""How often the daemon checks whether the container is still alive.

Tuned to balance ``acp list`` freshness against polling overhead.
"""


def acp_socket_is_live(path: Path) -> bool:
    """Return ``True`` when a peer is currently accepting on *path*.

    Distinguishes a live ACP daemon from a stale socket file left
    behind by a crash: a successful ``connect`` means a peer is
    listening, while ``ECONNREFUSED`` (and any other ``OSError``)
    means the file is safe to unlink.
    """
    if not path.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            probe.connect(str(path))
    except OSError:
        return False
    return True


def serve_acp(
    container_name: str,
    socket_path: Path,
    *,
    sandbox: Sandbox | None = None,
    poll_interval_sec: float = CONTAINER_POLL_INTERVAL_SEC,
) -> int:
    """Bind *socket_path* and run the ACP host-proxy until *container_name* stops.

    Returns the process exit code.  When *sandbox* is ``None``, builds
    one from the layered ``config.yml`` + env (the same
    ``SandboxConfig()`` defaults executor uses everywhere).
    """
    if sandbox is None:
        sandbox = Sandbox(config=SandboxConfig())
    return asyncio.run(_run(container_name, socket_path, sandbox, poll_interval_sec))


def main(argv: list[str] | None = None) -> int:
    """Argv entry — ``terok-executor acp <container_name> <socket_path>``."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(
            "usage: terok-executor acp <container_name> <socket_path>",
            file=sys.stderr,
        )
        return 2
    cname, sock_str = args
    logging.basicConfig(level=logging.INFO, format="acp[%(levelname)s] %(message)s")
    return serve_acp(cname, Path(sock_str))


# ── Internals ─────────────────────────────────────────────────────────────


async def _run(
    container_name: str,
    socket_path: Path,
    sandbox: Sandbox,
    poll_interval_sec: float,
) -> int:
    """Bind, accept, supervise, clean up.  Always exits cleanly."""
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    # Don't clobber a live socket: probe before unlinking.  Two spawn
    # attempts can race; the second would otherwise unlink the first
    # one's freshly bound socket.  If the probe connects, a peer is
    # already serving — exit cleanly; otherwise the file is stale.
    if socket_path.exists():
        if acp_socket_is_live(socket_path):
            _logger.info(
                "ACP proxy already active at %s for container=%s — exiting",
                socket_path,
                container_name,
            )
            return 0
        socket_path.unlink(missing_ok=True)

    container = sandbox.runtime.container(container_name)
    image = container.image
    if image is None:
        _logger.error("container %r has no image — aborting", container_name)
        return 1
    image_id = image.id or image.ref

    roster = ACPRoster(
        container_name=container_name,
        image_id=image_id,
        sandbox=sandbox,
    )

    # Avoid a world-readable socket: umask off group/other before bind.
    old_umask = os.umask(0o077)
    try:
        server = await asyncio.start_unix_server(
            _make_handler(roster),
            path=str(socket_path),
        )
    finally:
        os.umask(old_umask)
    _logger.info("ACP proxy listening at %s for container=%s", socket_path, container_name)

    stop_event = asyncio.Event()

    def _request_stop(*_: object) -> None:
        """Signal the main loop to exit cleanly (SIGTERM/SIGINT handler)."""
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:  # pragma: no cover — non-POSIX
            pass

    supervisor = asyncio.create_task(
        _watch_container(sandbox, container_name, stop_event, poll_interval_sec)
    )
    try:
        await stop_event.wait()
    finally:
        supervisor.cancel()
        server.close()
        await server.wait_closed()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        _logger.info("ACP proxy for container=%s exited cleanly", container_name)
    return 0


def _make_handler(roster: ACPRoster):
    """Return an ``asyncio.start_unix_server`` callback bound to *roster*.

    No daemon-level concurrency lock: the previous v1 single-client
    guard used an :class:`asyncio.Lock` plus ``writer.wait_closed()``,
    but the liveness probe in :func:`acp_socket_is_live` (the same
    one ``terok acp connect``'s ``_wait_for_socket`` calls before
    bridging) opens and immediately closes a real socket connection.
    That connection runs through the handler too and ends with
    ``wait_closed()`` — which doesn't always fire its
    ``connection_lost`` callback on Unix-socket half-closes — so the
    lock would stay held and the *real* bridge client got reject-and-
    RST.  Each connection now gets its own :class:`ACPProxy` and runs
    independently; concurrent ACP clients are unusual in practice
    and would only race over the backend agent, which the proxy's
    own ``_client_session_id`` check already rejects at the protocol
    level.
    """

    async def _handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Per-connection handler — runs the proxy attach loop for one client."""
        try:
            await roster.attach(reader, writer)
        except Exception:
            # Attach-loop crashes are bugs, not "expected" disconnect
            # paths — surface with traceback at error level so the
            # daemon log makes the cause obvious.
            _logger.exception("proxy: attach loop crashed")
        finally:
            # Fire-and-forget close: ``wait_closed`` can hang on Unix-
            # socket half-closes (the bug that motivated removing the
            # outer lock), and we have nothing to wait for here — the
            # proxy already ``drain``'d every ``_send_to_client`` call.
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001
                _logger.debug("proxy: writer close error: %s", exc)

    return _handler


async def _watch_container(
    sandbox: Sandbox,
    container_name: str,
    stop_event: asyncio.Event,
    poll_interval_sec: float,
) -> None:
    """Set *stop_event* when the container is no longer running.

    Polls until the state is not ``"running"`` (covers both
    ``exited`` and the no-such-container case).
    """
    while not stop_event.is_set():
        try:
            state = sandbox.runtime.container(container_name).state
        except Exception as exc:  # noqa: BLE001
            _logger.warning("proxy: container state probe failed: %s", exc)
            stop_event.set()
            return
        if state != "running":
            _logger.info(
                "proxy: container %s state=%r — shutting down",
                container_name,
                state,
            )
            stop_event.set()
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_sec)
        except TimeoutError:
            continue


if __name__ == "__main__":  # pragma: no cover — module entry point
    sys.exit(main())
