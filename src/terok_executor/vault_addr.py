# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container-side vault addressing — single source of truth.

Two transports reach the vault from inside a task container:

- **Socket mode** (default, preferred): the host's vault socket is
  bind-mounted into the container at :data:`CONTAINER_VAULT_SOCKET`.
  Clients that can speak HTTP-over-UNIX (``gh``, ``claude``) connect
  directly; everyone else goes through an in-container socat bridge that
  exposes the vault as plain TCP on ``localhost:LOOPBACK_VAULT_PORT``.

- **TCP mode** (legacy): the vault's token broker listens on a host TCP
  port (``host.containers.internal:<broker_port>``).  Socket-only
  clients reach it via a local socat bridge that presents a Unix socket
  at :data:`LOOPBACK_BRIDGE_SOCKET`.

These paths and port numbers have to agree across the python builders,
the shell bridge script, and the doctor probes — define them here so
every layer imports the same value.
"""

from __future__ import annotations

from terok_sandbox import CONTAINER_RUNTIME_DIR

CONTAINER_VAULT_SOCKET = f"{CONTAINER_RUNTIME_DIR}/vault.sock"
"""Host vault socket as seen inside the container (socket mode only)."""

LOOPBACK_BRIDGE_SOCKET = "/tmp/terok-vault.sock"  # nosec B108 — container-only path
"""Local socat-created socket that fronts the broker in TCP mode."""

LOOPBACK_VAULT_PORT = 9419
"""TCP port the in-container loopback bridge listens on (socket mode).

Sits next to the gate bridge on 9418; picked once, hardcoded everywhere.
"""

VAULT_LOOPBACK_PORT_ENV = "TEROK_VAULT_LOOPBACK_PORT"
"""Env var name the bridge script reads to find the loopback port."""
