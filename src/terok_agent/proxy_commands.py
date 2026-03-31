# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Credential proxy CLI commands for terok-agent.

Wraps terok-sandbox proxy lifecycle with agent-level concerns: route
generation from the YAML roster is performed before ``start`` and
``install`` so the proxy always has up-to-date provider config.
"""

from __future__ import annotations

import sys
from pathlib import Path

from terok_sandbox import CommandDef


def _ensure_routes() -> Path:
    """Generate routes.json from the YAML agent roster."""
    from .roster import ensure_proxy_routes

    return ensure_proxy_routes()


def _handle_start() -> None:
    """Generate routes and start the credential proxy daemon."""
    from terok_sandbox import is_proxy_running, start_proxy

    if is_proxy_running():
        print("Credential proxy is already running.")
        sys.exit(1)
    _ensure_routes()
    start_proxy()
    print("Credential proxy started.")


def _handle_stop() -> None:
    """Stop the credential proxy daemon."""
    from terok_sandbox import is_proxy_running, stop_proxy

    if not is_proxy_running():
        print("Credential proxy is not running.")
        return
    stop_proxy()
    print("Credential proxy stopped.")


def scan_leaked_credentials(mounts_base: Path) -> list[tuple[str, Path]]:
    """Return ``(provider, host_path)`` for credential files found in shared mounts.

    When the credential proxy is active, real secrets should only live in the
    proxy's sqlite DB — not in the shared config directories that get mounted
    into containers.  This function checks each routed provider's mount for
    credential files that would leak real tokens alongside phantom ones.
    """
    from .roster import get_roster

    roster = get_roster()
    leaked: list[tuple[str, Path]] = []
    for name, route in roster.proxy_routes.items():
        if not route.credential_file:
            continue
        auth = roster.auth_providers.get(name)
        if not auth:
            continue
        try:
            path = mounts_base / auth.host_dir_name / route.credential_file
            if path.is_file() and path.stat().st_size > 0:
                leaked.append((name, path))
        except (OSError, TypeError):
            continue
    return leaked


def _handle_status() -> None:
    """Show credential proxy status."""
    from terok_sandbox import get_proxy_status, is_proxy_systemd_available

    from .paths import mounts_dir

    status = get_proxy_status()
    state = "running" if status.running else "stopped"
    print(f"Mode:        {status.mode}")
    print(f"Status:      {state}")
    print(f"Socket:      {status.socket_path}")
    print(f"DB:          {status.db_path}")
    print(f"Routes:      {status.routes_path} ({status.routes_configured} configured)")
    if status.credentials_stored:
        print(f"Credentials: {', '.join(status.credentials_stored)}")
    else:
        print("Credentials: none stored")
    if not status.running and status.mode == "none" and is_proxy_systemd_available():
        print("\nHint: run 'install' to set up systemd socket activation.")

    leaked = scan_leaked_credentials(mounts_dir())
    if leaked:
        print("\nWARNING: Real credentials found in shared config mounts:")
        for provider, path in leaked:
            print(f"  {provider}: {path}")
        print("These files are mounted into containers alongside proxy phantom tokens.")
        print("Run 'clean' to remove them.")


def _handle_install() -> None:
    """Generate routes and install systemd socket activation."""
    from terok_sandbox import install_proxy_systemd, is_proxy_systemd_available

    if not is_proxy_systemd_available():
        print(
            "Error: systemd user services are not available on this host.\n"
            "Use 'start' to run the proxy without systemd."
        )
        sys.exit(1)
    _ensure_routes()
    install_proxy_systemd()
    print("Credential proxy systemd socket installed and started.")


def _handle_uninstall() -> None:
    """Remove credential proxy systemd units."""
    from terok_sandbox import is_proxy_systemd_available, uninstall_proxy_systemd

    if not is_proxy_systemd_available():
        print("Error: systemd user services are not available. Nothing to uninstall.")
        sys.exit(1)
    uninstall_proxy_systemd()
    print("Credential proxy systemd units removed.")


def _handle_routes() -> None:
    """Regenerate routes.json from the YAML agent roster."""
    path = _ensure_routes()
    if path:
        print(f"Routes written to {path}")


def _handle_clean() -> None:
    """Remove leaked credential files from shared config mounts."""
    from .paths import mounts_dir

    leaked = scan_leaked_credentials(mounts_dir())
    if not leaked:
        print("No leaked credential files found.")
        return
    for provider, path in leaked:
        path.unlink()
        print(f"Removed {provider}: {path}")


PROXY_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="start", help="Start the credential proxy daemon", handler=_handle_start, group="proxy"
    ),
    CommandDef(
        name="stop", help="Stop the credential proxy daemon", handler=_handle_stop, group="proxy"
    ),
    CommandDef(
        name="status", help="Show credential proxy status", handler=_handle_status, group="proxy"
    ),
    CommandDef(
        name="install",
        help="Install systemd socket activation",
        handler=_handle_install,
        group="proxy",
    ),
    CommandDef(
        name="uninstall", help="Remove systemd units", handler=_handle_uninstall, group="proxy"
    ),
    CommandDef(
        name="routes",
        help="Regenerate routes.json from YAML roster",
        handler=_handle_routes,
        group="proxy",
    ),
    CommandDef(
        name="clean",
        help="Remove leaked credential files from shared mounts",
        handler=_handle_clean,
        group="proxy",
    ),
)
