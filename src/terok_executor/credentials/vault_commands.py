# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Manages the vault daemon lifecycle via CLI subcommands.

Wraps terok-sandbox vault lifecycle with agent-level concerns: route
generation from the YAML roster is performed before ``start`` and
``install`` so the vault always has up-to-date provider config.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from terok_sandbox import CommandDef

if TYPE_CHECKING:
    from terok_sandbox import SandboxConfig


def _ensure_routes(cfg: SandboxConfig | None = None) -> Path:
    """Generate routes.json from the YAML agent roster."""
    from terok_executor.roster.loader import ensure_vault_routes

    return ensure_vault_routes(cfg=cfg)


def _handle_start(*, cfg: SandboxConfig | None = None) -> None:
    """Generate routes and start the vault daemon."""
    from terok_sandbox import is_vault_running, start_vault

    if is_vault_running(cfg=cfg):
        print("Vault is already running.")
        sys.exit(1)
    _ensure_routes(cfg=cfg)
    start_vault(cfg=cfg)
    print("Vault started.")


def _handle_stop(*, cfg: SandboxConfig | None = None) -> None:
    """Stop the vault daemon."""
    from terok_sandbox import is_vault_running, stop_vault

    if not is_vault_running(cfg=cfg):
        print("Vault is not running.")
        return
    stop_vault(cfg=cfg)
    print("Vault stopped.")


def _is_injected_credentials_file(path: Path) -> bool:
    """Check whether *path* is a terok-injected ``.credentials.json``.

    Returns ``True`` only when **all** of these hold:

    - ``claudeAiOauth.accessToken`` equals `PHANTOM_CREDENTIALS_MARKER`
    - ``claudeAiOauth.refreshToken`` is empty or absent

    Any parse error or unexpected structure → ``False`` (treat as real leak).
    """
    import json

    from terok_sandbox import PHANTOM_CREDENTIALS_MARKER

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        oauth = data.get("claudeAiOauth", {})
        if not isinstance(oauth, dict):
            return False
        return oauth.get("accessToken") == PHANTOM_CREDENTIALS_MARKER and not oauth.get(
            "refreshToken"
        )
    except (json.JSONDecodeError, OSError, ValueError):
        return False


def _is_injected_codex_auth_file(path: Path) -> bool:
    """Check whether *path* is a terok-injected shared Codex ``auth.json``."""
    import json

    from terok_sandbox import CODEX_SHARED_OAUTH_MARKER

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        tokens = data.get("tokens", {})
        if not isinstance(tokens, dict):
            return False
        return (
            tokens.get("access_token") == CODEX_SHARED_OAUTH_MARKER
            and tokens.get("refresh_token") == CODEX_SHARED_OAUTH_MARKER
        )
    except (json.JSONDecodeError, OSError, ValueError):
        return False


def scan_leaked_credentials(mounts_base: Path) -> list[tuple[str, Path]]:
    """Return ``(provider, host_path)`` for credential files found in shared mounts.

    When the vault is active, real secrets should only live in the
    vault's sqlite DB — not in the shared config directories that get mounted
    into containers.  This function checks each routed provider's mount for
    credential files that would leak real tokens alongside phantom ones.

    Files injected by `_write_claude_credentials_file`
    are recognised by their dummy ``accessToken`` marker and skipped.

    Symlinks are rejected to prevent a container from tricking the scan into
    reading arbitrary host files via a crafted symlink in the shared mount.
    """
    import stat

    from terok_executor.roster.loader import get_roster

    roster = get_roster()
    base_resolved = mounts_base.resolve(strict=False)
    leaked: list[tuple[str, Path]] = []
    for name, route in roster.vault_routes.items():
        if not route.credential_file:
            continue
        auth = roster.auth_providers.get(name)
        if not auth:
            continue
        try:
            path = mounts_base / auth.host_dir_name / route.credential_file
            # lstat: do not follow symlinks — reject them outright
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                continue
            # Ensure resolved path stays within the mounts base
            if base_resolved not in path.resolve(strict=True).parents:
                continue
            if st.st_size > 0 and not (
                _is_injected_credentials_file(path) or _is_injected_codex_auth_file(path)
            ):
                leaked.append((name, path))
        except (OSError, TypeError):
            continue
    return leaked


def _format_credentials(status: object) -> str:
    """Format stored credentials as ``name (type), ...`` for status display."""
    from terok_sandbox import CredentialDB, VaultStatus

    st: VaultStatus = status  # type: ignore[assignment]
    if not st.credentials_stored:
        return "none stored"
    try:
        db = CredentialDB(st.db_path)
        try:
            parts = []
            for name in st.credentials_stored:
                cred = db.load_credential("default", name)
                ctype = cred.get("type", "unknown") if cred else "unknown"
                parts.append(f"{name} ({ctype})")
        finally:
            db.close()
        return ", ".join(parts)
    except Exception:  # noqa: BLE001
        print(
            "Warning [vault]: credential type lookup failed; showing names only",
            file=sys.stderr,
        )
        return ", ".join(st.credentials_stored)


def _handle_status(*, cfg: SandboxConfig | None = None) -> None:
    """Show vault status."""
    from terok_sandbox import get_vault_status, is_vault_systemd_available

    from terok_executor.paths import mounts_dir

    status = get_vault_status(cfg=cfg)
    state = "running" if status.running else "stopped"
    print(f"Mode:        {status.mode}")
    print(f"Status:      {state}")
    print(f"Socket:      {status.socket_path}")
    print(f"DB:          {status.db_path}")
    print(f"Routes:      {status.routes_path} ({status.routes_configured} configured)")
    if status.credentials_stored:
        print(f"Credentials: {_format_credentials(status)}")
    else:
        print("Credentials: none stored")
    if not status.running and status.mode == "none" and is_vault_systemd_available():
        print("\nHint: run 'install' to set up systemd socket activation.")

    leaked = scan_leaked_credentials(mounts_dir())
    if leaked:
        print("\nWARNING: Real credentials found in shared config mounts:")
        for provider, path in leaked:
            print(f"  {provider}: {path}")
        print("These files are mounted into containers alongside vault phantom tokens.")
        print("Run 'clean' to remove them.")


def _handle_install(*, cfg: SandboxConfig | None = None) -> None:
    """Generate routes and install systemd socket activation."""
    from terok_sandbox import install_vault_systemd, is_vault_systemd_available

    if not is_vault_systemd_available():
        print(
            "Error: systemd user services are not available on this host.\n"
            "Use 'start' to run the vault without systemd."
        )
        sys.exit(1)
    _ensure_routes(cfg=cfg)
    install_vault_systemd(cfg=cfg)
    print("Vault installed via systemd socket activation.")


def _handle_uninstall(*, cfg: SandboxConfig | None = None) -> None:
    """Remove vault systemd units."""
    from terok_sandbox import is_vault_systemd_available, uninstall_vault_systemd

    if not is_vault_systemd_available():
        print("Error: systemd user services are not available. Nothing to uninstall.")
        sys.exit(1)
    uninstall_vault_systemd(cfg=cfg)
    print("Vault systemd units removed.")


def _handle_routes(*, cfg: SandboxConfig | None = None) -> None:
    """Regenerate routes.json from the YAML agent roster."""
    path = _ensure_routes(cfg=cfg)
    if path:
        print(f"Routes written to {path}")


def _handle_clean(*, cfg: SandboxConfig | None = None) -> None:  # noqa: ARG001
    """Remove leaked credential files from shared config mounts."""
    from terok_executor.paths import mounts_dir

    leaked = scan_leaked_credentials(mounts_dir())
    if not leaked:
        print("No leaked credential files found.")
        return
    for provider, path in leaked:
        path.unlink()
        print(f"Removed {provider}: {path}")


VAULT_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(name="start", help="Start the vault daemon", handler=_handle_start, group="vault"),
    CommandDef(name="stop", help="Stop the vault daemon", handler=_handle_stop, group="vault"),
    CommandDef(name="status", help="Show vault status", handler=_handle_status, group="vault"),
    CommandDef(
        name="install",
        help="Install systemd socket activation",
        handler=_handle_install,
        group="vault",
    ),
    CommandDef(
        name="uninstall", help="Remove systemd units", handler=_handle_uninstall, group="vault"
    ),
    CommandDef(
        name="routes",
        help="Regenerate routes.json from YAML roster",
        handler=_handle_routes,
        group="vault",
    ),
    CommandDef(
        name="clean",
        help="Remove leaked credential files from shared mounts",
        handler=_handle_clean,
        group="vault",
    ),
)
