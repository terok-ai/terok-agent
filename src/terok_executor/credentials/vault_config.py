# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Patches provider config files to route API traffic through the vault.

Applies ``shared_config_patch`` from the YAML roster after authentication
and — crucially — on every task start.  Writes vault URLs / socket paths
(not secrets) to provider config files so agents route traffic through the
vault instead of hitting upstream directly with phantom tokens.

Two template tokens are substituted into patch values:

- ``{vault_url}``    — HTTP URL the container should reach the vault on.
- ``{vault_socket}`` — filesystem path of a Unix socket the container can
  connect to for the vault.

The concrete values are mode-dependent (socket vs TCP transport) and
resolved centrally — agent YAMLs only need to reference the tokens.
"""

from __future__ import annotations

import errno
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terok_executor.roster.loader import AgentRoster

_logger = logging.getLogger(__name__)


class ConfigPatchError(RuntimeError):
    """Raised when a shared config patch fails and the task must not start."""


@dataclass(frozen=True, slots=True)
class VaultLocation:
    """Container-side addresses of the vault in both transports.

    One or both fields are set depending on the active transport:

    - Socket mode: *socket* points at the mounted host socket; *url* points
      at the in-container TCP→UNIX loopback bridge for HTTP-only clients.
    - TCP mode: *url* points at ``host.containers.internal:<broker_port>``;
      *socket* points at a local socat bridge that forwards to the same
      broker over TCP (for clients that can only speak HTTP-over-UNIX).
    """

    url: str
    """Base URL an in-container HTTP client should use (always non-empty)."""

    socket: str
    """Filesystem path for a Unix-socket-speaking HTTP client."""


def write_vault_config(provider_name: str) -> None:
    """Apply ``shared_config_patch`` from the YAML roster after auth.

    Patches a TOML or YAML config file in the provider's shared config dir
    to redirect API traffic through the vault.  The patch spec is declared
    in the agent YAML — no provider-specific code needed.
    """
    from terok_executor.roster.loader import get_roster

    roster = get_roster()
    route = roster.vault_routes.get(provider_name)
    if not route or not route.shared_config_patch:
        return

    auth_info = roster.auth_providers.get(provider_name)
    if not auth_info:
        return

    from terok_executor.paths import mounts_dir

    location = resolve_vault_location()

    patch = route.shared_config_patch
    shared_dir = mounts_dir() / auth_info.host_dir_name
    shared_dir.mkdir(parents=True, exist_ok=True)
    config_path = _safe_config_path(shared_dir, patch["file"])

    if "yaml_set" in patch:
        _apply_yaml_patch(config_path, patch, location)
    elif "toml_table" in patch:
        _apply_toml_patch(config_path, patch, location)

    print(f"Vault config written to {config_path}")


def apply_shared_config_patches(roster: AgentRoster, mounts_base: Path) -> None:
    """Re-apply every ``shared_config_patch`` for the whole roster.

    Called during task start so shared mount directories (which may have
    been recreated empty) always contain the correct vault addresses.
    Idempotent: safe to call on every launch.

    Raises :class:`ConfigPatchError` on failure — callers must not start
    the container if vault routing cannot be established.
    """
    location = resolve_vault_location()

    for name, route in roster.vault_routes.items():
        if not route.shared_config_patch:
            continue
        auth_info = roster.auth_providers.get(name)
        if not auth_info:
            continue

        patch = route.shared_config_patch
        try:
            shared_dir = mounts_base / auth_info.host_dir_name
            shared_dir.mkdir(parents=True, exist_ok=True)
            config_path = _safe_config_path(shared_dir, patch["file"])

            if "yaml_set" in patch:
                _apply_yaml_patch(config_path, patch, location)
            elif "toml_table" in patch:
                _apply_toml_patch(config_path, patch, location)
            _logger.debug("Applied config patch for %s → %s", name, config_path)
        except ConfigPatchError:
            raise
        except Exception as exc:
            raise ConfigPatchError(
                f"Failed to apply vault config patch for {name} (file={patch.get('file')!r})"
            ) from exc


def resolve_vault_location() -> VaultLocation:
    """Resolve the container-side vault addresses for the active transport.

    Reads the sandbox config once to decide whether we're in socket or TCP
    mode.  Exposed as a public helper so the env builder can use the same
    values it later writes to config files.
    """
    from terok_sandbox import SandboxConfig, get_token_broker_port

    from terok_executor.vault_addr import (
        CONTAINER_VAULT_SOCKET,
        LOOPBACK_BRIDGE_SOCKET,
        LOOPBACK_VAULT_PORT,
    )

    port = get_token_broker_port(SandboxConfig())
    if port is None:
        # Socket mode: container mounts the host vault socket directly; the
        # loopback bridge serves clients that can only speak HTTP-over-TCP.
        return VaultLocation(
            url=f"http://localhost:{LOOPBACK_VAULT_PORT}",
            socket=CONTAINER_VAULT_SOCKET,
        )
    # TCP mode: direct broker on the host; socat on the container side turns
    # a local Unix socket into a TCP connection for socket-only clients.
    return VaultLocation(
        url=f"http://host.containers.internal:{port}",
        socket=LOOPBACK_BRIDGE_SOCKET,
    )


# ── Private helpers ──────────────────────────────────────────────────────


def _safe_config_path(shared_dir: Path, filename: str) -> Path:
    """Resolve *filename* inside *shared_dir*, rejecting traversal attempts.

    Raises :class:`ConfigPatchError` if the resolved path escapes the
    intended directory (absolute paths, ``..`` components, symlinks).

    Note: this check is TOCTOU-racy against a container that can plant
    symlinks between the check here and the subsequent write.  Callers
    MUST use :func:`_read_nofollow` / :func:`_write_nofollow` to open
    the final file, so a symlink planted in the race window is rejected
    at open() time (``ELOOP``) instead of being silently followed.
    """
    rel = Path(filename)
    if rel.is_absolute() or ".." in rel.parts:
        raise ConfigPatchError(f"invalid patch file path: {filename!r}")

    target = (shared_dir / rel).resolve(strict=False)
    base = shared_dir.resolve(strict=True)
    if base not in target.parents and target != base:
        raise ConfigPatchError(f"patch target {target} escapes shared dir {base}")
    return target


def _read_nofollow(path: Path) -> bytes | None:
    """Read *path* refusing to follow symlinks; return ``None`` if missing.

    The shared config directories are bind-mounted read-write into task
    containers, so an attacker can plant a symlink between the
    :func:`_safe_config_path` check and this read.  ``O_NOFOLLOW``
    rejects that at open() time.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ConfigPatchError(f"refusing to read through symlink at {path}") from exc
        raise
    try:
        chunks: list[bytes] = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _write_nofollow(path: Path, data: bytes) -> None:
    """Write *data* to *path* refusing to follow symlinks.

    A planted symlink at *path* is rejected with :class:`ConfigPatchError`
    (via ``ELOOP``) rather than silently followed — protecting against a
    compromised container redirecting the executor's write to an
    arbitrary operator-owned file (CWE-367 / CWE-59).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o644)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ConfigPatchError(f"refusing to write through symlink at {path}") from exc
        raise
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _substitute(value: object, location: VaultLocation) -> object:
    """Expand ``{vault_url}`` / ``{vault_socket}`` tokens in a patch value."""
    if not isinstance(value, str):
        return value
    return value.replace("{vault_url}", location.url).replace("{vault_socket}", location.socket)


def _apply_toml_patch(config_path: Path, patch: dict, location: VaultLocation) -> None:
    """Patch a TOML array-of-tables entry."""
    import tomllib

    raw = _read_nofollow(config_path)
    if raw is not None:
        try:
            existing = tomllib.loads(raw.decode("utf-8"))
        except Exception as exc:
            print(
                f"Warning [vault-config]: failed to parse {config_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            existing = {}
    else:
        existing = {}

    table_key = patch["toml_table"]
    match_criteria = patch["toml_match"]
    values = {k: _substitute(v, location) for k, v in patch["toml_set"].items()}

    entries = existing.get(table_key, [])
    target = next(
        (e for e in entries if all(e.get(k) == v for k, v in match_criteria.items())),
        None,
    )
    if target:
        target.update(values)
    else:
        entries.append({**match_criteria, **values})
        existing[table_key] = entries

    import tomli_w

    _write_nofollow(config_path, tomli_w.dumps(existing).encode("utf-8"))


def _apply_yaml_patch(config_path: Path, patch: dict, location: VaultLocation) -> None:
    """Set top-level keys in a YAML config file."""
    import io

    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    raw = _read_nofollow(config_path)
    if raw is not None:
        try:
            existing = yaml.load(raw)
        except Exception as exc:
            print(
                f"Warning [vault-config]: failed to parse {config_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            existing = {}
    else:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}

    for k, v in patch["yaml_set"].items():
        existing[k] = _substitute(v, location)

    buf = io.BytesIO()
    yaml.dump(existing, buf)
    _write_nofollow(config_path, buf.getvalue())
