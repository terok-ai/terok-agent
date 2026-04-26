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
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terok_executor.roster.loader import AgentRoster

_logger = logging.getLogger(__name__)

_MANAGED_CONFIG_FILENAME = ".terok-managed-config.json"
"""Sidecar file that records config values last written by terok."""

_MANAGED_CONFIG_VERSION = 1
"""Schema version for :data:`_MANAGED_CONFIG_FILENAME`."""


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
    elif "toml_set" in patch:
        _apply_toml_patch(config_path, patch, location)

    print(f"Vault config written to {config_path}")


def apply_shared_config_patches(
    roster: AgentRoster,
    mounts_base: Path,
    *,
    providers: frozenset[str] | None = None,
    disabled_providers: frozenset[str] | None = None,
) -> None:
    """Reconcile ``shared_config_patch`` for enabled and disabled providers.

    Called during task start so shared mount directories (which may have
    been recreated empty) always contain the correct vault addresses.
    Idempotent: safe to call on every launch.  Disabled providers have
    previously managed values removed only when the live config still
    matches the sidecar value terok wrote last time; user-edited values
    are preserved and ownership is dropped.

    Args:
        roster: Loaded agent roster.
        mounts_base: Shared config mount root.
        providers:
            ``None`` means "all providers with a patch".  An empty set
            disables patching entirely.  A non-empty set restricts
            patching to that provider subset.
        disabled_providers:
            Provider subset whose previously managed patch values should
            be reconciled away.  ``None`` removes nothing; callers pass an
            explicit set when a feature mode disables provider routing.

    Raises [`ConfigPatchError`][terok_executor.credentials.vault_config.ConfigPatchError] on failure — callers must not start
    the container if vault routing cannot be established.
    """
    patched_routes = {
        name: route
        for name, route in roster.vault_routes.items()
        if route.shared_config_patch and (providers is None or name in providers)
    }
    disabled_routes = {
        name: route
        for name, route in roster.vault_routes.items()
        if route.shared_config_patch and disabled_providers and name in disabled_providers
    }

    for name in disabled_routes:
        auth_info = roster.auth_providers.get(name)
        if not auth_info:
            continue
        try:
            _remove_managed_patch_values(mounts_base / auth_info.host_dir_name, name)
            _logger.debug("Removed managed config patch for disabled provider %s", name)
        except ConfigPatchError:
            raise
        except Exception as exc:
            raise ConfigPatchError(f"Failed to remove vault config patch for {name}") from exc

    if not patched_routes:
        return

    location = resolve_vault_location()

    for name, route in patched_routes.items():
        auth_info = roster.auth_providers.get(name)
        if not auth_info:
            continue

        patch = route.shared_config_patch
        try:
            shared_dir = mounts_base / auth_info.host_dir_name
            shared_dir.mkdir(parents=True, exist_ok=True)
            config_path = _safe_config_path(shared_dir, patch["file"])

            if "yaml_set" in patch:
                records = _apply_yaml_patch(config_path, patch, location)
            elif "toml_set" in patch:
                records = _apply_toml_patch(config_path, patch, location)
            else:
                records = []
            _record_managed_patch_values(shared_dir, name, patch["file"], records)
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

    Raises [`ConfigPatchError`][terok_executor.credentials.vault_config.ConfigPatchError] if the resolved path escapes the
    intended directory (absolute paths, ``..`` components, symlinks).

    Note: this check is TOCTOU-racy against a container that can plant
    symlinks between the check here and the subsequent write.  Callers
    MUST use `_read_nofollow` / `_write_nofollow` to open
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
    `_safe_config_path` check and this read.  ``O_NOFOLLOW``
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

    A planted symlink at *path* is rejected with [`ConfigPatchError`][terok_executor.credentials.vault_config.ConfigPatchError]
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


def _delete_nofollow(path: Path) -> None:
    """Delete *path* without following symlinks; ignore missing files."""
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except IsADirectoryError as exc:
        raise ConfigPatchError(f"refusing to delete directory at {path}") from exc


def _substitute(value: object, location: VaultLocation) -> object:
    """Expand ``{vault_url}`` / ``{vault_socket}`` tokens in a patch value."""
    if not isinstance(value, str):
        return value
    return value.replace("{vault_url}", location.url).replace("{vault_socket}", location.socket)


def _empty_metadata() -> dict:
    """Return an empty managed-config sidecar payload."""
    return {"version": _MANAGED_CONFIG_VERSION, "files": {}}


def _managed_config_path(shared_dir: Path) -> Path:
    """Return the safe sidecar path inside *shared_dir*."""
    return _safe_config_path(shared_dir, _MANAGED_CONFIG_FILENAME)


def _load_metadata(shared_dir: Path) -> dict:
    """Load the managed-config sidecar, falling back to an empty payload."""
    path = _managed_config_path(shared_dir)
    raw = _read_nofollow(path)
    if raw is None:
        return _empty_metadata()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(
            f"Warning [vault-config]: failed to parse managed sidecar {path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return _empty_metadata()
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        return _empty_metadata()
    data["version"] = _MANAGED_CONFIG_VERSION
    return data


def _write_metadata(shared_dir: Path, metadata: dict) -> None:
    """Persist the managed-config sidecar, removing it when empty."""
    files = metadata.get("files")
    path = _managed_config_path(shared_dir)
    if not files:
        _delete_nofollow(path)
        return
    payload = {"version": _MANAGED_CONFIG_VERSION, "files": files}
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _write_nofollow(path, encoded)


def _record_managed_patch_values(
    shared_dir: Path, provider: str, filename: str, records: list[dict]
) -> None:
    """Remember which values *provider* owns in *filename*."""
    metadata = _load_metadata(shared_dir)
    files = metadata.setdefault("files", {})
    file_info = files.get(filename)
    if not isinstance(file_info, dict):
        file_info = {"providers": {}}
        files[filename] = file_info
    providers = file_info.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        file_info["providers"] = providers
    if records:
        providers[provider] = records
    else:
        providers.pop(provider, None)
    if not providers:
        files.pop(filename, None)
    _write_metadata(shared_dir, metadata)


def _remove_managed_patch_values(shared_dir: Path, provider: str) -> None:
    """Remove sidecar-owned config values for *provider* when still unchanged."""
    metadata = _load_metadata(shared_dir)
    files = metadata.setdefault("files", {})

    for filename, file_info in list(files.items()):
        if not isinstance(file_info, dict):
            files.pop(filename, None)
            continue
        providers = file_info.get("providers")
        if not isinstance(providers, dict) or provider not in providers:
            continue
        records = providers[provider]
        if isinstance(records, list):
            config_path = _safe_config_path(shared_dir, filename)
            if not _remove_records_from_config(config_path, records):
                continue
        providers.pop(provider, None)
        if not providers:
            files.pop(filename, None)

    _write_metadata(shared_dir, metadata)


def _read_toml_mapping(config_path: Path, *, warn_on_error: bool) -> dict | None:
    """Read TOML as a dict; return ``None`` on parse failure when requested."""
    import tomllib

    raw = _read_nofollow(config_path)
    if raw is None:
        return {}
    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except Exception as exc:
        if warn_on_error:
            print(
                f"Warning [vault-config]: failed to parse {config_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return {}
        print(
            f"Warning [vault-config]: cannot remove managed values from {config_path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
    return parsed if isinstance(parsed, dict) else {}


def _read_yaml_mapping(config_path: Path, *, warn_on_error: bool) -> dict | None:
    """Read YAML as a dict; return ``None`` on parse failure when requested."""
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    raw = _read_nofollow(config_path)
    if raw is None:
        return {}
    try:
        parsed = yaml.load(raw)
    except Exception as exc:
        if warn_on_error:
            print(
                f"Warning [vault-config]: failed to parse {config_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return {}
        print(
            f"Warning [vault-config]: cannot remove managed values from {config_path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
    return parsed if isinstance(parsed, dict) else {}


def _apply_toml_patch(config_path: Path, patch: dict, location: VaultLocation) -> list[dict]:
    """Patch top-level TOML keys or an array-of-tables entry."""
    existing = _read_toml_mapping(config_path, warn_on_error=True) or {}

    values = {k: _substitute(v, location) for k, v in patch["toml_set"].items()}
    if "toml_table" not in patch:
        existing.update(values)
        records = [{"kind": "toml_top", "values": values}]
    else:
        table_key = patch["toml_table"]
        match_criteria = patch["toml_match"]

        entries = existing.get(table_key, [])
        if not isinstance(entries, list):
            entries = []
        target = next(
            (
                e
                for e in entries
                if isinstance(e, dict) and all(e.get(k) == v for k, v in match_criteria.items())
            ),
            None,
        )
        created = target is None
        if target:
            target.update(values)
        else:
            entries.append({**match_criteria, **values})
            existing[table_key] = entries
        records = [
            {
                "kind": "toml_table",
                "table": table_key,
                "match": match_criteria,
                "values": values,
                "created": created,
            }
        ]

    import tomli_w

    _write_nofollow(config_path, tomli_w.dumps(existing).encode("utf-8"))
    return records


def _apply_yaml_patch(config_path: Path, patch: dict, location: VaultLocation) -> list[dict]:
    """Set top-level keys in a YAML config file."""
    import io

    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    existing = _read_yaml_mapping(config_path, warn_on_error=True) or {}

    values = {k: _substitute(v, location) for k, v in patch["yaml_set"].items()}
    for k in patch["yaml_set"]:
        existing[k] = values[k]

    buf = io.BytesIO()
    yaml.dump(existing, buf)
    _write_nofollow(config_path, buf.getvalue())
    return [{"kind": "yaml_top", "values": values}]


def _remove_records_from_config(config_path: Path, records: list[dict]) -> bool:
    """Remove managed records from *config_path* when live values still match."""
    toml_records = [record for record in records if str(record.get("kind", "")).startswith("toml")]
    yaml_records = [record for record in records if str(record.get("kind", "")).startswith("yaml")]
    ok = True
    if toml_records:
        ok = _remove_toml_records(config_path, toml_records) and ok
    if yaml_records:
        ok = _remove_yaml_records(config_path, yaml_records) and ok
    return ok


def _remove_toml_records(config_path: Path, records: list[dict]) -> bool:
    """Remove TOML sidecar-owned values if the current values still match."""
    existing = _read_toml_mapping(config_path, warn_on_error=False)
    if existing is None:
        return False
    changed = False

    for record in records:
        kind = record.get("kind")
        if kind == "toml_top":
            for key, value in dict(record.get("values", {})).items():
                if existing.get(key) == value:
                    existing.pop(key, None)
                    changed = True
        elif kind == "toml_table":
            table_key = record.get("table")
            match = dict(record.get("match", {}))
            values = dict(record.get("values", {}))
            entries = existing.get(table_key)
            if not isinstance(table_key, str) or not isinstance(entries, list):
                continue
            target = next(
                (
                    entry
                    for entry in entries
                    if isinstance(entry, dict)
                    and all(entry.get(key) == value for key, value in match.items())
                ),
                None,
            )
            if target is None:
                continue
            for key, value in values.items():
                if target.get(key) == value:
                    target.pop(key, None)
                    changed = True
            if record.get("created") and target == match:
                entries.remove(target)
                changed = True
            if not entries:
                existing.pop(table_key, None)

    if changed:
        import tomli_w

        _write_nofollow(config_path, tomli_w.dumps(existing).encode("utf-8"))
    return True


def _remove_yaml_records(config_path: Path, records: list[dict]) -> bool:
    """Remove YAML sidecar-owned top-level values when unchanged."""
    import io

    from ruamel.yaml import YAML

    existing = _read_yaml_mapping(config_path, warn_on_error=False)
    if existing is None:
        return False
    changed = False

    for record in records:
        if record.get("kind") != "yaml_top":
            continue
        for key, value in dict(record.get("values", {})).items():
            if existing.get(key) == value:
                existing.pop(key, None)
                changed = True

    if changed:
        yaml = YAML()
        yaml.preserve_quotes = True
        buf = io.BytesIO()
        yaml.dump(existing, buf)
        _write_nofollow(config_path, buf.getvalue())
    return True
