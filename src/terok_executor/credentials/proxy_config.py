# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Patches provider config files to route API traffic through the credential proxy.

Applies ``shared_config_patch`` from the YAML roster after authentication
and — crucially — on every task start.  Writes proxy URLs (not secrets) to
provider config files so that agents route API traffic through the
credential proxy instead of hitting upstream directly with phantom tokens.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terok_executor.roster.loader import AgentRoster

_logger = logging.getLogger(__name__)


class ConfigPatchError(RuntimeError):
    """Raised when a shared config patch fails and the task must not start."""


def write_proxy_config(provider_name: str) -> None:
    """Apply ``shared_config_patch`` from the YAML roster after auth.

    Patches a TOML or YAML config file in the provider's shared config dir
    to redirect API traffic through the credential proxy.  The patch spec
    is declared in the agent YAML — no provider-specific code needed.
    """
    from terok_executor.roster.loader import get_roster

    roster = get_roster()
    route = roster.proxy_routes.get(provider_name)
    if not route or not route.shared_config_patch:
        return

    auth_info = roster.auth_providers.get(provider_name)
    if not auth_info:
        return

    from terok_sandbox import SandboxConfig, get_proxy_port

    from terok_executor.paths import mounts_dir

    cfg = SandboxConfig()
    port = get_proxy_port(cfg)
    proxy_url = f"http://host.containers.internal:{port}"

    patch = route.shared_config_patch
    shared_dir = mounts_dir() / auth_info.host_dir_name
    shared_dir.mkdir(parents=True, exist_ok=True)
    config_path = _safe_config_path(shared_dir, patch["file"])

    if "yaml_set" in patch:
        _apply_yaml_patch(config_path, patch, proxy_url)
    elif "toml_table" in patch:
        _apply_toml_patch(config_path, patch, proxy_url)

    print(f"Proxy config written to {config_path}")


def apply_shared_config_patches(roster: AgentRoster, mounts_base: Path) -> None:
    """Re-apply every ``shared_config_patch`` for the whole roster.

    Called during task start so that shared mount directories (which may
    have been recreated empty) always contain the correct proxy URLs.
    Idempotent: safe to call on every launch.

    Raises :class:`ConfigPatchError` on failure — callers must not start
    the container if proxy routing cannot be established.
    """
    from terok_sandbox import SandboxConfig, get_proxy_port

    cfg = SandboxConfig()
    port = get_proxy_port(cfg)
    proxy_url = f"http://host.containers.internal:{port}"

    for name, route in roster.proxy_routes.items():
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
                _apply_yaml_patch(config_path, patch, proxy_url)
            elif "toml_table" in patch:
                _apply_toml_patch(config_path, patch, proxy_url)
            _logger.debug("Applied config patch for %s → %s", name, config_path)
        except ConfigPatchError:
            raise
        except Exception as exc:
            raise ConfigPatchError(
                f"Failed to apply proxy config patch for {name} (file={patch.get('file')!r})"
            ) from exc


# ── Private helpers ──────────────────────────────────────────────────────


def _safe_config_path(shared_dir: Path, filename: str) -> Path:
    """Resolve *filename* inside *shared_dir*, rejecting traversal attempts.

    Raises :class:`ConfigPatchError` if the resolved path escapes the
    intended directory (absolute paths, ``..`` components, symlinks).
    """
    rel = Path(filename)
    if rel.is_absolute() or ".." in rel.parts:
        raise ConfigPatchError(f"invalid patch file path: {filename!r}")

    target = (shared_dir / rel).resolve(strict=False)
    base = shared_dir.resolve(strict=True)
    if base not in target.parents and target != base:
        raise ConfigPatchError(f"patch target {target} escapes shared dir {base}")
    return target


def _apply_toml_patch(config_path: Path, patch: dict, proxy_url: str) -> None:
    """Patch a TOML array-of-tables entry."""
    import tomllib

    if config_path.is_file():
        try:
            existing = tomllib.loads(config_path.read_text())
        except Exception as exc:
            print(
                f"Warning [proxy-config]: failed to parse {config_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            existing = {}
    else:
        existing = {}

    table_key = patch["toml_table"]
    match_criteria = patch["toml_match"]
    values = {
        k: v.replace("{proxy_url}", proxy_url) if isinstance(v, str) else v
        for k, v in patch["toml_set"].items()
    }

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

    config_path.write_bytes(tomli_w.dumps(existing).encode())


def _apply_yaml_patch(config_path: Path, patch: dict, proxy_url: str) -> None:
    """Set top-level keys in a YAML config file."""
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    if config_path.is_file():
        try:
            existing = yaml.load(config_path)
        except Exception as exc:
            print(
                f"Warning [proxy-config]: failed to parse {config_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            existing = {}
    else:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}

    for k, v in patch["yaml_set"].items():
        existing[k] = v.replace("{proxy_url}", proxy_url) if isinstance(v, str) else v

    yaml.dump(existing, config_path)
