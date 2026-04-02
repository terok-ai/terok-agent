# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Post-auth shared config patching for the credential proxy.

Applies ``shared_config_patch`` from the YAML roster after authentication.
Writes proxy URLs (not secrets) to provider config files so that agents
route API traffic through the credential proxy.
"""

from __future__ import annotations

import sys
from pathlib import Path


def write_proxy_config(provider_name: str) -> None:
    """Apply ``shared_config_patch`` from the YAML roster after auth.

    Patches a TOML or YAML config file in the provider's shared config dir
    to redirect API traffic through the credential proxy.  The patch spec
    is declared in the agent YAML — no provider-specific code needed.
    """
    from .roster import get_roster

    roster = get_roster()
    route = roster.proxy_routes.get(provider_name)
    if not route or not route.shared_config_patch:
        return

    auth_info = roster.auth_providers.get(provider_name)
    if not auth_info:
        return

    from terok_sandbox import SandboxConfig, get_proxy_port

    from .paths import mounts_dir

    cfg = SandboxConfig()
    port = get_proxy_port(cfg)
    proxy_url = f"http://host.containers.internal:{port}"

    patch = route.shared_config_patch
    shared_dir = mounts_dir() / auth_info.host_dir_name
    config_path = shared_dir / patch["file"]
    shared_dir.mkdir(parents=True, exist_ok=True)

    if "yaml_set" in patch:
        _apply_yaml_patch(config_path, patch, proxy_url)
    elif "toml_table" in patch:
        _apply_toml_patch(config_path, patch, proxy_url)

    print(f"Proxy config written to {config_path}")


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
