# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-provider credential extractors for the auth interceptor.

Each extractor reads a vendor-specific credential file from a temporary
auth container mount and returns a normalized dict suitable for storage
in :class:`~terok_sandbox.CredentialDB`.  The dict must contain at least
one of ``access_token``, ``token``, or ``key`` — the credential proxy
server uses these fields to inject the real auth header.

All extractors are pure functions: ``Path → dict``.  They raise
``ValueError`` if the file is missing, malformed, or empty.
"""

from __future__ import annotations

import json
from pathlib import Path


def _expect_mapping(value: object, *, context: str) -> dict:
    """Validate that *value* is a dict, raising ``ValueError`` if not."""
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {context}, got {type(value).__name__}")
    return value


def extract_claude_oauth(base_dir: Path) -> dict:
    """Extract Claude OAuth tokens from ``.credentials.json``.

    Claude stores OAuth data under the ``claudeAiOauth`` key with
    nested ``token`` containing ``accessToken`` and ``refreshToken``.
    """
    cred_file = base_dir / ".credentials.json"
    if not cred_file.is_file():
        raise ValueError(f"Claude credential file not found: {cred_file}")

    data = _expect_mapping(
        json.loads(cred_file.read_text(encoding="utf-8")), context=str(cred_file)
    )
    oauth = _expect_mapping(data.get("claudeAiOauth", {}), context=f"{cred_file}:claudeAiOauth")
    token_data = _expect_mapping(oauth.get("token", {}), context=f"{cred_file}:token")
    access_token = token_data.get("accessToken")
    if not access_token:
        raise ValueError("Claude credential file has no accessToken")

    return {
        "type": "oauth",
        "access_token": access_token,
        "refresh_token": token_data.get("refreshToken", ""),
        "expires_at": token_data.get("expiresAt"),
    }


def extract_codex_oauth(base_dir: Path) -> dict:
    """Extract Codex (OpenAI) OAuth tokens from ``auth.json``."""
    cred_file = base_dir / "auth.json"
    if not cred_file.is_file():
        raise ValueError(f"Codex credential file not found: {cred_file}")

    data = _expect_mapping(
        json.loads(cred_file.read_text(encoding="utf-8")), context=str(cred_file)
    )
    tokens = _expect_mapping(data.get("tokens", {}), context=f"{cred_file}:tokens")
    access_token = tokens.get("access_token")
    if not access_token:
        raise ValueError("Codex credential file has no access_token")

    return {
        "type": "oauth",
        "access_token": access_token,
        "refresh_token": tokens.get("refresh_token", ""),
    }


def extract_api_key_env(base_dir: Path, filename: str = ".env", var_name: str = "") -> dict:
    """Extract an API key from a dotenv-style file (e.g. Vibe's ``.env``).

    Looks for ``VAR_NAME=value`` lines.  If *var_name* is empty, takes
    the first non-comment, non-empty value.
    """
    env_file = base_dir / filename
    if not env_file.is_file():
        raise ValueError(f"Env file not found: {env_file}")

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if var_name and key.strip() != var_name:
            continue
        value = value.strip().strip("'\"")
        if value:
            return {"type": "api_key", "key": value}

    raise ValueError(f"No API key found in {env_file}")


def extract_json_api_key(base_dir: Path, filename: str = "config.json") -> dict:
    """Extract an API key from a JSON config file (blablador, kisski).

    Expects ``{"api_key": "..."}`` at the top level.
    """
    cred_file = base_dir / filename
    if not cred_file.is_file():
        raise ValueError(f"JSON config not found: {cred_file}")

    data = _expect_mapping(
        json.loads(cred_file.read_text(encoding="utf-8")), context=str(cred_file)
    )
    key = data.get("api_key")
    if not key:
        raise ValueError(f"No api_key field in {cred_file}")

    return {"type": "api_key", "key": key}


def extract_gh_token(base_dir: Path) -> dict:
    """Extract GitHub token from ``hosts.yml``.

    The gh CLI stores tokens per host under ``github.com.oauth_token``.
    """
    hosts_file = base_dir / "hosts.yml"
    if not hosts_file.is_file():
        raise ValueError(f"GitHub hosts file not found: {hosts_file}")

    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    data = yaml.load(hosts_file)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected hosts.yml format: {type(data)}")

    # Try github.com first, then remaining hosts (deduplicated)
    ordered = ["github.com"] + [h for h in data if h != "github.com"]
    for host in ordered:
        host_data = data.get(host, {})
        if isinstance(host_data, dict):
            token = host_data.get("oauth_token")
            if token:
                return {"type": "oauth_token", "token": token, "host": host}

    raise ValueError("No oauth_token found in hosts.yml")


def extract_glab_token(base_dir: Path) -> dict:
    """Extract GitLab token from ``config.yml``.

    The glab CLI stores tokens per host under ``hosts.<host>.token``.
    """
    config_file = base_dir / "config.yml"
    if not config_file.is_file():
        raise ValueError(f"GitLab config not found: {config_file}")

    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    data = yaml.load(config_file)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected config.yml format: {type(data)}")

    hosts = _expect_mapping(data.get("hosts", {}), context=f"{config_file}:hosts")
    for host, host_data in hosts.items():
        if isinstance(host_data, dict):
            token = host_data.get("token")
            if token:
                return {"type": "pat", "token": token, "host": host}

    raise ValueError("No token found in glab config.yml")


# ---------------------------------------------------------------------------
# Extractor registry — maps provider names to their extraction function
# ---------------------------------------------------------------------------

#: Maps provider name → (extractor_fn, *extra_args).
#: The extractor receives (base_dir, *extra_args) and returns a credential dict.
EXTRACTORS: dict[str, tuple] = {
    "claude": (extract_claude_oauth,),
    "codex": (extract_codex_oauth,),
    "vibe": (extract_api_key_env, ".env", "MISTRAL_API_KEY"),
    "gh": (extract_gh_token,),
    "glab": (extract_glab_token,),
    "blablador": (extract_json_api_key, "config.json"),
    "kisski": (extract_json_api_key, "config.json"),
}


def extract_credential(provider: str, base_dir: Path) -> dict:
    """Run the appropriate extractor for *provider* against *base_dir*.

    Raises ``ValueError`` if no extractor is registered or extraction fails.
    """
    entry = EXTRACTORS.get(provider)
    if entry is None:
        raise ValueError(f"No credential extractor for provider {provider!r}")
    fn, *args = entry
    return fn(base_dir, *args)
