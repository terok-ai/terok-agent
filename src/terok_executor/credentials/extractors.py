# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Extracts vendor-specific credentials from auth container mounts.

Each extractor reads a vendor-specific credential file from a temporary
auth container mount and returns a normalized dict suitable for storage
in [`CredentialDB`][terok_sandbox.CredentialDB].  The dict must contain at least
one of ``access_token``, ``token``, or ``key`` --- the vault server
uses these fields to inject the real auth header.

All file shapes live in
[`vendor_files`][terok_executor.credentials.vendor_files] as Pydantic
models.  Vendors own those formats, so the models are deliberately lax
(``extra="ignore"``); only the fields we depend on are typed-checked.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from .vendor_files import (
    RawApiKeyJsonFile,
    RawClaudeCredentialsFile,
    RawCodexAuthFile,
    RawGhHostsFile,
    RawGlabConfigFile,
    load_vendor_json,
    load_vendor_yaml,
)

# ---------------------------------------------------------------------------
# Individual extractors (scannable catalog entries)
# ---------------------------------------------------------------------------


def extract_claude_oauth(base_dir: Path) -> dict:
    """Extract Claude credentials --- OAuth tokens or API key.

    Claude stores OAuth state in ``.credentials.json`` under
    ``claudeAiOauth``.  If that block is missing or token-less (for
    example, because the user authenticated with an API key only), we
    fall back to ``config.json``'s ``api_key`` field.
    """
    # OAuth-then-API-key fallback is per-design; suppress shape errors on
    # ``.credentials.json`` so a corrupt OAuth file doesn't block API-key login.
    try:
        cred = load_vendor_json(RawClaudeCredentialsFile, base_dir / ".credentials.json")
    except ValidationError:
        cred = None
    if cred is not None and cred.claudeAiOauth and cred.claudeAiOauth.accessToken:
        oauth = cred.claudeAiOauth
        return {
            "type": "oauth",
            "access_token": oauth.accessToken,
            "refresh_token": oauth.refreshToken,
            "expires_at": oauth.expiresAt,
            "scopes": oauth.scopes,
            "subscription_type": oauth.subscriptionType,
            "rate_limit_tier": oauth.rateLimitTier,
        }

    try:
        api_key_cfg = load_vendor_json(RawApiKeyJsonFile, base_dir / "config.json")
    except ValidationError:
        api_key_cfg = None
    if api_key_cfg is not None and api_key_cfg.api_key:
        return {"type": "api_key", "key": api_key_cfg.api_key}

    raise ValueError(
        f"No Claude credentials found in {base_dir} "
        "(checked .credentials.json for OAuth, config.json for API key)"
    )


def extract_codex_oauth(base_dir: Path) -> dict:
    """Extract Codex (OpenAI) OAuth tokens from ``auth.json``.

    Also preserves the source ``id_token`` JWT and ``account_id`` when
    present — the shared synthetic auth.json writer derives a safe
    claim-only JWT from it so Codex's plan-tier and workspace UI keeps
    working without leaking the real OAuth tokens.
    """
    cred_file = base_dir / "auth.json"
    cred = load_vendor_json(RawCodexAuthFile, cred_file)
    if cred is None:
        raise ValueError(f"Codex credential file not found or unreadable: {cred_file}")
    if cred.tokens is None or not cred.tokens.access_token:
        raise ValueError("Codex credential file has no access_token")

    out: dict = {
        "type": "oauth",
        "access_token": cred.tokens.access_token,
        "refresh_token": cred.tokens.refresh_token,
    }
    if cred.tokens.id_token:
        out["id_token"] = cred.tokens.id_token
    if cred.tokens.account_id:
        out["account_id"] = cred.tokens.account_id
    return out


def extract_api_key_env(base_dir: Path, filename: str = ".env", var_name: str = "") -> dict:
    """Extract an API key from a dotenv-style file (e.g. Vibe's ``.env``).

    Looks for ``VAR_NAME=value`` lines.  If *var_name* is empty, takes
    the first non-comment, non-empty value.  Dotenv is line-oriented and
    not JSON — a Pydantic model adds no value here, so the parse stays
    inline.
    """
    env_file = base_dir / filename
    try:
        text = env_file.read_text(encoding="utf-8")
    except OSError:
        raise ValueError(f"Env file not found or unreadable: {env_file}") from None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
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
    cfg = load_vendor_json(RawApiKeyJsonFile, cred_file)
    if cfg is None:
        raise ValueError(f"JSON config not found or unreadable: {cred_file}")
    if not cfg.api_key:
        raise ValueError(f"No api_key field in {cred_file}")
    return {"type": "api_key", "key": cfg.api_key}


def extract_gh_token(base_dir: Path) -> dict:
    """Extract GitHub token from ``hosts.yml``.

    Tokens are stored per host under ``<host>.oauth_token``; we prefer
    ``github.com`` and fall back to any other host present.
    """
    hosts_file = base_dir / "hosts.yml"
    cred = load_vendor_yaml(RawGhHostsFile, hosts_file)
    if cred is None:
        raise ValueError(f"GitHub hosts file not found or unreadable: {hosts_file}")

    hosts = cred.root
    ordered = ["github.com"] + [h for h in hosts if h != "github.com"]
    for host in ordered:
        block = hosts.get(host)
        if block is not None and block.oauth_token:
            return {"type": "oauth_token", "token": block.oauth_token, "host": host}

    raise ValueError("No oauth_token found in hosts.yml")


def extract_glab_token(base_dir: Path) -> dict:
    """Extract GitLab token from ``config.yml``.

    Tokens live under ``hosts.<host>.token``; the first non-empty entry wins.
    """
    config_file = base_dir / "config.yml"
    cred = load_vendor_yaml(RawGlabConfigFile, config_file)
    if cred is None:
        raise ValueError(f"GitLab config not found or unreadable: {config_file}")

    for host, block in cred.hosts.items():
        if block.token:
            return {"type": "pat", "token": block.token, "host": host}

    raise ValueError("No token found in glab config.yml")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EXTRACTORS: dict[str, tuple] = {
    "claude": (extract_claude_oauth,),
    "codex": (extract_codex_oauth,),
    "vibe": (extract_api_key_env, ".env", "MISTRAL_API_KEY"),
    "gh": (extract_gh_token,),
    "glab": (extract_glab_token,),
    "blablador": (extract_json_api_key, "config.json"),
    "kisski": (extract_json_api_key, "config.json"),
}
"""Maps provider name → ``(extractor_fn, *extra_args)``."""


def extract_credential(provider: str, base_dir: Path) -> dict:
    """Run the appropriate extractor for *provider* against *base_dir*.

    Raises ``ValueError`` if no extractor is registered or extraction fails.
    """
    entry = EXTRACTORS.get(provider)
    if entry is None:
        raise ValueError(f"No credential extractor for provider {provider!r}")
    fn, *args = entry
    return fn(base_dir, *args)
