# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models describing the credential files we read from third-party CLIs.

These files are owned by Anthropic, OpenAI, GitHub, GitLab, and a handful of
OpenAI-compatible providers — **not by us**.  Every model in this module
therefore uses ``extra="ignore"`` (Pydantic's default), and only the fields
we actually consume have any guarantees.  Vendor adds a new key, prunes a
side-field, renames an internal-only block?  Best-effort: we keep working
as long as the **fields we read** still hold their shape.

A single failure mode is loud: if a vendor renames or retypes a field we
*depend on* (e.g. ``claudeAiOauth.accessToken`` or ``tokens.access_token``),
the model raises [`ValidationError`][pydantic.ValidationError] — pointing at
the exact field, in the exact file.  Callers translate that into a clear
"vendor file format may have changed" surface.

For fields where we tolerate absence (most of them), the field is declared
optional with a default; the extractor checks truthiness rather than
``is not None``.

The file-loading helpers in this module ([`load_vendor_json`][terok_executor.credentials.vendor_files.load_vendor_json],
[`load_vendor_yaml`][terok_executor.credentials.vendor_files.load_vendor_yaml])
distinguish between "file is absent / unreadable / not a dict at the top
level" (silent fallback) and "file present but structure broke our
contract" (loud).  See those docstrings for the exact rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, RootModel

# ── Reusable building blocks ──────────────────────────────────────────────


def _normalize_js_timestamp(v: object) -> float | None:
    """Coerce a JavaScript-style millisecond timestamp to POSIX seconds.

    Claude Code is a JS app; ``expiresAt`` is ``Date.now()`` (milliseconds
    since epoch).  Values larger than ``1e12`` are unambiguously ms (a value
    in seconds that large would be year 33658).  Any non-numeric or boolean
    input collapses to ``None`` — best-effort, since the field is purely
    informational on our side (the vault re-checks expiry against the actual
    upstream).
    """
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return v / 1000 if v > 1e12 else float(v)


JsTimestamp = Annotated[float | None, BeforeValidator(_normalize_js_timestamp)]
"""POSIX-seconds timestamp coerced from a JS ``Date.now()`` ms value."""


class _VendorFile(BaseModel):
    """Base for vendor file models — lax (``extra="ignore"``) by design.

    Vendors own these formats; we never want a brand-new field they add to
    one of their files to break our login flow.
    """

    model_config = ConfigDict(extra="ignore")


# ── Anthropic (Claude) ────────────────────────────────────────────────────


class RawClaudeOauthBlock(_VendorFile):
    """``.credentials.json`` → ``claudeAiOauth`` — Claude OAuth state block."""

    accessToken: str = ""  # noqa: N815 — vendor uses camelCase
    refreshToken: str = ""  # noqa: N815
    expiresAt: JsTimestamp = None  # noqa: N815
    scopes: str = ""
    subscriptionType: str | None = None  # noqa: N815
    rateLimitTier: str | None = None  # noqa: N815


class RawClaudeCredentialsFile(_VendorFile):
    """Top-level shape of Claude Code's ``.credentials.json``.

    The OAuth block is optional — the file may exist without it (e.g. when
    the user authenticated via API key only).
    """

    claudeAiOauth: RawClaudeOauthBlock | None = None  # noqa: N815


# ── OpenAI (Codex) ────────────────────────────────────────────────────────


class RawCodexTokensBlock(_VendorFile):
    """``auth.json`` → ``tokens`` — Codex OAuth token block."""

    access_token: str = ""
    refresh_token: str = ""
    id_token: str | None = None
    account_id: str | None = None


class RawCodexAuthFile(_VendorFile):
    """Top-level shape of Codex's ``auth.json``.

    Both ``tokens`` (OAuth) and ``OPENAI_API_KEY`` (legacy) are optional;
    the extractor accepts whichever is present.
    """

    tokens: RawCodexTokensBlock | None = None
    OPENAI_API_KEY: str | None = None  # noqa: N815 — vendor's literal key name


# ── Generic API-key JSON ──────────────────────────────────────────────────


class RawApiKeyJsonFile(_VendorFile):
    """``{"api_key": "..."}``-shaped JSON config.

    Used by Claude's ``config.json`` (API-key fallback path) and by the
    OpenAI-compatible providers (blablador, kisski).
    """

    api_key: str = ""


# ── GitHub (gh CLI) ───────────────────────────────────────────────────────


class RawGhHostBlock(_VendorFile):
    """``hosts.yml`` → ``<host>`` — one entry in gh's per-host config."""

    oauth_token: str = ""


class RawGhHostsFile(RootModel[dict[str, RawGhHostBlock]]):
    """Top-level shape of gh's ``hosts.yml``.

    The YAML is a bare dict keyed by host name (``github.com``,
    ``ghe.example.com``, …) — no wrapper section.  ``RootModel`` lets us
    validate it without inventing a synthetic outer key.
    """


# ── GitLab (glab CLI) ─────────────────────────────────────────────────────


class RawGlabHostBlock(_VendorFile):
    """``config.yml`` → ``hosts.<host>`` — one entry in glab's per-host config."""

    token: str = ""


class RawGlabConfigFile(_VendorFile):
    """Top-level shape of glab's ``config.yml`` — has a ``hosts:`` map."""

    hosts: dict[str, RawGlabHostBlock] = Field(default_factory=dict)


# ── Loaders ───────────────────────────────────────────────────────────────


def load_vendor_json[T: BaseModel](model: type[T], path: Path) -> T | None:
    """Read a JSON vendor file from *path* and validate against *model*.

    Returns ``None`` only for "the file isn't there" cases — missing,
    unreadable, or unparseable as JSON.  When the JSON parses but the
    structure doesn't match *model* (wrong root type, missing nested
    field, wrong field type), raises [`pydantic.ValidationError`][pydantic.ValidationError]
    — a ``ValueError`` subclass — so the caller surfaces "your credential
    file is in a shape we don't recognize" rather than silently falling
    through to a generic "not found" path.  Callers that want a
    fall-through anyway (e.g. Claude tries OAuth then API key) catch
    [`ValidationError`][pydantic.ValidationError] explicitly.
    """
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None
    return model.model_validate(data)


def load_vendor_yaml[T: BaseModel](model: type[T], path: Path) -> T | None:
    """Read a YAML vendor file from *path* and validate against *model*.

    Same fallback / loud-fail rules as [`load_vendor_json`][terok_executor.credentials.vendor_files.load_vendor_json].
    Uses ruamel.yaml's safe loader; ``RootModel`` subclasses (like
    [`RawGhHostsFile`][terok_executor.credentials.vendor_files.RawGhHostsFile])
    accept the parsed dict as their root value.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError

    yaml = YAML(typ="safe")
    try:
        data = yaml.load(path)
    except (OSError, YAMLError):
        return None
    return model.model_validate(data)


__all__ = [
    "JsTimestamp",
    "RawApiKeyJsonFile",
    "RawClaudeCredentialsFile",
    "RawClaudeOauthBlock",
    "RawCodexAuthFile",
    "RawCodexTokensBlock",
    "RawGhHostBlock",
    "RawGhHostsFile",
    "RawGlabConfigFile",
    "RawGlabHostBlock",
    "load_vendor_json",
    "load_vendor_yaml",
]
