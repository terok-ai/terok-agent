# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Runtime dataclasses produced by the agent roster loader.

These are the immutable result types that consumers (env builder,
auth flow, image build) receive after a YAML file passes the
[`schema`][terok_executor.roster.schema] validation gate.  Kept in
their own module so both [`schema`][terok_executor.roster.schema]
(which projects onto them) and [`loader`][terok_executor.roster.loader]
(which orchestrates the projection) can import without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args


@dataclass(frozen=True)
class MountDef:
    """A shared directory mount derived from the agent roster."""

    host_dir: str
    """Directory name under ``mounts_dir()`` (e.g. ``"_codex-config"``)."""

    container_path: str
    """Mount point inside the container (e.g. ``"/home/dev/.codex"``)."""

    label: str
    """Human-readable label (e.g. ``"Codex config"``)."""

    credential_file: str = ""
    """Credential file path relative to the mount root (e.g. ``".credentials.json"``).

    Empty when the mount carries no auth artefact (e.g. opencode state dirs).
    Populated from the matching ``vault.credential_file`` so callers can
    layer a read-only shadow over the file without touching the rest of
    the shared mount.  See [terok-ai/terok#873](https://github.com/terok-ai/terok/issues/873).
    """

    provider: str = ""
    """Roster entry name that contributed this mount (e.g. ``"claude"``).

    Empty for explicit ``mounts:`` blocks that aren't tied to a single
    provider.  Used by the credential-shadow path to match against
    [`ContainerEnvSpec.expose_credential_providers`][terok_executor.ContainerEnvSpec.expose_credential_providers].
    """


@dataclass(frozen=True)
class VaultRoute:
    """Vault route config parsed from a ``vault:`` YAML section.

    Used to generate the ``routes.json`` that the vault server reads.
    """

    provider: str
    """Agent/tool name (e.g. ``"claude"``)."""

    route_prefix: str
    """Path prefix in the proxy (e.g. ``"claude"`` → ``/claude/v1/...``)."""

    upstream: str
    """Upstream API base URL (e.g. ``"https://api.anthropic.com"``)."""

    path_upstreams: dict[str, str] = field(default_factory=dict)
    """Optional request-path prefix → upstream-base overrides."""

    oauth_extra_headers: dict[str, str] = field(default_factory=dict)
    """Provider-specific headers added only when forwarding OAuth credentials."""

    auth_header: str = "Authorization"
    """HTTP header name for the real credential."""

    auth_prefix: str = "Bearer "
    """Prefix before the token value in the auth header."""

    credential_type: str = "api_key"
    """Type of credential: ``"oauth"``, ``"api_key"``, ``"oauth_token"``, ``"pat"``."""

    credential_file: str = ""
    """Credential file path relative to the auth mount."""

    phantom_env: dict[str, bool] = field(default_factory=dict)
    """Phantom env vars for API-key credentials (e.g. ``{"ANTHROPIC_API_KEY": true}``)."""

    oauth_phantom_env: dict[str, bool] = field(default_factory=dict)
    """Phantom env vars for OAuth credentials (e.g. ``{"CLAUDE_CODE_OAUTH_TOKEN": true}``).

    When the stored credential type is ``"oauth"`` and this is non-empty, these
    env vars are injected *instead of* [`phantom_env`][terok_executor.roster.types.VaultRoute.phantom_env].
    """

    base_url_env: str = ""
    """Env var to override with the vault's HTTP URL (e.g. ``"ANTHROPIC_BASE_URL"``)."""

    socket_env: str = ""
    """Env var that receives the container-side vault socket path.

    Set when the agent speaks HTTP-over-UNIX natively (e.g. Claude reads
    ``ANTHROPIC_UNIX_SOCKET``).  The resolved value is mode-dependent and
    injected centrally by the env builder.
    """

    shared_config_patch: dict | None = None
    """Optional shared config patch applied after auth (e.g. Vibe's config.toml)."""

    oauth_refresh: dict | None = None
    """OAuth refresh config: ``{token_url, client_id, scope}``."""


@dataclass(frozen=True)
class InstallSpec:
    """Roster-driven install snippets emitted into the L1 Dockerfile.

    The build template loops over the resolved selection and concatenates
    ``run_as_root`` snippets in the root section, ``run_as_dev`` snippets
    in the dev-user section.  Both fields are raw Dockerfile fragments
    (``RUN``, ``COPY`` — anything valid at top level after ``USER ...``).
    ``depends_on`` lists other roster names that must be installed
    alongside this one (transitively resolved at selection time).
    """

    depends_on: tuple[str, ...] = ()
    """Other roster entries this install requires (e.g. ``blablador → opencode``)."""

    run_as_root: str = ""
    """Dockerfile fragment emitted in the root section of the L1 image."""

    run_as_dev: str = ""
    """Dockerfile fragment emitted in the dev-user section of the L1 image."""


HelpSection = Literal["agent", "dev_tool"]
"""Section in the in-container help banner that an entry belongs to."""

HELP_SECTIONS: tuple[HelpSection, ...] = get_args(HelpSection)
"""All valid [`HelpSection`][terok_executor.roster.types.HelpSection] values, as a tuple (single source of truth)."""


@dataclass(frozen=True)
class HelpSpec:
    """One-line entry shown in the in-container help banner."""

    label: str = ""
    """Raw banner line (the agent owns its formatting, including ANSI codes)."""

    section: HelpSection = "agent"


@dataclass(frozen=True)
class SidecarSpec:
    """Sidecar container configuration parsed from a ``sidecar:`` YAML section.

    Tools with sidecar specs run in a separate lightweight L1 image
    (no agent CLIs) and receive the real API key instead of phantom tokens.
    """

    tool_name: str
    """Tool identifier used to select the Jinja2 install block in the template."""

    env_map: dict[str, str] = field(default_factory=dict)
    """Maps container env var names to credential dict keys.

    Example: ``{"CODERABBIT_API_KEY": "key"}`` reads ``cred["key"]`` and
    injects it as ``CODERABBIT_API_KEY``.
    """
