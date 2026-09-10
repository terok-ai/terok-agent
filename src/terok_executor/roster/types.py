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
from urllib.parse import urlsplit


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
    [`ContainerEnvSpec.expose_credential_providers`][terok_executor.container.env.ContainerEnvSpec.expose_credential_providers].
    """

    writable: bool = False
    """Mount [`credential_file`][terok_executor.roster.types.MountDef.credential_file]
    writable instead of under a read-only shadow.

    Set for tools whose credential file doubles as their settings file and is
    rewritten on startup (e.g. glab's ``config.yml``): the read-only shadow
    would abort the tool, so the file rides the shared mount writable and
    persists across tasks — trading the per-container credential containment of
    [terok-ai/terok#873](https://github.com/terok-ai/terok/issues/873) for a
    tool that runs at all.  Sourced from ``provider.credential_file_writable``.
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

    oauth_credential_headers: dict[str, str] = field(default_factory=dict)
    """Upstream header names mapped to fields in the stored OAuth credential."""

    auth_header: str = "Authorization"
    """HTTP header name for the real credential."""

    auth_prefix: str = "Bearer "
    """Prefix before the token value in the auth header."""

    credential_type: str = "api_key"
    """Type of credential: ``"oauth"``, ``"api_key"``, ``"oauth_token"``, ``"pat"``."""

    credential_file: str = ""
    """Credential file path relative to the auth mount."""

    token_env: dict[str, str] = field(default_factory=dict)
    """Phantom-token env var name, keyed by stored credential type.

    The named env var carries the phantom token the agent reads in place of
    the real credential.  Keys are credential types (``"oauth"``, ``"pat"``,
    …); ``"_default"`` is the fallback for any type without an explicit
    entry.  Most agents read one env var regardless of type
    (``{"_default": "MISTRAL_API_KEY"}``); Claude swaps the name when an
    OAuth token is stored
    (``{"oauth": "CLAUDE_CODE_OAUTH_TOKEN", "_default": "ANTHROPIC_API_KEY"}``).
    """

    token_env_aliases: tuple[str, ...] = ()
    """Additional env vars that receive the same phantom token."""

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

    shared_domain: bool = False
    """Whether the upstream host also serves non-API traffic.

    Set on entries whose ``upstream`` host is an apex (or otherwise mixed)
    domain that legitimately serves docs, dashboards, ``git push``, etc.
    Host-level egress denies can't separate paths, so terok's auth-protect
    layer skips these providers when re-applying denies after ``shield
    down`` — credential containment alone keeps the API safe.

    Examples: ``gitlab.com`` (API + ``git push``), ``sonarcloud.io``
    (API + project pages + docs + badges).
    """


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


@dataclass(frozen=True)
class OpenCodeProviderConfig:
    """OpenAI-compatible endpoint config for an OpenCode-driven provider.

    Carried by a [`Provider`][terok_executor.roster.types.Provider] that a
    harness (OpenCode, Pi) drives rather than a native CLI — the curated
    research endpoints (Blablador, KISSKI) and the OpenRouter aggregator.
    Supplies the display name, model defaults and per-provider config dir the
    ``opencode-provider`` wrapper writes into an ``opencode.json`` when that
    provider is selected at runtime.
    """

    display_name: str
    """Human-readable display name (e.g. ``"Helmholtz Blablador"``)."""

    base_url: str
    """Base URL for the OpenAI-compatible API (e.g. ``".../v1"``)."""

    preferred_model: str
    """Preferred model ID (e.g. ``"alias-huge"``)."""

    fallback_model: str
    """Fallback model ID used when the preferred one is unavailable."""

    env_var_prefix: str
    """API-key env-var prefix (e.g. ``"BLABLADOR"`` → ``BLABLADOR_API_KEY``)."""

    config_dir: str
    """Per-provider config directory name (e.g. ``".blablador"``)."""

    auth_key_url: str
    """URL where users obtain an API key (documentation / auth hint)."""

    api_key_hint: str = ""
    """Custom API-key prompt hint; falls back to ``auth_key_url`` when empty."""

    def to_env(self, name: str) -> dict[str, str]:
        """Return the ``TEROK_OC_{NAME}_*`` env vars for container injection."""
        prefix = f"TEROK_OC_{name.upper()}_"
        return {
            f"{prefix}BASE_URL": self.base_url,
            f"{prefix}PREFERRED_MODEL": self.preferred_model,
            f"{prefix}FALLBACK_MODEL": self.fallback_model,
            f"{prefix}DISPLAY_NAME": self.display_name,
            f"{prefix}ENV_VAR_PREFIX": self.env_var_prefix,
            f"{prefix}CONFIG_DIR": self.config_dir,
        }


@dataclass(frozen=True)
class ProviderAuth:
    """How a provider attaches the real credential to an upstream request.

    One instance per supported mode.  A provider may carry an OAuth mode, an
    API-key mode, or both — see [`Provider.wire_auth`][terok_executor.roster.types.Provider.wire_auth].
    """

    header: str
    """HTTP header that carries the credential (e.g. ``"Authorization"``, ``"x-api-key"``)."""

    prefix: str = ""
    """String prepended to the token value (e.g. ``"Bearer "``, ``"token "``)."""

    extra_headers: dict[str, str] = field(default_factory=dict)
    """Headers added only for this mode (e.g. Anthropic's ``anthropic-beta``)."""


@dataclass(frozen=True)
class ProviderModel:
    """Store provider-neutral data for one model from an LLM provider."""

    name: str
    """Model name that users see, or the provider model ID by default."""

    context_limit: int | None = None
    """Maximum context-window size in tokens, if the provider specifies it."""

    output_limit: int | None = None
    """Maximum output size in tokens, if the provider specifies it."""


@dataclass(frozen=True)
class Provider:
    """A vault-routed upstream — an LLM endpoint or a tool API.

    Carries the *endpoint* concern lifted out of the agent ``vault:`` blocks:
    where requests go (``upstream`` / ``path_upstreams``) and how the real
    credential is attached (``oauth_auth`` / ``api_key_auth``).  This is the
    single source the ``routes.json`` consumed by the sandbox vault is
    generated from.  The per-agent *delivery* concern (which env var receives
    the phantom token or base URL) stays on the agent binding, not here.
    """

    name: str
    """Clean provider name — also the ``routes.json`` key and the vault DB
    ``credentials.provider`` value (e.g. ``"anthropic"``, ``"github"``)."""

    upstream: str
    """Upstream API base URL (e.g. ``"https://api.anthropic.com"``)."""

    label: str = ""
    """Provider name that users see, or the roster entry name by default."""

    default_model: str | None = None
    """Model ID that OpenCode or Pi uses when the operator does not select one."""

    models: dict[str, ProviderModel] = field(default_factory=dict)
    """Static model data, keyed by provider model ID."""

    api_key_auth: ProviderAuth | None = None
    """Wire auth used when an API key is the stored credential, if supported."""

    oauth_auth: ProviderAuth | None = None
    """Wire auth used when an OAuth token is the stored credential, if supported."""

    path_upstreams: dict[str, str] = field(default_factory=dict)
    """Request-path prefix → upstream-base overrides (e.g. Codex's ``/backend-api/``)."""

    oauth_credential_headers: dict[str, str] = field(default_factory=dict)
    """Upstream header names mapped to fields in the stored OAuth credential."""

    oauth_refresh: dict[str, str] | None = None
    """OAuth refresh config: ``{token_url, client_id, scope}``."""

    shared_domain: bool = False
    """Whether the upstream host also serves non-API traffic (docs, ``git push``…).

    See [`VaultRoute.shared_domain`][terok_executor.roster.types.VaultRoute.shared_domain].
    """

    serves: dict[str, str] = field(default_factory=dict)
    """Wire protocol → container-facing base path (LLM providers only).

    Empty for tool providers (github, gitlab, …).  A provider that serves more
    than one protocol (openrouter: ``anthropic-messages`` + ``openai-chat``)
    backs several agent protocols from one endpoint.  Consumed by the wrapper
    when resolving an agent×provider combo; not part of the ``routes.json``
    contract.
    """

    egress_allow: tuple[str, ...] = ()
    """Extra hosts this provider's runtime legitimately reaches *directly*.

    Projected into the shield's ``provider_allow`` tier (t30) — distinct from
    the vault-relayed API host in ``upstream``: these are *additional* runtime
    destinations (telemetry, a model-listing endpoint, …) the agent reaches
    without vault mediation.  Build-time fetches do **not** belong here: image
    build is unshielded.  Empty for most providers.
    """

    opencode_config: OpenCodeProviderConfig | None = None
    """OpenCode wrapper config when a harness drives this provider (Blablador,
    KISSKI, OpenRouter).  ``None`` for native LLM providers and tool providers,
    which are reached through their own CLI or a native protocol binding."""

    install_spec: InstallSpec | None = None
    """Dockerfile install fragment when this provider ships its own pinned-alias
    command (the curated harness providers symlink ``blablador`` → the OpenCode
    wrapper family).  ``None`` for providers reached only through an agent."""

    help_spec: HelpSpec | None = None
    """Help-listing entry when this provider ships a command; ``None`` otherwise."""

    def wire_auth(self) -> tuple[str, str, dict[str, str]]:
        """Resolve the ``routes.json`` ``(auth_header, auth_prefix, oauth_extra_headers)``.

        When the provider offers both OAuth and API-key auth under *different*
        headers (Anthropic: ``Authorization: Bearer`` for OAuth vs ``x-api-key``
        for the key), the vault must pick the header *and* prefix per stored
        credential type at request time — signalled by the sentinel header
        ``"dynamic"`` with an empty prefix.  A single mode emits its header and
        prefix verbatim.

        Two modes sharing a header are only representable if they're wire-
        identical: ``routes.json`` carries a single ``auth_prefix`` /
        ``oauth_extra_headers``, so modes on the same header but with different
        prefixes (or extra headers) can't both be serialised — that raises
        rather than silently wiring one mode with the other's prefix.

        This reproduces the historical ``auth_header: dynamic`` wire contract
        without the magic string ever appearing in a provider definition.
        """
        oauth, api_key = self.oauth_auth, self.api_key_auth
        if oauth is not None and api_key is not None:
            if oauth.header != api_key.header:
                # Distinct headers → the vault selects header + prefix by the
                # stored credential type at request time.
                return "dynamic", "", dict(oauth.extra_headers)
            if oauth.prefix != api_key.prefix or oauth.extra_headers != api_key.extra_headers:
                raise ValueError(
                    f"Provider {self.name!r} declares oauth and api_key auth on the same "
                    f"header ({oauth.header!r}) but with different prefix/extra_headers; the "
                    f"routes.json contract carries only one, so one mode would be wired with "
                    f"the other's prefix. Give the modes distinct headers or unify them."
                )
        mode = oauth or api_key
        if mode is None:
            raise ValueError(f"Provider {self.name!r} declares no auth mode")
        return mode.header, mode.prefix, dict(mode.extra_headers)

    def relayed_hosts(self) -> frozenset[str]:
        """Every host the vault relays to for this provider.

        The union of the ``upstream`` host, every ``path_upstreams`` override
        host, and the OAuth-refresh ``token_url`` host when present — exactly
        the hosts the agent must reach *only* through the loopback vault, and
        thus the raw material for the shield's ``security_deny`` tier.  Values
        that carry no parseable host are dropped.
        """
        urls = [self.upstream, *self.path_upstreams.values()]
        if self.oauth_refresh and (token_url := self.oauth_refresh.get("token_url")):
            urls.append(token_url)
        return frozenset(host for url in urls if (host := urlsplit(url).hostname))


@dataclass(frozen=True)
class EgressProjection:
    """Roster-derived egress policy for a task's shield bundle.

    Produced by [`AgentRoster.compose_egress`][terok_executor.roster.loader.AgentRoster.compose_egress]
    and handed (via the sandbox) to the shield's tier writer: ``deny_to_vault``
    seeds the ``security_deny`` tier (t20), ``provider_allow`` the
    ``provider_allow`` tier (t30).  Both tuples are sorted and de-duplicated so
    the projection is deterministic across runs.
    """

    deny_to_vault: tuple[str, ...] = ()
    """Relayed provider endpoints the agent may reach *only* through the vault,
    denied directly at the egress firewall (shared-domain and exposed-credential
    providers excluded)."""

    provider_allow: tuple[str, ...] = ()
    """Extra runtime-egress hosts the providers legitimately need
    (``egress.allow`` in the roster), allowed at the egress firewall."""
