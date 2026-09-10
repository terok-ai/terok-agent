# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Pydantic v2 schema for agent-roster YAML files.

Each ``resources/agents/*.yaml`` (and any user override under
``~/.config/terok/agent/agents/*.yaml``) is parsed into [`RawAgentYaml`][terok_executor.roster.schema.RawAgentYaml];
each ``resources/providers/*.yaml`` into [`RawProvider`][terok_executor.roster.schema.RawProvider].
These project onto the runtime dataclasses
([`Agent`][terok_executor.provider.providers.Agent],
[`Provider`][terok_executor.roster.types.Provider],
[`AuthProvider`][terok_executor.credentials.auth.AuthProvider], …).

Validation guarantees:

- **Strict keys**: every section uses ``extra="forbid"``, so a typo
  (``headles:``, ``oauth_refesh:``) fails fast with a precise error
  instead of silently falling back to defaults.
- **Type-checked values**: ``modes`` only accepts ``oauth``/``api_key``,
  ``credential_type`` only accepts the four known kinds, ``help.section``
  only accepts ``agent``/``dev_tool``.
- **Required fields**: ``provider.upstream``, ``provider.auth`` (≥1 mode),
  ``auth.host_dir``, ``auth.container_mount`` raise on missing.
- **Coercions**: ``install.depends_on`` accepts a single string or a list.

Each ``Raw…`` model exposes a ``to_dataclass(...)`` method that produces
the corresponding frozen runtime object.  The roster loader threads
context (the agent's name and label) through these methods rather than
encoding it in the schema itself — keeps the schema purely declarative
and matches the per-file YAML shape.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PositiveInt, model_validator

from terok_executor.credentials.auth import AuthKeyConfig, AuthProvider, api_key_command
from terok_executor.provider.providers import Agent, Launcher, ProviderBinding

from .types import (
    HELP_SECTIONS,
    HelpSection,
    HelpSpec,
    InstallSpec,
    OpenCodeProviderConfig,
    Provider,
    ProviderAuth,
    ProviderModel,
    SidecarSpec,
)

# ── Reusable building blocks ──────────────────────────────────────────────


def _coerce_str_to_list(v: object) -> object:
    """Accept a single string in place of a list — a YAML ergonomics hack."""
    if isinstance(v, str):
        return [v]
    return v


def _coerce_none_to_empty_dict(v: object) -> object:
    """Accept ``~``/``None`` in place of an empty mapping — YAML ergonomics."""
    return {} if v is None else v


StrOrStrList = Annotated[list[str], BeforeValidator(_coerce_str_to_list)]
"""Permits ``foo: bar`` as a shorthand for ``foo: [bar]``."""

PostCaptureState = Annotated[dict[str, dict], BeforeValidator(_coerce_none_to_empty_dict)]
"""Mapping of ``filename`` → JSON patch.  ``post_capture_state: ~`` collapses to ``{}``."""


class StrictModel(BaseModel):
    """Base for every roster section — forbids unknown keys."""

    model_config = ConfigDict(extra="forbid")


# ── Sub-section models ────────────────────────────────────────────────────


class RawGitIdentity(StrictModel):
    """``git_identity:`` — author/committer override per agent."""

    name: str | None = Field(default=None, description="Git author/committer name")
    email: str | None = Field(default=None, description="Git author/committer email")


class RawHeadless(StrictModel):
    """``headless:`` — flags and subcommand for non-interactive prompt invocation."""

    subcommand: str | None = Field(
        default=None, description="Subcommand for headless mode (e.g. ``exec`` for codex)"
    )
    prompt_flag: str = Field(default="-p", description='Flag for the prompt; ``""`` for positional')
    model_flag: str | None = Field(default=None, description="Flag for model override")
    max_turns_flag: str | None = Field(default=None, description="Flag for maximum turns")
    verbose_flag: str | None = Field(default=None, description="Flag for verbose output")
    output_format_flags: list[str] = Field(
        default_factory=list, description="Flags for structured output"
    )


class RawAutoApprove(StrictModel):
    """``auto_approve:`` — env vars and flags injected when ``TEROK_UNRESTRICTED=1``."""

    env: dict[str, str] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)


class RawSession(StrictModel):
    """``session:`` — session resume / continue capability flags."""

    supports_resume: bool = False
    resume_flag: str | None = None
    resume_subcommand: str | None = None
    continue_flag: str | None = None
    session_file: str | None = None
    supports_hook: bool = False

    @model_validator(mode="after")
    def _one_resume_mechanism(self) -> RawSession:
        """Require exactly one resume mechanism when the agent resumes.

        The runtime [`Agent`][terok_executor.provider.providers.Agent] resumes
        by a flag or by a subcommand, never both — the wrapper reads whichever
        is set.  Reject a roster (bundled or user override) that claims resume
        support with neither, or that declares both, before it can render an
        ambiguous wrapper.
        """
        if self.resume_flag and self.resume_subcommand:
            raise ValueError("resume_flag and resume_subcommand are mutually exclusive")
        if self.supports_resume and not (self.resume_flag or self.resume_subcommand):
            raise ValueError("supports_resume requires a resume_flag or a resume_subcommand")
        return self


class RawCapabilities(StrictModel):
    """``capabilities:`` — agent-specific feature toggles."""

    add_dir: bool = False
    log_format: Literal["plain", "claude-stream-json"] = "plain"


class RawLauncher(StrictModel):
    """``wrapper.launcher:`` — the per-task launcher script and when it runs."""

    script: str
    mode: Literal["on_provider_select", "always"]

    def to_dataclass(self) -> Launcher:
        """Project to a runtime [`Launcher`][terok_executor.provider.providers.Launcher]."""
        return Launcher(script=self.script, mode=self.mode)


class RawWrapper(StrictModel):
    """``wrapper:`` — in-container shell-wrapper behavior."""

    refuse_subcommands: list[str] = Field(default_factory=list)
    launcher: RawLauncher | None = None


class RawOpenCode(StrictModel):
    """``opencode:`` — OpenAI-compatible provider config for OpenCode-based agents."""

    display_name: str
    base_url: str
    preferred_model: str
    fallback_model: str
    env_var_prefix: str
    config_dir: str
    auth_key_url: str
    api_key_hint: str | None = Field(
        default=None,
        description="Override for the auto-derived auth provider's API-key hint",
    )

    def to_dataclass(self) -> OpenCodeProviderConfig:
        """Project to the runtime [`OpenCodeProviderConfig`][terok_executor.roster.types.OpenCodeProviderConfig]."""
        return OpenCodeProviderConfig(
            display_name=self.display_name,
            base_url=self.base_url,
            preferred_model=self.preferred_model,
            fallback_model=self.fallback_model,
            env_var_prefix=self.env_var_prefix,
            config_dir=self.config_dir,
            auth_key_url=self.auth_key_url,
            api_key_hint=self.api_key_hint or "",
        )


class RawAuthKey(StrictModel):
    """``auth.auth_key:`` — printf-template-driven API-key prompt for tools."""

    label: str | None = None
    key_url: str
    env_var: str
    config_path: str
    printf_template: str
    tool_name: str | None = None


class RawAuth(StrictModel):
    """``auth:`` — credential-capture behavior (OAuth container or API-key prompt)."""

    host_dir: str = Field(
        description="Single-segment dir under mounts_dir() (e.g. ``_codex-config``)"
    )
    container_mount: str = Field(description="Mount point inside the container")
    command: list[str] | None = Field(
        default=None,
        description="Container command for OAuth mode; derived from auth_key when absent",
    )
    auth_key: RawAuthKey | None = None
    banner_hint: str = ""
    extra_run_args: list[str] = Field(default_factory=list)
    modes: list[Literal["oauth", "api_key"]] = Field(
        default_factory=lambda: ["api_key"]  # type: ignore[arg-type]
    )
    device_auth: bool = Field(
        default=False,
        description="OAuth flow has a headless device-code variant, offered in the auth prompt",
    )
    api_key_hint: str = ""
    post_capture_state: PostCaptureState = Field(
        default_factory=dict,
        description="JSON state files to merge into the auth mount post-capture",
    )

    def to_dataclass(self, *, name: str, label: str) -> AuthProvider:
        """Project to the runtime [`AuthProvider`][terok_executor.credentials.auth.AuthProvider].

        *name* and *label* come from the parent file (the YAML doesn't repeat
        them inside ``auth:``).
        """
        if self.command is not None:
            cmd = list(self.command)
        elif self.auth_key is not None:
            cmd = api_key_command(
                AuthKeyConfig(
                    label=self.auth_key.label or label,
                    key_url=self.auth_key.key_url,
                    env_var=self.auth_key.env_var,
                    config_path=self.auth_key.config_path,
                    printf_template=self.auth_key.printf_template,
                    tool_name=self.auth_key.tool_name or name,
                )
            )
        else:
            cmd = []
        return AuthProvider(
            name=name,
            label=label,
            host_dir_name=self.host_dir,
            container_mount=self.container_mount,
            command=cmd,
            banner_hint=self.banner_hint,
            extra_run_args=tuple(self.extra_run_args),
            modes=tuple(self.modes),
            device_auth=self.device_auth,
            api_key_hint=self.api_key_hint,
            post_capture_state=dict(self.post_capture_state),
        )


class RawOAuthRefresh(StrictModel):
    """``vault.oauth_refresh:`` — token-refresh endpoint and client config."""

    token_url: str
    client_id: str
    scope: str | None = None


class RawSidecar(StrictModel):
    """``sidecar:`` — separate L1 image + env-mapped credentials for tool runners."""

    tool_name: str | None = None
    env_map: dict[str, str] = Field(default_factory=dict)

    def to_dataclass(self, *, default_name: str) -> SidecarSpec:
        """Project to a runtime [`SidecarSpec`][terok_executor.roster.types.SidecarSpec]."""
        return SidecarSpec(
            tool_name=self.tool_name or default_name,
            env_map=dict(self.env_map),
        )


class RawInstall(StrictModel):
    """``install:`` — Dockerfile fragments emitted into the L1 image."""

    depends_on: StrOrStrList = Field(default_factory=list)
    run_as_root: str = ""
    run_as_dev: str = ""

    def to_dataclass(self) -> InstallSpec:
        """Project to a runtime [`InstallSpec`][terok_executor.roster.types.InstallSpec]."""
        return InstallSpec(
            depends_on=tuple(self.depends_on),
            run_as_root=self.run_as_root or "",
            run_as_dev=self.run_as_dev or "",
        )


class RawHelp(StrictModel):
    """``help:`` — one-line entry shown in the in-container help banner."""

    label: str = ""
    section: HelpSection = "agent"

    def to_dataclass(self) -> HelpSpec:
        """Project to a runtime [`HelpSpec`][terok_executor.roster.types.HelpSpec]."""
        return HelpSpec(label=self.label or "", section=self.section)


class RawMountSpec(StrictModel):
    """One entry in the ``mounts:`` list — explicit shared-config mount."""

    host_dir: str
    container_path: str
    label: str | None = None


# ── Generated routes.json contract ────────────────────────────────────────


class VaultRouteEntry(StrictModel):
    """One entry in the generated ``routes.json`` consumed by the sandbox vault.

    The on-disk file is a top-level ``{provider_name: VaultRouteEntry}`` dict.
    Empty optional fields (``path_upstreams``, ``oauth_extra_headers``,
    ``oauth_credential_headers``,
    ``oauth_refresh``) are dropped from the serialized output via
    ``exclude_none``, keeping the produced file small and diff-friendly.
    """

    upstream: str = Field(description="Upstream API base URL")
    auth_header: str = Field(description="HTTP header name for the real credential")
    auth_prefix: str = Field(description='Prefix prepended to the token (e.g. ``"Bearer "``)')
    path_upstreams: dict[str, str] | None = Field(
        default=None, description="Path-prefix → upstream-base overrides"
    )
    oauth_extra_headers: dict[str, str] | None = Field(
        default=None, description="Headers added when forwarding OAuth credentials"
    )
    oauth_credential_headers: dict[str, str] | None = Field(
        default=None, description="Upstream headers sourced from OAuth credential fields"
    )
    oauth_refresh: dict[str, str] | None = Field(
        default=None,
        description="Token-refresh endpoint config (``token_url``, ``client_id``, optional ``scope``)",
    )


# ── Provider schema (resources/providers/*.yaml) ──────────────────────────


class RawProviderAuthMode(StrictModel):
    """Define wire authentication for an ``auth.api_key`` or ``auth.oauth`` block."""

    header: str = Field(
        default="",
        description=(
            "HTTP header that contains the credential. For an empty block, the default "
            "is ``Authorization``. For a block that is not empty, this field is required."
        ),
    )
    prefix: str = Field(
        default="",
        description=(
            "Text before the credential. For an empty block, the default is ``Bearer ``. "
            "If a block contains only ``header``, the prefix stays empty. This behavior "
            "supports legacy files."
        ),
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict, description="Additional headers for this authentication mode."
    )

    @model_validator(mode="before")
    @classmethod
    def _expand_bearer_shorthand(cls, value: object) -> object:
        """Expand an empty block to the standard Authorization and Bearer values.

        A block that is not empty keeps the legacy rules. The block must specify
        its header. If the block does not specify a prefix, the prefix is empty.
        This behavior makes ``api_key: {}`` simple. It does not change existing
        blocks such as ``header: PRIVATE-TOKEN``.
        """
        if isinstance(value, dict) and not value:
            return {"header": "Authorization", "prefix": "Bearer "}
        return value

    @model_validator(mode="after")
    def _require_header(self) -> RawProviderAuthMode:
        """Reject an authentication block that is not empty and has no header."""
        if not self.header:
            raise ValueError("Specify a provider auth header unless the block is empty")
        return self

    def to_dataclass(self) -> ProviderAuth:
        """Project to a runtime [`ProviderAuth`][terok_executor.roster.types.ProviderAuth]."""
        return ProviderAuth(
            header=self.header, prefix=self.prefix, extra_headers=dict(self.extra_headers)
        )


class RawProviderAuth(StrictModel):
    """``auth:`` — at least one of ``api_key`` / ``oauth`` must be present.

    Declaring both with *different* headers is how a provider expresses the
    OAuth-or-API-key header switch (Anthropic); the projection collapses that
    to the ``auth_header: dynamic`` wire sentinel
    (see [`Provider.wire_auth`][terok_executor.roster.types.Provider.wire_auth]).
    """

    api_key: RawProviderAuthMode | None = None
    oauth: RawProviderAuthMode | None = None

    @model_validator(mode="after")
    def _at_least_one_mode(self) -> RawProviderAuth:
        """Reject a provider with no auth mode — a route with no credential is meaningless."""
        if self.api_key is None and self.oauth is None:
            raise ValueError("provider auth must declare at least one of 'api_key' / 'oauth'")
        return self


class RawEgress(StrictModel):
    """``egress:`` — extra runtime destinations a provider legitimately reaches.

    ``allow`` hosts feed the shield's ``provider_allow`` tier (t30) — traffic
    the agent makes *directly* at runtime, additional to the vault-relayed API
    host (e.g. telemetry or a model-listing endpoint).  Build-time fetches do
    **not** belong here: image build runs before the shield attaches.
    """

    allow: list[str] = Field(
        default_factory=list, description="Hosts allowed directly at egress (t30)"
    )


class RawProviderModelLimit(StrictModel):
    """Define optional token limits in ``models.<id>.limit``."""

    context: PositiveInt | None = Field(
        default=None, description="Maximum number of tokens in the context window"
    )
    output: PositiveInt | None = Field(
        default=None, description="Maximum number of tokens in generated output"
    )


class RawProviderModel(StrictModel):
    """Define the name and limits for one model in ``models.<id>``."""

    name: str | None = Field(default=None, description="Model name that users see")
    limit: RawProviderModelLimit = Field(default_factory=RawProviderModelLimit)

    def to_dataclass(self, *, model_id: str) -> ProviderModel:
        """Project to [`ProviderModel`][terok_executor.roster.types.ProviderModel]."""
        return ProviderModel(
            name=self.name or model_id,
            context_limit=self.limit.context,
            output_limit=self.limit.output,
        )


class RawProvider(StrictModel):
    """Full schema for one ``resources/providers/*.yaml`` file.

    The file's stem (``anthropic.yaml`` → ``"anthropic"``) supplies the provider
    name; the YAML never repeats it.  Strict keys reject typos
    (``upstreams:``, ``oath:``) the same way the agent schema does.
    """

    label: str | None = Field(default=None, description="Provider name that users see")
    upstream: str = Field(description="Upstream API base URL")
    auth: RawProviderAuth
    path_upstreams: dict[str, str] = Field(default_factory=dict)
    oauth_credential_headers: dict[str, str] = Field(default_factory=dict)
    oauth_refresh: RawOAuthRefresh | None = None
    shared_domain: bool = False
    egress: RawEgress | None = Field(
        default=None, description="Extra runtime egress hosts allowed at t30"
    )
    serves: dict[str, str] = Field(
        default_factory=dict,
        description="Wire protocol → container-facing base path (LLM providers only)",
    )
    default_model: str | None = Field(
        default=None, description="Default model ID for OpenCode and Pi"
    )
    models: dict[str, RawProviderModel] = Field(
        default_factory=dict, description="Model data. Each key is a model ID."
    )
    opencode: RawOpenCode | None = Field(
        default=None,
        description="OpenCode wrapper config for a harness-driven provider (Blablador, …)",
    )
    install: RawInstall | None = Field(
        default=None, description="Pinned-alias install fragment for a harness-driven provider"
    )
    help: RawHelp | None = Field(default=None, description="Help-listing entry for the command")

    def to_dataclass(self, *, name: str) -> Provider:
        """Project to a runtime [`Provider`][terok_executor.roster.types.Provider]."""
        refresh: dict[str, str] | None = None
        if self.oauth_refresh is not None:
            r = self.oauth_refresh
            refresh = {"token_url": r.token_url, "client_id": r.client_id}
            if r.scope is not None:
                refresh["scope"] = r.scope
        return Provider(
            name=name,
            upstream=self.upstream,
            label=self.label or (self.opencode.display_name if self.opencode else name),
            default_model=self.default_model
            or (self.opencode.preferred_model if self.opencode else None),
            models={
                model_id: model.to_dataclass(model_id=model_id)
                for model_id, model in self.models.items()
            },
            api_key_auth=self.auth.api_key.to_dataclass() if self.auth.api_key else None,
            oauth_auth=self.auth.oauth.to_dataclass() if self.auth.oauth else None,
            path_upstreams=dict(self.path_upstreams),
            oauth_credential_headers=dict(self.oauth_credential_headers),
            oauth_refresh=refresh,
            shared_domain=self.shared_domain,
            egress_allow=tuple(self.egress.allow) if self.egress else (),
            serves=dict(self.serves),
            opencode_config=self.opencode.to_dataclass() if self.opencode else None,
            install_spec=self.install.to_dataclass() if self.install else None,
            help_spec=self.help.to_dataclass() if self.help else None,
        )


class RawProviderBinding(StrictModel):
    """``provider:`` — how an agent routes to (delivers) a provider.

    The *delivery* concern lifted out of the agent ``vault:`` block: which
    provider this agent uses (``default``) plus the per-agent env-var /
    config-patch plumbing.  The *endpoint* concern (upstream, wire auth) lives
    in the named [`RawProvider`][terok_executor.roster.schema.RawProvider].
    """

    default: str | None = Field(
        default=None, description="Provider name this agent routes to (None for harnesses)"
    )
    token_env: dict[str, str] = Field(default_factory=dict)
    token_env_aliases: list[str] = Field(default_factory=list)
    base_url_env: str = ""
    socket_env: str = ""
    credential_file: str = ""
    credential_file_writable: bool = Field(
        default=False,
        description=(
            "Mount the credential file writable instead of under a read-only shadow. "
            "Set when the tool stores credentials and settings in the same file it must "
            "rewrite on startup (e.g. glab's config.yml) — the shadow would block the "
            "write, and the file persists in the shared mount instead of being contained."
        ),
    )
    credential_type: Literal["api_key", "oauth", "oauth_token", "pat"] = "api_key"
    config_patch: dict | None = None

    def to_dataclass(self) -> ProviderBinding:
        """Project to a runtime [`ProviderBinding`][terok_executor.provider.providers.ProviderBinding]."""
        return ProviderBinding(
            default=self.default,
            token_env=dict(self.token_env),
            token_env_aliases=tuple(self.token_env_aliases),
            base_url_env=self.base_url_env,
            socket_env=self.socket_env,
            credential_file=self.credential_file,
            credential_type=self.credential_type,
            config_patch=self.config_patch,
        )


# ── Top-level model ───────────────────────────────────────────────────────


AgentKind = Literal["native", "harness", "frontend", "tool", "infra"]
"""Kind of roster entry, chosen to predict behaviour:

- ``native`` — a CLI with a fixed protocol + default provider (claude, codex, vibe, copilot)
- ``harness`` — a multi-provider driver, provider supplied at runtime (opencode, pi)
- ``frontend`` — drives co-installed agents in-container (toad)
- ``tool`` — a non-LLM API client (gh, glab, sonar, coderabbit)
- ``infra`` — a support service, not user-invoked (caddy)
"""


# Default sub-section instances reused for agents that omit a section —
# all-defaults Pydantic models are immutable in practice for our reads,
# so one shared instance per type avoids re-validating empty input on
# every load.
_DEFAULT_HEADLESS = RawHeadless()
_DEFAULT_AUTO_APPROVE = RawAutoApprove()
_DEFAULT_SESSION = RawSession()
_DEFAULT_CAPABILITIES = RawCapabilities()
_DEFAULT_WRAPPER = RawWrapper()
_DEFAULT_GIT_IDENTITY = RawGitIdentity()

# Git author email for any agent whose roster entry omits ``git_identity.email``.
# Deliberately terok's own domain — never a fabricated vendor address — so an
# unconfigured agent is honestly attributed rather than impersonating a provider.
_FALLBACK_GIT_EMAIL = "noreply@terok.ai"


class RawAgentYaml(StrictModel):
    """Full schema for one agent YAML file.

    The file's stem (e.g. ``claude.yaml`` → ``"claude"``) supplies the
    roster name; the YAML never repeats it inside.  ``roster_version``
    is stripped before validation by the loader's compat check, so it
    is intentionally absent here.
    """

    kind: AgentKind = "native"
    label: str | None = Field(default=None, description="Human-readable display name")
    binary: str | None = Field(
        default=None, description="CLI binary name (defaults to roster name)"
    )
    git_identity: RawGitIdentity | None = None
    headless: RawHeadless | None = None
    auto_approve: RawAutoApprove | None = None
    session: RawSession | None = None
    capabilities: RawCapabilities | None = None
    wrapper: RawWrapper | None = None
    auth: RawAuth | None = None
    protocol: str | None = Field(
        default=None,
        description="Wire protocol the agent speaks (anthropic-messages / openai-chat / …)",
    )
    provider: RawProviderBinding | None = None
    sidecar: RawSidecar | None = None
    install: RawInstall | None = None
    help: RawHelp | None = None
    mounts: list[RawMountSpec] = Field(default_factory=list)
    web_ingress: bool = Field(
        default=False, description="Whether this entry publishes a host HTTP port"
    )

    # ── Derived helpers ──

    def resolve_label(self, name: str) -> str:
        """Return ``label`` or fall back to *name*."""
        return self.label or name

    def to_agent(self, name: str) -> Agent:
        """Project to a runtime [`Agent`][terok_executor.provider.providers.Agent]."""
        hl = self.headless or _DEFAULT_HEADLESS
        aa = self.auto_approve or _DEFAULT_AUTO_APPROVE
        sess = self.session or _DEFAULT_SESSION
        caps = self.capabilities or _DEFAULT_CAPABILITIES
        wrap = self.wrapper or _DEFAULT_WRAPPER
        gi = self.git_identity or _DEFAULT_GIT_IDENTITY
        return Agent(
            name=name,
            label=self.resolve_label(name),
            binary=self.binary or name,
            git_author_name=gi.name or name.capitalize(),
            git_author_email=gi.email or _FALLBACK_GIT_EMAIL,
            headless_subcommand=hl.subcommand,
            prompt_flag=hl.prompt_flag,
            auto_approve_env=dict(aa.env),
            auto_approve_flags=tuple(aa.flags),
            output_format_flags=tuple(hl.output_format_flags),
            model_flag=hl.model_flag,
            max_turns_flag=hl.max_turns_flag,
            verbose_flag=hl.verbose_flag,
            supports_session_resume=sess.supports_resume,
            resume_flag=sess.resume_flag,
            resume_subcommand=sess.resume_subcommand,
            continue_flag=sess.continue_flag,
            session_file=sess.session_file,
            supports_session_hook=sess.supports_hook,
            supports_add_dir=caps.add_dir,
            log_format=caps.log_format,
            refuse_subcommands=tuple(wrap.refuse_subcommands),
            protocol=self.protocol,
            provider_binding=self.provider.to_dataclass() if self.provider else None,
            launcher=wrap.launcher.to_dataclass() if wrap.launcher else None,
        )


__all__ = [
    "HELP_SECTIONS",
    "AgentKind",
    "RawAgentYaml",
    "RawAuth",
    "RawAuthKey",
    "RawAutoApprove",
    "RawCapabilities",
    "RawEgress",
    "RawGitIdentity",
    "RawHeadless",
    "RawHelp",
    "RawInstall",
    "RawMountSpec",
    "RawOAuthRefresh",
    "RawLauncher",
    "RawOpenCode",
    "RawProvider",
    "RawProviderAuth",
    "RawProviderAuthMode",
    "RawProviderBinding",
    "RawProviderModel",
    "RawProviderModelLimit",
    "RawSession",
    "RawSidecar",
    "RawWrapper",
    "VaultRouteEntry",
]
