# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Pydantic v2 schema for agent-roster YAML files.

Each ``resources/agents/*.yaml`` (and any user override under
``~/.config/terok/agent/agents/*.yaml``) is parsed into [`RawAgentYaml`][terok_executor.roster.schema.RawAgentYaml]
before being projected onto the runtime dataclasses
([`AgentProvider`][terok_executor.provider.providers.AgentProvider],
[`AuthProvider`][terok_executor.credentials.auth.AuthProvider],
[`VaultRoute`][terok_executor.roster.types.VaultRoute], …).

Validation guarantees:

- **Strict keys**: every section uses ``extra="forbid"``, so a typo
  (``headles:``, ``oauth_refesh:``) fails fast with a precise error
  instead of silently falling back to defaults.
- **Type-checked values**: ``modes`` only accepts ``oauth``/``api_key``,
  ``credential_type`` only accepts the four known kinds, ``help.section``
  only accepts ``agent``/``dev_tool``.
- **Required fields**: ``vault.route_prefix``, ``vault.upstream``,
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

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from terok_executor.credentials.auth import AuthKeyConfig, AuthProvider, api_key_command
from terok_executor.provider.providers import AgentProvider, OpenCodeProviderConfig

from .types import HELP_SECTIONS, HelpSection, HelpSpec, InstallSpec, SidecarSpec, VaultRoute

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
    continue_flag: str | None = None
    session_file: str | None = None
    supports_hook: bool = False


class RawCapabilities(StrictModel):
    """``capabilities:`` — agent-specific feature toggles."""

    agents_json: bool = False
    add_dir: bool = False
    log_format: Literal["plain", "claude-stream-json"] = "plain"


class RawWrapper(StrictModel):
    """``wrapper:`` — in-container shell-wrapper behavior."""

    refuse_subcommands: list[str] = Field(default_factory=list)


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
        """Project to the runtime [`OpenCodeProviderConfig`][terok_executor.provider.providers.OpenCodeProviderConfig]."""
        return OpenCodeProviderConfig(
            display_name=self.display_name,
            base_url=self.base_url,
            preferred_model=self.preferred_model,
            fallback_model=self.fallback_model,
            env_var_prefix=self.env_var_prefix,
            config_dir=self.config_dir,
            auth_key_url=self.auth_key_url,
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
    modes: list[Literal["oauth", "api_key"]] = Field(default_factory=lambda: ["api_key"])
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
            api_key_hint=self.api_key_hint,
            post_capture_state=dict(self.post_capture_state),
        )


class RawOAuthRefresh(StrictModel):
    """``vault.oauth_refresh:`` — token-refresh endpoint and client config."""

    token_url: str
    client_id: str
    scope: str | None = None


class RawVault(StrictModel):
    """``vault:`` — proxy route + credential-injection rules."""

    route_prefix: str = Field(description="Path prefix in the proxy (e.g. ``claude``)")
    upstream: str = Field(description="Upstream API base URL")
    path_upstreams: dict[str, str] = Field(default_factory=dict)
    oauth_extra_headers: dict[str, str] = Field(default_factory=dict)
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    credential_type: Literal["api_key", "oauth", "oauth_token", "pat"] = "api_key"
    credential_file: str = ""
    phantom_env: dict[str, bool] = Field(default_factory=dict)
    oauth_phantom_env: dict[str, bool] = Field(default_factory=dict)
    base_url_env: str = ""
    socket_env: str = ""
    shared_config_patch: dict | None = None
    """Free-form dict consumed by the post-auth config patcher (TOML/YAML set ops)."""
    oauth_refresh: RawOAuthRefresh | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_socket_path(cls, data: object) -> object:
        if isinstance(data, dict) and "socket_path" in data:
            raise ValueError(
                "'socket_path' is no longer configurable — "
                "remove it; the env builder resolves the vault socket path centrally"
            )
        return data

    def to_dataclass(self, *, provider: str) -> VaultRoute:
        """Project to a runtime [`VaultRoute`][terok_executor.roster.types.VaultRoute]."""
        refresh: dict[str, str] | None = None
        if self.oauth_refresh is not None:
            r = self.oauth_refresh
            refresh = {"token_url": r.token_url, "client_id": r.client_id}
            if r.scope is not None:
                refresh["scope"] = r.scope
        return VaultRoute(
            provider=provider,
            route_prefix=self.route_prefix,
            upstream=self.upstream,
            path_upstreams=dict(self.path_upstreams),
            oauth_extra_headers=dict(self.oauth_extra_headers),
            auth_header=self.auth_header,
            auth_prefix=self.auth_prefix,
            credential_type=self.credential_type,
            credential_file=self.credential_file,
            phantom_env=dict(self.phantom_env),
            oauth_phantom_env=dict(self.oauth_phantom_env),
            base_url_env=self.base_url_env,
            socket_env=self.socket_env,
            shared_config_patch=self.shared_config_patch,
            oauth_refresh=refresh,
        )


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


# ── Top-level model ───────────────────────────────────────────────────────


AgentKind = Literal["native", "opencode", "bridge", "tool", "runtime"]
"""Kind of roster entry: agents (native/opencode/bridge), tools, or runtime helpers."""


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
    opencode: RawOpenCode | None = None
    auth: RawAuth | None = None
    vault: RawVault | None = None
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

    def to_agent_provider(self, name: str) -> AgentProvider:
        """Project to a runtime [`AgentProvider`][terok_executor.provider.providers.AgentProvider]."""
        hl = self.headless or _DEFAULT_HEADLESS
        aa = self.auto_approve or _DEFAULT_AUTO_APPROVE
        sess = self.session or _DEFAULT_SESSION
        caps = self.capabilities or _DEFAULT_CAPABILITIES
        wrap = self.wrapper or _DEFAULT_WRAPPER
        gi = self.git_identity or _DEFAULT_GIT_IDENTITY
        return AgentProvider(
            name=name,
            label=self.resolve_label(name),
            binary=self.binary or name,
            git_author_name=gi.name or name.capitalize(),
            git_author_email=gi.email or f"noreply@{name}.ai",
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
            continue_flag=sess.continue_flag,
            session_file=sess.session_file,
            supports_agents_json=caps.agents_json,
            supports_session_hook=sess.supports_hook,
            supports_add_dir=caps.add_dir,
            log_format=caps.log_format,
            opencode_config=self.opencode.to_dataclass() if self.opencode else None,
            refuse_subcommands=tuple(wrap.refuse_subcommands),
        )

    def derive_opencode_auth(self, name: str) -> AuthProvider | None:
        """Auto-derive an [`AuthProvider`][terok_executor.credentials.auth.AuthProvider] for OpenCode-based agents."""
        if self.opencode is None:
            return None
        hint = self.opencode.api_key_hint or f"Get your API key at: {self.opencode.auth_key_url}"
        return AuthProvider(
            name=name,
            label=self.resolve_label(name),
            host_dir_name=f"_{name}-config",
            container_mount=f"/home/dev/{self.opencode.config_dir}",
            command=[],
            banner_hint="",
            modes=("api_key",),
            api_key_hint=hint,
        )


__all__ = [
    "HELP_SECTIONS",
    "AgentKind",
    "RawAgentYaml",
    "RawAuth",
    "RawAuthKey",
    "RawAutoApprove",
    "RawCapabilities",
    "RawGitIdentity",
    "RawHeadless",
    "RawHelp",
    "RawInstall",
    "RawMountSpec",
    "RawOAuthRefresh",
    "RawOpenCode",
    "RawSession",
    "RawSidecar",
    "RawVault",
    "RawWrapper",
]
