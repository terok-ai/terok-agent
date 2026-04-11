# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Andreas Knüpfer
# SPDX-License-Identifier: Apache-2.0

"""Agent provider registry: definitions, lookup, and environment collection.

Each supported AI coding agent is described by an :class:`AgentProvider`
dataclass.  The ``AGENT_PROVIDERS`` dict maps short names to descriptors
and is populated at package load time from the YAML roster.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpenCodeProviderConfig:
    """Immutable descriptor for an OpenCode-based provider wrapper."""

    display_name: str
    """Human-readable display name (e.g., 'Helmholtz Blablador')."""

    base_url: str
    """Base URL for the OpenAI-compatible API (e.g., 'https://api.helmholtz-blablador.fz-juelich.de/v1')."""

    preferred_model: str
    """Preferred model ID (e.g., 'alias-huge')."""

    fallback_model: str
    """Fallback model ID if preferred is unavailable (e.g., 'alias-code')."""

    env_var_prefix: str
    """Environment variable prefix for API key (e.g., 'BLABLADOR' → BLABLADOR_API_KEY)."""

    config_dir: str
    """Configuration directory name (e.g., '.blablador')."""

    auth_key_url: str
    """URL where users can obtain API keys for documentation."""

    def to_env(self, name: str) -> dict[str, str]:
        """Return env vars for container injection, keyed by TEROK_OC_{NAME}_*."""
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
class AgentProvider:
    """Describes how to run one AI coding agent (all modes: interactive + headless)."""

    name: str
    """Short key used in CLI dispatch (e.g. ``"claude"``, ``"codex"``)."""

    label: str
    """Human-readable display name (e.g. ``"Claude"``, ``"Codex"``)."""

    binary: str
    """CLI binary name (e.g. ``"claude"``, ``"codex"``, ``"opencode"``)."""

    git_author_name: str
    """AI identity name for Git author/committer policy application."""

    git_author_email: str
    """AI identity email for Git author/committer policy application."""

    # -- Headless command construction --

    headless_subcommand: str | None
    """Subcommand for headless mode (e.g. ``"exec"`` for codex, ``"run"`` for opencode).

    ``None`` means the binary uses flags only (e.g. ``claude -p``).
    """

    prompt_flag: str
    """Flag for passing the prompt.

    ``"-p"`` for flag-based, ``""`` for positional (after subcommand).
    """

    auto_approve_env: dict[str, str]
    """Environment variables for fully autonomous execution.

    Injected into the container env by ``_apply_unrestricted_env()`` when
    ``TEROK_UNRESTRICTED=1``.  Read by agents regardless of launch path.
    Claude uses ``/etc/claude-code/managed-settings.json`` instead.
    """

    auto_approve_flags: tuple[str, ...]
    """CLI flags injected by the shell wrapper when ``TEROK_UNRESTRICTED=1``.

    Only for agents that lack an env var or managed config mechanism
    (currently Codex only).  Empty for all other agents — their env vars
    and ``/etc/`` config files handle permissions across all launch paths.
    """

    output_format_flags: tuple[str, ...]
    """Flags for structured output (e.g. ``("--output-format", "stream-json")``)."""

    model_flag: str | None
    """Flag for model override (``"--model"``, ``"--agent"``, or ``None``)."""

    max_turns_flag: str | None
    """Flag for maximum turns (``"--max-turns"`` or ``None``)."""

    verbose_flag: str | None
    """Flag for verbose output (``"--verbose"`` or ``None``)."""

    # -- Session support --

    supports_session_resume: bool
    """Whether the provider supports resuming a previous session."""

    resume_flag: str | None
    """Flag to resume a session (e.g. ``"--resume"``, ``"--session"``)."""

    continue_flag: str | None
    """Flag to continue a session (e.g. ``"--continue"``)."""

    session_file: str | None
    """Filename in ``/home/dev/.terok/`` for stored session ID.

    Providers that capture session IDs via plugin or post-run parsing set this
    to a filename (e.g. ``"opencode-session.txt"``).  Providers with their own
    hook mechanism (Claude) or no session support set this to ``None``.
    """

    # -- Claude-specific capabilities --

    supports_agents_json: bool
    """Whether the provider supports ``--agents`` JSON (Claude only)."""

    supports_session_hook: bool
    """Whether the provider supports SessionStart hooks (Claude only)."""

    supports_add_dir: bool
    """Whether the provider supports ``--add-dir "/"`` (Claude only)."""

    # -- Log formatting --

    log_format: str
    """Log format identifier: ``"claude-stream-json"`` or ``"plain"``."""

    opencode_config: OpenCodeProviderConfig | None = None
    """Configuration for OpenCode-based providers (Blablador, KISSKI, etc.).

    When set, this provider uses OpenCode with a custom OpenAI-compatible API.
    The configuration includes API endpoints, model preferences, and provider-specific
    settings that are injected into the container environment.
    """

    @property
    def uses_opencode_instructions(self) -> bool:
        """Whether the provider uses OpenCode's instruction system."""
        return self.opencode_config is not None or self.name == "opencode"


# ---------------------------------------------------------------------------
# Provider registry — populated from YAML by __init__.py at package load time
# ---------------------------------------------------------------------------

AGENT_PROVIDERS: dict[str, AgentProvider] = {}
"""All agent providers, keyed by name.  Loaded from ``resources/agents/*.yaml``."""

PROVIDER_NAMES: tuple[str, ...] = ()


# ── Public API ──────────────────────────────────────────────────────────────


def resolve_provider(
    providers: dict[str, AgentProvider],
    name: str | None,
    *,
    default_agent: str | None = None,
) -> AgentProvider:
    """Look up a provider by name from *providers*, with fallback chain.

    Resolution order: explicit *name* → *default_agent* → ``"claude"``.
    Raises ``SystemExit`` if the resolved name is not found.
    """
    resolved = name or default_agent or "claude"
    provider = providers.get(resolved)
    if provider is None:
        valid = ", ".join(sorted(providers))
        raise SystemExit(f"Unknown provider {resolved!r}. Valid providers: {valid}")
    return provider


def get_provider(name: str | None, *, default_agent: str | None = None) -> AgentProvider:
    """Resolve a provider name against the global :data:`AGENT_PROVIDERS` registry.

    Convenience wrapper around :func:`resolve_provider`.
    """
    return resolve_provider(AGENT_PROVIDERS, name, default_agent=default_agent)


def collect_all_auto_approve_env() -> dict[str, str]:
    """Collect ``auto_approve_env`` from all providers into one dict.

    Used by task runners to inject these env vars at the container level
    (not just inside shell wrappers) so that ACP-spawned agents also
    inherit unrestricted permissions.
    """
    merged: dict[str, str] = {}
    for p in AGENT_PROVIDERS.values():
        for key, value in p.auto_approve_env.items():
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"Conflicting auto_approve_env for {key!r}: "
                    f"{merged[key]!r} vs {value!r} (provider {p.name!r})"
                )
            merged[key] = value
    return merged


def collect_opencode_provider_env() -> dict[str, str]:
    """Collect environment variables for all OpenCode-based providers.

    Returns a dictionary of environment variables that will be injected into containers
    to configure OpenCode-based providers. Each provider with opencode_config set
    contributes variables prefixed with TEROK_OC_{PROVIDER_NAME}_*.
    """
    env: dict[str, str] = {}
    for provider in AGENT_PROVIDERS.values():
        if provider.opencode_config is not None:
            env.update(provider.opencode_config.to_env(provider.name))
    return env
