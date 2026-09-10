# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Andreas Knüpfer
# SPDX-License-Identifier: Apache-2.0

"""Agent registry + per-agent behaviour.

Each supported AI coding agent is described by an
[`Agent`][terok_executor.provider.providers.Agent]
dataclass that owns both its capability shape (flags, environment,
session handling) and the behaviour bound to that shape — config
resolution ([`Agent.apply_config`][terok_executor.provider.providers.Agent.apply_config])
and headless-command assembly ([`Agent.build_headless_command`][terok_executor.provider.providers.Agent.build_headless_command]).

The ``AGENTS`` dict maps short names to descriptors and is
populated at package load time from the YAML roster.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentRunConfig:
    """Resolved per-run config for a headless provider.

    Produced by [`Agent.apply_config`][terok_executor.provider.providers.Agent.apply_config]
    after best-effort feature mapping.
    """

    model: str | None
    """Model override for providers that support it, else ``None``."""

    max_turns: int | None
    """Max turns for providers that support it, else ``None``."""

    timeout: int
    """Effective timeout in seconds."""

    prompt_extra: str
    """Extra text to append to the prompt (best-effort feature analogues)."""

    warnings: tuple[str, ...]
    """Warnings about unsupported features (for user display)."""


@dataclass(frozen=True)
class CLIOverrides:
    """CLI flag overrides for a headless agent run."""

    model: str | None = None
    """Explicit ``--model`` from CLI (takes precedence over config)."""

    max_turns: int | None = None
    """Explicit ``--max-turns`` from CLI."""

    timeout: int | None = None
    """Explicit ``--timeout`` from CLI."""

    instructions: str | None = None
    """Resolved instructions text. Delivery is provider-aware."""


def resolve_agent_value(
    key: str,
    config: dict[str, Any],
    agent_name: str,
) -> Any | None:
    """Extract an agent-aware config value.

    Supports two forms:

    * **Flat value** — ``model: opus`` → same for all agents.
    * **Per-agent dict** — ``model: {claude: opus, codex: o3, _default: fast}``
      → looks up *agent_name*, falls back to ``_default``, then ``None``.

    Returns ``None`` when the key is absent or has no match for the agent.

    **Null override behaviour**: when a per-agent dict maps an agent to
    ``null`` (Python ``None``), that ``None`` is treated as "no value" and the
    resolver falls back to ``_default``.  This is intentional — it allows a
    lower-priority config layer to set an agent-specific value that a
    higher-priority layer can effectively *unset* by mapping it to ``null``,
    letting the ``_default`` (or ``None``) bubble up instead.

    Internal to agent config resolution — full config-stack composition
    (``build_agent_config_stack``, ``resolve_agent_config``) lives in terok,
    which owns the global/project/preset layer semantics.
    """
    val = config.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        agent_val = val.get(agent_name)
        if agent_val is not None:
            return agent_val
        return val.get("_default")
    return val


@dataclass(frozen=True)
class ProviderBinding:
    """How one agent consumes a provider — the *delivery* concern.

    Lifted out of the agent ``vault:`` block: which provider this agent routes
    to by default, and the per-agent env-var / config-patch plumbing that
    carries the phantom token and vault base URL into the agent's own
    configuration.  The *endpoint* concern (upstream, wire auth) lives on the
    [`Provider`][terok_executor.roster.types.Provider] this binding names.
    """

    default: str | None = None
    """Provider name this agent routes to by default.

    Set for agents with a fixed provider (natives → their LLM provider; tools
    → their API provider; the OpenCode-shim agents → their endpoint).  ``None``
    for harnesses (opencode, pi), which take a provider at runtime.
    """

    token_env: dict[str, str] = field(default_factory=dict)
    """Phantom-token env var name, keyed by stored credential type.

    Keys are credential types (``"oauth"``, ``"pat"``, …); ``"_default"`` is the
    fallback.  Most agents read one var (``{"_default": "MISTRAL_API_KEY"}``);
    Claude swaps on OAuth (``{"oauth": "CLAUDE_CODE_OAUTH_TOKEN", "_default":
    "ANTHROPIC_API_KEY"}``).
    """

    token_env_aliases: tuple[str, ...] = ()
    """Additional env vars that receive the same phantom token."""

    base_url_env: str = ""
    """Env var overridden with the vault's HTTP URL (e.g. ``"ANTHROPIC_BASE_URL"``)."""

    socket_env: str = ""
    """Env var that receives the container-side vault socket path (HTTP-over-UNIX agents)."""

    credential_file: str = ""
    """Credential file path relative to the auth mount (e.g. ``".credentials.json"``).

    Drives the read-only credential shadow ([terok-ai/terok#873](https://github.com/terok-ai/terok/issues/873))
    and the doctor's leaked-secret probe.
    """

    credential_type: str = "api_key"
    """Default credential type captured for this agent (``"oauth"``/``"api_key"``/
    ``"oauth_token"``/``"pat"``).  The runtime token selection uses the *stored*
    type from the vault DB; this is the authoring default."""

    config_patch: dict | None = None
    """Optional shared-config patch applied after auth (e.g. Codex/Vibe config.toml)."""


LAUNCHER_ON_PROVIDER_SELECT = "on_provider_select"
"""Launcher mode: run only when a provider override is selected (default = bare binary)."""

LAUNCHER_ALWAYS = "always"
"""Launcher mode: run on every invocation (the launcher does per-run prep)."""


@dataclass(frozen=True)
class Launcher:
    """How an agent's wrapper hands off to a per-task launcher script.

    A launcher script preps the agent — rewrites provider config, scopes a
    picker, injects instructions — before exec-ing the binary.  ``mode`` decides
    when the wrapper routes through it:

    - [`LAUNCHER_ON_PROVIDER_SELECT`][terok_executor.provider.providers.LAUNCHER_ON_PROVIDER_SELECT]
      — only when a provider override is selected; the default path runs the bare
      binary (a config-rewriting launcher has nothing to do without a selection).
    - [`LAUNCHER_ALWAYS`][terok_executor.provider.providers.LAUNCHER_ALWAYS] —
      every invocation (the launcher does per-run prep the bare binary can't,
      such as instruction injection).
    """

    script: str
    """Launcher script name on PATH (e.g. ``"pi-provider"``)."""

    mode: str
    """When the launcher runs — one of the ``LAUNCHER_*`` modes."""


@dataclass(frozen=True)
class Agent:
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
    """Whether the agent supports resuming a previous session."""

    resume_flag: str | None
    """Flag to resume a session (e.g. ``"--resume"``, ``"--session"``)."""

    resume_subcommand: str | None
    """Subcommand that resumes a session, for CLIs that phrase it that way.

    Codex: ``codex resume <id>`` interactively, ``codex exec resume <id> …``
    headless — the wrapper leads with it, or nests it under the headless
    subcommand.  Mutually exclusive with ``resume_flag``.
    """

    continue_flag: str | None
    """Flag to continue a session (e.g. ``"--continue"``)."""

    session_file: str | None
    """Filename in ``/home/dev/.terok/`` for stored session ID.

    Providers that capture session IDs via plugin, post-run parsing, or a
    hook installed in the image (Codex) set this to a filename (e.g.
    ``"opencode-session.txt"``).  Claude's hook writes a path the wrapper
    hard-codes, and providers without session support set this to ``None``.
    """

    # -- Claude-specific capabilities --

    supports_session_hook: bool
    """Whether the agent supports SessionStart hooks (Claude only)."""

    supports_add_dir: bool
    """Whether the agent supports ``--add-dir "/"`` (Claude only)."""

    # -- Log formatting --

    log_format: str
    """Log format identifier: ``"claude-stream-json"`` or ``"plain"``."""

    refuse_subcommands: tuple[str, ...] = ()
    """Subcommands the in-container wrapper refuses with a friendly error.

    Used to block credential-handling flows (``login``, ``logout``,
    ``setup-token``) that would otherwise pollute the host-shared mount —
    operators authenticate on the host via ``terok auth`` instead.  Best
    effort only; the firewall is the actual enforcement
    ([terok-ai/terok#873](https://github.com/terok-ai/terok/issues/873)).
    """

    protocol: str | None = None
    """Wire protocol the agent speaks (``"anthropic-messages"`` / ``"openai-chat"`` /
    ``"openai-responses"``), or ``None`` for non-LLM tools.  Matched against a
    provider's ``serves`` to resolve a runtime agent×provider combo."""

    provider_binding: ProviderBinding | None = None
    """How this agent routes to a provider — see [`ProviderBinding`][terok_executor.provider.providers.ProviderBinding].

    ``None`` for entries with no vault route of their own (harnesses, copilot)."""

    launcher: Launcher | None = None
    """Per-task launcher the wrapper hands off to — see [`Launcher`][terok_executor.provider.providers.Launcher].

    ``None`` runs the bare binary (any provider override arriving via env instead,
    e.g. Claude)."""

    @property
    def uses_opencode_instructions(self) -> bool:
        """Whether the agent uses OpenCode's instruction system.

        Only the OpenCode harness itself; the curated OpenCode-driven providers
        (Blablador, KISSKI, OpenRouter) are no longer agents — they run through
        the same ``opencode`` wrapper, which already carries this behaviour.
        """
        return self.name == "opencode"

    # ── Headless behaviour ───────────────────────────────────────────

    def apply_config(
        self,
        config: dict[str, Any],
        overrides: CLIOverrides | None = None,
    ) -> AgentRunConfig:
        """Resolve config values for this provider with best-effort feature mapping.

        CLI flag *overrides* take precedence over *config* values.  When this
        provider lacks a feature, an analogue is used where possible (e.g.
        injecting max-turns guidance into the prompt), and a warning is
        emitted for features that have no analogue.
        """
        if overrides is None:
            overrides = CLIOverrides()

        warnings: list[str] = []
        prompt_parts: list[str] = []

        # --- Model ---
        cfg_model = resolve_agent_value("model", config, self.name)
        model = overrides.model or (str(cfg_model) if cfg_model is not None else None)
        if model and not self.model_flag:
            warnings.append(
                f"{self.label} does not support model selection; ignoring model={model!r}"
            )
            model = None

        # --- Max turns ---
        cfg_turns = resolve_agent_value("max_turns", config, self.name)
        max_turns_raw = overrides.max_turns if overrides.max_turns is not None else cfg_turns
        max_turns: int | None = int(max_turns_raw) if max_turns_raw is not None else None
        if max_turns is not None and not self.max_turns_flag:
            # Best-effort: inject into prompt as guidance
            prompt_parts.append(f"Important: complete this task in no more than {max_turns} steps.")
            warnings.append(
                f"{self.label} does not support --max-turns; "
                f"added guidance to prompt instead ({max_turns} steps)"
            )
            max_turns = None

        # --- Timeout ---
        cfg_timeout = resolve_agent_value("timeout", config, self.name)
        timeout = (
            overrides.timeout
            if overrides.timeout is not None
            else (int(cfg_timeout) if cfg_timeout is not None else 1800)
        )

        # --- Instructions ---
        # Claude receives instructions via --append-system-prompt in the wrapper.
        # Codex receives instructions via -c model_instructions_file=... in the wrapper.
        # OpenCode-based providers receive instructions via opencode.json `instructions`
        # array (injected by prepare_agent_config_dir).
        # Remaining providers get best-effort prompt prepending.
        instructions = overrides.instructions
        if (
            instructions
            and self.name not in {"claude", "codex"}
            and not self.uses_opencode_instructions
        ):
            prompt_parts.insert(0, instructions)

        return AgentRunConfig(
            model=model,
            max_turns=max_turns,
            timeout=timeout,
            prompt_extra="\n".join(prompt_parts),
            warnings=tuple(warnings),
        )

    def build_headless_command(
        self,
        *,
        timeout: int,
        model: str | None = None,
        max_turns: int | None = None,
    ) -> str:
        """Assemble the bash command string for a headless agent run.

        The command assumes:

        - ``init-ssh-and-repo.sh`` has already set up the workspace
        - The prompt is in ``/home/dev/.terok/prompt.txt``
        - For Claude, the ``claude()`` wrapper function is sourced via ``bash -l``

        Returns a bash command string suitable for ``["bash", "-lc", cmd]``.
        Dispatches to provider-specific assembly: Claude routes through the
        shell wrapper (which adds ``--add-dir``, git env); everything else
        uses the generic shape with subcommand + flags.
        """
        if self.name == "claude":
            return self._build_claude_command(timeout=timeout, model=model, max_turns=max_turns)
        return self._build_generic_command(timeout=timeout, model=model, max_turns=max_turns)

    def _build_claude_command(
        self,
        *,
        timeout: int,
        model: str | None,
        max_turns: int | None,
    ) -> str:
        """Build the headless command for Claude using the wrapper function.

        Claude uses the ``claude()`` wrapper from ``terok-executor.sh`` which
        handles ``--add-dir``, git env, and timeout.
        """
        flags = ""
        if model:
            flags += f" --model {shlex.quote(model)}"
        if max_turns:
            flags += f" --max-turns {int(max_turns)}"

        return (
            f"init-ssh-and-repo.sh &&"
            f" claude --terok-timeout {timeout}"
            f" -p "
            '"$(cat /home/dev/.terok/prompt.txt)"'
            f"{flags} --output-format stream-json --verbose"
        )

    def _build_generic_command(
        self,
        *,
        timeout: int,
        model: str | None,
        max_turns: int | None,
    ) -> str:
        """Build the headless command for non-Claude providers.

        Uses the shell wrapper function (e.g. ``codex()``) instead of invoking the
        binary directly, so that git env vars and session resume logic from
        ``terok-executor.sh`` are applied.  The wrapper parses ``--terok-timeout``
        to wrap the actual invocation with ``timeout``.
        """
        parts = ["init-ssh-and-repo.sh &&"]

        # Call the wrapper function (sourced via bash -l from profile.d);
        # it handles git identity env vars and session resume args.
        parts.append(self.binary)
        parts.append("--terok-timeout")
        parts.append(str(int(timeout)))

        # Subcommand (e.g. "exec" for codex, "run" for opencode)
        if self.headless_subcommand:
            parts.append(self.headless_subcommand)

        # Auto-approve flags are injected by the shell wrapper (wrappers.py)
        # based on TEROK_UNRESTRICTED env var — not here.

        # Model
        if model and self.model_flag:
            parts.append(self.model_flag)
            parts.append(shlex.quote(model))

        # Max turns
        if max_turns and self.max_turns_flag:
            parts.append(self.max_turns_flag)
            parts.append(str(int(max_turns)))

        # Output format
        for flag in self.output_format_flags:
            parts.append(flag)

        # Verbose
        if self.verbose_flag:
            parts.append(self.verbose_flag)

        # Prompt — flag-based or positional
        if self.prompt_flag:
            parts.append(self.prompt_flag)
        parts.append('"$(cat /home/dev/.terok/prompt.txt)"')

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Agent registry — populated from YAML by __init__.py at package load time
# ---------------------------------------------------------------------------

AGENTS: dict[str, Agent] = {}
"""All agents, keyed by name.  Loaded from ``resources/agents/*.yaml``."""

AGENT_NAMES: tuple[str, ...] = ()

OPENCODE_PROVIDERS: dict[str, str] = {}
"""OpenCode-driven provider name → config-dir, for the curated harness providers
(Blablador, KISSKI, OpenRouter).  Populated from the roster's providers at load
time so this layer can drive per-provider config injection without importing the
roster (which would cycle)."""


# ── Public API ──────────────────────────────────────────────────────────────


def resolve_agent(
    agents: dict[str, Agent],
    name: str | None,
    *,
    default_agent: str | None = None,
) -> Agent:
    """Look up an agent by name from *agents*, with fallback chain.

    Resolution order: explicit *name* → *default_agent* → ``"claude"``.
    Raises ``SystemExit`` if the resolved name is not found.
    """
    resolved = name or default_agent or "claude"
    agent = agents.get(resolved)
    if agent is None:
        valid = ", ".join(sorted(agents))
        raise SystemExit(f"Unknown agent {resolved!r}. Valid agents: {valid}")
    return agent


def get_agent(name: str | None, *, default_agent: str | None = None) -> Agent:
    """Resolve an agent name against the global [`AGENTS`][terok_executor.provider.providers.AGENTS] registry.

    Convenience wrapper around [`resolve_agent`][terok_executor.provider.providers.resolve_agent].
    """
    return resolve_agent(AGENTS, name, default_agent=default_agent)
