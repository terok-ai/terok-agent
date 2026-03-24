# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""YAML-driven agent and tool registry.

Loads per-agent definition files from bundled package resources and optional
user extensions, deserializes them into the existing dataclass types, and
provides the same query API that ``headless_providers`` and ``auth`` expose
today.

Directory layout::

    resources/agents/claude.yaml      (bundled, shipped in wheel)
    resources/agents/codex.yaml
    ...
    ~/.config/terok-agent/agents/     (user overrides / additions)
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from .config_stack import deep_merge

if TYPE_CHECKING:
    from .auth import AuthProvider
    from .headless_providers import HeadlessProvider, OpenCodeProviderConfig

# ---------------------------------------------------------------------------
# User config root
# ---------------------------------------------------------------------------

_USER_AGENTS_DIR_NAME = "agents"


def _user_agents_dir() -> Path:
    """Return ``~/.config/terok-agent/agents/``."""
    try:
        from platformdirs import user_config_dir
    except ImportError:  # pragma: no cover
        return Path.home() / ".config" / "terok-agent" / _USER_AGENTS_DIR_NAME
    return Path(user_config_dir("terok-agent")) / _USER_AGENTS_DIR_NAME


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------


def _load_yaml(text: str) -> dict:
    """Parse YAML text into a dict via ruamel.yaml round-trip loader."""
    from ._util import yaml_load

    result = yaml_load(text)
    return result if isinstance(result, dict) else {}


def _load_bundled_agents() -> dict[str, dict]:
    """Load all ``*.yaml`` files from the bundled ``resources/agents/`` package."""
    agents: dict[str, dict] = {}
    pkg = importlib.resources.files("terok_agent.resources.agents")
    for item in pkg.iterdir():
        if not hasattr(item, "name") or not item.name.endswith(".yaml"):
            continue
        name = item.name.removesuffix(".yaml")
        data = _load_yaml(item.read_text(encoding="utf-8"))
        if data:
            agents[name] = data
    return agents


def _load_user_agents() -> dict[str, dict]:
    """Load user override/addition YAML files from ``~/.config/terok-agent/agents/``."""
    agents: dict[str, dict] = {}
    user_dir = _user_agents_dir()
    if not user_dir.is_dir():
        return agents
    for path in sorted(user_dir.glob("*.yaml")):
        name = path.stem
        data = _load_yaml(path.read_text(encoding="utf-8"))
        if data:
            agents[name] = data
    return agents


# ---------------------------------------------------------------------------
# Deserialization: YAML dict → dataclass
# ---------------------------------------------------------------------------


def _to_opencode_config(data: dict) -> OpenCodeProviderConfig:
    """Deserialize the ``opencode:`` YAML section."""
    from .headless_providers import OpenCodeProviderConfig

    return OpenCodeProviderConfig(
        display_name=data["display_name"],
        base_url=data["base_url"],
        preferred_model=data["preferred_model"],
        fallback_model=data["fallback_model"],
        env_var_prefix=data["env_var_prefix"],
        config_dir=data["config_dir"],
        auth_key_url=data["auth_key_url"],
    )


def _to_headless_provider(name: str, data: dict) -> HeadlessProvider:
    """Deserialize a full agent YAML dict into a ``HeadlessProvider``."""
    from .headless_providers import HeadlessProvider

    hl = data.get("headless", {})
    aa = data.get("auto_approve", {})
    sess = data.get("session", {})
    caps = data.get("capabilities", {})
    gi = data.get("git_identity", {})

    oc_data = data.get("opencode")
    oc = _to_opencode_config(oc_data) if oc_data else None

    return HeadlessProvider(
        name=name,
        label=data.get("label", name),
        binary=data.get("binary", name),
        git_author_name=gi.get("name", name.capitalize()),
        git_author_email=gi.get("email", f"noreply@{name}.ai"),
        # Headless command shape
        headless_subcommand=hl.get("subcommand"),
        prompt_flag=hl.get("prompt_flag", "-p"),
        auto_approve_env=aa.get("env", {}),
        auto_approve_flags=tuple(aa.get("flags", ())),
        output_format_flags=tuple(hl.get("output_format_flags", ())),
        model_flag=hl.get("model_flag"),
        max_turns_flag=hl.get("max_turns_flag"),
        verbose_flag=hl.get("verbose_flag"),
        # Session
        supports_session_resume=sess.get("supports_resume", False),
        resume_flag=sess.get("resume_flag"),
        continue_flag=sess.get("continue_flag"),
        session_file=sess.get("session_file"),
        supports_agents_json=caps.get("agents_json", False),
        supports_session_hook=sess.get("supports_hook", False),
        supports_add_dir=caps.get("add_dir", False),
        # Log format
        log_format=caps.get("log_format", "plain"),
        opencode_config=oc,
    )


def _to_auth_provider(name: str, data: dict) -> AuthProvider | None:
    """Deserialize the ``auth:`` YAML section into an ``AuthProvider``."""
    from .auth import AuthKeyConfig, AuthProvider, _api_key_command

    auth = data.get("auth", {})
    if not auth:
        return None

    # Determine command: explicit command list, or build from auth_key config
    auth_key_data = auth.get("auth_key")
    if "command" in auth:
        command = list(auth["command"])
    elif auth_key_data:
        command = _api_key_command(
            AuthKeyConfig(
                label=auth_key_data.get("label", data.get("label", name)),
                key_url=auth_key_data["key_url"],
                env_var=auth_key_data["env_var"],
                config_path=auth_key_data["config_path"],
                printf_template=auth_key_data["printf_template"],
                tool_name=auth_key_data.get("tool_name", name),
            )
        )
    else:
        return None

    return AuthProvider(
        name=name,
        label=data.get("label", name),
        host_dir_name=auth["host_dir"],
        container_mount=auth["container_mount"],
        command=command,
        banner_hint=auth.get("banner_hint", ""),
        extra_run_args=tuple(auth.get("extra_run_args", ())),
    )


def _derive_opencode_auth(name: str, data: dict) -> AuthProvider | None:
    """Auto-derive an auth provider for an OpenCode-based agent."""
    from .auth import AuthKeyConfig, AuthProvider, _api_key_command

    oc = data.get("opencode")
    if not oc:
        return None

    return AuthProvider(
        name=name,
        label=data.get("label", name),
        host_dir_name=f"_{name}-config",
        container_mount=f"/home/dev/{oc['config_dir']}",
        command=_api_key_command(
            AuthKeyConfig(
                label=data.get("label", name),
                key_url=oc["auth_key_url"],
                env_var=f"{oc['env_var_prefix']}_API_KEY",
                config_path=f"~/{oc['config_dir']}/config.json",
                printf_template='{"api_key": "%s"}',
                tool_name=name,
            )
        ),
        banner_hint=(
            f"You will be prompted to enter your {data.get('label', name)} API key.\n"
            f"Get your API key at: {oc['auth_key_url']}"
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MountDef:
    """A shared directory mount derived from the agent registry."""

    host_dir: str
    """Directory name under ``envs_base_dir`` (e.g. ``"_codex-config"``)."""

    container_path: str
    """Mount point inside the container (e.g. ``"/home/dev/.codex"``)."""

    label: str
    """Human-readable label (e.g. ``"Codex config"``)."""


@dataclass(frozen=True)
class AgentRegistry:
    """Loaded registry of agents and tools from YAML definitions.

    Provides the same query API as the legacy hardcoded dicts.
    """

    _providers: dict[str, HeadlessProvider] = field(default_factory=dict)
    _auth_providers: dict[str, AuthProvider] = field(default_factory=dict)
    _mounts: tuple[MountDef, ...] = ()
    _agent_names: tuple[str, ...] = ()
    _all_names: tuple[str, ...] = ()

    @property
    def providers(self) -> dict[str, HeadlessProvider]:
        """All headless agent providers (``kind: agent`` only)."""
        return dict(self._providers)

    @property
    def auth_providers(self) -> dict[str, AuthProvider]:
        """All auth providers (agents + tools with ``auth:`` section)."""
        return dict(self._auth_providers)

    @property
    def agent_names(self) -> tuple[str, ...]:
        """Names of ``kind: agent`` entries (for CLI completion)."""
        return self._agent_names

    @property
    def all_names(self) -> tuple[str, ...]:
        """Names of all entries (agents + tools)."""
        return self._all_names

    @property
    def mounts(self) -> tuple[MountDef, ...]:
        """All shared directory mounts (auth dirs + explicit ``mounts:`` sections).

        Deduplicated by ``host_dir`` — if auth and mounts define the same
        directory, only one entry is returned.
        """
        return self._mounts

    def get_provider(
        self, name: str | None, *, default_agent: str | None = None
    ) -> HeadlessProvider:
        """Resolve a provider name to a ``HeadlessProvider``.

        Falls back to *default_agent*, then ``"claude"``.
        Raises ``SystemExit`` if the resolved name is unknown.
        """
        resolved = name or default_agent or "claude"
        provider = self._providers.get(resolved)
        if provider is None:
            valid = ", ".join(sorted(self._providers))
            raise SystemExit(f"Unknown headless provider {resolved!r}. Valid providers: {valid}")
        return provider

    def get_auth_provider(self, name: str) -> AuthProvider:
        """Look up an auth provider by name.

        Raises ``SystemExit`` if the name is unknown.
        """
        info = self._auth_providers.get(name)
        if info is None:
            available = ", ".join(sorted(self._auth_providers))
            raise SystemExit(f"Unknown auth provider: {name!r}. Available: {available}")
        return info

    def collect_all_auto_approve_env(self) -> dict[str, str]:
        """Merge ``auto_approve.env`` from all providers into one dict."""
        merged: dict[str, str] = {}
        for p in self._providers.values():
            for key, value in p.auto_approve_env.items():
                if key in merged and merged[key] != value:
                    raise ValueError(
                        f"Conflicting auto_approve_env for {key!r}: "
                        f"{merged[key]!r} vs {value!r} (provider {p.name!r})"
                    )
                merged[key] = value
        return merged

    def collect_opencode_provider_env(self) -> dict[str, str]:
        """Collect env vars for all OpenCode-based providers."""
        env: dict[str, str] = {}
        for p in self._providers.values():
            if p.opencode_config is not None:
                env.update(p.opencode_config.to_env(p.name))
        return env


def load_registry() -> AgentRegistry:
    """Load the agent registry from bundled YAML + user overrides.

    Bundled agents in ``resources/agents/*.yaml`` are loaded first, then
    user files in ``~/.config/terok-agent/agents/*.yaml`` are deep-merged
    on top (allowing field-level overrides or entirely new agents).
    """
    raw = _load_bundled_agents()

    # Deep-merge user overrides on top of bundled definitions
    for name, user_data in _load_user_agents().items():
        if name in raw:
            raw[name] = deep_merge(raw[name], user_data)
        else:
            raw[name] = user_data

    providers: dict[str, HeadlessProvider] = {}
    auth_providers: dict[str, AuthProvider] = {}
    agent_names: list[str] = []
    all_names: list[str] = []

    # Collect mounts from all entries — deduplicate by host_dir
    seen_mounts: dict[str, MountDef] = {}

    for name, data in sorted(raw.items()):
        kind = data.get("kind", "native")
        if kind != "runtime":
            all_names.append(name)

        # Agent kinds (native, opencode, bridge) get a HeadlessProvider;
        # tools and runtime entries only contribute auth/mounts.
        if kind not in ("tool", "runtime"):
            agent_names.append(name)
            providers[name] = _to_headless_provider(name, data)

        # Auth: explicit auth section, or auto-derived from opencode config
        auth_prov = _to_auth_provider(name, data)
        if auth_prov is not None:
            auth_providers[name] = auth_prov
            # Auth providers also contribute a mount
            if auth_prov.host_dir_name not in seen_mounts:
                seen_mounts[auth_prov.host_dir_name] = MountDef(
                    host_dir=auth_prov.host_dir_name,
                    container_path=auth_prov.container_mount,
                    label=f"{auth_prov.label} config",
                )
        elif kind not in ("tool", "runtime"):
            oc_auth = _derive_opencode_auth(name, data)
            if oc_auth is not None:
                auth_providers[name] = oc_auth
                if oc_auth.host_dir_name not in seen_mounts:
                    seen_mounts[oc_auth.host_dir_name] = MountDef(
                        host_dir=oc_auth.host_dir_name,
                        container_path=oc_auth.container_mount,
                        label=f"{oc_auth.label} config",
                    )

        # Explicit mounts section
        for m in data.get("mounts", ()):
            hd = m["host_dir"]
            if hd not in seen_mounts:
                seen_mounts[hd] = MountDef(
                    host_dir=hd,
                    container_path=m["container_path"],
                    label=m.get("label", name),
                )

    return AgentRegistry(
        _providers=providers,
        _auth_providers=auth_providers,
        _mounts=tuple(seen_mounts.values()),
        _agent_names=tuple(agent_names),
        _all_names=tuple(all_names),
    )


@lru_cache(maxsize=1)
def get_registry() -> AgentRegistry:
    """Return the singleton registry instance (loaded once, cached)."""
    return load_registry()
