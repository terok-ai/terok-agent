# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""YAML-driven agent and tool roster.

Loads per-agent definition files from bundled package resources and optional
user extensions, deserializes them into the existing dataclass types, and
provides the same query API that ``headless_providers`` and ``auth`` expose
today.

Directory layout::

    resources/agents/claude.yaml      (bundled, shipped in wheel)
    resources/agents/codex.yaml
    ...
    ~/.config/terok/agent/agents/      (user overrides / additions)
"""

from __future__ import annotations

import importlib.resources
import sys
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
    """Return ``~/.config/terok/agent/agents/``."""
    from .paths import _SUBDIR, _UMBRELLA

    try:
        from platformdirs import user_config_dir
    except ImportError:  # pragma: no cover
        return Path.home() / ".config" / _UMBRELLA / _SUBDIR / _USER_AGENTS_DIR_NAME
    return Path(user_config_dir(_UMBRELLA)) / _SUBDIR / _USER_AGENTS_DIR_NAME


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
        try:
            data = _load_yaml(item.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                f"Warning [roster]: failed to parse bundled agent {name!r}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        if data:
            agents[name] = data
    return agents


def _load_user_agents() -> dict[str, dict]:
    """Load user override/addition YAML files from ``~/.config/terok/agent/agents/``."""
    agents: dict[str, dict] = {}
    user_dir = _user_agents_dir()
    if not user_dir.is_dir():
        return agents
    for path in sorted(user_dir.glob("*.yaml")):
        name = path.stem
        try:
            data = _load_yaml(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                f"Warning [roster]: failed to parse user agent file {path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
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

    # Determine command: explicit command list, or build from auth_key config.
    # API-key-only providers may have no command — that's fine.
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
        command = []

    modes = tuple(auth.get("modes", ("api_key",)))

    return AuthProvider(
        name=name,
        label=data.get("label", name),
        host_dir_name=auth["host_dir"],
        container_mount=auth["container_mount"],
        command=command,
        banner_hint=auth.get("banner_hint", ""),
        extra_run_args=tuple(auth.get("extra_run_args", ())),
        modes=modes,
        api_key_hint=auth.get("api_key_hint", ""),
    )


def _derive_opencode_auth(name: str, data: dict) -> AuthProvider | None:
    """Auto-derive an auth provider for an OpenCode-based agent."""
    from .auth import AuthProvider

    oc = data.get("opencode")
    if not oc:
        return None

    return AuthProvider(
        name=name,
        label=data.get("label", name),
        host_dir_name=f"_{name}-config",
        container_mount=f"/home/dev/{oc['config_dir']}",
        command=[],  # API-key-only — no container command needed
        banner_hint="",
        modes=("api_key",),
        api_key_hint=f"Get your API key at: {oc['auth_key_url']}",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MountDef:
    """A shared directory mount derived from the agent roster."""

    host_dir: str
    """Directory name under ``mounts_dir()`` (e.g. ``"_codex-config"``)."""

    container_path: str
    """Mount point inside the container (e.g. ``"/home/dev/.codex"``)."""

    label: str
    """Human-readable label (e.g. ``"Codex config"``)."""


@dataclass(frozen=True)
class CredentialProxyRoute:
    """Proxy route config parsed from a ``credential_proxy:`` YAML section.

    Used to generate the ``routes.json`` that the credential proxy server reads.
    """

    provider: str
    """Agent/tool name (e.g. ``"claude"``)."""

    route_prefix: str
    """Path prefix in the proxy (e.g. ``"claude"`` → ``/claude/v1/...``)."""

    upstream: str
    """Upstream API base URL (e.g. ``"https://api.anthropic.com"``)."""

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
    env vars are injected *instead of* :attr:`phantom_env`.
    """

    base_url_env: str = ""
    """Env var to override with proxy URL (e.g. ``"ANTHROPIC_BASE_URL"``)."""

    socket_path: str = ""
    """Unix socket path for socat bridge (e.g. ``"/tmp/terok-claude-proxy.sock"``)."""

    socket_env: str = ""
    """Env var that receives :attr:`socket_path` (e.g. ``"ANTHROPIC_UNIX_SOCKET"``)."""

    shared_config_patch: dict | None = None
    """Optional shared config patch applied after auth (e.g. Vibe's config.toml)."""

    oauth_refresh: dict | None = None
    """OAuth refresh config: ``{token_url, client_id, scope}``."""


def _validated_oauth_refresh(name: str, raw: dict | None) -> dict | None:
    """Validate ``oauth_refresh`` section and return it, or ``None``."""
    if raw is None:
        return None
    for key in ("token_url", "client_id"):
        if key not in raw:
            raise ValueError(f"Agent {name!r}: oauth_refresh missing required key {key!r}")
    return raw


def _to_proxy_route(name: str, data: dict) -> CredentialProxyRoute | None:
    """Parse the ``credential_proxy:`` YAML section into a route config."""
    cp = data.get("credential_proxy")
    if not cp:
        return None
    if not isinstance(cp, dict):
        raise ValueError(
            f"Agent {name!r}: credential_proxy must be a mapping, got {type(cp).__name__}"
        )
    for required in ("route_prefix", "upstream"):
        if required not in cp:
            raise ValueError(f"Agent {name!r}: credential_proxy missing required key {required!r}")
    oauth_phantom_env = cp.get("oauth_phantom_env") or {}
    socket_path = cp.get("socket_path") or ""
    socket_env = cp.get("socket_env") or ""
    if bool(socket_path) != bool(socket_env):
        raise ValueError(
            f"Agent {name!r}: credential_proxy requires both 'socket_path' and 'socket_env' together"
        )
    return CredentialProxyRoute(
        provider=name,
        route_prefix=cp["route_prefix"],
        upstream=cp["upstream"],
        auth_header=cp.get("auth_header", "Authorization"),
        auth_prefix=cp.get("auth_prefix", "Bearer "),
        credential_type=cp.get("credential_type", "api_key"),
        credential_file=cp.get("credential_file", ""),
        phantom_env=cp.get("phantom_env", {}),
        oauth_phantom_env=oauth_phantom_env,
        base_url_env=cp.get("base_url_env", ""),
        socket_path=socket_path,
        socket_env=socket_env,
        shared_config_patch=cp.get("shared_config_patch"),
        oauth_refresh=_validated_oauth_refresh(name, cp.get("oauth_refresh")),
    )


@dataclass(frozen=True)
class AgentRoster:
    """Loaded roster of agents and tools from YAML definitions.

    Provides the same query API as the legacy hardcoded dicts.
    """

    _providers: dict[str, HeadlessProvider] = field(default_factory=dict)
    _auth_providers: dict[str, AuthProvider] = field(default_factory=dict)
    _proxy_routes: dict[str, CredentialProxyRoute] = field(default_factory=dict)
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

    @property
    def proxy_routes(self) -> dict[str, CredentialProxyRoute]:
        """All credential proxy routes, keyed by provider name."""
        return dict(self._proxy_routes)

    def generate_routes_json(self) -> str:
        """Generate the ``routes.json`` content for the credential proxy server.

        Returns a JSON string mapping route prefixes to upstream config.
        """
        import json

        routes: dict[str, dict[str, object]] = {}
        prefix_owners: dict[str, str] = {}
        for route in self._proxy_routes.values():
            existing = prefix_owners.get(route.route_prefix)
            if existing is not None:
                raise ValueError(
                    f"Duplicate route prefix {route.route_prefix!r}: "
                    f"providers {existing!r} and {route.provider!r}"
                )
            prefix_owners[route.route_prefix] = route.provider
            entry: dict[str, object] = {
                "upstream": route.upstream,
                "auth_header": route.auth_header,
                "auth_prefix": route.auth_prefix,
            }
            if route.oauth_refresh:
                entry["oauth_refresh"] = route.oauth_refresh
            routes[route.provider] = entry
        return json.dumps(routes, indent=2)

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


def load_roster() -> AgentRoster:
    """Load the agent roster from bundled YAML + user overrides.

    Bundled agents in ``resources/agents/*.yaml`` are loaded first, then
    user files in ``~/.config/terok/agent/agents/*.yaml`` are deep-merged
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
    proxy_routes: dict[str, CredentialProxyRoute] = {}
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

        # Credential proxy route
        proxy_route = _to_proxy_route(name, data)
        if proxy_route is not None:
            proxy_routes[name] = proxy_route

    return AgentRoster(
        _providers=providers,
        _auth_providers=auth_providers,
        _proxy_routes=proxy_routes,
        _mounts=tuple(seen_mounts.values()),
        _agent_names=tuple(agent_names),
        _all_names=tuple(all_names),
    )


@lru_cache(maxsize=1)
def get_roster() -> AgentRoster:
    """Return the singleton roster instance (loaded once, cached)."""
    return load_roster()


def ensure_proxy_routes() -> Path:
    """Generate ``routes.json`` from the YAML roster and write it to disk.

    The routes file is written to the path configured in
    :class:`~terok_sandbox.SandboxConfig` (typically
    ``~/.local/share/terok/proxy/routes.json``).

    Returns the path to the written file.
    """
    from terok_sandbox import SandboxConfig

    cfg = SandboxConfig()
    path = cfg.proxy_routes_path
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    content = get_roster().generate_routes_json() + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
