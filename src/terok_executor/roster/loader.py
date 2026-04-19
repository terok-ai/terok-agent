# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Loads agent and tool definitions from YAML and assembles them into a queryable roster.

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
from typing import TYPE_CHECKING, Literal, get_args

from terok_sandbox.config_stack import deep_merge

if TYPE_CHECKING:
    from terok_sandbox import SandboxConfig

    from terok_executor.credentials.auth import AuthProvider
    from terok_executor.provider.providers import AgentProvider, OpenCodeProviderConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENTS_DIR_NAME = "agents"

ROSTER_VERSION = 1
"""Schema version of the agent-roster YAML format.

Bundled agent YAMLs and user override files declare a top-level
``roster_version: 1`` that matches this constant.  A file with no
``roster_version`` is treated as version 1 (forward-compat for existing
user overrides written before the marker existed).  A file declaring a
future version is still loaded but the loader logs a warning — the host
and container may be on incompatible contracts.  Bumped only on breaking
changes to the roster schema, never per release."""


# ── Domain model ──────────────────────────────────────────────────────────


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
"""All valid :data:`HelpSection` values, as a tuple (single source of truth)."""


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
class AgentRoster:
    """Loaded roster of agents and tools from YAML definitions.

    Provides the same query API as the legacy hardcoded dicts.
    """

    _providers: dict[str, AgentProvider] = field(default_factory=dict)
    _auth_providers: dict[str, AuthProvider] = field(default_factory=dict)
    _vault_routes: dict[str, VaultRoute] = field(default_factory=dict)
    _sidecar_specs: dict[str, SidecarSpec] = field(default_factory=dict)
    _installs: dict[str, InstallSpec] = field(default_factory=dict)
    _helps: dict[str, HelpSpec] = field(default_factory=dict)
    _mounts: tuple[MountDef, ...] = ()
    _agent_names: tuple[str, ...] = ()
    _all_names: tuple[str, ...] = ()
    _web_ingress: frozenset[str] = frozenset()

    # ── Properties ──

    @property
    def providers(self) -> dict[str, AgentProvider]:
        """All headless agent providers (``kind: agent`` only)."""
        return dict(self._providers)

    @property
    def auth_providers(self) -> dict[str, AuthProvider]:
        """All auth providers (agents + tools with ``auth:`` section)."""
        return dict(self._auth_providers)

    @property
    def vault_routes(self) -> dict[str, VaultRoute]:
        """All vault routes, keyed by provider name."""
        return dict(self._vault_routes)

    @property
    def sidecar_specs(self) -> dict[str, SidecarSpec]:
        """All sidecar tool specs, keyed by tool name."""
        return dict(self._sidecar_specs)

    @property
    def agent_names(self) -> tuple[str, ...]:
        """Names of ``kind: agent`` entries (for CLI completion)."""
        return self._agent_names

    @property
    def all_names(self) -> tuple[str, ...]:
        """Names of all entries (agents + tools)."""
        return self._all_names

    @property
    def installs(self) -> dict[str, InstallSpec]:
        """All install specs, keyed by roster name (entries without one are absent)."""
        return dict(self._installs)

    @property
    def helps(self) -> dict[str, HelpSpec]:
        """All help blurbs, keyed by roster name (entries without one are absent)."""
        return dict(self._helps)

    @property
    def web_ingress(self) -> frozenset[str]:
        """Names of entries that publish a host HTTP port (``web_ingress: true``).

        Consumers (e.g. terok's task launcher) use this to decide whether
        to allocate a published port and drop a per-task auth token into
        the container-visible config dir.
        """
        return self._web_ingress

    # ── Selection ──

    def resolve_selection(self, names: str | tuple[str, ...]) -> tuple[str, ...]:
        """Resolve a user-supplied selection into the full set of roster names to install.

        Accepts the literal string ``"all"`` (every roster entry that has an
        :class:`InstallSpec`) or a tuple of names.  Expands ``depends_on``
        transitively.  Returns the names sorted alphabetically — the canonical
        order used for the OCI label, the tag suffix, and the in-container
        manifest.

        Raises ``ValueError`` if a requested name is not in the roster, or
        ``TypeError`` if *names* is a string other than ``"all"`` (a bare
        name like ``"claude"`` would otherwise be iterated into characters).
        """
        if isinstance(names, str):
            if names != "all":
                raise TypeError(
                    f"Selection must be the literal string 'all' or a tuple of names, got {names!r}"
                )
            seed = set(self._installs)
        else:
            seed = set(names)

        unknown = seed - set(self._installs)
        if unknown:
            avail = ", ".join(sorted(self._installs))
            raise ValueError(f"Unknown roster entries: {sorted(unknown)!r}. Available: {avail}")

        resolved: set[str] = set()
        stack = list(seed)
        while stack:
            name = stack.pop()
            if name in resolved:
                continue
            resolved.add(name)
            spec = self._installs.get(name)
            if spec is None:
                continue
            for dep in spec.depends_on:
                if dep not in self._installs:
                    raise ValueError(
                        f"Agent {name!r} declares depends_on {dep!r}, "
                        f"which has no install: section in the roster"
                    )
                if dep not in resolved:
                    stack.append(dep)
        return tuple(sorted(resolved))

    @property
    def mounts(self) -> tuple[MountDef, ...]:
        """All shared directory mounts (auth dirs + explicit ``mounts:`` sections).

        Deduplicated by ``host_dir`` — if auth and mounts define the same
        directory, only one entry is returned.
        """
        return self._mounts

    # ── Keyed lookups ──

    def get_provider(self, name: str | None, *, default_agent: str | None = None) -> AgentProvider:
        """Resolve a provider name to an ``AgentProvider``.

        Falls back to *default_agent*, then ``"claude"``.
        Raises ``SystemExit`` if the resolved name is unknown.
        """
        from terok_executor.provider.providers import resolve_provider

        return resolve_provider(self._providers, name, default_agent=default_agent)

    def get_auth_provider(self, name: str) -> AuthProvider:
        """Look up an auth provider by name.

        Raises ``SystemExit`` if the name is unknown.
        """
        info = self._auth_providers.get(name)
        if info is None:
            available = ", ".join(sorted(self._auth_providers))
            raise SystemExit(f"Unknown auth provider: {name!r}. Available: {available}")
        return info

    def get_sidecar_spec(self, name: str) -> SidecarSpec:
        """Look up a sidecar spec by tool name.

        Raises ``SystemExit`` if the name has no sidecar configuration.
        """
        spec = self._sidecar_specs.get(name)
        if spec is None:
            available = ", ".join(sorted(self._sidecar_specs)) or "(none)"
            raise SystemExit(f"No sidecar config for {name!r}. Available: {available}")
        return spec

    # ── Domain operations ──

    def generate_routes_json(self) -> str:
        """Generate the ``routes.json`` content for the vault server.

        Returns a JSON string mapping route prefixes to upstream config.
        """
        import json

        routes: dict[str, dict[str, object]] = {}
        prefix_owners: dict[str, str] = {}
        for route in self._vault_routes.values():
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


# ── Public API ────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_roster() -> AgentRoster:
    """Return the singleton roster instance (loaded once, cached)."""
    return load_roster()


def parse_agent_selection(raw: str) -> str | tuple[str, ...]:
    """Normalise a user-supplied agent selection string.

    Accepts a comma-list (``"claude,codex"``) or the literal ``"all"``.
    Whitespace is stripped, empty / whitespace-only entries dropped,
    and case folded.  Empty or all-whitespace input collapses to
    ``"all"`` — the same shape :meth:`AgentRoster.resolve_selection`
    expects.  Unknown names are not checked here; ``resolve_selection``
    does that.
    """
    folded = raw.strip().lower()
    if folded == "all" or not folded:
        return "all"
    names = tuple(n.strip() for n in folded.split(",") if n.strip())
    return names or "all"


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

    providers: dict[str, AgentProvider] = {}
    auth_providers: dict[str, AuthProvider] = {}
    vault_routes: dict[str, VaultRoute] = {}
    sidecar_specs: dict[str, SidecarSpec] = {}
    installs: dict[str, InstallSpec] = {}
    helps: dict[str, HelpSpec] = {}
    agent_names: list[str] = []
    all_names: list[str] = []
    web_ingress_names: set[str] = set()

    # Collect mounts from all entries — deduplicate by host_dir
    seen_mounts: dict[str, MountDef] = {}

    for name, data in sorted(raw.items()):
        kind = data.get("kind", "native")
        if kind != "runtime":
            all_names.append(name)

        # Agent kinds (native, opencode, bridge) get a AgentProvider;
        # tools and runtime entries only contribute auth/mounts.
        if kind not in ("tool", "runtime"):
            agent_names.append(name)
            providers[name] = _to_agent_provider(name, data)

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

        # Vault route
        vault_route = _to_vault_route(name, data)
        if vault_route is not None:
            vault_routes[name] = vault_route

        # Sidecar spec
        sidecar = _to_sidecar_spec(name, data)
        if sidecar is not None:
            sidecar_specs[name] = sidecar

        # Install snippets and help blurb (both optional)
        install = _to_install_spec(name, data)
        if install is not None:
            installs[name] = install
        help_spec = _to_help_spec(name, data)
        if help_spec is not None:
            helps[name] = help_spec

        raw_web_ingress = data.get("web_ingress", False)
        if not isinstance(raw_web_ingress, bool):
            raise ValueError(
                f"Agent {name!r}: web_ingress must be a boolean, got "
                f"{type(raw_web_ingress).__name__}"
            )
        if raw_web_ingress:
            web_ingress_names.add(name)

    return AgentRoster(
        _providers=providers,
        _auth_providers=auth_providers,
        _vault_routes=vault_routes,
        _sidecar_specs=sidecar_specs,
        _installs=installs,
        _helps=helps,
        _mounts=tuple(seen_mounts.values()),
        _agent_names=tuple(agent_names),
        _all_names=tuple(all_names),
        _web_ingress=frozenset(web_ingress_names),
    )


def ensure_vault_routes(cfg: SandboxConfig | None = None) -> Path:
    """Generate ``routes.json`` from the YAML roster and write it to disk.

    The routes file is written to the path configured in
    :class:`~terok_sandbox.SandboxConfig` (typically
    ``~/.local/share/terok/vault/routes.json``).

    When *cfg* is ``None``, falls back to standalone defaults.

    Returns the path to the written file.
    """
    from terok_sandbox import SandboxConfig

    if cfg is None:
        cfg = SandboxConfig()
    path = cfg.routes_path
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


# ── YAML loading ──────────────────────────────────────────────────────────


def _user_agents_dir() -> Path:
    """Return ``~/.config/terok/agent/agents/``."""
    from terok_sandbox.paths import namespace_config_dir

    return namespace_config_dir("agent") / _USER_AGENTS_DIR_NAME


def _load_yaml(text: str) -> dict:
    """Parse YAML text into a dict via ruamel.yaml round-trip loader."""
    from terok_executor._util import yaml_load

    result = yaml_load(text)
    return result if isinstance(result, dict) else {}


def _load_bundled_agents() -> dict[str, dict]:
    """Load all ``*.yaml`` files from the bundled ``resources/agents/`` package."""
    agents: dict[str, dict] = {}
    pkg = importlib.resources.files("terok_executor.resources.agents")
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
        _add_agent(agents, name, data, source=f"bundled {name}.yaml")
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
        _add_agent(agents, name, data, source=str(path))
    return agents


def _add_agent(agents: dict[str, dict], name: str, data: dict | None, *, source: str) -> None:
    """Validate, strip version metadata, and add an agent entry if it has content.

    Files that turn out to be pure metadata (e.g. only ``roster_version``)
    are skipped — they contribute no agent definition and would otherwise
    land as an empty dict downstream and surprise the deserializer.
    """
    if not data:
        return
    _check_roster_version(name, data, source=source)
    if not data:
        # Purely metadata file (only ``roster_version`` was present).
        print(
            f"Info [roster]: skipping metadata-only file ({source}) — "
            f"no agent definition to register for {name!r}.",
            file=sys.stderr,
        )
        return
    agents[name] = data


def _check_roster_version(name: str, data: dict, *, source: str) -> None:
    """Strip the ``roster_version`` marker and warn only on a *future* version.

    Missing or older versions still load silently — existing user overrides
    written before the marker existed must keep working, and older-but-still-
    understood roster files are the backward-compat path.  A declared
    version strictly greater than :data:`ROSTER_VERSION` prints a warning,
    because the host may not speak every field the file uses.
    """
    declared = data.pop("roster_version", None)
    if declared is None:
        return
    try:
        declared_int = int(declared)
    except (TypeError, ValueError):
        print(
            f"Warning [roster]: {source} declares roster_version={declared!r}, "
            f"which is not a valid integer version; treating as current.",
            file=sys.stderr,
        )
        return
    if declared_int > ROSTER_VERSION:
        print(
            f"Warning [roster]: {source} declares roster_version={declared_int}, "
            f"but this terok-executor speaks version {ROSTER_VERSION}.  "
            f"Some fields in {name!r} may be ignored; upgrade terok-executor "
            f"or adjust the file.",
            file=sys.stderr,
        )


# ── Deserialization ───────────────────────────────────────────────────────


def _to_opencode_config(data: dict) -> OpenCodeProviderConfig:
    """Deserialize the ``opencode:`` YAML section."""
    from terok_executor.provider.providers import OpenCodeProviderConfig

    return OpenCodeProviderConfig(
        display_name=data["display_name"],
        base_url=data["base_url"],
        preferred_model=data["preferred_model"],
        fallback_model=data["fallback_model"],
        env_var_prefix=data["env_var_prefix"],
        config_dir=data["config_dir"],
        auth_key_url=data["auth_key_url"],
    )


def _to_agent_provider(name: str, data: dict) -> AgentProvider:
    """Deserialize a full agent YAML dict into an ``AgentProvider``."""
    from terok_executor.provider.providers import AgentProvider

    hl = data.get("headless", {})
    aa = data.get("auto_approve", {})
    sess = data.get("session", {})
    caps = data.get("capabilities", {})
    gi = data.get("git_identity", {})

    oc_data = data.get("opencode")
    oc = _to_opencode_config(oc_data) if oc_data else None

    return AgentProvider(
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
    from terok_executor.credentials.auth import AuthKeyConfig, AuthProvider, _api_key_command

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

    raw_pcs = auth.get("post_capture_state", {})
    if raw_pcs is None:
        post_capture_state: dict[str, dict] = {}
    elif isinstance(raw_pcs, dict):
        post_capture_state = {}
        for filename, patch in raw_pcs.items():
            if not isinstance(filename, str) or not isinstance(patch, dict):
                raise ValueError(
                    f"Agent {name!r}: auth.post_capture_state must map filename -> mapping"
                )
            post_capture_state[filename] = patch
    else:
        raise ValueError(
            f"Agent {name!r}: auth.post_capture_state must be a mapping, "
            f"got {type(raw_pcs).__name__}"
        )

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
        post_capture_state=post_capture_state,
    )


def _derive_opencode_auth(name: str, data: dict) -> AuthProvider | None:
    """Auto-derive an auth provider for an OpenCode-based agent."""
    from terok_executor.credentials.auth import AuthProvider

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


def _validated_oauth_refresh(name: str, raw: dict | None) -> dict | None:
    """Validate ``oauth_refresh`` section and return it, or ``None``."""
    if raw is None:
        return None
    for key in ("token_url", "client_id"):
        if key not in raw:
            raise ValueError(f"Agent {name!r}: oauth_refresh missing required key {key!r}")
    return raw


def _to_vault_route(name: str, data: dict) -> VaultRoute | None:
    """Parse the ``vault:`` YAML section into a route config."""
    cp = data.get("vault")
    if not cp:
        return None
    if not isinstance(cp, dict):
        raise ValueError(f"Agent {name!r}: vault must be a mapping, got {type(cp).__name__}")
    for required in ("route_prefix", "upstream"):
        if required not in cp:
            raise ValueError(f"Agent {name!r}: vault missing required key {required!r}")
    oauth_phantom_env = cp.get("oauth_phantom_env") or {}
    socket_path = cp.get("socket_path") or ""
    socket_env = cp.get("socket_env") or ""
    if bool(socket_path) != bool(socket_env):
        raise ValueError(
            f"Agent {name!r}: vault requires both 'socket_path' and 'socket_env' together"
        )
    return VaultRoute(
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


def _to_sidecar_spec(name: str, data: dict) -> SidecarSpec | None:
    """Parse the optional ``sidecar:`` YAML section into a :class:`SidecarSpec`."""
    sc = data.get("sidecar")
    if not sc:
        return None
    return SidecarSpec(
        tool_name=sc.get("tool_name", name),
        env_map=dict(sc.get("env_map", {})),
    )


def _to_install_spec(name: str, data: dict) -> InstallSpec | None:
    """Parse the optional ``install:`` YAML section into an :class:`InstallSpec`.

    Both snippet fields and ``depends_on`` are optional individually; the
    section as a whole is omitted only for entries that need no install
    work in L1 (typically sidecar tools and runtime kinds with no payload).
    """
    inst = data.get("install")
    if not inst:
        return None
    if not isinstance(inst, dict):
        raise ValueError(f"Agent {name!r}: install must be a mapping, got {type(inst).__name__}")
    deps = inst.get("depends_on", ())
    if isinstance(deps, str):
        deps = (deps,)
    return InstallSpec(
        depends_on=tuple(deps),
        run_as_root=inst.get("run_as_root", "") or "",
        run_as_dev=inst.get("run_as_dev", "") or "",
    )


def _to_help_spec(name: str, data: dict) -> HelpSpec | None:
    """Parse the optional ``help:`` YAML section into a :class:`HelpSpec`."""
    h = data.get("help")
    if not h:
        return None
    if not isinstance(h, dict):
        raise ValueError(f"Agent {name!r}: help must be a mapping, got {type(h).__name__}")
    section = h.get("section") or "agent"
    if section not in HELP_SECTIONS:
        raise ValueError(
            f"Agent {name!r}: help.section must be one of {list(HELP_SECTIONS)!r}, got {section!r}"
        )
    return HelpSpec(label=h.get("label", "") or "", section=section)
