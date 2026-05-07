# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Loads agent and tool definitions from YAML and assembles them into a queryable roster.

Loads per-agent definition files from bundled package resources and
optional user extensions, validates them through the strict
[`schema`][terok_executor.roster.schema] (typo-rejecting Pydantic
models), and projects each entry onto the runtime
[`types`][terok_executor.roster.types] dataclasses.

Directory layout::

    resources/agents/claude.yaml      (bundled, shipped in wheel)
    resources/agents/codex.yaml
    ...
    ~/.config/terok/agent/agents/      (user overrides / additions)
"""

from __future__ import annotations

import importlib.resources
import os
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError
from terok_sandbox import SandboxConfig
from terok_sandbox.config_stack import deep_merge
from terok_sandbox.paths import namespace_config_dir

from terok_executor._util import yaml_load
from terok_executor.credentials.auth import AuthProvider
from terok_executor.provider.providers import AgentProvider

from .schema import RawAgentYaml, VaultRouteEntry
from .types import HelpSpec, InstallSpec, MountDef, SidecarSpec, VaultRoute

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


@dataclass(frozen=True)
class AgentRoster:
    """Queryable view over the loaded set of agents and tools.

    Returned by [`load_roster`][terok_executor.roster.loader.load_roster];
    grouped accessors expose providers, auth providers, vault routes,
    sidecar specs, install snippets, and help blurbs by name.
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
        [`InstallSpec`][terok_executor.roster.types.InstallSpec]) or a tuple of names.  Expands ``depends_on``
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
        """Generate the ``routes.json`` content for the sandbox vault server.

        Returns a JSON object mapping provider name → [`VaultRouteEntry`][terok_executor.roster.schema.VaultRouteEntry]
        with empty/absent optional fields stripped.
        """
        from pydantic import TypeAdapter

        routes: dict[str, VaultRouteEntry] = {}
        prefix_owners: dict[str, str] = {}
        for route in self._vault_routes.values():
            existing = prefix_owners.get(route.route_prefix)
            if existing is not None:
                raise ValueError(
                    f"Duplicate route prefix {route.route_prefix!r}: "
                    f"providers {existing!r} and {route.provider!r}"
                )
            prefix_owners[route.route_prefix] = route.provider
            routes[route.provider] = VaultRouteEntry(
                upstream=route.upstream,
                auth_header=route.auth_header,
                auth_prefix=route.auth_prefix,
                path_upstreams=route.path_upstreams or None,
                oauth_extra_headers=route.oauth_extra_headers or None,
                oauth_refresh=route.oauth_refresh or None,
            )
        return (
            TypeAdapter(dict[str, VaultRouteEntry])
            .dump_json(routes, indent=2, exclude_none=True)
            .decode()
        )

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
    ``"all"`` — the same shape [`AgentRoster.resolve_selection`][terok_executor.roster.loader.AgentRoster.resolve_selection]
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
    on top (allowing field-level overrides or entirely new agents).  Each
    merged entry is then validated through [`RawAgentYaml`][terok_executor.roster.schema.RawAgentYaml]
    — typos in section keys, wrong types, or unknown fields fail loud
    instead of silently defaulting.
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
        try:
            spec = RawAgentYaml.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Agent {name!r}: invalid roster YAML\n{exc}") from exc

        label = spec.resolve_label(name)
        is_agent_kind = spec.kind not in ("tool", "runtime")

        if spec.kind != "runtime":
            all_names.append(name)
        if is_agent_kind:
            agent_names.append(name)
            providers[name] = spec.to_agent_provider(name)

        credential_file = spec.vault.credential_file if spec.vault else ""

        auth_prov: AuthProvider | None
        if spec.auth is not None:
            auth_prov = spec.auth.to_dataclass(name=name, label=label)
        elif is_agent_kind:
            auth_prov = spec.derive_opencode_auth(name)
        else:
            auth_prov = None

        if auth_prov is not None:
            auth_providers[name] = auth_prov
            if auth_prov.host_dir_name not in seen_mounts:
                seen_mounts[auth_prov.host_dir_name] = MountDef(
                    host_dir=auth_prov.host_dir_name,
                    container_path=auth_prov.container_mount,
                    label=f"{auth_prov.label} config",
                    credential_file=credential_file,
                    provider=name,
                )

        for m in spec.mounts:
            if m.host_dir not in seen_mounts:
                seen_mounts[m.host_dir] = MountDef(
                    host_dir=m.host_dir,
                    container_path=m.container_path,
                    label=m.label or name,
                )

        if spec.vault is not None:
            vault_routes[name] = spec.vault.to_dataclass(provider=name)

        if spec.sidecar is not None:
            sidecar_specs[name] = spec.sidecar.to_dataclass(default_name=name)

        if spec.install is not None:
            installs[name] = spec.install.to_dataclass()

        if spec.help is not None:
            helps[name] = spec.help.to_dataclass()

        if spec.web_ingress:
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
    [`SandboxConfig`][terok_sandbox.SandboxConfig] (typically
    ``~/.local/share/terok/vault/routes.json``).

    When *cfg* is ``None``, falls back to standalone defaults.

    Returns the path to the written file.
    """
    if cfg is None:
        cfg = SandboxConfig()
    path = cfg.routes_path

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
    return namespace_config_dir("agent") / _USER_AGENTS_DIR_NAME


def _load_yaml(text: str) -> dict:
    """Parse YAML text into a dict via ruamel.yaml round-trip loader."""
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
    version strictly greater than [`ROSTER_VERSION`][terok_executor.roster.loader.ROSTER_VERSION] prints a warning,
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
