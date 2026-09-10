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
    ~/.config/terok/providers/         (user provider overrides / additions)
"""

from __future__ import annotations

import importlib.resources
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError
from terok_util import deep_merge, namespace_config_dir, yaml

from terok_executor.credentials.auth import AuthProvider
from terok_executor.integrations.sandbox import DoctorCheck, SandboxConfig
from terok_executor.provider.providers import Agent

from .schema import RawAgentYaml, RawProvider, RawProviderBinding, VaultRouteEntry
from .types import (
    EgressProjection,
    HelpSpec,
    InstallSpec,
    MountDef,
    Provider,
    SidecarSpec,
    VaultRoute,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENTS_DIR_NAME = "agents"
_USER_PROVIDERS_DIR_NAME = "providers"
_YAML_SUFFIX = ".yaml"
_CONTAINER_HOME = "/home/dev"
# Use names that the ``pi-provider`` environment-variable parser accepts.
_PROVIDER_NAME_PATTERN = re.compile(r"[a-z0-9]+")

ROSTER_VERSION = 3
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
    grouped accessors expose agents, auth providers, vault routes,
    sidecar specs, install snippets, and help blurbs by name.
    """

    _agents: dict[str, Agent] = field(default_factory=dict)
    _providers: dict[str, Provider] = field(default_factory=dict)
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
    def agents(self) -> dict[str, Agent]:
        """All headless agents (``kind: agent`` only)."""
        return dict(self._agents)

    @property
    def providers(self) -> dict[str, Provider]:
        """All vault-routed providers (LLM endpoints + tool APIs), keyed by clean name.

        The endpoint axis: where requests go and how the real credential is
        attached.  Loaded from ``resources/providers/*.yaml``.  The
        ``routes.json`` the sandbox vault reads is generated from these.
        """
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
        """Return the names of agents, tools, and selectable LLM providers."""
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

    def resolve_selection(self, selection: str | tuple[str, ...]) -> tuple[str, ...]:
        """Resolve a user-supplied selection into the full set of roster names to install.

        Accepts the literal string ``"all"`` (every roster entry that has an
        [`InstallSpec`][terok_executor.roster.types.InstallSpec]) or a tuple of
        selection tokens.  Each token is either a roster name (include) or a
        name prefixed with ``-`` (exclude).  The pseudo-name ``"all"`` is also
        valid as an include token, meaning "seed from every installable
        entry"; this combines naturally with excludes, e.g. ``("all",
        "-vibe")`` installs everything except vibe.  When no include tokens
        are present (only excludes), the seed is the full roster.

        Includes are expanded transitively via ``depends_on`` *before*
        excludes are applied, so an exclude that names a dependency of a
        kept agent will silently drop that dependency — likely producing a
        broken image, but matching the user's literal request.

        Returns the names sorted alphabetically — the canonical order used
        for the OCI label, the tag suffix, and the in-container manifest.

        This method raises ``ValueError`` if an include or exclude specifies a
        provider-only name or an unknown name. It raises ``TypeError`` if
        *selection* is a string other than ``"all"``. This check prevents
        iteration through the characters of a bare name such as ``"claude"``.
        An exclusion has no effect if the agent is not in the resolved include
        set.
        """
        if isinstance(selection, str):
            if selection != "all":
                raise TypeError(
                    f"Selection must be the literal string 'all' or a tuple of "
                    f"tokens, got {selection!r}"
                )
            return tuple(sorted(self._installs))

        includes = {t for t in selection if not t.startswith("-")}
        excludes = {t[1:] for t in selection if t.startswith("-")}

        referenced = (includes | excludes) - {"all"}
        runtime_endpoints = {
            name
            for name, provider in self._providers.items()
            if provider.serves and name not in self._installs
        }
        endpoint_only = referenced & runtime_endpoints
        if endpoint_only:
            raise ValueError(
                f"These names are providers, not installable agents: "
                f"{sorted(endpoint_only)!r}. Add 'opencode' or 'pi' to image.agents. "
                "Then select a provider with --provider <name>."
            )
        unknown = referenced - set(self._installs)
        if unknown:
            avail = ", ".join(sorted(self._installs))
            raise ValueError(f"Unknown roster entries: {sorted(unknown)!r}. Available: {avail}")

        seed = set(self._installs) if "all" in includes or not includes else includes

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
        return tuple(sorted(resolved - excludes))

    @property
    def mounts(self) -> tuple[MountDef, ...]:
        """All shared directory mounts (auth dirs + explicit ``mounts:`` sections).

        Deduplicated by ``host_dir`` — if auth and mounts define the same
        directory, only one entry is returned.
        """
        return self._mounts

    # ── Keyed lookups ──

    def get_agent(self, name: str | None, *, default_agent: str | None = None) -> Agent:
        """Resolve an agent name to an ``Agent``.

        Falls back to *default_agent*, then ``"claude"``.
        Raises ``SystemExit`` if the resolved name is unknown.
        """
        from terok_executor.provider.providers import resolve_agent

        return resolve_agent(self._agents, name, default_agent=default_agent)

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

        Emits one entry per **provider** — keyed by its clean name — so the
        vault can route to any authenticated provider, not just the one some
        agent binds by default.  This is what lets a harness (opencode, pi)
        reach a provider no agent owns, and what keeps a provider routable
        after its shim agent is collapsed away.  Empty/absent optional fields
        are stripped.
        """
        from pydantic import TypeAdapter

        routes = {
            name: _provider_route_entry(provider) for name, provider in self._providers.items()
        }
        return (
            TypeAdapter(dict[str, VaultRouteEntry])
            .dump_json(routes, indent=2, exclude_none=True)
            .decode()
        )

    def deny_to_vault_hosts(
        self, *, exposed_credential_providers: frozenset[str] = frozenset()
    ) -> frozenset[str]:
        """Hosts to deny directly at the egress firewall (shield ``security_deny``, t20).

        Every provider the vault relays contributes its
        [`relayed_hosts`][terok_executor.roster.types.Provider.relayed_hosts]
        (upstream + path overrides + OAuth-refresh endpoint) — the agent must
        reach those *only* through the loopback vault.  Two classes are skipped:

        - **``shared_domain`` providers** (``gitlab.com``, ``sonarcloud.io``):
          the API rides an apex that also serves ``git push`` / docs, so a
          host-level deny would kill legitimate traffic; credential containment
          carries the weight instead.
        - the providers bound by **``exposed_credential_providers``** — roster-entry
          (agent/tool) names in terok's experimental ``expose_oauth_token`` mode,
          where the agent holds its *real* credential in-container (vault bypassed)
          and so must reach the endpoint directly.  Each entry is mapped to the
          provider it binds (``provider_binding.default``).

        A pure function of the roster and *exposed_credential_providers*, mirroring
        [`generate_routes_json`][terok_executor.roster.loader.AgentRoster.generate_routes_json]:
        the vault routes every provider, so every relayed host is denied.
        """
        exposed = self._exposed_provider_names(exposed_credential_providers)
        hosts: set[str] = set()
        for name, provider in self._providers.items():
            if provider.shared_domain or name in exposed:
                continue
            hosts |= provider.relayed_hosts()
        return frozenset(hosts)

    def _exposed_provider_names(self, exposed_entries: frozenset[str]) -> frozenset[str]:
        """Map exposed roster-entry (agent/tool) names to the providers they bind.

        The exposed set is keyed by roster entry (``claude``, ``codex``), but a
        deny targets a *provider* (``anthropic``, ``openai``).  Resolve each entry
        through its ``provider_binding.default``; entries with no binding (harnesses)
        contribute nothing.
        """
        names: set[str] = set()
        for entry in exposed_entries:
            agent = self._agents.get(entry)
            binding = agent.provider_binding if agent else None
            if binding and binding.default:
                names.add(binding.default)
        return frozenset(names)

    def compose_egress(
        self, *, exposed_credential_providers: frozenset[str] = frozenset()
    ) -> EgressProjection:
        """Project the roster into the shield's egress tiers.

        Bundles the [`deny_to_vault_hosts`][terok_executor.roster.loader.AgentRoster.deny_to_vault_hosts]
        set (t20) with the union of every provider's
        [`egress_allow`][terok_executor.roster.types.Provider.egress_allow]
        (t30) into one [`EgressProjection`][terok_executor.roster.types.EgressProjection];
        both tuples are sorted and de-duplicated for a deterministic bundle.

        *exposed_credential_providers* — roster-entry (agent/tool) names whose real
        credential is exposed in-container; the providers they bind are left
        reachable (see
        [`deny_to_vault_hosts`][terok_executor.roster.loader.AgentRoster.deny_to_vault_hosts]).
        """
        deny = self.deny_to_vault_hosts(exposed_credential_providers=exposed_credential_providers)
        provider_allow = {host for p in self._providers.values() for host in p.egress_allow}
        return EgressProjection(
            deny_to_vault=tuple(sorted(deny)),
            provider_allow=tuple(sorted(provider_allow)),
        )

    def collect_all_auto_approve_env(self) -> dict[str, str]:
        """Merge ``auto_approve.env`` from all agents into one dict."""
        merged: dict[str, str] = {}
        for p in self._agents.values():
            for key, value in p.auto_approve_env.items():
                if key in merged and merged[key] != value:
                    raise ValueError(
                        f"Conflicting auto_approve_env for {key!r}: "
                        f"{merged[key]!r} vs {value!r} (agent {p.name!r})"
                    )
                merged[key] = value
        return merged

    def collect_opencode_provider_env(self) -> dict[str, str]:
        """Collect the ``TEROK_OC_{NAME}_*`` env vars for all OpenCode-driven providers."""
        env: dict[str, str] = {}
        for p in self._providers.values():
            if p.opencode_config is not None:
                env.update(p.opencode_config.to_env(p.name))
        return env

    # ── Process-singleton + selection parsing ──

    @staticmethod
    def load() -> AgentRoster:
        """Load a fresh roster from the current configuration.

        This method reads the configuration on each call. It does not replace
        the process-wide snapshot from
        [`AgentRoster.shared`][terok_executor.roster.loader.AgentRoster.shared].
        """
        return load_roster()

    @staticmethod
    def shared() -> AgentRoster:
        """Return the process-wide cached roster.

        Loaded on first access; every subsequent call returns the same
        instance.  Use this from anywhere that just needs the global
        view. Call
        [`AgentRoster.load`][terok_executor.roster.loader.AgentRoster.load]
        when the caller needs the current configuration.
        """
        return _shared_roster()

    @staticmethod
    def parse_selection(raw: str) -> str | tuple[str, ...]:
        """Normalise a user-supplied agent selection string.

        Accepts a comma-list of selection tokens or the literal ``"all"``.
        Each token is either an agent name (``"claude"``) or a name
        prefixed with ``-`` to exclude it from the selection
        (``"-vibe"``).  The pseudo-name ``"all"`` is also valid as a
        token, so ``"all,-vibe"`` means "everything except vibe".  When
        the input contains only excludes (``"-vibe"``), the selection
        seeds from every installable entry — same effect as
        ``"all,-vibe"``.

        Whitespace is stripped, empty / whitespace-only entries dropped,
        and case folded.  Empty or all-whitespace input collapses to
        ``"all"`` — the same shape
        [`AgentRoster.resolve_selection`][terok_executor.roster.loader.AgentRoster.resolve_selection]
        expects.  Unknown names are not checked here;
        ``resolve_selection`` does that.
        """
        folded = raw.strip().lower()
        if folded == "all" or not folded:
            return "all"
        tokens = tuple(n.strip() for n in folded.split(",") if n.strip())
        return tokens or "all"

    def validate_selection(self, raw: str) -> None:
        """Reject *raw* with ``SystemExit(2)`` if it names roster entries we don't have.

        CLI-flavoured: prints a ``Invalid agent selection: …`` line on
        stderr and exits.  Domain callers that just want the parsed
        tuple should use
        [`parse_selection`][terok_executor.roster.loader.AgentRoster.parse_selection]
        + [`resolve_selection`][terok_executor.roster.loader.AgentRoster.resolve_selection]
        and handle ``ValueError`` themselves.
        """
        try:
            self.resolve_selection(self.parse_selection(raw))
        except ValueError as exc:
            print(f"Invalid agent selection: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    def prompt_selection(self) -> str:
        """Print the installed roster and read one line of executor grammar.

        Empty input → ``"all"``.  Non-interactive stdin (closed pipe)
        exits with a hint to pass the selection positionally instead.
        """
        agents = self.agents
        print("\nAvailable agents:")
        for name in sorted(self.agent_names):
            agent = agents.get(name)
            label = agent.label if agent is not None else name
            print(f"  · {name}  — {label}")
        try:
            raw = input("\nType a comma list, or '-name' to exclude [all]: ").strip()
        except EOFError as exc:
            raise SystemExit(
                "No interactive stdin available.  Pass the selection positionally "
                "instead, e.g. `terok agents set all`."
            ) from exc
        return raw or "all"

    # ── Vault routes + doctor checks ──

    def ensure_vault_routes(self, cfg: SandboxConfig | None = None) -> Path:
        """Generate ``routes.json`` from this roster and write it to disk.

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
        content = self.generate_routes_json() + "\n"
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

    def doctor_checks(self, *, token_broker_port: int | None = None) -> list[DoctorCheck]:
        """Return agent-level health checks for in-container diagnostics.

        Delegates to
        [`terok_executor.doctor`][terok_executor.doctor] for the actual
        check factories; this method is the canonical entry point so
        consumers can discover the checks through the roster.

        Args:
            token_broker_port: Host-side vault broker TCP port.  ``None``
                selects socket mode; any integer selects TCP mode.  Base
                URL checks use the port (or the in-container loopback
                port) to derive the expected host.
        """
        from terok_executor.doctor import _build_agent_doctor_checks

        return _build_agent_doctor_checks(self, token_broker_port=token_broker_port)


@lru_cache(maxsize=1)
def _shared_roster() -> AgentRoster:
    """Process-wide cached roster — backing for ``AgentRoster.shared``."""
    return load_roster()


# ── Public API ────────────────────────────────────────────────────────────


def providers_config_dir() -> Path:
    """Return the current directory for user provider files.

    The function uses the shared XDG-aware Terok configuration root. The
    default directory is ``~/.config/terok/providers/``. Terok also reads the
    legacy ``~/.config/terok/agent/providers/`` directory. Do not use the
    legacy directory for new files.
    """
    return namespace_config_dir() / _USER_PROVIDERS_DIR_NAME


def load_roster() -> AgentRoster:
    """Load the agent roster from bundled YAML + user overrides.

    Bundled agents in ``resources/agents/*.yaml`` are loaded first, then
    user files in ``~/.config/terok/agent/agents/*.yaml`` are deep-merged
    on top (allowing field-level overrides or entirely new agents).  Each
    merged entry is then validated through [`RawAgentYaml`][terok_executor.roster.schema.RawAgentYaml]
    — typos in section keys, wrong types, or unknown fields fail loud
    instead of silently defaulting. Providers use the bundled,
    legacy-user, and current-user layers. See
    [`providers_config_dir`][terok_executor.roster.loader.providers_config_dir].
    """
    raw = _load_bundled_agents()

    # Deep-merge user overrides on top of bundled definitions
    for name, user_data in _load_user_agents().items():
        if name in raw:
            raw[name] = deep_merge(raw[name], user_data)
        else:
            raw[name] = user_data

    providers = _load_providers(reserved_names=frozenset(raw))

    agents: dict[str, Agent] = {}
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
        is_agent_kind = spec.kind not in ("tool", "frontend", "infra")

        if spec.kind not in ("frontend", "infra"):
            all_names.append(name)
        if is_agent_kind:
            agent_names.append(name)
            agents[name] = spec.to_agent(name)

        credential_file = spec.provider.credential_file if spec.provider else ""
        credential_file_writable = bool(spec.provider and spec.provider.credential_file_writable)

        # Agents capture credentials only through an explicit ``auth:`` block;
        # the harness-driven providers' API-key capture is synthesized from
        # their OpenCode config in the provider loop below.
        auth_prov: AuthProvider | None = (
            spec.auth.to_dataclass(name=name, label=label) if spec.auth is not None else None
        )

        if auth_prov is not None:
            # Stamp the provider the captured credential is keyed under (claude →
            # anthropic), so the auth layer can resolve it without importing the
            # roster.  Falls back to the entry's own name when unbound.
            if spec.provider is not None and spec.provider.default:
                auth_prov = replace(auth_prov, credential_provider=spec.provider.default)
            auth_providers[name] = auth_prov
            if auth_prov.host_dir_name not in seen_mounts:
                seen_mounts[auth_prov.host_dir_name] = MountDef(
                    host_dir=auth_prov.host_dir_name,
                    container_path=auth_prov.container_mount,
                    label=f"{auth_prov.label} config",
                    credential_file=credential_file,
                    provider=name,
                    writable=credential_file_writable,
                )

        for m in spec.mounts:
            if m.host_dir not in seen_mounts:
                seen_mounts[m.host_dir] = MountDef(
                    host_dir=m.host_dir,
                    container_path=m.container_path,
                    label=m.label or name,
                )

        if spec.provider is not None and spec.provider.default:
            pname = spec.provider.default
            prov = providers.get(pname)
            if prov is None:
                raise ValueError(
                    f"Agent {name!r} binds provider {pname!r}, which has no "
                    f"resources/providers/{pname}.yaml"
                )
            if pname in vault_routes:
                raise ValueError(
                    f"Provider {pname!r} is bound by more than one agent (second: {name!r}); "
                    f"a provider maps to exactly one vault route"
                )
            vault_routes[pname] = _vault_route_from_binding(pname, prov, spec.provider)

        if spec.sidecar is not None:
            sidecar_specs[name] = spec.sidecar.to_dataclass(default_name=name)

        if spec.install is not None:
            installs[name] = spec.install.to_dataclass()

        if spec.help is not None:
            helps[name] = spec.help.to_dataclass()

        if spec.web_ingress:
            web_ingress_names.add(name)

    # LLM endpoints without an agent binding still need credential delivery.
    # New provider-neutral entries opt in through ``serves`` + API-key auth;
    # legacy curated providers remain eligible through their ``opencode`` block.
    # Install/help are independent decorations for the curated one-word aliases.
    for pname, provider in providers.items():
        oc = provider.opencode_config
        unbound_api_endpoint = (
            pname not in vault_routes
            and bool(provider.serves)
            and provider.api_key_auth is not None
        )
        delivers_to_harness = oc is not None or unbound_api_endpoint
        if delivers_to_harness:
            if pname not in vault_routes:
                vault_routes[pname] = _harness_provider_route(provider)
            if pname not in auth_providers:
                auth_prov = replace(_harness_provider_auth(provider), credential_provider=pname)
                auth_providers[pname] = auth_prov
                if auth_prov.host_dir_name not in seen_mounts:
                    seen_mounts[auth_prov.host_dir_name] = MountDef(
                        host_dir=auth_prov.host_dir_name,
                        container_path=auth_prov.container_mount,
                        label=f"{auth_prov.label} config",
                        credential_file=_PROVIDER_CREDENTIAL_FILE,
                        provider=pname,
                    )
            if pname not in all_names:
                all_names.append(pname)
        if provider.install_spec is not None and pname not in installs:
            installs[pname] = provider.install_spec
        if provider.help_spec is not None and pname not in helps:
            helps[pname] = provider.help_spec

    return AgentRoster(
        _agents=agents,
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


# ── YAML loading ──────────────────────────────────────────────────────────


def _user_agents_dir() -> Path:
    """Return ``~/.config/terok/agent/agents/``."""
    return namespace_config_dir("agent") / _USER_AGENTS_DIR_NAME


def _load_yaml(text: str) -> dict:
    """Parse YAML text into a dict via ruamel.yaml round-trip loader."""
    result = yaml.load(text)
    return result if isinstance(result, dict) else {}


def _legacy_user_providers_dir() -> Path:
    """Return the legacy provider-directory path."""
    return namespace_config_dir("agent") / _USER_PROVIDERS_DIR_NAME


def _load_raw_yaml_dir(pkg_or_path: object, *, label: str) -> dict[str, dict]:
    """Load every ``*.yaml`` in a bundled package or user dir into ``{stem: data}``.

    A parse failure on one file is warned about and skipped rather than
    aborting the whole load — a single broken override must not take the
    roster down.
    """
    out: dict[str, dict] = {}
    items = pkg_or_path.iterdir() if hasattr(pkg_or_path, "iterdir") else []
    for item in items:
        if not hasattr(item, "name") or not item.name.endswith(_YAML_SUFFIX):
            continue
        name = item.name.removesuffix(_YAML_SUFFIX)
        try:
            data = _load_yaml(item.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                f"Warning [{label}]: failed to parse {name!r}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        if data:
            out[name] = data
    return out


def _load_providers(*, reserved_names: frozenset[str] = frozenset()) -> dict[str, Provider]:
    """Load and validate bundled and user provider files.

    Terok loads the bundled definitions first. It then loads the legacy
    ``~/.config/terok/agent/providers/`` directory. It loads the current
    ``~/.config/terok/providers/`` directory last. Each later layer merges into
    the earlier layers. Terok validates each result with
    [`RawProvider`][terok_executor.roster.schema.RawProvider]. Thus, an invalid
    field causes an error instead of a silent default.
    """
    raw = _load_raw_yaml_dir(
        importlib.resources.files("terok_executor.resources.providers"),
        label="providers",
    )
    bundled_names = frozenset(raw)
    user_layers = (
        (_legacy_user_providers_dir(), "legacy providers"),
        (providers_config_dir(), "providers"),
    )
    for user_dir, label in user_layers:
        if not user_dir.is_dir():
            continue
        for name, user_data in _load_raw_yaml_dir(user_dir, label=label).items():
            if name in reserved_names and name not in bundled_names:
                raise ValueError(
                    f"Provider {name!r} conflicts with an existing agent or tool. "
                    "Rename the provider YAML file."
                )
            raw[name] = deep_merge(raw[name], user_data) if name in raw else user_data

    providers: dict[str, Provider] = {}
    for name, data in sorted(raw.items()):
        if _PROVIDER_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(
                f"Provider {name!r} has an invalid name. The YAML file name, without "
                "'.yaml', must match '[a-z0-9]+'. Use only lowercase ASCII letters "
                "and digits."
            )
        try:
            spec = RawProvider.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Provider {name!r}: invalid provider YAML\n{exc}") from exc
        providers[name] = spec.to_dataclass(name=name)
    return providers


def _provider_route_entry(provider: Provider) -> VaultRouteEntry:
    """Build the ``routes.json`` entry for a *provider*.

    The single projection from a [`Provider`][terok_executor.roster.types.Provider]
    to the on-disk vault contract; empty optional fields collapse to ``None``
    so ``exclude_none`` drops them.  Keyed by the provider's clean name when
    assembled into the full file.
    """
    auth_header, auth_prefix, oauth_extra_headers = provider.wire_auth()
    return VaultRouteEntry(
        upstream=provider.upstream,
        auth_header=auth_header,
        auth_prefix=auth_prefix,
        path_upstreams=provider.path_upstreams or None,
        oauth_extra_headers=oauth_extra_headers or None,
        oauth_credential_headers=provider.oauth_credential_headers or None,
        oauth_refresh=provider.oauth_refresh or None,
    )


def _vault_route_from_binding(
    name: str, provider: Provider, binding: RawProviderBinding
) -> VaultRoute:
    """Join a [`Provider`][terok_executor.roster.types.Provider] endpoint with an agent's
    delivery binding into a [`VaultRoute`][terok_executor.roster.types.VaultRoute], keyed by
    the provider name.

    The endpoint half (upstream, wire auth, path overrides, refresh,
    shared-domain) comes from *provider*; the delivery half (phantom-token env
    var, base-URL/socket env, config patch, credential file/type) from
    *binding*.  This is what env assembly, the config patcher, and the doctor
    consume — and what ``generate_routes_json`` serialises.
    """
    auth_header, auth_prefix, oauth_extra_headers = provider.wire_auth()
    return VaultRoute(
        provider=name,
        route_prefix=name,
        upstream=provider.upstream,
        path_upstreams=dict(provider.path_upstreams),
        oauth_extra_headers=oauth_extra_headers,
        oauth_credential_headers=dict(provider.oauth_credential_headers),
        auth_header=auth_header,
        auth_prefix=auth_prefix,
        credential_type=binding.credential_type,
        credential_file=binding.credential_file,
        token_env=dict(binding.token_env),
        token_env_aliases=tuple(binding.token_env_aliases),
        base_url_env=binding.base_url_env,
        socket_env=binding.socket_env,
        shared_config_patch=binding.config_patch,
        oauth_refresh=provider.oauth_refresh,
        shared_domain=provider.shared_domain,
    )


_PROVIDER_CREDENTIAL_FILE = "config.json"
"""File that stores the captured API key in the provider configuration mount.

The read-only credential shadow (terok-ai/terok#873) covers this file.
"""


def _harness_provider_route(provider: Provider) -> VaultRoute:
    """Build a route that delivers provider credentials to OpenCode or Pi.

    Legacy OpenCode configuration uses its specified environment-variable
    prefix. A provider-neutral entry uses an uppercase roster name. For
    example, ``example`` creates ``EXAMPLE_API_KEY``.
    """
    auth_header, auth_prefix, oauth_extra_headers = provider.wire_auth()
    oc = provider.opencode_config
    env_prefix = oc.env_var_prefix if oc else provider.name.upper()
    return VaultRoute(
        provider=provider.name,
        route_prefix=provider.name,
        upstream=provider.upstream,
        path_upstreams=dict(provider.path_upstreams),
        oauth_extra_headers=oauth_extra_headers,
        auth_header=auth_header,
        auth_prefix=auth_prefix,
        credential_type="api_key",
        credential_file=_PROVIDER_CREDENTIAL_FILE,
        token_env={"_default": f"{env_prefix}_API_KEY"},
        base_url_env="",
        socket_env="",
        shared_config_patch=None,
        oauth_refresh=provider.oauth_refresh,
        shared_domain=provider.shared_domain,
    )


def _harness_provider_auth(provider: Provider) -> AuthProvider:
    """Build API-key capture settings for a provider.

    Legacy OpenCode fields specify the key hint and configuration directory. A
    provider-neutral entry uses defaults that Terok derives from the name.
    """
    oc = provider.opencode_config
    hint = (oc.api_key_hint or f"Get your API key at: {oc.auth_key_url}") if oc else ""
    config_dir = oc.config_dir if oc else f".{provider.name}"
    return AuthProvider(
        name=provider.name,
        label=provider.label or provider.name,
        host_dir_name=f"_{provider.name}-config",
        container_mount=f"{_CONTAINER_HOME}/{config_dir}",
        command=[],
        banner_hint="",
        modes=("api_key",),
        api_key_hint=hint,
    )


def _load_bundled_agents() -> dict[str, dict]:
    """Load all ``*.yaml`` files from the bundled ``resources/agents/`` package."""
    agents: dict[str, dict] = {}
    pkg = importlib.resources.files("terok_executor.resources.agents")
    for item in pkg.iterdir():
        if not hasattr(item, "name") or not item.name.endswith(_YAML_SUFFIX):
            continue
        name = item.name.removesuffix(_YAML_SUFFIX)
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
