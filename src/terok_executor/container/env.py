# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Assembles container environment variables and volume mounts for agent launches.

Both ``terok-executor run`` (standalone) and ``terok`` (project orchestrator)
construct identical container environments — shared config mounts, vault
tokens, git identity, unrestricted-mode flags.  This module provides
the canonical assembly function so that logic lives in one place.

Usage::

    from terok_executor.container.env import ContainerEnvSpec, assemble_container_env
    from terok_executor import AgentRoster

    result = assemble_container_env(
        ContainerEnvSpec(task_id="abc", agent_name="claude"),
        AgentRoster.shared(),
    )
    # result.env, result.volumes
"""

from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from terok_executor._util import detect_host_timezone
from terok_executor.integrations.sandbox import (
    CONTAINER_SSH_SIGNER_SOCKET,
    CONTAINER_VAULT_SOCKET,
    Sharing,
    VolumeSpec,
)
from terok_executor.roster.types import EgressProjection

CONTAINER_PROTOCOL = 3
"""Version of the host↔container env/script contract.

Emitted to every container as ``TEROK_CONTAINER_PROTOCOL``, and stamped on
each container at creation, so an operator can read off which contract a
long-lived container was built against.  Bumped on breaking changes to the
env-var or script-interface contract, not on every release.  Old containers
on protocol N keep running; new containers get protocol N+1.

Protocol 3 records terok-sandbox #491, which moved each service's socket
into its own ``/run/terok`` subdirectory and so changed the values of
``TEROK_VAULT_SOCKET`` / ``TEROK_SSH_SIGNER_SOCKET`` / ``TEROK_GATE_SOCKET``.
A container stamped below that starts anyway and is warned about, never
refused — [`MIN_RUNTIME_PROTOCOL`][terok_sandbox.supervision.MIN_RUNTIME_PROTOCOL]
is the matching floor, so the two constants move together."""

if TYPE_CHECKING:
    from terok_executor.integrations.sandbox import (
        CredentialDB,
        PerContainerResources,
        SandboxConfig,
    )
    from terok_executor.roster.loader import AgentRoster
    from terok_executor.roster.types import Provider

_logger = logging.getLogger(__name__)


# ── Vocabulary ──


@dataclass(frozen=True)
class ContainerEnvSpec:
    """Specification for container environment assembly.

    All fields use primitives or [`Path`][pathlib.Path] — no terok-specific
    types.  Callers pre-resolve domain-specific decisions (security class,
    authorship mode, SSH mount, gate mirror creation) and pass results here.
    """

    # -- Required ----------------------------------------------------------

    task_id: str
    """Unique task identifier."""

    agent_name: str
    """Agent name (e.g. ``"claude"``, ``"codex"``)."""

    # -- Workspace location --------------------------------------------------

    workspace_host_path: Path | None = None
    """Host-side workspace directory — caller pre-creates, mounted as
    ``/workspace:Z``.  ``None`` (the default) keeps the workspace in the
    container's writable layer; the init script clones ``code_repo``
    there."""

    # -- Repository setup --------------------------------------------------

    code_repo: str | None = None
    """Git URL to clone inside the container (→ ``CODE_REPO``)."""

    clone_from: str | None = None
    """Secondary clone source for online-mode gate optimization (→ ``CLONE_FROM``)."""

    branch: str | None = None
    """Git branch to check out (→ ``GIT_BRANCH``)."""

    # -- Git identity (container-level defaults) ---------------------------

    git_author_name: str | None = None
    """Resolved from roster provider if ``None``."""

    git_author_email: str | None = None
    git_committer_name: str | None = None
    git_committer_email: str | None = None

    # -- Authorship mode (for per-agent shell wrappers) --------------------

    authorship: str = "agent"
    """Authorship mode consumed by in-container wrappers (→ ``TEROK_GIT_AUTHORSHIP``)."""

    human_name: str = "Nobody"
    """Human operator name (→ ``HUMAN_GIT_NAME``).  terok resolves from project
    config / ``git config``; standalone uses the default or ``--git-identity-from-host``."""

    human_email: str = "nobody@localhost"
    """Human operator email (→ ``HUMAN_GIT_EMAIL``)."""

    # -- Vault --------------------------------------------------

    credential_scope: str = "standalone"
    """Scope for vault token creation.  terok passes ``project.id``."""

    credential_set: str = "default"
    """Vault storage namespace to read credentials from.  Pairs with
    [`Authenticator.run`][terok_executor.credentials.auth.Authenticator.run]'s
    ``credential_set`` — if the auth flow stored a token under set
    ``foo``, the runtime must read from set ``foo`` too or the container
    will see empty env.  Default ``"default"`` matches the shared
    host-wide bucket every standalone caller uses; terok overrides for
    per-project credentials."""

    vault_transport: Literal["direct", "socket"] = "direct"
    """Vault transport mode: ``"direct"`` (HTTP base URL) or ``"socket"``
    (Unix socket path via [`socket_env`][terok_executor.roster.types.VaultRoute.socket_env])."""

    vault_required: bool = False
    """When ``True``, raise ``SystemExit`` if the vault is
    unreachable.  When ``False`` (default), soft-fail to empty env."""

    scan_leaked_creds: bool = False
    """When ``True``, scan shared mounts for real credential files and emit
    warnings.  Standalone mode defaults to off; terok enables this."""

    enabled_vault_patch_providers: frozenset[str] | None = None
    """Provider subset whose shared config patches should be applied.

    ``None`` means "all providers with patches".  An empty set disables
    vault config patching entirely.  terok uses this to gate experimental
    OAuth routing without affecting standalone executor defaults.
    """

    disabled_vault_patch_providers: frozenset[str] | None = None
    """Provider subset whose previously managed config patch values should
    be removed if still owned by terok.  ``None`` removes nothing."""

    expose_credential_providers: frozenset[str] = frozenset()
    """Providers whose credential file should remain writable in-container.

    By default every provider with a [`vault.credential_file`][terok_executor.roster.types.VaultRoute.credential_file]
    gets the file mounted read-only on top of its shared config dir, so an
    in-container ``/login`` cannot taint the host copy
    ([terok-ai/terok#873](https://github.com/terok-ai/terok/issues/873)).
    Providers in this set keep the writable bind — used by terok's
    experimental ``expose_oauth_token`` mode where the agent intentionally
    manages its own token.
    """

    # -- Provider selection ------------------------------------------------

    provider: str | None = None
    """LLM endpoint provider the agent routes to at runtime (→ ``TEROK_PROVIDER``).

    ``None`` leaves the agent on its own default endpoint.  When set, harness
    wrappers (OpenCode today) select this provider unless overridden by an
    explicit ``--provider`` flag.  The per-provider credentials/URLs are
    materialized separately as ``TEROK_PROVIDER_{NAME}_*``."""

    # -- Permissions -------------------------------------------------------

    unrestricted: bool = True
    """Enable auto-approve flags for all agents."""

    # -- Locale ------------------------------------------------------------

    timezone: str | None = None
    """IANA timezone name propagated to the container as ``TZ``.

    ``None`` (the default) means *detect the host's timezone* via
    `terok_executor._util.detect_host_timezone` — the container
    then follows the host.  Pass an explicit string (``"UTC"``,
    ``"Europe/Prague"``) to override, including to pin the container to
    UTC for reproducible runs.  If neither detection nor an override
    yields a zone, ``TZ`` is not set and the image default applies."""

    # -- Agent config ------------------------------------------------------

    agent_config_dir: Path | None = None
    """Pre-prepared agent config directory (→ ``/home/dev/.terok:Z``)."""

    # -- Shared task directory (multi-agent IPC) ---------------------------

    shared_dir: Path | None = None
    """Host-side shared directory.  Created by the assembly function if set."""

    shared_mount: str = "/shared"
    """Container-side mount point for the shared directory."""

    # -- Directories -------------------------------------------------------

    envs_dir: Path | None = None
    """Base directory for shared config mounts.  Uses [`paths.mounts_dir`][terok_executor.paths.mounts_dir]
    if ``None``."""

    # -- Caller-specific mounts --------------------------------------------

    extra_volumes: tuple[VolumeSpec, ...] = ()
    """Additional volume specs from the caller (e.g. SSH mounts from terok)."""


@dataclass(frozen=True)
class ContainerEnvResult:
    """Assembled container environment ready for RunSpec construction.

    Not a ``RunSpec`` — omits launch-time concerns (container name, image,
    command, GPU, shield kill-switch).  Callers add those and construct ``RunSpec``.
    """

    env: dict[str, str]
    """Environment variables for the container."""

    volumes: tuple[VolumeSpec, ...]
    """Typed volume specs — the sandbox decides whether to mount or inject."""

    egress: EgressProjection = field(default_factory=EgressProjection)
    """Roster-derived egress projection — shield's generated t20 / t30 tiers.

    The firewall half of credential containment: relayed provider endpoints
    denied direct (the vault relays them), exposed-credential providers left
    reachable.  Empty by construction when no provider is vault-routed."""


# ── Public API ──


def assemble_container_env(
    spec: ContainerEnvSpec,
    roster: AgentRoster,
    *,
    caller_manages_vault: bool = False,
    per_container: PerContainerResources | None = None,
) -> ContainerEnvResult:
    """Assemble container environment variables and volume mounts.

    This is the **single source of truth** for container env/volume assembly.
    Both ``AgentRunner._run()`` and terok's ``build_task_env_and_volumes()``
    delegate here.

    Args:
        spec: What the caller wants — all host↔container contract fields.
        roster: Agent roster for shared mounts, vault routes, provider identity.
        caller_manages_vault: When ``True``, skip phantom-token injection
            here — the caller injects richer vault tokens itself (e.g.
            terok's per-provider OAuth tiers, socket transport, SSH signer).
            Shared config patches (``api_base`` rewrites) still run because
            the vault **is** in use; only token injection is
            delegated.

    Returns:
        Assembled env dict, volume tuple, and the roster egress projection
        (shield's generated t20 / t30 tiers).
    """
    from terok_executor.paths import mounts_dir as _mounts_dir

    env: dict[str, str] = {}
    volumes: list[VolumeSpec] = []

    # 1. Base env
    env["TASK_ID"] = spec.task_id
    env["REPO_ROOT"] = "/workspace"
    env["GIT_RESET_MODE"] = "none"
    env["TEROK_CONTAINER_PROTOCOL"] = str(CONTAINER_PROTOCOL)
    env["CLAUDE_CONFIG_DIR"] = "/home/dev/.claude"

    # 1b. Timezone — explicit override wins, otherwise follow the host
    if tz := spec.timezone or detect_host_timezone():
        env["TZ"] = tz

    # 1c. Selected provider — harness wrappers default their runtime --provider
    # to this when no explicit flag is passed (see agent-wrappers.sh.j2).
    if spec.provider:
        env["TEROK_PROVIDER"] = spec.provider

    # 2. OpenCode provider env
    env.update(roster.collect_opencode_provider_env())

    # 3. Git identity
    env.update(_resolve_git_identity(spec, roster))

    # 4. Authorship env (for per-agent wrappers inside container)
    env["TEROK_GIT_AUTHORSHIP"] = spec.authorship
    env["HUMAN_GIT_NAME"] = spec.human_name
    env["HUMAN_GIT_EMAIL"] = spec.human_email

    # 5. Branch
    if spec.branch:
        env["GIT_BRANCH"] = spec.branch

    # 6. Repo URLs
    if spec.code_repo:
        env["CODE_REPO"] = spec.code_repo
    if spec.clone_from:
        env["CLONE_FROM"] = spec.clone_from

    # 7. Workspace volume — only the host-mounted cell needs one; the
    #    in-container workspace is just the writable layer at /workspace.
    if spec.workspace_host_path is not None:
        volumes.append(VolumeSpec(spec.workspace_host_path, "/workspace", sharing=Sharing.PRIVATE))

    # 8. Shared config mounts from roster
    mounts_base = spec.envs_dir or _mounts_dir()
    volumes += _shared_config_mounts(
        roster,
        mounts_base,
        expose_credential_providers=spec.expose_credential_providers,
    )

    # 8b. Re-apply vault config patches (idempotent — ensures shared mount
    #     dirs contain correct vault addresses even after state wipe).
    #
    #     NOT gated by caller_manages_vault: that flag only skips
    #     phantom-token injection here because the caller (terok) injects
    #     richer tokens itself — the vault is still in use and
    #     agents still need their config files rewritten to route through
    #     it.  Providers whose credential is exposed directly (Claude OAuth
    #     tier 3) are safe because they have no shared_config_patch.
    from terok_executor.credentials.vault_config import apply_shared_config_patches

    apply_shared_config_patches(
        roster,
        mounts_base,
        providers=spec.enabled_vault_patch_providers,
        disabled_providers=spec.disabled_vault_patch_providers,
        credential_set=spec.credential_set,
    )

    # 9. Vault
    if not caller_manages_vault:
        env.update(
            _inject_vault_tokens(
                roster,
                spec.credential_scope,
                spec.task_id,
                vault_transport=spec.vault_transport,
                vault_required=spec.vault_required,
                credential_set=spec.credential_set,
                per_container=per_container,
            )
        )

    # 9b. Leaked credential scan (runs regardless of caller_manages_vault —
    #     the shared mounts exist either way)
    if spec.scan_leaked_creds:
        _warn_leaked_credentials(mounts_base)

    # 9c. Materialize exposed-credential providers into the same generic handle
    #     as vault-routed ones, so harnesses (pi) select them uniformly via
    #     --provider — no consumer special-cases the exposed set.  Runs regardless
    #     of caller_manages_vault (exposed creds bypass the vault either way).
    if spec.expose_credential_providers:
        env.update(
            _materialize_exposed_providers(roster, mounts_base, spec.expose_credential_providers)
        )

    # 10. Agent config mount
    if spec.agent_config_dir:
        volumes.append(
            VolumeSpec(spec.agent_config_dir, "/home/dev/.terok", sharing=Sharing.PRIVATE)
        )

    # 11. Unrestricted mode
    if spec.unrestricted:
        env["TEROK_UNRESTRICTED"] = "1"
        env.update(roster.collect_all_auto_approve_env())

    # 12. Shared task directory
    if spec.shared_dir:
        spec.shared_dir.mkdir(parents=True, exist_ok=True)
        volumes.append(VolumeSpec(spec.shared_dir, spec.shared_mount))
        env["TEROK_SHARED_DIR"] = spec.shared_mount

    # 13. Extra volumes
    volumes.extend(spec.extra_volumes)

    # 14. Egress projection — the firewall half of credential containment.
    #     The vault contains the credential; shield's generated deny tier keeps
    #     the agent from bypassing it by reaching the provider host directly.
    #     Derived from the same (roster, exposed-providers) inputs as the vault
    #     work above; exposed providers hold the real credential in-container, so
    #     they are skipped (they need the direct route).
    egress = roster.compose_egress(exposed_credential_providers=spec.expose_credential_providers)

    return ContainerEnvResult(env=env, volumes=tuple(volumes), egress=egress)


# ── Private helpers ──


def _warn_leaked_credentials(mounts_base: Path) -> None:
    """Log a warning for each real credential found in a shared mount."""
    from terok_executor.credentials.vault_commands import scan_leaked_credentials

    for provider, path in scan_leaked_credentials(mounts_base):
        _logger.warning("Real credential in shared mount: %s: %s", provider, path)


def _resolve_git_identity(spec: ContainerEnvSpec, roster: AgentRoster) -> dict[str, str]:
    """Resolve the four git identity env vars.

    Uses explicit spec fields when provided, otherwise falls back to the
    roster agent's configured identity (same name for both author and
    committer — standalone default).
    """
    agent = roster.agents.get(spec.agent_name)

    author_name = spec.git_author_name or (agent.git_author_name if agent else "AI Agent")
    author_email = spec.git_author_email or (agent.git_author_email if agent else "ai@localhost")
    committer_name = spec.git_committer_name or author_name
    committer_email = spec.git_committer_email or author_email

    return {
        "GIT_AUTHOR_NAME": author_name,
        "GIT_AUTHOR_EMAIL": author_email,
        "GIT_COMMITTER_NAME": committer_name,
        "GIT_COMMITTER_EMAIL": committer_email,
    }


def _shared_config_mounts(
    roster: AgentRoster,
    mounts_base: Path,
    *,
    expose_credential_providers: frozenset[str] = frozenset(),
) -> list[VolumeSpec]:
    """Derive shared volume specs from the agent roster.

    Uses ``roster.mounts`` — the already-deduplicated list that merges auth
    provider mounts and explicit ``mounts:`` YAML sections.  Creates host
    directories on demand.

    For mounts that carry a [`MountDef.credential_file`][terok_executor.roster.loader.MountDef.credential_file],
    a read-only file mount is layered on top of the shared directory so an
    in-container ``/login`` cannot rewrite the host-side phantom token
    (terok-ai/terok#873).  We ``touch()`` the host-side credential file
    when missing so podman has a real bind source — otherwise podman
    materialises the destination inside the container's userns as root
    with mode 0700, leaving the file unreadable to the agent (and breaking
    e.g. ``gh`` whose ``hosts.yml`` is normally absent on fresh installs).
    Credential files are created — and pre-existing ones re-clamped —
    owner-only, because glab aborts on a ``config.yml`` looser than 0600.

    Providers in *expose_credential_providers* keep the writable mount —
    used by terok's experimental ``expose_oauth_token`` mode where the
    agent intentionally manages its own token.  Mounts flagged
    [`writable`][terok_executor.roster.types.MountDef.writable] skip the shadow
    too: their credential file doubles as a settings file the tool rewrites on
    startup (e.g. glab's ``config.yml``), so the read-only shadow would abort
    the tool — it rides the shared mount writable instead.
    """
    specs: list[VolumeSpec] = []

    for m in roster.mounts:
        host_dir = mounts_base / m.host_dir
        host_dir.mkdir(parents=True, exist_ok=True)
        specs.append(VolumeSpec(host_dir, m.container_path))

        if not m.credential_file:
            continue

        host_file = host_dir / m.credential_file
        _clamp_credential_mode(host_file)

        if m.writable or m.provider in expose_credential_providers:
            continue

        host_file.parent.mkdir(parents=True, exist_ok=True)
        if not host_file.exists():
            host_file.touch(mode=0o600)

        container_file = f"{m.container_path}/{m.credential_file}"
        specs.append(VolumeSpec(host_file, container_file, read_only=True))

    return specs


def _clamp_credential_mode(host_file: Path) -> None:
    """Strip group/other permission bits from an existing credential file.

    Releases up to 0.2.x ``touch()``-ed credential files with the umask
    default (0644), and those files persist in the shared mounts across
    upgrades.  glab refuses to start when its ``config.yml`` is looser
    than 0600, so every launch re-clamps to owner-only — idempotent, and
    never *adds* bits (a deliberately read-only 0400 file stays 0400).

    Opens with ``O_NOFOLLOW`` + ``fchmod`` so a symlink planted in the
    container-writable shared dir cannot redirect the chmod to an
    arbitrary operator-owned file (CWE-59).  The clamp is best-effort:
    any open failure (missing file, planted symlink, permissions) leaves
    the file as-is rather than aborting container assembly — a file we
    cannot even open is not one the old ``touch()`` created loose.
    """
    try:
        fd = os.open(host_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return
    try:
        st = os.fstat(fd)
        if stat.S_ISREG(st.st_mode) and stat.S_IMODE(st.st_mode) & 0o077:
            os.fchmod(fd, stat.S_IMODE(st.st_mode) & ~0o077)
    finally:
        os.close(fd)


def _set_provider_handle(
    env: dict[str, str],
    provider: Provider,
    token: str,
    base_prefix: str,
) -> None:
    """Set the generic provider variables that OpenCode and Pi read.

    The variables contain a token, model data, and a base URL for each
    protocol. The ``--provider`` option selects *provider* at run time. For a
    routed provider, *base_prefix* is the vault loopback and *token* is a
    phantom token. For an exposed provider, these values are the upstream URL
    and the real token. The variable format is the same for both provider
    types.
    """
    handle = provider.name.upper()
    env[f"TEROK_PROVIDER_{handle}_TOKEN"] = token
    env[f"TEROK_PROVIDER_{handle}_LABEL"] = provider.label or provider.name
    env[f"TEROK_PROVIDER_{handle}_MODELS"] = json.dumps(
        {
            model_id: {
                key: value
                for key, value in {
                    "name": model.name,
                    "context_limit": model.context_limit,
                    "output_limit": model.output_limit,
                }.items()
                if value is not None
            }
            for model_id, model in sorted(provider.models.items())
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    if provider.default_model:
        env[f"TEROK_PROVIDER_{handle}_DEFAULT_MODEL"] = provider.default_model
    for proto, path in provider.serves.items():
        proto_var = proto.upper().replace("-", "_")
        env[f"TEROK_PROVIDER_{handle}_BASE_{proto_var}"] = f"{base_prefix}{path}"


def _materialize_exposed_providers(
    roster: AgentRoster, mounts_base: Path, expose_credential_providers: frozenset[str]
) -> dict[str, str]:
    """Materialize ``TEROK_PROVIDER_<NAME>_*`` handles for exposed providers.

    Exposed mode bypasses the vault: the real token is written only to the
    provider's credential mount.  Read it (host-side) and emit the same handle a
    vault-routed provider gets — real token, base pointing at the upstream
    directly — so a harness selects exposed and proxied providers uniformly.
    Missing or unreadable credentials are skipped.
    """
    from terok_executor.credentials.extractors import extract_credential

    env: dict[str, str] = {}
    for entry in expose_credential_providers:
        auth_info = roster.auth_providers.get(entry)
        if not auth_info:
            continue
        provider_name = auth_info.credential_provider or entry
        provider = roster.providers.get(provider_name)
        if not (provider and provider.serves):
            continue
        try:
            cred = extract_credential(entry, mounts_base / auth_info.host_dir_name)
        except ValueError:
            continue
        token = cred.get("access_token") or cred.get("token") or cred.get("key")
        if token:
            _set_provider_handle(env, provider, token, provider.upstream)
    return env


def _inject_vault_tokens(
    roster: AgentRoster,
    scope: str,
    task_id: str,
    *,
    vault_transport: Literal["direct", "socket"] = "direct",
    vault_required: bool = False,
    credential_set: str = "default",
    per_container: PerContainerResources | None = None,
) -> dict[str, str]:
    """Inject vault phantom tokens for routed providers.

    Handles three orthogonal concerns:

    - **Auth**: picks the phantom-token env var from ``token_env`` keyed by
      the stored credential type (Phase 1).
    - **Transport**: resolves the shared vault address and writes it into
      every route's ``socket_env`` / ``base_url_env`` (Phase 2).
    - **SSH signer**: creates a phantom token for the SSH signer when the
      scope has registered SSH keys (Phase 3).

    *credential_set* selects which vault DB namespace to query — the auth
    flow stores credentials keyed by ``(credential_set, provider)``, and
    the runtime must use the same value to look them back up.

    The per-container supervisor (spawned by the terok-sandbox OCI hook
    on container start) embeds the vault proxy and binds to the
    socket/port written into the sidecar config.  Token creation only
    needs DB access; the broker port is config-derived.  *vault_required*
    gates DB-level failure handling: a hard failure (terok project mode)
    versus a soft-fail to an empty dict (standalone default).
    """
    from terok_executor.credentials.extractors import credential_type
    from terok_executor.integrations.sandbox import SandboxConfig

    cfg = SandboxConfig()

    # Provider YAML is live host configuration, not image content. Update vault
    # routes before token assembly. Then, the next task can use a new provider
    # without a manual command or an image rebuild.
    try:
        roster.ensure_vault_routes(cfg=cfg)
    except Exception:
        _logger.exception("Vault route generation failed")
        if vault_required:
            raise SystemExit(
                "Terok could not generate vault routes. Read the logs for details."
            ) from None
        return {}

    vault_routes = roster.vault_routes
    try:
        db = cfg.open_credential_db()
    except Exception:
        _logger.exception("Vault DB unavailable")
        if vault_required:
            raise SystemExit("Vault DB unavailable. Check logs for details.") from None
        return {}

    try:
        stored = set(db.list_credentials(credential_set))
        routed = stored & vault_routes.keys()

        # SSH signer token is independent of provider credentials — a project
        # with SSH keys but no API creds should still get TEROK_SSH_SIGNER_*.
        ssh_token = _load_ssh_signer_token(db, cfg, scope, task_id)

        if not routed and not ssh_token:
            return {}

        credential_types: dict[str, str] = {}
        tokens: dict[str, str] = {}
        for name in routed:
            cred = db.load_credential(credential_set, name)
            credential_types[name] = credential_type(cred)
            tokens[name] = db.create_token(scope, task_id, credential_set, name)

        # Per-container port if the caller allocated one (production
        # path); fallback to the host-singleton ``cfg`` field for tests
        # + ad-hoc callers that don't pass one.  In production the runner
        # always provides ``per_container`` so concurrent containers get
        # distinct ports.
        allocated_port = (
            per_container.token_broker_port if per_container is not None else cfg.token_broker_port
        )
    except Exception:
        _logger.exception("Vault token injection failed")
        if vault_required:
            raise SystemExit("Vault token injection failed. Check logs for details.") from None
        return {}
    finally:
        db.close()

    use_socket = vault_transport == "socket"
    port = None if use_socket else allocated_port
    # Resolve the single container-side vault address used by every
    # socket/URL injection below.  Socket mode points at the mounted host
    # socket + a loopback HTTP bridge; TCP mode points at the broker's TCP
    # endpoint + a socat-fronted Unix socket.  Agents don't decide per-route
    # any more — addressing is centralised.
    from terok_executor.credentials.vault_config import resolve_vault_location
    from terok_executor.vault_addr import LOOPBACK_VAULT_PORT, VAULT_LOOPBACK_PORT_ENV

    location = resolve_vault_location(token_broker_port=port)
    host_tcp = f"host.containers.internal:{port}" if port else None
    env: dict[str, str] = {}

    for name, route in vault_routes.items():
        if name not in routed:
            continue

        stored_type = credential_types.get(name, "")
        token_var = route.token_env.get(stored_type) or route.token_env.get("_default")
        if token_var:
            env[token_var] = tokens[name]
        env.update(dict.fromkeys(route.token_env_aliases, tokens[name]))

        if route.socket_env:
            env[route.socket_env] = location.socket
        if route.base_url_env:
            env[route.base_url_env] = location.url

        # OpenCode base URL override for proxied providers — point the wrapper
        # at the vault loopback (with the provider's served path) instead of
        # its direct endpoint.  Using the served path is what gets OpenRouter's
        # mandatory ``/api`` segment right (``/api/v1``, not ``/v1``).
        provider = roster.providers.get(name)
        if provider and provider.opencode_config:
            oc_path = provider.serves.get("openai-chat", "/v1")
            env[f"TEROK_OC_{name.upper()}_BASE_URL"] = f"{location.url}{oc_path}"

        # Materialize a generic per-provider handle so a harness (opencode, pi)
        # can select ANY authenticated, protocol-compatible provider at runtime
        # via --provider — not just its bound default.  Routed providers go
        # through the vault, so the base is the loopback plus the served path.
        if provider and provider.serves:
            _set_provider_handle(env, provider, tokens[name], location.url)
        # The gitlab provider (glab CLI) uses its own host+protocol split; in
        # socket mode it rides the same in-container loopback bridge as every
        # other HTTP-only client.
        if name == "gitlab":
            env["API_PROTOCOL"] = "http"
            env["GITLAB_API_HOST"] = host_tcp if host_tcp else f"localhost:{LOOPBACK_VAULT_PORT}"

    if routed:
        if use_socket:
            env["TEROK_VAULT_SOCKET"] = CONTAINER_VAULT_SOCKET
        elif port:
            env["TEROK_TOKEN_BROKER_PORT"] = str(port)
        # The in-container loopback bridge runs in both transports
        # (socat unix→vault.sock or socat tcp→host:broker), so the URL
        # http://localhost:9419/v1 is uniform — agents and patched
        # config files never see per-container host details.
        env[VAULT_LOOPBACK_PORT_ENV] = str(LOOPBACK_VAULT_PORT)

    if ssh_token:
        env["TEROK_SSH_SIGNER_TOKEN"] = ssh_token
        if use_socket:
            env["TEROK_SSH_SIGNER_SOCKET"] = CONTAINER_SSH_SIGNER_SOCKET
        else:
            # Per-container port (production); fallback to cfg (tests).
            # See ``port`` resolution above for the same shape.
            ssh_port = (
                per_container.ssh_signer_port if per_container is not None else cfg.ssh_signer_port
            )
            if ssh_port is not None:
                env["TEROK_SSH_SIGNER_PORT"] = str(ssh_port)

    _logger.debug("Vault: injected %d env vars for %s", len(env), routed)
    return env


def _load_ssh_signer_token(
    db: CredentialDB, cfg: SandboxConfig, scope: str, task_id: str
) -> str | None:
    """Mint an SSH signer phantom token when *scope* has at least one assigned key.

    SSH keys live as rows in the vault's ``credentials.db`` (see
    ``ssh_keys`` / ``ssh_key_assignments`` in terok-sandbox); no sidecar
    JSON to consult.  *cfg* is retained for API symmetry — callers pass
    it regardless, and future policy knobs may use it.
    """
    del cfg  # currently unused; part of a stable call-site signature
    if db.list_ssh_keys_for_scope(scope):
        return db.create_token(scope, task_id, scope, "ssh")
    return None
