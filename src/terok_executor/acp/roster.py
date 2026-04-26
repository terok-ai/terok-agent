# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-task ACP roster: aggregates in-container agents into one endpoint.

:class:`ACPRoster` owns the per-task state for the ACP host-proxy:

- the cache lookup that answers "what models does this agent advertise?"
- the live walk that answers "what agents are currently authenticated for
  this image?" — re-evaluated on every ``session/new`` so newly-authed
  agents appear without daemon restart
- the proxy attach loop (delegated to :mod:`.proxy`) that brokers JSON-RPC
  frames between the connected client and the chosen backend

The class follows the shape of
:class:`terok_executor.container.runner.AgentRunner`: lazy-init
properties for cross-cutting subsystems, OOP over free functions, no
mutable state in ``__init__`` beyond the parameters themselves.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from .cache import GLOBAL_CACHE, AgentRosterCache, CacheKey
from .probe import ProbeError, probe_agent_models
from .proxy import ACPProxy

if TYPE_CHECKING:
    from pathlib import Path

    from terok_sandbox import Sandbox

_logger = logging.getLogger(__name__)

DEFAULT_AUTH_IDENTITY = "global"
"""Sentinel used everywhere terok auth is currently process-wide.

Future per-project auth makes this variable; the cache key already
accommodates the change without a schema migration.
"""

DEFAULT_CREDENTIAL_SCOPE = "default"
"""Scope name used by :class:`terok_sandbox.CredentialDB` for the
process-wide credential set.  Mirrors what
:func:`terok_executor.credentials.auth.authenticate` writes."""


class ACPEndpointStatus(StrEnum):
    """Surface state of a per-task ACP endpoint, as visible to ``acp list``.

    Tracks daemon presence and readiness without forcing the listing path
    to actually probe or connect.
    """

    ACTIVE = "active"
    """Daemon up, socket bound, ready for client connections."""

    READY = "ready"
    """Task running with at least one authenticated agent — a daemon
    will spawn on first ``terok acp connect``."""

    UNSUPPORTED = "unsupported"
    """Task running but no in-image agents are authenticated.  Connect
    would fail; surface honestly so the user knows to authenticate."""


@dataclass(frozen=True)
class _AgentMatrix:
    """Outcome of a live walk of the image's agents and the credential DB.

    Kept private — callers consume :meth:`ACPRoster.list_available_agents`
    instead.  Stored as a record so the proxy can ask "is agent X
    authenticated?" without re-querying the DB.
    """

    configured: tuple[str, ...]
    """Agents declared in the image's ``ai.terok.agents`` label."""

    authenticated: frozenset[str]
    """Subset of *configured* with credentials in the vault."""


def list_authenticated_agents(
    *,
    db_path: Path | None = None,
    scope: str = DEFAULT_CREDENTIAL_SCOPE,
) -> list[str]:
    """Return provider names that have stored credentials in *scope*.

    Pure query against :class:`terok_sandbox.CredentialDB` — no probing,
    no container exec.  Used by :class:`ACPRoster` and by the host-side
    ``acp list`` to classify endpoints as ``ready`` vs ``unsupported``.
    """
    from terok_sandbox import CredentialDB, SandboxConfig

    path = db_path or SandboxConfig().db_path
    db = CredentialDB(path)
    try:
        return list(db.list_credentials(scope))
    finally:
        db.close()


class ACPRoster:
    """Per-task ACP aggregator.

    Construct one per running task — the roster owns the per-agent
    probe cache lookups, the live "who is authenticated right now?"
    walk, and the attach loop that brokers a connected ACP client.

    Heavy subsystems (sandbox handle, credential DB, agent label) are
    resolved lazily so unit tests can exercise the roster without
    actually opening a container.
    """

    def __init__(
        self,
        *,
        task_id: str,
        container_name: str,
        image_id: str,
        sandbox: Sandbox,
        auth_identity: str = DEFAULT_AUTH_IDENTITY,
        credential_scope: str = DEFAULT_CREDENTIAL_SCOPE,
        cache: AgentRosterCache | None = None,
    ) -> None:
        self._task_id = task_id
        self._container_name = container_name
        self._image_id = image_id
        self._sandbox = sandbox
        self._auth_identity = auth_identity
        self._credential_scope = credential_scope
        self._cache = cache or GLOBAL_CACHE
        self._configured_agents_cache: tuple[str, ...] | None = None

    # ── Lazy-init properties (mirrors AgentRunner) ─────────────────────

    @property
    def configured_agents(self) -> tuple[str, ...]:
        """Agents declared in the image's ``ai.terok.agents`` label.

        Parsed once per roster instance — the image label is stable for
        the lifetime of the running task.  The label format is a comma-
        separated list (see ``terok_executor.container.build:63``).
        """
        if self._configured_agents_cache is None:
            from terok_executor.container.build import AGENTS_LABEL

            image = self._sandbox.runtime.image(self._image_id)
            raw = image.labels().get(AGENTS_LABEL, "")
            self._configured_agents_cache = tuple(
                token for token in (s.strip() for s in raw.split(",")) if token
            )
        return self._configured_agents_cache

    # ── Public surface ────────────────────────────────────────────────

    def agent_matrix(self) -> _AgentMatrix:
        """Return the live (configured, authenticated) snapshot for this task.

        Cheap — one credential-DB query, no probing.  Recomputed on
        every call so newly-authed agents are reflected immediately.
        """
        configured = self.configured_agents
        authed = frozenset(list_authenticated_agents(scope=self._credential_scope)).intersection(
            configured
        )
        return _AgentMatrix(configured=configured, authenticated=authed)

    def list_available_agents(self) -> list[str]:
        """Return ``agent:model`` ids ready to surface to a client.

        Walks the configured agents, intersects with current auth, and
        emits namespaced ids drawing models from the cache.  Cache misses
        trigger a one-shot probe via :meth:`warm`.  Probe failures cache
        an empty roster (so we don't hammer a misconfigured agent every
        ``session/new``) and the agent is silently skipped.
        """
        matrix = self.agent_matrix()
        out: list[str] = []
        for agent in matrix.configured:
            if agent not in matrix.authenticated:
                continue
            models = self._cache.get(self._cache_key(agent))
            if models is None:
                models = self.warm(agent)
            for model in models:
                out.append(f"{agent}:{model}")
        return out

    def warm(self, agent_id: str) -> tuple[str, ...]:
        """Probe *agent_id* and store its roster in the cache.

        Returns the probed model tuple (possibly empty on failure).
        Callers don't normally need this — :meth:`list_available_agents`
        warms lazily — but workflows can call it eagerly after auth
        completion to pre-populate the cache.
        """
        key = self._cache_key(agent_id)
        try:
            models = asyncio.run(self._probe(agent_id))
        except ProbeError as exc:
            _logger.warning("ACP probe failed for agent %r: %s", agent_id, exc)
            models = ()
        self._cache.put(key, models)
        return models

    async def attach(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Run the proxy loop for one connected client until disconnect.

        Delegates the JSON-RPC state machine to :class:`ACPProxy`.  The
        roster owns the data (cache + live walk); the proxy owns the
        protocol.
        """
        proxy = ACPProxy(roster=self)
        await proxy.run(reader, writer)

    # ── Internals ─────────────────────────────────────────────────────

    def _cache_key(self, agent_id: str) -> CacheKey:
        return CacheKey(
            image_id=self._image_id,
            auth_identity=self._auth_identity,
            agent_id=agent_id,
        )

    async def _probe(self, agent_id: str) -> tuple[str, ...]:
        """Drive a single probe in the current event loop."""
        container = self._sandbox.runtime.container(self._container_name)
        return await probe_agent_models(
            agent_id=agent_id,
            container=container,
            sandbox=self._sandbox,
        )

    # ── Accessors used by the proxy ──────────────────────────────────

    @property
    def container_name(self) -> str:
        """Container the bound agent will exec into."""
        return self._container_name

    @property
    def sandbox(self) -> Sandbox:
        """Backing sandbox handle (provides the runtime + exec_stdio)."""
        return self._sandbox

    @property
    def task_id(self) -> str:
        """Identifier of the running task this roster aggregates."""
        return self._task_id
