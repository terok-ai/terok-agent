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
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
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


def _default_auth_source(scope: str) -> list[str]:
    """The default ``auth_source`` :class:`ACPRoster` uses — wraps the DB.

    A free function so tests can inject any ``Callable[[str], list[str]]``
    via the constructor without monkey-patching this module.
    """
    return list_authenticated_agents(scope=scope)


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
        auth_source: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._task_id = task_id
        self._container_name = container_name
        self._image_id = image_id
        self._sandbox = sandbox
        self._auth_identity = auth_identity
        self._credential_scope = credential_scope
        # Don't ``cache or GLOBAL_CACHE`` here — ``AgentRosterCache`` defines
        # ``__len__``, so an empty cache is falsy and would silently swap in
        # the global singleton.  Explicit ``is None`` check.
        self._cache = cache if cache is not None else GLOBAL_CACHE
        self._auth_source: Callable[[str], list[str]] = auth_source or _default_auth_source

    # ── Lazy-init properties (mirrors AgentRunner) ─────────────────────

    @cached_property
    def configured_agents(self) -> tuple[str, ...]:
        """Agents declared in the image's ``ai.terok.agents`` label.

        Parsed once per roster instance — the image label is stable for
        the lifetime of the running task.  The label format is a comma-
        separated list (see ``terok_executor.container.build:63``).
        """
        from terok_executor.container.build import AGENTS_LABEL

        image = self._sandbox.runtime.image(self._image_id)
        raw = image.labels().get(AGENTS_LABEL, "")
        return tuple(token for token in (s.strip() for s in raw.split(",")) if token)

    # ── Public surface ────────────────────────────────────────────────

    def agent_matrix(self) -> _AgentMatrix:
        """Return the live (configured, authenticated) snapshot for this task.

        Cheap — one credential-DB query, no probing.  Recomputed on
        every call so newly-authed agents are reflected immediately.
        """
        configured = self.configured_agents
        authed = frozenset(self._auth_source(self._credential_scope)).intersection(configured)
        return _AgentMatrix(configured=configured, authenticated=authed)

    async def list_available_agents(self) -> list[str]:
        """Return ``agent:model`` ids ready to surface to a client.

        Walks configured agents, intersects with current auth, draws
        models from the cache.  Cold-cache agents are probed in
        parallel via :func:`asyncio.gather`, so first-call latency is
        ``max(probe_time)`` rather than ``sum(probe_time)``.  Probe
        failures cache an empty roster (so we don't hammer a
        misconfigured agent every ``session/new``) and the agent is
        silently skipped.
        """
        matrix = self.agent_matrix()
        agents_in_order = [a for a in matrix.configured if a in matrix.authenticated]
        cold = [a for a in agents_in_order if self._cache.get(self._cache_key(a)) is None]
        if cold:
            await asyncio.gather(*(self.warm(a) for a in cold))
        out: list[str] = []
        for agent in agents_in_order:
            for model in self._cache.get(self._cache_key(agent)) or ():
                out.append(f"{agent}:{model}")
        return out

    async def warm(self, agent_id: str) -> tuple[str, ...]:
        """Probe *agent_id* and store its roster in the cache.

        Returns the probed model tuple (possibly empty on failure).
        Callers don't normally need this — :meth:`list_available_agents`
        warms lazily — but workflows can call it eagerly after auth
        completion to pre-populate the cache.
        """
        key = self._cache_key(agent_id)
        try:
            models = await self._probe(agent_id)
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
    def task_id(self) -> str:
        """Identifier of the running task this roster aggregates."""
        return self._task_id

    def exec_wrapper(self, agent_id: str, *, stdin: object, stdout: object) -> int:
        """Run ``terok-{agent_id}-acp`` in the task container with bridged stdio.

        Sync entry point the proxy drives via ``loop.run_in_executor`` —
        keeps the sandbox + container handle behind one method instead of
        leaking them as public properties just so the proxy can spawn
        backends.  *stdin* / *stdout* are the host-side ends of an
        ``os.pipe()`` pair (as :class:`BinaryIO`).
        """
        runtime = self._sandbox.runtime
        return runtime.exec_stdio(
            runtime.container(self._container_name),
            [f"terok-{agent_id}-acp"],
            stdin=stdin,
            stdout=stdout,
        )
