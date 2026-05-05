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
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from terok_sandbox import CredentialDB, SandboxConfig

from terok_executor.container.build import AGENTS_LABEL

from .cache import GLOBAL_CACHE, AgentRosterCache, CacheKey
from .probe import ProbeError, probe_agent_models
from .proxy import ACPProxy

if TYPE_CHECKING:
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


def list_authenticated_agents(
    *,
    db_path: Path | None = None,
    scope: str = DEFAULT_CREDENTIAL_SCOPE,
) -> list[str]:
    """Return provider names that have stored credentials in *scope*.

    Pure query against :class:`terok_sandbox.CredentialDB` — no probing,
    no container exec.  Used by the host-side ``acp list`` to classify
    endpoints in its status display; the roster itself doesn't gate
    probing on this anymore (file-based auth like Claude's OAuth lives
    outside the vault, so a vault-only filter would silently hide
    working agents).
    """
    path = db_path or SandboxConfig().db_path
    db = CredentialDB(path)
    try:
        return list(db.list_credentials(scope))
    finally:
        db.close()


class ACPRoster:
    """Per-task ACP aggregator.

    Construct one per running task — the roster owns the per-agent
    probe cache lookups and the attach loop that brokers a connected
    ACP client.  It probes every agent declared in the image's
    ``ai.terok.agents`` label; failed probes (missing wrapper, no
    credentials, agent crashed) cache empty so a misbehaving agent
    doesn't get re-probed every ``session/new``.  The roster
    deliberately does *not* consult the credential vault: that view
    is incomplete (file-mounted creds aren't there) and the proxy
    has nothing useful to do with the answer anyway — a probe that
    succeeds is, by definition, an authed agent.
    """

    def __init__(
        self,
        *,
        container_name: str,
        image_id: str,
        sandbox: Sandbox,
        auth_identity: str = DEFAULT_AUTH_IDENTITY,
        cache: AgentRosterCache | None = None,
    ) -> None:
        self._container_name = container_name
        self._image_id = image_id
        self._sandbox = sandbox
        self._auth_identity = auth_identity
        # Don't ``cache or GLOBAL_CACHE`` here — ``AgentRosterCache`` defines
        # ``__len__``, so an empty cache is falsy and would silently swap in
        # the global singleton.  Explicit ``is None`` check.
        self._cache = cache if cache is not None else GLOBAL_CACHE

    # ── Lazy-init properties (mirrors AgentRunner) ─────────────────────

    @cached_property
    def configured_agents(self) -> tuple[str, ...]:
        """Agents declared in the image's ``ai.terok.agents`` label.

        Parsed once per roster instance — the image label is stable for
        the lifetime of the running task.  The label is a comma-
        separated list (see :data:`terok_executor.container.build.AGENTS_LABEL`).
        """
        image = self._sandbox.runtime.image(self._image_id)
        raw = image.labels().get(AGENTS_LABEL, "")
        return tuple(token for token in (s.strip() for s in raw.split(",")) if token)

    @cached_property
    def acp_capable_agents(self) -> tuple[str, ...]:
        """Subset of ``configured_agents`` that ship a ``terok-{agent}-acp`` wrapper.

        The image label lists every agent in the runtime — claude,
        opencode, gh, sonar, blablador, etc.  Of those, only the
        ones that actually install an ACP wrapper script (currently
        claude, codex, copilot, opencode, vibe) can be probed by the
        proxy; the rest are tools or LLM gateways that don't speak
        the protocol at all.  Probing them anyway costs a full
        ``probe_timeout`` per agent for nothing — and worse, leaves
        their wrappers as zombie subprocess threads in the executor
        pool until exec_stdio's own timeout kills them.

        Resolved by a single in-container shell call at first use
        (``command -v`` is built-in to bash, near-zero cost).  The
        property is cached for the roster's lifetime; new wrappers
        installed mid-task aren't picked up without a daemon restart.
        """
        agents = self.configured_agents
        if not agents:
            return ()
        # One ``bash -c`` exec instead of N — the latency is dominated
        # by ``podman exec`` round-trip, so coalescing matters.
        # ``command -v`` prints the resolved path on success and
        # silently fails on missing; we echo the agent name only on
        # success so the result is a newline-separated whitelist.
        script = "; ".join(
            f"command -v 'terok-{agent}-acp' >/dev/null 2>&1 && echo {agent}" for agent in agents
        )
        try:
            result = self._sandbox.runtime.exec(
                self._sandbox.runtime.container(self._container_name),
                ["bash", "-c", script],
                timeout=5.0,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "ACP roster: wrapper-existence check failed: %s — falling back to full list",
                exc,
            )
            return agents
        present = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        kept = tuple(a for a in agents if a in present)
        skipped = tuple(a for a in agents if a not in present)
        if skipped:
            _logger.info(
                "ACP roster: skipping agents without an ACP wrapper: %s",
                ", ".join(skipped),
            )
        return kept

    # ── Domain operations ────────────────────────────────────────────

    async def list_available_agents(self) -> list[str]:
        """Return ``agent:model`` ids ready to surface to a client.

        Probes every agent in the image's ``ai.terok.agents`` label
        (filtered through the cache) and concatenates the namespaced
        model ids of those that responded.  Cold-cache agents are
        probed in parallel via :func:`asyncio.gather`, so first-call
        latency is ``max(probe_time)`` rather than ``sum(probe_time)``.
        Successful probes cache the model tuple for the daemon's
        lifetime; failed probes are *not* cached so a transient cold
        start (Node wrapper warming up, OAuth refresh in flight) can
        recover on the next ``session/new`` instead of wedging the
        roster empty until the daemon restarts.
        """
        agents_in_order = self.acp_capable_agents
        cold = [a for a in agents_in_order if self._cache.get(self._cache_key(a)) is None]
        if cold:
            await asyncio.gather(*(self.warm(a) for a in cold))
        out: list[str] = []
        for agent in agents_in_order:
            for model in self._cache.get(self._cache_key(agent)) or ():
                out.append(f"{agent}:{model}")
        return out

    async def warm(self, agent_id: str) -> tuple[str, ...]:
        """Probe *agent_id* and cache the result on success only.

        Returns the probed model tuple (possibly empty on failure).
        Failures are deliberately *not* cached — caching ``()`` on
        timeout used to wedge the daemon's roster empty after a
        single bad first-probe (cold container, slow Node start, OAuth
        refresh) and only a daemon restart could recover.  The trade-
        off is paid in cold-start latency: an agent that's genuinely
        unavailable will be re-probed every ``session/new`` and add
        its full timeout to the response.  The cache is per-daemon,
        not per-connection, so once an agent probes successfully its
        roster is reused across every Zed reconnect.
        """
        key = self._cache_key(agent_id)
        try:
            models = await self._probe(agent_id)
        except ProbeError as exc:
            _logger.warning("ACP probe failed for agent %r: %s", agent_id, exc)
            return ()
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

    def exec_wrapper(self, agent_id: str, *, stdin: object, stdout: object) -> int:
        """Run ``terok-{agent_id}-acp`` in the task container with bridged stdio.

        The proxy spawns backends through this method so the sandbox
        and container handles stay private to the roster.  Sync because
        :meth:`Sandbox.runtime.exec_stdio` is sync — callers in async
        contexts wrap the call in ``loop.run_in_executor``.
        Stderr is left at exec_stdio's default (DEVNULL); capturing it
        through a long-lived pipe was the structural difference between
        bind (hung) and probe (worked) while debugging the silent-bind
        regression — keep it off here and reach for an ad-hoc invocation
        if you ever need to inspect a wrapper's stderr directly.
        """
        runtime = self._sandbox.runtime
        return runtime.exec_stdio(
            runtime.container(self._container_name),
            [f"terok-{agent_id}-acp"],
            stdin=stdin,
            stdout=stdout,
        )

    # ── Lower-level operations ───────────────────────────────────────

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

    # ── Queries ──────────────────────────────────────────────────────
