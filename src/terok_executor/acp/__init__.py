# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-task host-side ACP (Agent Client Protocol) aggregator.

Bridges a single ACP client (Zed, Toad, …) to one of several in-container
agents (claude, codex, copilot, …) by collapsing "pick agent + pick model"
into ACP's standard ``category: "model"`` configOption.  Models are surfaced
namespaced as ``agent:model`` (e.g. ``claude:opus-4.6``).

The aggregator binds to a single agent on the first
``session/set_config_option`` and forwards subsequent JSON-RPC traffic to
that backend; the option list collapses to the bound agent's models on the
next response.  Cross-agent switching is out of scope for v1.

Public surface: :class:`ACPRoster`, :class:`AgentRosterCache`,
:func:`list_authenticated_agents`.  Re-exported from ``terok_executor`` for
``ACPRoster`` and the query function; the cache is exposed for tests.
"""

from __future__ import annotations

from .cache import AgentRosterCache, CacheKey
from .probe import ProbeError, probe_agent_models
from .proxy import AgentBindError
from .roster import ACPEndpointStatus, ACPRoster, list_authenticated_agents

__all__ = [
    "ACPEndpointStatus",
    "ACPRoster",
    "AgentBindError",
    "AgentRosterCache",
    "CacheKey",
    "ProbeError",
    "list_authenticated_agents",
    "probe_agent_models",
]
