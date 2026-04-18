# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Loads agent and tool definitions from layered YAML config into a queryable roster.

Delegates to :mod:`.loader` for YAML deserialization and roster construction,
and to :mod:`.config_stack` for generic layered config resolution.
"""

from .loader import (
    AgentRoster,
    CredentialProxyRoute,
    MountDef,
    SidecarSpec,
    ensure_proxy_routes,
    get_roster,
    load_roster,
    parse_agent_selection,
)

__all__ = [
    "AgentRoster",
    "CredentialProxyRoute",
    "MountDef",
    "SidecarSpec",
    "ensure_proxy_routes",
    "get_roster",
    "load_roster",
    "parse_agent_selection",
]
