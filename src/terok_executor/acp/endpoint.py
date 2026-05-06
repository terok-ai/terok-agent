# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Surface state of a per-task ACP endpoint as the host's discovery view sees it.

The host CLI (``terok acp list``) and the future TUI panel both render
endpoints by status.  This catalog defines the enum without dragging
the whole roster module in — useful because the host imports the
status earlier in its startup than it imports the roster.
"""

from __future__ import annotations

from enum import StrEnum


class ACPEndpointStatus(StrEnum):
    """Live state of a per-task ACP endpoint.

    The host classifier (``Project.acp_endpoints``) attaches one of
    these to every running task; the value drives both the rendered
    row in ``acp list`` and the decision ``acp connect`` makes about
    whether to spawn a daemon.
    """

    ACTIVE = "active"
    """Daemon up, socket bound, ready for client connections."""

    READY = "ready"
    """Task running with at least one authenticated agent — a daemon
    will spawn on first ``terok acp connect``."""

    UNSUPPORTED = "unsupported"
    """Task running but no in-image agents are authenticated.  Connect
    would fail; surface honestly so the user knows to authenticate."""
