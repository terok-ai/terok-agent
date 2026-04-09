# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Resolves filesystem paths for agent state and bind-mount directories.

Delegates to :func:`terok_sandbox.paths.umbrella_state_dir` for the
shared XDG/FHS resolution logic — no vendored copy of the platform
detection code.
"""

from pathlib import Path

from terok_sandbox.paths import umbrella_state_dir

_UMBRELLA = "terok"
_SUBDIR = "agent"


def state_root() -> Path:
    """Writable state root for agent-owned data.

    Priority: ``TEROK_AGENT_STATE_DIR`` → ``/var/lib/terok/agent`` (root)
    → ``platformdirs`` → ``$XDG_DATA_HOME/terok/agent``
    → ``~/.local/share/terok/agent``.
    """
    return umbrella_state_dir(_SUBDIR, "TEROK_AGENT_STATE_DIR")


def mounts_dir() -> Path:
    """Base directory for agent config bind-mounts (container-writable).

    Lives under the ``sandbox-live/`` tree alongside task workspaces,
    grouping all container-writable content for security-aware
    partitioning (``noexec,nosuid,nodev``).

    Each agent/tool gets a subdirectory (e.g. ``_claude-config/``) that is
    bind-mounted read-write into task containers.  These directories are
    intentionally separated from the credentials store since they are
    container-exposed and subject to potential poisoning.
    """
    return umbrella_state_dir("sandbox-live", "TEROK_SANDBOX_LIVE_DIR") / "mounts"
