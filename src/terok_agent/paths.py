# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Platform-aware path resolution for terok-agent directories.

Provides XDG / FHS resolution for the agent's own state directory,
independent of terok-sandbox's namespace.
"""

import getpass
import os
from pathlib import Path

try:
    from platformdirs import user_data_dir as _user_data_dir
except ImportError:  # optional dependency
    _user_data_dir = None  # type: ignore[assignment]


APP_NAME = "terok-agent"


def _is_root() -> bool:
    """Return True if the current process is running as root."""
    try:
        return os.geteuid() == 0  # type: ignore[attr-defined]
    except AttributeError:
        return getpass.getuser() == "root"


def state_root() -> Path:
    """Writable state root for agent-owned data.

    Priority: ``TEROK_AGENT_STATE_DIR`` → ``/var/lib/terok-agent`` (root)
    → ``platformdirs.user_data_dir()`` → ``$XDG_DATA_HOME/terok-agent``
    → ``~/.local/share/terok-agent``.
    """
    env = os.getenv("TEROK_AGENT_STATE_DIR")
    if env:
        return Path(env).expanduser()
    if _is_root():
        return Path("/var/lib") / APP_NAME
    if _user_data_dir is not None:
        return Path(_user_data_dir(APP_NAME))
    xdg = os.getenv("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


def mounts_dir() -> Path:
    """Base directory for agent config bind-mounts.

    Each agent/tool gets a subdirectory (e.g. ``_claude-config/``) that is
    bind-mounted read-write into task containers.  These directories are
    intentionally separated from the credentials store since they are
    container-exposed and subject to potential poisoning.
    """
    return state_root() / "mounts"
