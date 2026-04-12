# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Filesystem storage queries for agent-owned directories.

Every task leaves a footprint on the host: a workspace, agent config
files, and shared config mounts that survive across containers.  This
module measures those footprints so the orchestrator can report them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import mounts_dir
from .roster import MountDef, get_roster


@dataclass(frozen=True)
class TaskStorageInfo:
    """Disk usage snapshot for a single task's host directories."""

    task_id: str
    workspace_bytes: int
    agent_config_bytes: int

    @property
    def total_bytes(self) -> int:
        """Combined footprint of workspace and agent config."""
        return self.workspace_bytes + self.agent_config_bytes


@dataclass(frozen=True)
class SharedMountStorageInfo:
    """Disk usage for one shared config mount directory."""

    name: str
    label: str
    bytes: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dir_bytes(path: Path) -> int:
    """Recursive disk usage of *path* in bytes, zero if absent."""
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _mount_label(name: str, roster_mounts: tuple[MountDef, ...]) -> str:
    """Look up the human-readable label for a mount dir name, or title-case it."""
    for m in roster_mounts:
        if m.host_dir == name:
            return m.label
    # Fallback: "_claude-config" → "Claude config"
    return name.lstrip("_").replace("-", " ").capitalize()


# ---------------------------------------------------------------------------
# Public queries
# ---------------------------------------------------------------------------


def get_task_storage(task_dir: Path) -> TaskStorageInfo:
    """Measure a single task's disk footprint.

    Expects the standard layout: ``<task_dir>/workspace-dangerous/`` for
    the agent-writable code, ``<task_dir>/agent-config/`` for per-task
    configuration.
    """
    return TaskStorageInfo(
        task_id=task_dir.name,
        workspace_bytes=_dir_bytes(task_dir / "workspace-dangerous"),
        agent_config_bytes=_dir_bytes(task_dir / "agent-config"),
    )


def get_tasks_storage(tasks_root: Path) -> list[TaskStorageInfo]:
    """Measure all tasks under *tasks_root*, sorted by task ID."""
    if not tasks_root.is_dir():
        return []
    return sorted(
        (get_task_storage(d) for d in tasks_root.iterdir() if d.is_dir()),
        key=lambda t: t.task_id,
    )


def get_shared_mounts_storage(
    mounts_base: Path | None = None,
) -> list[SharedMountStorageInfo]:
    """Measure each shared config mount directory.

    Labels come from the agent roster when available, falling back to
    a title-cased version of the directory name.
    """
    base = mounts_base or mounts_dir()
    if not base.is_dir():
        return []

    roster_mounts = get_roster().mounts
    return sorted(
        (
            SharedMountStorageInfo(
                name=d.name,
                label=_mount_label(d.name, roster_mounts),
                bytes=_dir_bytes(d),
            )
            for d in base.iterdir()
            if d.is_dir()
        ),
        key=lambda m: m.name,
    )
