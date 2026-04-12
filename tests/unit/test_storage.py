# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent filesystem storage queries.

We build small directory trees in tmp and verify the module reports their
sizes faithfully — no mocking the filesystem, just real (tiny) files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from terok_agent.roster.loader import MountDef
from terok_agent.storage import (
    _dir_bytes,
    _mount_label,
    get_shared_mounts_storage,
    get_task_storage,
    get_tasks_storage,
)

# ---------------------------------------------------------------------------
# Fixtures: tiny directory trees with known byte counts
# ---------------------------------------------------------------------------


@pytest.fixture
def task_tree(tmp_path: Path) -> Path:
    """A minimal task directory with a workspace and agent config."""
    ws = tmp_path / "workspace-dangerous"
    ws.mkdir()
    (ws / "main.py").write_text("x" * 100)
    (ws / "lib.py").write_text("y" * 50)

    cfg = tmp_path / "agent-config"
    cfg.mkdir()
    (cfg / "instructions.md").write_text("z" * 30)
    return tmp_path


@pytest.fixture
def tasks_root(tmp_path: Path) -> Path:
    """Two task dirs under a common root."""
    for tid in ("aaa111", "bbb222"):
        t = tmp_path / tid
        ws = t / "workspace-dangerous"
        ws.mkdir(parents=True)
        (ws / "code.py").write_text(tid * 10)
        cfg = t / "agent-config"
        cfg.mkdir()
        (cfg / "notes.md").write_text(tid)
    return tmp_path


@pytest.fixture
def mounts_tree(tmp_path: Path) -> Path:
    """Shared mount dirs mimicking the roster layout."""
    for name, content in (("_claude-config", "abc"), ("_codex-config", "de")):
        d = tmp_path / name
        d.mkdir()
        (d / "settings.json").write_text(content)
    return tmp_path


# ---------------------------------------------------------------------------
# _dir_bytes — the measuring tape
# ---------------------------------------------------------------------------


class TestDirBytes:
    """Raw byte counting for directories."""

    def test_counts_file_sizes(self, task_tree: Path):
        ws = task_tree / "workspace-dangerous"
        assert _dir_bytes(ws) == 150  # 100 + 50

    def test_absent_dir_is_zero(self):
        assert _dir_bytes(Path("/nonexistent/path")) == 0

    def test_empty_dir_is_zero(self, tmp_path: Path):
        assert _dir_bytes(tmp_path) == 0

    def test_nested_files_counted(self, tmp_path: Path):
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        (sub / "deep.txt").write_text("deep")
        assert _dir_bytes(tmp_path) == 4


# ---------------------------------------------------------------------------
# _mount_label — translating dir names to human prose
# ---------------------------------------------------------------------------


class TestMountLabel:
    """The roster provides labels; the fallback title-cases the dir name."""

    ROSTER = (
        MountDef(host_dir="_claude-config", container_path="/home/dev/.claude", label="Claude"),
        MountDef(host_dir="_codex-config", container_path="/home/dev/.codex", label="Codex"),
    )

    def test_known_mount(self):
        assert _mount_label("_claude-config", self.ROSTER) == "Claude"

    def test_unknown_mount_fallback(self):
        assert _mount_label("_mystery-tool", self.ROSTER) == "Mystery tool"


# ---------------------------------------------------------------------------
# get_task_storage — one task's footprint
# ---------------------------------------------------------------------------


class TestGetTaskStorage:
    """Measuring a single task directory."""

    def test_measures_workspace_and_config(self, task_tree: Path):
        info = get_task_storage(task_tree)
        assert info.workspace_bytes == 150
        assert info.agent_config_bytes == 30
        assert info.total_bytes == 180

    def test_task_id_from_dirname(self, task_tree: Path):
        info = get_task_storage(task_tree)
        assert info.task_id == task_tree.name

    def test_missing_subdirs_yield_zero(self, tmp_path: Path):
        info = get_task_storage(tmp_path)
        assert info.total_bytes == 0


# ---------------------------------------------------------------------------
# get_tasks_storage — all tasks under a root
# ---------------------------------------------------------------------------


class TestGetTasksStorage:
    """Bulk measurement across all task directories."""

    def test_finds_both_tasks(self, tasks_root: Path):
        results = get_tasks_storage(tasks_root)
        assert len(results) == 2
        assert results[0].task_id == "aaa111"
        assert results[1].task_id == "bbb222"

    def test_each_task_has_bytes(self, tasks_root: Path):
        results = get_tasks_storage(tasks_root)
        assert all(t.total_bytes > 0 for t in results)

    def test_missing_root_returns_empty(self):
        assert get_tasks_storage(Path("/nonexistent")) == []


# ---------------------------------------------------------------------------
# get_shared_mounts_storage — the common ground between containers
# ---------------------------------------------------------------------------


class TestGetSharedMountsStorage:
    """Shared mount measurement with roster label resolution."""

    def test_measures_mount_dirs(self, mounts_tree: Path):
        roster_mounts = (
            MountDef("_claude-config", "/home/dev/.claude", "Claude"),
            MountDef("_codex-config", "/home/dev/.codex", "Codex"),
        )
        with patch("terok_agent.storage.get_roster") as mock_roster:
            mock_roster.return_value.mounts = roster_mounts
            results = get_shared_mounts_storage(mounts_tree)

        assert len(results) == 2
        claude = next(m for m in results if m.name == "_claude-config")
        assert claude.label == "Claude"
        assert claude.bytes == 3  # "abc"

    def test_missing_base_returns_empty(self):
        with patch("terok_agent.storage.get_roster"):
            assert get_shared_mounts_storage(Path("/nonexistent")) == []

    def test_sorted_by_name(self, mounts_tree: Path):
        with patch("terok_agent.storage.get_roster") as mock_roster:
            mock_roster.return_value.mounts = ()
            results = get_shared_mounts_storage(mounts_tree)
        names = [m.name for m in results]
        assert names == sorted(names)
