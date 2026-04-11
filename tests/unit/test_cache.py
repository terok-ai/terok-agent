# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for clone-cache workspace seeding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from terok_agent.container.cache import (
    _copy_tree,
    _resolve_cache_dir,
    _wipe_workspace_contents,
    seed_workspace_from_clone_cache,
)


class TestResolveCacheDir:
    """Verify cache directory resolution."""

    def test_returns_path_when_exists(self, tmp_path: Path) -> None:
        cache_base = tmp_path / "clone-cache"
        proj_cache = cache_base / "myproj"
        proj_cache.mkdir(parents=True)

        cfg = SimpleNamespace(clone_cache_base_path=cache_base)
        assert _resolve_cache_dir("myproj", cfg) == proj_cache

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        cfg = SimpleNamespace(clone_cache_base_path=tmp_path / "clone-cache")
        assert _resolve_cache_dir("no-such-scope", cfg) is None


class TestCopyTree:
    """Verify copy with reflink fallback."""

    def test_uses_cp_reflink(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()

        with patch("terok_agent.container.cache.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            _copy_tree(src, dst)

        cmd = mock_run.call_args[0][0]
        assert "--reflink=auto" in cmd

    def test_falls_back_to_shutil(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "file.txt").write_text("hello")

        with patch(
            "terok_agent.container.cache.subprocess.run",
            side_effect=FileNotFoundError("cp"),
        ):
            _copy_tree(src, dst)

        assert (dst / "file.txt").read_text() == "hello"


class TestWipeWorkspaceContents:
    """Verify workspace cleanup removes all contents."""

    def test_removes_all_contents(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main")
        (tmp_path / "README.md").write_text("hello")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("pass")
        _wipe_workspace_contents(tmp_path)
        assert not any(tmp_path.iterdir())

    def test_noop_when_empty(self, tmp_path: Path) -> None:
        _wipe_workspace_contents(tmp_path)  # should not raise


class TestSeedWorkspaceFromCloneCache:
    """Verify the public seed entry point."""

    def test_seeds_from_cache(self, tmp_path: Path) -> None:
        cache_base = tmp_path / "clone-cache"
        cache_dir = cache_base / "proj"
        cache_dir.mkdir(parents=True)
        (cache_dir / ".git").mkdir()
        (cache_dir / ".git" / "HEAD").write_text("ref: refs/heads/main")
        (cache_dir / "README.md").write_text("# Hello")

        ws = tmp_path / "workspace"
        ws.mkdir()

        cfg = SimpleNamespace(clone_cache_base_path=cache_base)

        with (
            patch("terok_agent.container.cache._copy_tree") as mock_copy,
            patch("terok_agent.container.cache._rewrite_origin") as mock_rewrite,
        ):
            # Simulate _copy_tree creating .git in workspace
            def fake_copy(src, dst):
                (dst / ".git").mkdir(parents=True, exist_ok=True)

            mock_copy.side_effect = fake_copy
            result = seed_workspace_from_clone_cache(
                ws, "proj", origin_url="git@github.com:org/repo.git", cfg=cfg
            )

        assert result is True
        mock_copy.assert_called_once()
        mock_rewrite.assert_called_once_with(ws, "git@github.com:org/repo.git")

    def test_skips_when_git_exists(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / ".git").mkdir()

        cfg = SimpleNamespace(clone_cache_base_path=tmp_path / "clone-cache")
        assert seed_workspace_from_clone_cache(ws, "proj", cfg=cfg) is False

    def test_skips_when_no_cache(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()

        cfg = SimpleNamespace(clone_cache_base_path=tmp_path / "clone-cache")
        assert seed_workspace_from_clone_cache(ws, "proj", cfg=cfg) is False

    def test_cleans_up_on_failure(self, tmp_path: Path) -> None:
        cache_base = tmp_path / "clone-cache"
        cache_dir = cache_base / "proj"
        cache_dir.mkdir(parents=True)
        (cache_dir / ".git").mkdir()

        ws = tmp_path / "workspace"
        ws.mkdir()

        cfg = SimpleNamespace(clone_cache_base_path=cache_base)

        def partial_copy(src, dst):
            """Simulate a partial copy that leaves files before failing."""
            (dst / "README.md").write_text("partial")
            raise OSError("disk full")

        with patch("terok_agent.container.cache._copy_tree", side_effect=partial_copy):
            result = seed_workspace_from_clone_cache(ws, "proj", cfg=cfg)

        assert result is False
        # Entire workspace must be empty for fallback clone
        assert not any(ws.iterdir())
