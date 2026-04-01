# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for terok_agent.paths — umbrella directory resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_agent import paths

from ..constants import MOCK_BASE


class TestStateRoot:
    """Verify ``state_root()`` resolution across all priority tiers."""

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """TEROK_AGENT_STATE_DIR takes first priority."""
        monkeypatch.setenv("TEROK_AGENT_STATE_DIR", str(tmp_path))
        assert paths.state_root() == tmp_path

    def test_env_tilde_expansion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tilde in TEROK_AGENT_STATE_DIR is expanded."""
        monkeypatch.setenv("TEROK_AGENT_STATE_DIR", "~/agent-state")
        result = paths.state_root()
        assert "~" not in str(result)
        assert result == Path.home() / "agent-state"

    def test_root_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Root user gets /var/lib/terok/agent."""
        monkeypatch.delenv("TEROK_AGENT_STATE_DIR", raising=False)
        monkeypatch.setattr(paths, "_is_root", lambda: True)
        assert paths.state_root() == Path("/var/lib/terok/agent")

    def test_platformdirs_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-root with platformdirs uses umbrella/subdir."""
        monkeypatch.delenv("TEROK_AGENT_STATE_DIR", raising=False)
        monkeypatch.setattr(paths, "_is_root", lambda: False)
        monkeypatch.setattr(paths, "_user_data_dir", lambda name: f"{MOCK_BASE}/data/{name}")
        assert paths.state_root() == MOCK_BASE / "data" / "terok" / "agent"

    def test_xdg_data_home_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without platformdirs, XDG_DATA_HOME is honored."""
        monkeypatch.delenv("TEROK_AGENT_STATE_DIR", raising=False)
        monkeypatch.setattr(paths, "_is_root", lambda: False)
        monkeypatch.setattr(paths, "_user_data_dir", None)
        monkeypatch.setenv("XDG_DATA_HOME", str(MOCK_BASE / "xdg-data"))
        assert paths.state_root() == MOCK_BASE / "xdg-data" / "terok" / "agent"

    def test_bare_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Last resort: ~/.local/share/terok/agent."""
        monkeypatch.delenv("TEROK_AGENT_STATE_DIR", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(paths, "_is_root", lambda: False)
        monkeypatch.setattr(paths, "_user_data_dir", None)
        assert paths.state_root() == Path.home() / ".local" / "share" / "terok" / "agent"


class TestMountsDir:
    """Verify ``mounts_dir()`` is a child of ``state_root()``."""

    def test_is_subdir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """mounts_dir() returns state_root() / 'mounts'."""
        monkeypatch.setenv("TEROK_AGENT_STATE_DIR", str(tmp_path))
        assert paths.mounts_dir() == tmp_path / "mounts"


class TestUmbrellaConstants:
    """Verify umbrella namespace constants."""

    def test_umbrella_is_terok(self) -> None:
        """_UMBRELLA is 'terok'."""
        assert paths._UMBRELLA == "terok"

    def test_subdir_is_agent(self) -> None:
        """_SUBDIR is 'agent'."""
        assert paths._SUBDIR == "agent"
