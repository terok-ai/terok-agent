# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the terok-agent CLI."""

from __future__ import annotations

import subprocess
import sys


class TestAgentsCommand:
    """Verify ``terok-agent agents`` output."""

    def test_agents_lists_agents_only(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "terok_agent.cli", "agents"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "claude" in result.stdout
        assert "codex" in result.stdout
        # Tools should NOT appear without --all
        assert "gh" not in result.stdout.split("\n")[1:]  # skip header

    def test_agents_all_includes_tools(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "terok_agent.cli", "agents", "--all"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "gh" in result.stdout
        assert "glab" in result.stdout
        assert "tool" in result.stdout

    def test_agents_shows_kind_types(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "terok_agent.cli", "agents", "--all"],
            capture_output=True,
            text=True,
        )
        assert "native" in result.stdout
        assert "opencode" in result.stdout
        assert "bridge" in result.stdout
        assert "tool" in result.stdout

    def test_agents_has_header(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "terok_agent.cli", "agents"],
            capture_output=True,
            text=True,
        )
        assert "NAME" in result.stdout
        assert "LABEL" in result.stdout

    def test_no_command_shows_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "terok_agent.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "usage:" in result.stderr.lower() or "usage:" in result.stdout.lower()
