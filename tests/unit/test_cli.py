# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the terok-agent CLI."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from terok_agent.cli import main


def _run_cli(*args: str) -> tuple[str, str, int]:
    """Run CLI in-process, capturing stdout/stderr and exit code."""
    stdout, stderr = StringIO(), StringIO()
    code = 0
    with (
        patch("sys.argv", ["terok-agent", *args]),
        patch("sys.stdout", stdout),
        patch("sys.stderr", stderr),
    ):
        try:
            main()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    return stdout.getvalue(), stderr.getvalue(), code


class TestAgentsCommand:
    """Verify ``terok-agent agents`` output."""

    def test_agents_lists_agents_only(self) -> None:
        out, _, rc = _run_cli("agents")
        assert rc == 0
        assert "claude" in out
        assert "codex" in out
        # Tools should NOT appear without --all — check data lines only
        data_lines = out.strip().split("\n")[1:]
        assert not any(line.startswith("gh ") for line in data_lines)

    def test_agents_all_includes_tools(self) -> None:
        out, _, rc = _run_cli("agents", "--all")
        assert rc == 0
        assert "gh" in out
        assert "glab" in out
        assert "tool" in out

    def test_agents_shows_kind_types(self) -> None:
        out, _, _ = _run_cli("agents", "--all")
        assert "native" in out
        assert "opencode" in out
        assert "bridge" in out
        assert "tool" in out

    def test_agents_has_header(self) -> None:
        out, _, _ = _run_cli("agents")
        assert "NAME" in out
        assert "LABEL" in out
        assert "TYPE" in out

    def test_no_command_shows_help(self) -> None:
        out, err, rc = _run_cli()
        assert rc == 1
        combined = (out + err).lower()
        assert "usage:" in combined
        assert "agents" in combined

    def test_agents_column_alignment(self) -> None:
        """Header and data columns should be aligned."""
        out, _, _ = _run_cli("agents")
        lines = out.strip().split("\n")
        header_name_end = lines[0].index("LABEL")
        for line in lines[1:]:
            assert len(line) >= header_name_end, f"Short line: {line!r}"

    def test_unknown_subcommand_exits_nonzero(self) -> None:
        _, _, rc = _run_cli("nonexistent")
        assert rc != 0
