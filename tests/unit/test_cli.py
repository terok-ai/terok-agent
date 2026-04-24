# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the terok-executor CLI."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from terok_executor.cli import main


def _run_cli(*args: str) -> tuple[str, str, int]:
    """Run CLI in-process, capturing stdout/stderr and exit code."""
    stdout, stderr = StringIO(), StringIO()
    code = 0
    with (
        patch("sys.argv", ["terok-executor", *args]),
        patch("sys.stdout", stdout),
        patch("sys.stderr", stderr),
    ):
        try:
            main()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    return stdout.getvalue(), stderr.getvalue(), code


class TestAgentsCommand:
    """Verify ``terok-executor agents`` output."""

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
        assert "LABEL" in lines[0], "Header missing LABEL column"
        label_col = lines[0].index("LABEL")
        for line in lines[1:]:
            assert len(line) >= label_col, f"Short line: {line!r}"

    def test_unknown_subcommand_exits_nonzero(self) -> None:
        _, _, rc = _run_cli("nonexistent")
        assert rc != 0


class TestSharedDirArgs:
    """Verify --shared-dir and --shared-mount are accepted by the run command."""

    def test_shared_dir_arg_accepted(self) -> None:
        """--shared-dir is recognized by the argument parser."""
        from terok_executor.commands import RUN_COMMAND

        arg_names = [a.name for a in RUN_COMMAND.args]
        assert "--shared-dir" in arg_names

    def test_shared_mount_default(self) -> None:
        """--shared-mount defaults to /shared."""
        from terok_executor.commands import RUN_COMMAND

        for a in RUN_COMMAND.args:
            if a.name == "--shared-mount":
                assert a.default == "/shared"
                break
        else:
            pytest.fail("--shared-mount not found in RUN_COMMAND args")

    def test_handle_run_forwards_shared_dir(self) -> None:
        """_handle_run passes shared_dir to runner, omits shared_mount when no dir."""
        from terok_executor.commands import _handle_run

        with (
            patch("terok_executor.commands._setup_verdict_or_exit"),
            patch("terok_executor.commands._preflight_or_exit", return_value=True),
            patch("terok_executor.container.runner.AgentRunner") as mock_cls,
        ):
            mock_runner = mock_cls.return_value
            mock_runner.run_headless.return_value = "terok-executor-test"
            _handle_run(
                agent="claude",
                repo=".",
                prompt="test",
                shared_dir="/tmp/terok-testing/shared",
                shared_mount="/data",
            )
        call_kwargs = mock_runner.run_headless.call_args
        assert call_kwargs.kwargs["shared_dir"] == Path("/tmp/terok-testing/shared")
        assert call_kwargs.kwargs["shared_mount"] == "/data"

    def test_handle_run_omits_shared_mount_when_no_dir(self) -> None:
        """_handle_run omits shared_mount from common dict when shared_dir is None."""
        from terok_executor.commands import _handle_run

        with (
            patch("terok_executor.commands._setup_verdict_or_exit"),
            patch("terok_executor.commands._preflight_or_exit", return_value=True),
            patch("terok_executor.container.runner.AgentRunner") as mock_cls,
        ):
            mock_runner = mock_cls.return_value
            mock_runner.run_headless.return_value = "terok-executor-test"
            _handle_run(agent="claude", repo=".", prompt="test")
        call_kwargs = mock_runner.run_headless.call_args
        assert call_kwargs.kwargs["shared_dir"] is None
        assert "shared_mount" not in call_kwargs.kwargs


class TestResolveHostGitIdentity:
    """Verify _resolve_host_git_identity reads from host git config."""

    def test_reads_name_and_email(self) -> None:
        from terok_executor.commands import _resolve_host_git_identity

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                type("R", (), {"returncode": 0, "stdout": b"Jane Doe\n"})(),
                type("R", (), {"returncode": 0, "stdout": b"jane@example.com\n"})(),
            ]
            name, email = _resolve_host_git_identity()

        assert name == "Jane Doe"
        assert email == "jane@example.com"

    def test_returns_none_on_missing_config(self) -> None:
        from terok_executor.commands import _resolve_host_git_identity

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                type("R", (), {"returncode": 1, "stdout": b""})(),
                type("R", (), {"returncode": 1, "stdout": b""})(),
            ]
            name, email = _resolve_host_git_identity()

        assert name is None
        assert email is None

    def test_returns_none_when_git_not_found(self) -> None:
        from terok_executor.commands import _resolve_host_git_identity

        with patch("subprocess.run", side_effect=FileNotFoundError):
            name, email = _resolve_host_git_identity()

        assert name is None
        assert email is None
