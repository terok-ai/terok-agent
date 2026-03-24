# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgentRunner facade."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from terok_agent.runner import AgentRunner, _generate_task_id, _resolve_repo


class TestResolveRepo:
    """Verify repo argument classification."""

    def test_local_dir(self, tmp_path: Path) -> None:
        code_repo, local_path = _resolve_repo(str(tmp_path))
        assert code_repo is None
        assert local_path == tmp_path

    def test_git_url_ssh(self) -> None:
        code_repo, local_path = _resolve_repo("git@github.com:user/repo.git")
        assert code_repo == "git@github.com:user/repo.git"
        assert local_path is None

    def test_git_url_https(self) -> None:
        code_repo, local_path = _resolve_repo("https://github.com/user/repo.git")
        assert code_repo == "https://github.com/user/repo.git"
        assert local_path is None

    def test_nonexistent_local_path_exits(self) -> None:
        with pytest.raises(SystemExit, match="not found"):
            _resolve_repo("./nonexistent-dir-xyz")


class TestTaskId:
    """Verify task ID generation."""

    def test_length(self) -> None:
        assert len(_generate_task_id()) == 12

    def test_unique(self) -> None:
        ids = {_generate_task_id() for _ in range(100)}
        assert len(ids) == 100


class TestAgentRunner:
    """Verify AgentRunner construction and delegation."""

    def test_default_construction(self) -> None:
        runner = AgentRunner()
        assert runner._base_image == "ubuntu:24.04"

    def test_custom_base_image(self) -> None:
        runner = AgentRunner(base_image="nvidia/cuda:12.4")
        assert runner._base_image == "nvidia/cuda:12.4"

    def test_lazy_registry(self) -> None:
        runner = AgentRunner()
        reg = runner.registry
        assert "claude" in reg.agent_names

    def test_shared_mounts_from_registry(self, tmp_path: Path) -> None:
        runner = AgentRunner()
        mounts = runner._shared_mounts(tmp_path)
        # Should have mounts for agents with auth (claude, codex, vibe, gh, glab, etc.)
        mount_str = " ".join(mounts)
        assert "_claude-config" in mount_str
        assert "/home/dev/.claude" in mount_str

    def test_base_env_has_essentials(self) -> None:
        runner = AgentRunner()
        env = runner._base_env("task123", "claude")
        assert env["TASK_ID"] == "task123"
        assert env["REPO_ROOT"] == "/workspace"
        assert env["GIT_AUTHOR_NAME"] == "Claude"

    def test_base_env_opencode_vars(self) -> None:
        runner = AgentRunner()
        env = runner._base_env("task123", "blablador")
        # Should include OpenCode provider env vars
        assert any(k.startswith("TEROK_OC_") for k in env)

    def test_run_headless_assembles_command(self, tmp_path: Path) -> None:
        """Verify headless run composes the right pieces (mocked podman)."""
        runner = AgentRunner(sandbox=_mock_sandbox())

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch("subprocess.run") as mock_run,
        ):
            cname = runner.run_headless(
                "claude",
                str(tmp_path),
                prompt="Fix the bug",
                follow=False,
            )

        assert cname.startswith("terok-agent-")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "podman"
        assert "run" in cmd
        assert any("/home/dev/.terok" in arg for arg in cmd)

    def test_run_interactive_command(self, tmp_path: Path) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch("subprocess.run") as mock_run,
        ):
            runner.run_interactive("claude", str(tmp_path))

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "init-ssh-and-repo.sh" in cmd_str
        assert "__CLI_READY__" in cmd_str

    def test_run_web_publishes_port(self, tmp_path: Path) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch("subprocess.run") as mock_run,
        ):
            runner.run_web(str(tmp_path), port=9999)

        cmd = mock_run.call_args[0][0]
        assert "-p" in cmd
        idx = cmd.index("-p")
        assert "9999:8080" in cmd[idx + 1]

    def test_run_web_auto_allocates_port(self, tmp_path: Path) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch("terok_sandbox.find_free_port", return_value=12345),
            patch("subprocess.run") as mock_run,
        ):
            runner.run_web(str(tmp_path))  # no port= arg

        cmd = mock_run.call_args[0][0]
        assert "-p" in cmd
        idx = cmd.index("-p")
        assert "12345:8080" in cmd[idx + 1]

    def test_launch_redacts_code_repo(self, tmp_path: Path) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch("subprocess.run") as mock_run,
            patch("builtins.print") as mock_print,
        ):
            runner.run_headless(
                "claude",
                "git@github.com:user/repo.git",
                prompt="test",
                gate=False,
                follow=False,
            )

        # The printed command should have CODE_REPO redacted
        printed = mock_print.call_args[0][1]
        assert "CODE_REPO=<redacted>" in printed
        # But the actual podman command should have the real value
        cmd = mock_run.call_args[0][0]
        assert any("CODE_REPO=git@github.com:user/repo.git" in arg for arg in cmd)

    def test_lazy_sandbox_init(self) -> None:
        runner = AgentRunner()
        # Access sandbox property — should create a default Sandbox
        with patch("terok_sandbox.Sandbox") as mock_cls:
            mock_cls.return_value = _mock_sandbox()
            s = runner.sandbox
            assert s is not None


class TestGateIntegration:
    """Verify gate wiring in AgentRunner."""

    def test_setup_gate_calls_sandbox(self) -> None:
        sandbox = _mock_sandbox()
        sandbox.ensure_gate.return_value = None
        sandbox.create_token.return_value = "tok123"
        sandbox.gate_url.return_value = "http://tok123@host:9418/repo"
        runner = AgentRunner(sandbox=sandbox)

        with patch("terok_sandbox.GitGate") as mock_gate_cls:
            mock_gate = Mock()
            mock_gate_cls.return_value = mock_gate
            url = runner._setup_gate("git@github.com:user/repo.git", "task1")

        mock_gate.sync.assert_called_once()
        sandbox.ensure_gate.assert_called_once()
        sandbox.create_token.assert_called_once()
        assert url == "http://tok123@host:9418/repo"

    def test_gate_true_with_git_url_uses_gate(self) -> None:
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch.object(
                runner, "_setup_gate", return_value="http://tok@host:9418/repo"
            ) as mock_gate,
            patch("subprocess.run") as mock_run,
        ):
            runner.run_headless(
                "claude",
                "git@github.com:user/repo.git",
                prompt="test",
                gate=True,
                follow=False,
            )

        # Verify _setup_gate was called with the repo URL
        mock_gate.assert_called_once()
        assert mock_gate.call_args[0][0] == "git@github.com:user/repo.git"

        # Verify the gate URL ended up in the podman command as CODE_REPO
        cmd = mock_run.call_args[0][0]
        assert any("CODE_REPO=http://tok@host:9418/repo" in arg for arg in cmd)

    def test_gate_false_skips_gate(self) -> None:
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch.object(runner, "_setup_gate") as mock_gate,
            patch("subprocess.run") as mock_run,
        ):
            runner.run_headless(
                "claude",
                "git@github.com:user/repo.git",
                prompt="test",
                gate=False,
                follow=False,
            )

        mock_gate.assert_not_called()
        # CODE_REPO should be the raw URL
        cmd = mock_run.call_args[0][0]
        assert any("git@github.com:user/repo.git" in arg for arg in cmd)


class TestCommandRegistry:
    """Verify the command registry is well-formed."""

    def test_all_commands_have_handlers(self) -> None:
        from terok_agent.commands import COMMANDS

        for cmd in COMMANDS:
            assert cmd.handler is not None, f"Command '{cmd.name}' has no handler"

    def test_run_command_has_agent_arg(self) -> None:
        from terok_agent.commands import RUN_COMMAND

        arg_names = [a.name for a in RUN_COMMAND.args]
        assert "agent" in arg_names

    def test_commands_exported_from_package(self) -> None:
        from terok_agent import AGENT_COMMANDS

        assert len(AGENT_COMMANDS) >= 5  # run, auth, agents, build, ls, stop


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_sandbox() -> Mock:
    """Create a mock Sandbox with the right interface."""
    from terok_sandbox import SandboxConfig

    sandbox = Mock()
    sandbox.config = SandboxConfig()
    sandbox.pre_start_args.return_value = []
    sandbox.stream_logs.return_value = True
    sandbox.wait_for_exit.return_value = 0
    return sandbox
