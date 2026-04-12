# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgentRunner facade."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from terok_agent.container.runner import AgentRunner, _generate_task_id, _resolve_repo


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

    def test_lazy_roster(self) -> None:
        runner = AgentRunner()
        reg = runner.roster
        assert "claude" in reg.agent_names

    # test_shared_mounts_from_roster, test_base_env_has_essentials,
    # test_base_env_opencode_vars moved to test_env_builder.py

    def test_run_headless_delegates_to_sandbox_run(self, tmp_path: Path) -> None:
        """Verify headless run delegates to sandbox.run() with correct RunSpec."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            cname = runner.run_headless(
                "claude",
                str(tmp_path),
                prompt="Fix the bug",
                follow=False,
            )

        assert cname.startswith("terok-agent-")
        sandbox.run.assert_called_once()
        spec = sandbox.run.call_args[0][0]
        assert spec.image == "terok-l1-cli:test"
        assert any("/home/dev/.terok" in v.container_path for v in spec.volumes)

    def test_restricted_mode_sets_unrestricted_false(self, tmp_path: Path) -> None:
        """Restricted mode passes unrestricted=False in RunSpec."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless(
                "claude", str(tmp_path), prompt="test", follow=False, unrestricted=False
            )

        spec = sandbox.run.call_args[0][0]
        assert spec.unrestricted is False
        # TEROK_UNRESTRICTED should NOT be in env
        assert "TEROK_UNRESTRICTED" not in spec.env

    def test_unrestricted_mode_sets_env(self, tmp_path: Path) -> None:
        """Unrestricted spec includes TEROK_UNRESTRICTED=1."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless(
                "claude", str(tmp_path), prompt="test", follow=False, unrestricted=True
            )

        spec = sandbox.run.call_args[0][0]
        assert spec.unrestricted is True
        assert spec.env.get("TEROK_UNRESTRICTED") == "1"

    def test_gpu_flag_sets_gpu_enabled(self, tmp_path: Path) -> None:
        """GPU flag propagates to RunSpec.gpu_enabled."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless("claude", str(tmp_path), prompt="test", follow=False, gpu=True)

        spec = sandbox.run.call_args[0][0]
        assert spec.gpu_enabled is True

    def test_run_interactive_command(self, tmp_path: Path) -> None:
        """Interactive mode includes init-ssh-and-repo.sh in command."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_interactive("claude", str(tmp_path))

        spec = sandbox.run.call_args[0][0]
        cmd_str = " ".join(spec.command)
        assert "init-ssh-and-repo.sh" in cmd_str
        assert "__CLI_READY__" in cmd_str

    def test_run_web_publishes_port(self, tmp_path: Path) -> None:
        """Web mode includes port publishing in extra_args."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_web(str(tmp_path), port=9999)

        spec = sandbox.run.call_args[0][0]
        assert "-p" in spec.extra_args
        idx = list(spec.extra_args).index("-p")
        assert "9999:8080" in spec.extra_args[idx + 1]

    def test_run_web_auto_allocates_port(self, tmp_path: Path) -> None:
        """Web mode auto-allocates a port when none given."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch("terok_sandbox.find_free_port", return_value=12345),
        ):
            runner.run_web(str(tmp_path))  # no port= arg

        spec = sandbox.run.call_args[0][0]
        assert "-p" in spec.extra_args
        idx = list(spec.extra_args).index("-p")
        assert "12345:8080" in spec.extra_args[idx + 1]

    def test_hooks_passed_to_sandbox_run(self, tmp_path: Path) -> None:
        """Lifecycle hooks are forwarded to sandbox.run()."""
        from terok_sandbox import LifecycleHooks

        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)
        hooks = LifecycleHooks(pre_start=lambda: None)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless("claude", str(tmp_path), prompt="test", follow=False, hooks=hooks)

        assert sandbox.run.call_args.kwargs["hooks"] is hooks

    def test_lazy_sandbox_init(self) -> None:
        runner = AgentRunner()
        # Access sandbox property — should create a default Sandbox
        with patch("terok_sandbox.Sandbox") as mock_cls:
            mock_cls.return_value = _mock_sandbox()
            s = runner.sandbox
            assert s is not None

    def test_gpu_config_error_becomes_build_error(self, tmp_path: Path) -> None:
        """GpuConfigError from sandbox.run() is wrapped as BuildError."""
        from terok_sandbox import GpuConfigError

        from terok_agent.container.build import BuildError

        sandbox = _mock_sandbox()
        sandbox.run.side_effect = GpuConfigError("CDI broken")
        runner = AgentRunner(sandbox=sandbox)

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            pytest.raises(BuildError, match="CDI broken"),
        ):
            runner.run_headless("claude", str(tmp_path), prompt="test", follow=False)

    def test_shared_mount_must_be_absolute(self, tmp_path: Path) -> None:
        """Relative shared_mount is rejected with SystemExit."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)
        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            pytest.raises(SystemExit, match="absolute path"),
        ):
            runner.run_headless(
                "claude",
                str(tmp_path),
                prompt="test",
                follow=False,
                shared_dir=tmp_path / "ipc",
                shared_mount="relative/path",
            )

    def test_shared_mount_rejects_colon(self, tmp_path: Path) -> None:
        """shared_mount containing ':' is rejected (volume spec injection)."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)
        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            pytest.raises(SystemExit, match="':'"),
        ):
            runner.run_headless(
                "claude",
                str(tmp_path),
                prompt="test",
                follow=False,
                shared_dir=tmp_path / "ipc",
                shared_mount="/data:ro",
            )

    def test_shared_dir_file_rejected(self, tmp_path: Path) -> None:
        """shared_dir that exists as a file is rejected."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)
        existing_file = tmp_path / "not-a-dir"
        existing_file.touch()
        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            pytest.raises(SystemExit, match="exists as a file"),
        ):
            runner.run_headless(
                "claude",
                str(tmp_path),
                prompt="test",
                follow=False,
                shared_dir=existing_file,
            )

    def test_shared_dir_in_container_env(self, tmp_path: Path) -> None:
        """shared_dir kwarg produces TEROK_SHARED_DIR and a volume mount."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        shared = tmp_path / "ipc"
        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless(
                "claude",
                str(tmp_path),
                prompt="test",
                follow=False,
                shared_dir=shared,
                shared_mount="/data",
            )

        spec = sandbox.run.call_args[0][0]
        assert spec.env["TEROK_SHARED_DIR"] == "/data"
        assert any(v.host_path == shared and v.container_path == "/data" for v in spec.volumes)


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

        # Verify the gate URL ended up in the RunSpec env as CODE_REPO
        spec = sandbox.run.call_args[0][0]
        assert spec.env.get("CODE_REPO") == "http://tok@host:9418/repo"

    def test_gate_false_skips_gate(self) -> None:
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch.object(runner, "_setup_gate") as mock_gate,
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
        spec = sandbox.run.call_args[0][0]
        assert spec.env.get("CODE_REPO") == "git@github.com:user/repo.git"


class TestCredentialProxyEnv:
    """Verify credential proxy integration (now via env_builder).

    Detailed token injection tests are in test_env_builder.py.
    These tests verify the runner delegates correctly.
    """

    def test_proxy_not_running_no_tokens_in_run(self, tmp_path: Path) -> None:
        """When proxy is not running, headless run has no TEROK_PROXY_PORT."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch("terok_sandbox.is_proxy_socket_active", return_value=False),
            patch("terok_sandbox.is_proxy_running", return_value=False),
        ):
            runner.run_headless("claude", str(tmp_path), prompt="test", follow=False)

        spec = sandbox.run.call_args[0][0]
        assert "TEROK_PROXY_PORT" not in spec.env


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
