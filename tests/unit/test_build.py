# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for build resource staging functions."""

from __future__ import annotations

from pathlib import Path

from terok_agent.build import stage_scripts, stage_tmux_config, stage_toad_agents


class TestStageScripts:
    """Verify script staging into build context."""

    def test_stages_init_script(self, tmp_path: Path) -> None:
        dest = tmp_path / "scripts"
        stage_scripts(dest)
        assert (dest / "init-ssh-and-repo.sh").is_file()

    def test_stages_env_scripts(self, tmp_path: Path) -> None:
        dest = tmp_path / "scripts"
        stage_scripts(dest)
        assert (dest / "terok-env.sh").is_file()
        assert (dest / "terok-env-git-identity.sh").is_file()
        assert (dest / "terok-acp-env.sh").is_file()

    def test_stages_acp_wrappers(self, tmp_path: Path) -> None:
        dest = tmp_path / "scripts"
        stage_scripts(dest)
        for wrapper in [
            "terok-claude-acp",
            "terok-codex-acp",
            "terok-copilot-acp",
            "terok-vibe-acp",
            "terok-opencode-acp",
        ]:
            assert (dest / wrapper).is_file(), f"Missing ACP wrapper: {wrapper}"

    def test_stages_opencode_provider(self, tmp_path: Path) -> None:
        dest = tmp_path / "scripts"
        stage_scripts(dest)
        assert (dest / "opencode-provider").is_file()
        assert (dest / "opencode-provider-acp").is_file()
        assert (dest / "opencode-toad").is_file()

    def test_stages_toad_and_hilfe(self, tmp_path: Path) -> None:
        dest = tmp_path / "scripts"
        stage_scripts(dest)
        assert (dest / "toad").is_file()
        assert (dest / "hilfe").is_file()

    def test_stages_auth_and_sync(self, tmp_path: Path) -> None:
        dest = tmp_path / "scripts"
        stage_scripts(dest)
        assert (dest / "setup-codex-auth.sh").is_file()
        assert (dest / "mistral-model-sync.py").is_file()
        assert (dest / "vibe-model-sync.sh").is_file()

    def test_excludes_pycache(self, tmp_path: Path) -> None:
        dest = tmp_path / "scripts"
        stage_scripts(dest)
        assert not list(dest.rglob("__pycache__"))

    def test_excludes_init_py(self, tmp_path: Path) -> None:
        dest = tmp_path / "scripts"
        stage_scripts(dest)
        assert not (dest / "__init__.py").exists()

    def test_replaces_existing_dest(self, tmp_path: Path) -> None:
        dest = tmp_path / "scripts"
        dest.mkdir()
        (dest / "stale-file.txt").write_text("old")
        stage_scripts(dest)
        assert not (dest / "stale-file.txt").exists()
        assert (dest / "init-ssh-and-repo.sh").is_file()


class TestStageToadAgents:
    """Verify toad agent TOML staging."""

    def test_stages_blablador(self, tmp_path: Path) -> None:
        dest = tmp_path / "toad-agents"
        stage_toad_agents(dest)
        assert (dest / "blablador.helmholtz.de.toml").is_file()

    def test_stages_kisski(self, tmp_path: Path) -> None:
        dest = tmp_path / "toad-agents"
        stage_toad_agents(dest)
        assert (dest / "kisski.academiccloud.de.toml").is_file()

    def test_excludes_init_py(self, tmp_path: Path) -> None:
        dest = tmp_path / "toad-agents"
        stage_toad_agents(dest)
        assert not (dest / "__init__.py").exists()


class TestStageTmuxConfig:
    """Verify tmux config staging."""

    def test_stages_container_config(self, tmp_path: Path) -> None:
        dest = tmp_path / "tmux"
        stage_tmux_config(dest)
        assert (dest / "container-tmux.conf").is_file()

    def test_container_config_has_content(self, tmp_path: Path) -> None:
        dest = tmp_path / "tmux"
        stage_tmux_config(dest)
        content = (dest / "container-tmux.conf").read_text()
        assert "status-bg" in content or "prefix" in content

    def test_excludes_init_py(self, tmp_path: Path) -> None:
        dest = tmp_path / "tmux"
        stage_tmux_config(dest)
        assert not (dest / "__init__.py").exists()
