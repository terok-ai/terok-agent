# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for build module: image naming, template rendering, resource staging."""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_agent.build import (
    BuildError,
    ImageSet,
    _base_tag,
    _normalize_base_image,
    build_base_images,
    l0_image_tag,
    l1_image_tag,
    prepare_build_context,
    render_l0,
    render_l1,
    stage_scripts,
    stage_tmux_config,
    stage_toad_agents,
)

# ---------------------------------------------------------------------------
# Image naming
# ---------------------------------------------------------------------------


class TestImageNaming:
    """Verify OCI tag derivation from base image strings."""

    def test_ubuntu_default(self) -> None:
        assert _base_tag("ubuntu:24.04") == "ubuntu-24.04"

    def test_empty_falls_back(self) -> None:
        assert _base_tag("") == "ubuntu-24.04"

    def test_whitespace_falls_back(self) -> None:
        assert _base_tag("   ") == "ubuntu-24.04"

    def test_nvidia_cuda(self) -> None:
        tag = _base_tag("nvidia/cuda:12.4.1-devel-ubuntu24.04")
        assert tag == "nvidia-cuda-12.4.1-devel-ubuntu24.04"

    def test_long_tag_truncated(self) -> None:
        long_name = "a" * 200
        tag = _base_tag(long_name)
        assert len(tag) <= 120

    def test_l0_tag(self) -> None:
        assert l0_image_tag("ubuntu:24.04") == "terok-l0:ubuntu-24.04"

    def test_l1_tag(self) -> None:
        assert l1_image_tag("ubuntu:24.04") == "terok-l1-cli:ubuntu-24.04"

    def test_image_set(self) -> None:
        s = ImageSet(l0="terok-l0:test", l1="terok-l1-cli:test")
        assert s.l0 == "terok-l0:test"
        assert s.l1 == "terok-l1-cli:test"


class TestNormalization:
    """Verify base image normalization."""

    def test_strips_whitespace(self) -> None:
        assert _normalize_base_image("  ubuntu:24.04  ") == "ubuntu:24.04"

    def test_empty_to_default(self) -> None:
        assert _normalize_base_image("") == "ubuntu:24.04"

    def test_none_to_default(self) -> None:
        assert _normalize_base_image(None) == "ubuntu:24.04"

    def test_passthrough(self) -> None:
        assert _normalize_base_image("nvidia/cuda:12.4") == "nvidia/cuda:12.4"


class TestBuildError:
    """Verify BuildError is a proper exception."""

    def test_is_runtime_error(self) -> None:
        assert issubclass(BuildError, RuntimeError)

    def test_message(self) -> None:
        err = BuildError("podman not found")
        assert "podman" in str(err)


class TestBuildDirGuard:
    """Verify build_dir safety checks."""

    def test_rejects_nonempty_dir(self, tmp_path: Path) -> None:
        (tmp_path / "existing-file.txt").write_text("data")
        with pytest.raises(ValueError, match="must be empty"):
            build_base_images(build_dir=tmp_path)

    def test_rejects_file_as_build_dir(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not-a-dir"
        file_path.write_text("data")
        with pytest.raises(ValueError, match="not a directory"):
            build_base_images(build_dir=file_path)


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestTemplateRendering:
    """Verify Jinja2 Dockerfile template rendering."""

    def test_l0_is_valid_dockerfile(self) -> None:
        content = render_l0("ubuntu:24.04")
        assert content.startswith("# syntax=docker")
        assert "FROM" in content

    def test_l0_contains_init_script(self) -> None:
        content = render_l0()
        assert "init-ssh-and-repo.sh" in content

    def test_l0_contains_tmux(self) -> None:
        content = render_l0()
        assert "container-tmux.conf" in content

    def test_l0_contains_base_image_arg(self) -> None:
        # Templates use Dockerfile ARG/FROM ${VAR} — Jinja2 is a pass-through
        # for now. Verify the ARG directive and default value are present.
        content = render_l0("busybox:1.36")
        assert "ARG BASE_IMAGE=" in content
        # The default in the template is ubuntu:24.04; the render arg becomes
        # a --build-arg at podman build time, not a template substitution.
        assert "FROM ${BASE_IMAGE}" in content

    def test_l0_renders_with_custom_base(self) -> None:
        # Should not crash with any base image string
        content = render_l0("nvidia/cuda:12.4.1-devel-ubuntu24.04")
        assert "FROM" in content

    def test_l1_is_valid_dockerfile(self) -> None:
        content = render_l1("terok-l0:test")
        assert content.startswith("# syntax=docker")
        assert "FROM" in content

    def test_l1_contains_agent_installs(self) -> None:
        content = render_l1("terok-l0:test")
        assert "@openai/codex" in content
        assert "claude" in content.lower()

    def test_l1_contains_cache_bust_arg(self) -> None:
        content = render_l1("terok-l0:test")
        assert "ARG AGENT_CACHE_BUST=" in content

    def test_l1_renders_with_different_base(self) -> None:
        content = render_l1("terok-l0:nvidia-cuda-12.4")
        assert "FROM" in content


# ---------------------------------------------------------------------------
# Build context preparation
# ---------------------------------------------------------------------------


class TestPrepareBuildContext:
    """Verify full build context staging."""

    def test_stages_all_resources(self, tmp_path: Path) -> None:
        prepare_build_context(tmp_path)
        assert (tmp_path / "scripts" / "init-ssh-and-repo.sh").is_file()
        assert (tmp_path / "toad-agents" / "blablador.helmholtz.de.toml").is_file()
        assert (tmp_path / "tmux" / "container-tmux.conf").is_file()

    def test_creates_dest_if_missing(self, tmp_path: Path) -> None:
        dest = tmp_path / "nested" / "build"
        prepare_build_context(dest)
        assert dest.is_dir()
        assert (dest / "scripts").is_dir()


# ---------------------------------------------------------------------------
# Resource staging
# ---------------------------------------------------------------------------


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
