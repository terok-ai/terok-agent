# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for build module: image naming, template rendering, resource staging."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from terok_executor.container.build import (
    BuildError,
    ImageSet,
    _base_tag,
    _normalize_base_image,
    _split_image_ref,
    build_base_images,
    build_sidecar_image,
    detect_family,
    l0_image_tag,
    l1_image_tag,
    l1_sidecar_image_tag,
    prepare_build_context,
    render_l0,
    render_l1,
    render_l1_sidecar,
    stage_help_fragments,
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

    def test_l1_tag_with_agents(self) -> None:
        assert (
            l1_image_tag("ubuntu:24.04", ("claude", "codex"))
            == "terok-l1-cli:ubuntu-24.04-claude-codex"
        )

    def test_l1_tag_agents_sorted(self) -> None:
        # Suffix is canonicalised regardless of input order.
        assert l1_image_tag("ubuntu:24.04", ("codex", "claude")) == l1_image_tag(
            "ubuntu:24.04", ("claude", "codex")
        )

    def test_l1_tag_empty_selection(self) -> None:
        # Edge case: empty selection still produces a distinct, addressable tag.
        assert l1_image_tag("ubuntu:24.04", ()) == "terok-l1-cli:ubuntu-24.04-empty"

    def test_l1_tag_fits_oci_limit_for_worst_realistic_base(self) -> None:
        # Longest base we currently presets (podman+all-agents) must fit.
        tag = l1_image_tag(
            "quay.io/podman/stable:latest",
            (
                "blablador",
                "claude",
                "codex",
                "copilot",
                "gh",
                "glab",
                "kisski",
                "opencode",
                "sonar",
                "toad",
                "vibe",
            ),
        )
        _, _, tag_part = tag.partition(":")
        assert len(tag_part) <= 128  # OCI spec
        assert "blablador" in tag  # readable form still won

    def test_l1_tag_digests_when_combined_overflows(self) -> None:
        # Synthesise a base that by itself is near the cap so the readable
        # agent suffix would push past the limit; digest takes over.
        long_base = "registry.example.internal/" + "x" * 100 + ":latest"
        tag = l1_image_tag(long_base, ("claude", "codex", "gh"))
        _, _, tag_part = tag.partition(":")
        assert len(tag_part) <= 128
        # Agent suffix replaced by a 12-char hex digest — no readable names.
        assert "claude" not in tag and "codex" not in tag

    def test_l1_tag_digest_is_stable_and_selection_sensitive(self) -> None:
        long_base = "registry.example.internal/" + "x" * 100 + ":latest"
        a = l1_image_tag(long_base, ("claude", "codex"))
        b = l1_image_tag(long_base, ("codex", "claude"))  # same set, reordered
        c = l1_image_tag(long_base, ("claude", "gh"))  # different set
        assert a == b
        assert a != c

    def test_image_set(self) -> None:
        s = ImageSet(l0="terok-l0:test", l1="terok-l1-cli:test")
        assert s.l0 == "terok-l0:test"
        assert s.l1 == "terok-l1-cli:test"

    def test_l1_sidecar_tag(self) -> None:
        assert l1_sidecar_image_tag("ubuntu:24.04") == "terok-l1-sidecar:ubuntu-24.04"

    def test_l1_sidecar_tag_custom(self) -> None:
        tag = l1_sidecar_image_tag("nvidia/cuda:12.4")
        assert tag == "terok-l1-sidecar:nvidia-cuda-12.4"

    def test_image_set_with_sidecar(self) -> None:
        s = ImageSet(l0="terok-l0:test", l1="terok-l1-cli:test", l1_sidecar="terok-l1-sidecar:test")
        assert s.l1_sidecar == "terok-l1-sidecar:test"

    def test_image_set_sidecar_defaults_none(self) -> None:
        s = ImageSet(l0="terok-l0:test", l1="terok-l1-cli:test")
        assert s.l1_sidecar is None


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
# build_base_images (mocked podman)
# ---------------------------------------------------------------------------


class TestBuildProjectImage:
    """Verify the generic podman-build primitive used by all three factories.

    ``build_project_image`` is the single ``podman build`` invocation site
    that the agent-aware factories (L0+L1, sidecar) and terok's project/L2
    builds share.  Exercises flag assembly and BuildError translation.
    """

    def test_minimal_invocation(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from terok_executor.container.build import build_project_image

        dockerfile = tmp_path / "Dockerfile"
        dockerfile.touch()
        with patch("subprocess.run") as run_mock:
            build_project_image(
                dockerfile=dockerfile,
                context_dir=tmp_path,
                target_tag="proj:tag",
            )
        cmd = run_mock.call_args[0][0]
        assert cmd[:3] == ["podman", "build", "-f"]
        assert cmd[-1] == str(tmp_path)
        assert "-t" in cmd and "proj:tag" in cmd

    def test_build_args_and_labels(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from terok_executor.container.build import build_project_image

        dockerfile = tmp_path / "Dockerfile"
        dockerfile.touch()
        with patch("subprocess.run") as run_mock:
            build_project_image(
                dockerfile=dockerfile,
                context_dir=tmp_path,
                target_tag="proj:tag",
                build_args={"BASE_IMAGE": "ubuntu:24.04", "CACHE": "123"},
                labels={"terok.build_context_hash": "abc"},
            )
        cmd = run_mock.call_args[0][0]
        assert "--build-arg" in cmd
        assert "BASE_IMAGE=ubuntu:24.04" in cmd
        assert "CACHE=123" in cmd
        assert "--label" in cmd
        assert "terok.build_context_hash=abc" in cmd

    def test_extra_tags_applied_once(self, tmp_path: Path) -> None:
        """Multiple tags on the same build become multiple ``-t`` flags."""
        from unittest.mock import patch

        from terok_executor.container.build import build_project_image

        dockerfile = tmp_path / "Dockerfile"
        dockerfile.touch()
        with patch("subprocess.run") as run_mock:
            build_project_image(
                dockerfile=dockerfile,
                context_dir=tmp_path,
                target_tag="main:1",
                extra_tags=("alias:1", "other:1"),
            )
        cmd = run_mock.call_args[0][0]
        t_indices = [i for i, a in enumerate(cmd) if a == "-t"]
        assert len(t_indices) == 3
        assert cmd[t_indices[0] + 1] == "main:1"
        assert cmd[t_indices[1] + 1] == "alias:1"
        assert cmd[t_indices[2] + 1] == "other:1"

    def test_no_cache_and_pull_always(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from terok_executor.container.build import build_project_image

        dockerfile = tmp_path / "Dockerfile"
        dockerfile.touch()
        with patch("subprocess.run") as run_mock:
            build_project_image(
                dockerfile=dockerfile,
                context_dir=tmp_path,
                target_tag="proj:tag",
                no_cache=True,
                pull_always=True,
            )
        cmd = run_mock.call_args[0][0]
        assert "--no-cache" in cmd
        assert "--pull=always" in cmd

    def test_missing_podman_becomes_build_error(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from terok_executor.container.build import build_project_image

        dockerfile = tmp_path / "Dockerfile"
        dockerfile.touch()
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            pytest.raises(BuildError, match="podman not found"),
        ):
            build_project_image(dockerfile=dockerfile, context_dir=tmp_path, target_tag="x:1")

    def test_failed_build_becomes_build_error(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from terok_executor.container.build import build_project_image

        dockerfile = tmp_path / "Dockerfile"
        dockerfile.touch()
        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "podman"),
            ),
            pytest.raises(BuildError, match="Image build failed"),
        ):
            build_project_image(dockerfile=dockerfile, context_dir=tmp_path, target_tag="x:1")


class TestBuildBaseImages:
    """Verify build_base_images orchestration with mocked podman."""

    def test_skips_when_images_exist(self) -> None:
        from unittest.mock import patch

        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=True),
        ):
            result = build_base_images()
        assert result.l0.startswith("terok-l0:")
        assert result.l1.startswith("terok-l1-cli:")

    def test_builds_when_images_missing(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=False),
            patch("subprocess.run") as mock_run,
        ):
            result = build_base_images(build_dir=build_dir)

        # Two podman build calls (L0 + L1)
        assert mock_run.call_count == 2
        l0_cmd = mock_run.call_args_list[0][0][0]
        l1_cmd = mock_run.call_args_list[1][0][0]
        assert l0_cmd[0] == "podman"
        assert "-t" in l0_cmd
        assert result.l0 in l0_cmd
        assert l1_cmd[0] == "podman"
        assert result.l1 in l1_cmd

    def test_rebuild_forces_build(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=True),
            patch("subprocess.run") as mock_run,
        ):
            build_base_images(rebuild=True, build_dir=build_dir)

        # Should build even though images exist
        assert mock_run.call_count == 2

    def test_full_rebuild_passes_no_cache(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=False),
            patch("subprocess.run") as mock_run,
        ):
            build_base_images(full_rebuild=True, build_dir=build_dir)

        l0_cmd = mock_run.call_args_list[0][0][0]
        assert "--no-cache" in l0_cmd
        assert "--pull=always" in l0_cmd

        # L1 gets --no-cache but NOT --pull=always (it builds FROM L0, not a remote)
        l1_cmd = mock_run.call_args_list[1][0][0]
        assert "--no-cache" in l1_cmd
        assert "--pull=always" not in l1_cmd

    def test_build_failure_raises_build_error(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=False),
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "podman"),
            ),
            pytest.raises(BuildError, match="build failed"),
        ):
            build_base_images(build_dir=build_dir)

    def test_os_error_raises_build_error(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=False),
            patch(
                "terok_executor.container.build.prepare_build_context",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(BuildError, match="disk full"),
        ):
            build_base_images(build_dir=build_dir)

    def test_missing_podman_raises_build_error(self) -> None:
        from unittest.mock import patch

        with (
            patch("shutil.which", return_value=None),
            pytest.raises(BuildError, match="podman not found"),
        ):
            build_base_images()

    def test_custom_base_image(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=False),
            patch("subprocess.run") as mock_run,
        ):
            result = build_base_images("nvidia/cuda:12.4", build_dir=build_dir)

        assert "nvidia-cuda-12.4" in result.l0
        l0_cmd = mock_run.call_args_list[0][0][0]
        assert any("nvidia/cuda:12.4" in arg for arg in l0_cmd)

    def test_build_context_has_dockerfiles(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=False),
            patch("subprocess.run"),
        ):
            build_base_images(build_dir=build_dir)

        assert (build_dir / "L0.Dockerfile").is_file()
        assert (build_dir / "L1.cli.Dockerfile").is_file()
        assert (build_dir / "scripts" / "init-ssh-and-repo.sh").is_file()


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
        # Templates rely on Dockerfile ARG/FROM ${VAR} for the BASE_IMAGE
        # passthrough; Jinja2 only switches the package-manager branch.
        # Use family override to keep the test independent of the prefix
        # allowlist.
        content = render_l0("busybox:1.36", family="deb")
        assert "ARG BASE_IMAGE=busybox:1.36" in content
        assert "FROM ${BASE_IMAGE}" in content

    def test_l0_renders_with_custom_base(self) -> None:
        # Should not crash with any base image string
        content = render_l0("nvidia/cuda:12.4.1-devel-ubuntu24.04")
        assert "FROM" in content

    def test_l1_is_valid_dockerfile(self) -> None:
        content = render_l1("terok-l0:test", family="deb")
        assert content.startswith("# syntax=docker")
        assert "FROM" in content

    def test_l1_contains_agent_installs(self) -> None:
        content = render_l1("terok-l0:test", family="deb")
        assert "@openai/codex" in content
        assert "claude" in content.lower()

    def test_l1_contains_cache_bust_arg(self) -> None:
        content = render_l1("terok-l0:test", family="deb")
        assert "ARG AGENT_CACHE_BUST=" in content

    def test_l1_renders_with_different_base(self) -> None:
        content = render_l1("terok-l0:nvidia-cuda-12.4", family="deb")
        assert "FROM" in content

    def test_l1_label_lists_selection(self) -> None:
        content = render_l1("terok-l0:test", family="deb", agents=("claude", "codex"))
        assert 'LABEL ai.terok.agents="claude,codex"' in content

    def test_l1_omits_unselected_agents(self) -> None:
        # Selecting only claude should not pull in vibe's pipx install.
        content = render_l1("terok-l0:test", family="deb", agents=("claude",))
        assert "claude.ai/install" in content
        assert "pipx install mistral-vibe" not in content

    def test_l1_resolves_transitive_deps(self) -> None:
        # blablador depends_on opencode → opencode install must appear.
        content = render_l1("terok-l0:test", family="deb", agents=("blablador",))
        assert "opencode.ai/install" in content
        assert 'LABEL ai.terok.agents="blablador,opencode"' in content

    def test_l1_banner_targets_family_specific_bashrc(self) -> None:
        # Debian patches bash to source /etc/bash.bashrc; Fedora's bash never
        # reads that path and instead sources /etc/bashrc.  Wiring the help
        # banner into the wrong file silently drops it on the other family.
        deb = render_l1("terok-l0:test", family="deb")
        rpm = render_l1("terok-l0:test", family="rpm")
        assert "bashrc=/etc/bash.bashrc" in deb
        assert "bashrc=/etc/bashrc" in rpm
        # Banner snippet is concatenated from a sidecar file (see
        # scripts/terok-bash-banner.sh) into whichever bashrc the family uses.
        assert 'cat /tmp/terok-bash-banner.sh >> "$bashrc"' in deb
        assert 'cat /tmp/terok-bash-banner.sh >> "$bashrc"' in rpm

    def test_l1_unknown_agent_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown roster entries"):
            render_l1("terok-l0:test", family="deb", agents=("not-a-real-agent",))

    def test_l1_bare_string_selection_rejected(self) -> None:
        # A bare agent name passed as a string would otherwise be iterated
        # into characters ({'c','l','a','u','d','e'}) — the guard catches
        # that misuse instead of raising a confusing ValueError.
        with pytest.raises(TypeError, match="'all' or a tuple"):
            render_l1("terok-l0:test", family="deb", agents="claude")

    def test_l1_sidecar_is_valid_dockerfile(self) -> None:
        content = render_l1_sidecar("terok-l0:test", family="deb")
        assert content.startswith("# syntax=docker")
        assert "FROM" in content

    def test_l1_sidecar_contains_coderabbit(self) -> None:
        content = render_l1_sidecar("terok-l0:test", family="deb", tool_name="coderabbit")
        assert "coderabbit" in content.lower()

    def test_l1_sidecar_no_agent_installs(self) -> None:
        content = render_l1_sidecar("terok-l0:test", family="deb")
        assert "@openai/codex" not in content
        assert "claude.ai/install" not in content

    def test_l1_sidecar_contains_cache_bust_arg(self) -> None:
        content = render_l1_sidecar("terok-l0:test", family="deb")
        assert "ARG TOOL_CACHE_BUST=" in content

    def test_l1_sidecar_unknown_tool_empty(self) -> None:
        """Unknown tool_name renders a valid but tool-less Dockerfile."""
        content = render_l1_sidecar("terok-l0:test", family="deb", tool_name="nonexistent")
        assert "FROM" in content
        assert "coderabbit" not in content.lower()


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


class TestStageHelpFragments:
    """Verify per-section help fragment rendering for ``hilfe``."""

    def test_splits_by_section(self, tmp_path: Path) -> None:
        dest = tmp_path / "help.d"
        stage_help_fragments(dest, ("claude", "gh"))
        assert (dest / "agents.txt").is_file()
        assert (dest / "dev-tools.txt").is_file()
        assert "claude" in (dest / "agents.txt").read_text()
        assert "gh" in (dest / "dev-tools.txt").read_text()

    def test_decodes_ansi_escapes(self, tmp_path: Path) -> None:
        dest = tmp_path / "help.d"
        stage_help_fragments(dest, ("claude",))
        # \033 in the YAML must land as the literal ESC byte in the file so
        # that hilfe just `cat`s and gets coloured output.
        assert "\x1b[" in (dest / "agents.txt").read_text()
        assert r"\033[" not in (dest / "agents.txt").read_text()

    def test_omits_empty_sections(self, tmp_path: Path) -> None:
        # Selecting only an agent (no dev_tool) leaves dev-tools.txt absent.
        dest = tmp_path / "help.d"
        stage_help_fragments(dest, ("claude",))
        assert not (dest / "dev-tools.txt").exists()

    def test_decoder_preserves_non_ascii(self) -> None:
        """The escape decoder must not mojibake non-ASCII characters."""
        from terok_executor.container.build import _decode_label_escapes

        # bytes(s, "utf-8").decode("unicode_escape") would produce 'Ã¤' here.
        assert _decode_label_escapes(r"\033[36m→ ähnlich\033[0m") == ("\x1b[36m→ ähnlich\x1b[0m")


# ---------------------------------------------------------------------------
# Sidecar image build
# ---------------------------------------------------------------------------


class TestBuildSidecarImage:
    """Verify sidecar image build orchestration with mocked podman."""

    def test_skips_when_images_exist(self) -> None:
        from unittest.mock import patch

        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=True),
        ):
            tag = build_sidecar_image()
        assert tag.startswith("terok-l1-sidecar:")

    def test_builds_when_missing(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        # L0 exists but sidecar does not
        def image_exists_side_effect(image: str) -> bool:
            return "l0" in image

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch(
                "terok_executor.container.build._image_exists", side_effect=image_exists_side_effect
            ),
            patch("subprocess.run") as mock_run,
        ):
            tag = build_sidecar_image(build_dir=build_dir)

        # One podman build call (sidecar only — L0 already exists)
        assert mock_run.call_count == 1
        cmd = mock_run.call_args_list[0][0][0]
        assert "podman" in cmd[0]
        assert tag in cmd

    def test_sidecar_dockerfile_in_context(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        def image_exists_side_effect(image: str) -> bool:
            return "l0" in image

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch(
                "terok_executor.container.build._image_exists", side_effect=image_exists_side_effect
            ),
            patch("subprocess.run"),
        ):
            build_sidecar_image(build_dir=build_dir)

        assert (build_dir / "L1.sidecar.Dockerfile").is_file()
        content = (build_dir / "L1.sidecar.Dockerfile").read_text()
        assert "coderabbit" in content.lower()

    def test_build_failure_raises_build_error(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        def image_exists_side_effect(image: str) -> bool:
            return "l0" in image

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch(
                "terok_executor.container.build._image_exists", side_effect=image_exists_side_effect
            ),
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "podman"),
            ),
            pytest.raises(BuildError, match="Image build failed"),
        ):
            build_sidecar_image(build_dir=build_dir)

    def test_os_error_preparing_context_raises_build_error(self, tmp_path: Path) -> None:
        """Filesystem failures during sidecar context prep (disk full, permission
        denied, etc.) normalize to BuildError with the sidecar tag and context
        path in the message — same contract as build_base_images."""
        from unittest.mock import patch

        def image_exists_side_effect(image: str) -> bool:
            return "l0" in image

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch(
                "terok_executor.container.build._image_exists", side_effect=image_exists_side_effect
            ),
            patch(
                "terok_executor.container.build.prepare_build_context",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(BuildError, match=r"sidecar.*disk full"),
        ):
            build_sidecar_image(build_dir=build_dir)


# ---------------------------------------------------------------------------
# Package family detection + family-aware template rendering
# ---------------------------------------------------------------------------


class TestSplitImageRef:
    """Verify port-aware OCI ref parsing in `_split_image_ref`."""

    @pytest.mark.parametrize(
        ("ref", "expected"),
        [
            # Plain refs
            ("ubuntu:24.04", ("ubuntu", "24.04")),
            ("ubuntu", ("ubuntu", "")),
            ("nvcr.io/nvidia/cuda:13.0.0-devel-ubi9", ("nvcr.io/nvidia/cuda", "13.0.0-devel-ubi9")),
            # Registry ports — colon before the last '/' is *not* a tag
            ("localhost:5000/ubuntu:24.04", ("localhost:5000/ubuntu", "24.04")),
            ("localhost:5000/ubuntu", ("localhost:5000/ubuntu", "")),
            ("myreg.example.com:8443/fedora:43", ("myreg.example.com:8443/fedora", "43")),
            # Digests (everything after '@' is dropped before tag parsing)
            ("ubuntu@sha256:abc", ("ubuntu", "")),
            ("fedora:43@sha256:abc", ("fedora", "43")),
            (
                "localhost:5000/ubuntu:24.04@sha256:abc",
                ("localhost:5000/ubuntu", "24.04"),
            ),
        ],
    )
    def test_split(self, ref: str, expected: tuple[str, str]) -> None:
        assert _split_image_ref(ref) == expected


class TestDetectFamily:
    """Verify the deb/rpm allowlist + override behaviour."""

    @pytest.mark.parametrize(
        ("base_image", "expected"),
        [
            ("ubuntu:24.04", "deb"),
            ("ubuntu", "deb"),
            ("debian:12", "deb"),
            ("fedora:43", "rpm"),
            ("registry.fedoraproject.org/fedora:43", "rpm"),
            ("quay.io/podman/stable:latest", "rpm"),
            ("quay.io/podman/upstream:latest", "rpm"),
        ],
    )
    def test_known_prefixes(self, base_image: str, expected: str) -> None:
        assert detect_family(base_image) == expected

    @pytest.mark.parametrize(
        ("base_image", "expected"),
        [
            # Ubuntu marker → deb
            ("nvcr.io/nvidia/nvhpc:25.9-devel-cuda13.0-ubuntu24.04", "deb"),
            ("nvidia/cuda:12.4.1-devel-ubuntu24.04", "deb"),
            # UBI marker → rpm
            ("nvcr.io/nvidia/cuda:13.0.0-devel-ubi9", "rpm"),
            ("nvidia/cuda:12.4.0-devel-ubi8", "rpm"),
            # No marker → deb (historical NVIDIA convention)
            ("nvidia/cuda:latest", "deb"),
            ("nvcr.io/nvidia/pytorch:25.04-py3", "deb"),
        ],
    )
    def test_nvidia_disambiguates_via_tag(self, base_image: str, expected: str) -> None:
        assert detect_family(base_image) == expected

    @pytest.mark.parametrize(
        ("base_image", "expected"),
        [
            # Digest-only refs — tag parsing must not consume the digest.
            (
                "ubuntu@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "deb",
            ),
            ("fedora:43@sha256:0123456789abcdef", "rpm"),
        ],
    )
    def test_digest_refs_do_not_break_detection(self, base_image: str, expected: str) -> None:
        assert detect_family(base_image) == expected

    def test_registry_port_does_not_confuse_tag_parser(self) -> None:
        # Registry ports are preserved in the parsed name; private mirrors
        # don't match a known prefix, so the user sets ``family:`` explicitly.
        with pytest.raises(BuildError, match="Cannot infer package family"):
            detect_family("localhost:5000/ubuntu:24.04")
        assert detect_family("localhost:5000/ubuntu:24.04", override="deb") == "deb"

    def test_override_wins(self) -> None:
        # Override forces the family even when the prefix would resolve
        # to the other branch.
        assert detect_family("ubuntu:24.04", override="rpm") == "rpm"

    def test_override_allows_unknown(self) -> None:
        assert detect_family("rockylinux:9", override="rpm") == "rpm"

    def test_unknown_raises_with_hint(self) -> None:
        with pytest.raises(BuildError, match="family: deb"):
            detect_family("rockylinux:9")

    def test_invalid_override_rejected(self) -> None:
        with pytest.raises(BuildError, match="must be 'deb' or 'rpm'"):
            detect_family("ubuntu:24.04", override="alpine")

    def test_blank_falls_back_to_default(self) -> None:
        # _normalize_base_image turns blank into ubuntu:24.04 → deb.
        assert detect_family("") == "deb"


class TestRenderFamilyAware:
    """Verify L0/L1/sidecar templates emit the right package-manager branch."""

    def test_l0_deb_uses_apt(self) -> None:
        content = render_l0("ubuntu:24.04")
        assert "apt-get install" in content
        assert "locale-gen" in content
        assert "DEBIAN_FRONTEND=noninteractive" in content
        assert "dnf install" not in content

    def test_l0_rpm_uses_dnf(self) -> None:
        content = render_l0("fedora:43")
        assert "dnf install" in content
        assert "glibc-langpack-en" in content
        assert "openssh-clients" in content
        assert "apt-get" not in content
        assert "DEBIAN_FRONTEND" not in content

    def test_l0_podman_image_is_rpm(self) -> None:
        content = render_l0("quay.io/podman/stable:latest")
        assert "dnf install" in content
        assert "apt-get" not in content

    def test_l0_unknown_image_raises(self) -> None:
        with pytest.raises(BuildError, match="Cannot infer package family"):
            render_l0("rockylinux:9")

    def test_l0_explicit_family_overrides(self) -> None:
        content = render_l0("rockylinux:9", family="rpm")
        assert "dnf install" in content

    def test_l1_deb_uses_apt_and_deb_repos(self) -> None:
        content = render_l1("terok-l0:test", family="deb")
        assert "apt-get install" in content
        assert "deb.nodesource.com/setup_22.x" in content
        assert "/etc/apt/keyrings/githubcli" in content
        assert "glab_${GLAB_VERSION}_linux_${ARCH}.deb" in content
        assert "dpkg -i /tmp/glab.deb" in content

    def test_l1_rpm_uses_dnf_and_rpm_repos(self) -> None:
        content = render_l1("terok-l0:test", family="rpm")
        assert "dnf install" in content
        assert "rpm.nodesource.com/setup_22.x" in content
        assert "/etc/yum.repos.d/gh-cli.repo" in content
        assert "glab_${GLAB_VERSION}_linux_${ARCH}.rpm" in content
        # Filename must carry .rpm extension — dnf5 refuses local installs
        # of files named anything else (treats them as repo queries).
        assert "dnf install -y /tmp/glab.rpm" in content
        assert "apt-get" not in content
        assert "dpkg" not in content

    def test_l1_uses_uname_for_binary_tools(self) -> None:
        # Binary-tool installs (yq, glab, sonar) detect arch via uname so
        # the same shell snippet works on both families.  The deb branch
        # still uses dpkg in the apt-source registration line — that's
        # apt-specific and unrelated to binary downloads.
        for fam in ("deb", "rpm"):
            content = render_l1("terok-l0:test", family=fam)
            assert 'case "$(uname -m)"' in content
        rpm_content = render_l1("terok-l0:test", family="rpm")
        assert "dpkg" not in rpm_content

    def test_l1_sidecar_deb_uses_apt(self) -> None:
        content = render_l1_sidecar("terok-l0:test", family="deb")
        assert "apt-get install" in content
        assert "DEBIAN_FRONTEND" in content

    def test_l1_sidecar_rpm_uses_dnf(self) -> None:
        content = render_l1_sidecar("terok-l0:test", family="rpm")
        assert "dnf install" in content
        assert "apt-get" not in content
        assert "DEBIAN_FRONTEND" not in content


class TestBuildBaseImagesFamily:
    """Verify build_base_images resolves and threads the family kwarg."""

    def test_fedora_renders_dnf_dockerfile(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=False),
            patch("subprocess.run"),
        ):
            build_base_images("fedora:43", build_dir=build_dir)

        l0 = (build_dir / "L0.Dockerfile").read_text()
        l1 = (build_dir / "L1.cli.Dockerfile").read_text()
        assert "dnf install" in l0 and "apt-get" not in l0
        assert "dnf install" in l1 and "apt-get" not in l1

    def test_unknown_image_raises_buildError(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=False),
            pytest.raises(BuildError, match="Cannot infer package family"),
        ):
            build_base_images("rockylinux:9", build_dir=build_dir)

    def test_family_override_unblocks_unknown_image(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        build_dir = tmp_path / "ctx"
        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=False),
            patch("subprocess.run"),
        ):
            build_base_images("rockylinux:9", family="rpm", build_dir=build_dir)

        assert "dnf install" in (build_dir / "L0.Dockerfile").read_text()

    def test_cached_unknown_image_skips_family_detection(self) -> None:
        """The fast-path early return must not invoke detect_family.

        Otherwise a prebuilt L0+L1 for an unknown base would fail to be
        reused unless the user re-supplies ``family:`` on every call.
        """
        from unittest.mock import patch

        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=True),
            patch("terok_executor.container.build.detect_family") as mock_detect,
        ):
            result = build_base_images("rockylinux:9")

        assert result.l0.endswith(":rockylinux-9")
        # L1 tag carries the default agent-suffix on top of the base fragment.
        assert result.l1.startswith("terok-l1-cli:rockylinux-9-")
        mock_detect.assert_not_called()


class TestBuildSidecarImageFamily:
    """Verify the same fast-path applies to the sidecar build."""

    def test_cached_unknown_image_skips_family_detection(self) -> None:
        from unittest.mock import patch

        with (
            patch("terok_executor.container.build._check_podman"),
            patch("terok_executor.container.build._image_exists", return_value=True),
            patch("terok_executor.container.build.detect_family") as mock_detect,
        ):
            tag = build_sidecar_image("rockylinux:9")

        assert tag.endswith(":rockylinux-9")
        mock_detect.assert_not_called()
