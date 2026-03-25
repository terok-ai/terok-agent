# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the YAML agent registry loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from terok_agent.auth import AuthProvider
from terok_agent.headless_providers import HeadlessProvider
from terok_agent.registry import (
    _load_bundled_agents,
    _to_auth_provider,
    _to_headless_provider,
    load_registry,
)


@pytest.fixture(autouse=True)
def _isolate_user_agents_dir(tmp_path: Path) -> None:
    """Prevent real ~/.config/terok-agent/agents/ from leaking into tests."""
    isolated = tmp_path / "empty-agents"
    with patch("terok_agent.registry._user_agents_dir", return_value=isolated):
        yield


# ---------------------------------------------------------------------------
# Bundled YAML loading
# ---------------------------------------------------------------------------


class TestLoadBundledAgents:
    """Verify bundled agent YAML files are loadable and well-formed."""

    def test_loads_all_bundled_agents(self) -> None:
        agents = _load_bundled_agents()
        expected = {
            "claude",
            "codex",
            "copilot",
            "vibe",
            "blablador",
            "kisski",
            "opencode",
            "gh",
            "glab",
            "toad",
        }
        assert set(agents.keys()) == expected

    def test_each_agent_has_kind(self) -> None:
        valid_kinds = {"native", "opencode", "bridge", "tool", "runtime"}
        for name, data in _load_bundled_agents().items():
            assert "kind" in data, f"{name}.yaml missing 'kind' field"
            assert data["kind"] in valid_kinds, f"{name}.yaml has invalid kind={data['kind']!r}"

    def test_agents_have_required_sections(self) -> None:
        for name, data in _load_bundled_agents().items():
            if data["kind"] in ("tool", "runtime"):
                continue
            assert "label" in data, f"{name}: missing label"
            assert "binary" in data, f"{name}: missing binary"
            assert "git_identity" in data, f"{name}: missing git_identity"
            assert "headless" in data, f"{name}: missing headless"
            assert "session" in data, f"{name}: missing session"
            assert "capabilities" in data, f"{name}: missing capabilities"

    def test_tools_have_auth(self) -> None:
        for name, data in _load_bundled_agents().items():
            if data["kind"] != "tool":
                continue
            assert "auth" in data, f"tool {name}: missing auth"
            auth = data["auth"]
            assert "host_dir" in auth, f"tool {name}: missing auth.host_dir"
            assert "container_mount" in auth, f"tool {name}: missing auth.container_mount"


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


class TestDeserializeProvider:
    """Verify YAML → HeadlessProvider conversion."""

    def test_claude_full_fidelity(self) -> None:
        agents = _load_bundled_agents()
        p = _to_headless_provider("claude", agents["claude"])

        assert isinstance(p, HeadlessProvider)
        assert p.name == "claude"
        assert p.label == "Claude"
        assert p.binary == "claude"
        assert p.git_author_name == "Claude"
        assert p.git_author_email == "noreply@anthropic.com"
        assert p.headless_subcommand is None
        assert p.prompt_flag == "-p"
        assert p.auto_approve_env == {}
        assert p.auto_approve_flags == ()
        assert p.output_format_flags == ("--output-format", "stream-json")
        assert p.model_flag == "--model"
        assert p.max_turns_flag == "--max-turns"
        assert p.verbose_flag == "--verbose"
        assert p.supports_session_resume is True
        assert p.resume_flag == "--resume"
        assert p.continue_flag is None
        assert p.session_file is None
        assert p.supports_agents_json is True
        assert p.supports_session_hook is True
        assert p.supports_add_dir is True
        assert p.log_format == "claude-stream-json"
        assert p.opencode_config is None

    def test_codex_subcommand_and_flags(self) -> None:
        agents = _load_bundled_agents()
        p = _to_headless_provider("codex", agents["codex"])

        assert p.headless_subcommand == "exec"
        assert p.prompt_flag == ""
        assert p.auto_approve_flags == ("--yolo",)
        assert p.supports_session_resume is False

    def test_blablador_opencode_config(self) -> None:
        agents = _load_bundled_agents()
        p = _to_headless_provider("blablador", agents["blablador"])

        assert p.opencode_config is not None
        assert p.opencode_config.display_name == "Helmholtz Blablador"
        assert p.opencode_config.env_var_prefix == "BLABLADOR"
        assert p.opencode_config.config_dir == ".blablador"

    def test_vibe_session_support(self) -> None:
        agents = _load_bundled_agents()
        p = _to_headless_provider("vibe", agents["vibe"])

        assert p.supports_session_resume is True
        assert p.resume_flag == "--resume"
        assert p.continue_flag == "--continue"
        assert p.session_file == "vibe-session.txt"
        assert p.model_flag == "--agent"

    def test_defaults_for_omitted_fields(self) -> None:
        """Omitted optional fields get sensible defaults."""
        p = _to_headless_provider("minimal", {"label": "Test", "binary": "test"})

        assert p.headless_subcommand is None
        assert p.auto_approve_env == {}
        assert p.auto_approve_flags == ()
        assert p.output_format_flags == ()
        assert p.model_flag is None
        assert p.supports_session_resume is False
        assert p.log_format == "plain"


class TestDeserializeAuth:
    """Verify YAML → AuthProvider conversion."""

    def test_claude_auth_uses_native_cli(self) -> None:
        agents = _load_bundled_agents()
        ap = _to_auth_provider("claude", agents["claude"])

        assert isinstance(ap, AuthProvider)
        assert ap.name == "claude"
        assert ap.host_dir_name == "_claude-config"
        assert ap.container_mount == "/home/dev/.claude"
        assert ap.command == ["claude"]

    def test_codex_auth_command(self) -> None:
        agents = _load_bundled_agents()
        ap = _to_auth_provider("codex", agents["codex"])

        assert ap.command == ["setup-codex-auth.sh"]
        assert ap.extra_run_args == ("-p", "127.0.0.1:1455:1455")

    def test_gh_tool_auth(self) -> None:
        agents = _load_bundled_agents()
        ap = _to_auth_provider("gh", agents["gh"])

        assert ap.name == "gh"
        assert ap.command == ["gh", "auth", "login"]
        assert ap.host_dir_name == "_gh-config"

    def test_no_auth_section_returns_none(self) -> None:
        result = _to_auth_provider("test", {"label": "Test"})
        assert result is None


# ---------------------------------------------------------------------------
# Full registry
# ---------------------------------------------------------------------------


class TestLoadRegistry:
    """Integration tests for the complete registry load cycle."""

    def test_loads_all_agents(self) -> None:
        reg = load_registry()
        expected_agents = {"claude", "codex", "copilot", "vibe", "blablador", "kisski", "opencode"}
        assert set(reg.agent_names) == expected_agents

    def test_all_names_includes_tools(self) -> None:
        reg = load_registry()
        assert "gh" in reg.all_names
        assert "glab" in reg.all_names
        assert "claude" in reg.all_names

    def test_providers_only_agents(self) -> None:
        reg = load_registry()
        assert "gh" not in reg.providers
        assert "glab" not in reg.providers
        assert "claude" in reg.providers

    def test_auth_includes_tools(self) -> None:
        reg = load_registry()
        assert "gh" in reg.auth_providers
        assert "glab" in reg.auth_providers

    def test_auth_includes_opencode_derived(self) -> None:
        reg = load_registry()
        # blablador has no explicit auth section but has opencode config → auto-derived
        assert "blablador" in reg.auth_providers
        assert "kisski" in reg.auth_providers

    def test_mounts_include_auth_dirs(self) -> None:
        reg = load_registry()
        mount_dirs = {m.host_dir for m in reg.mounts}
        assert "_claude-config" in mount_dirs
        assert "_codex-config" in mount_dirs
        assert "_gh-config" in mount_dirs
        assert "_glab-config" in mount_dirs

    def test_mounts_include_extra_dirs(self) -> None:
        reg = load_registry()
        mount_dirs = {m.host_dir for m in reg.mounts}
        assert "_opencode-config" in mount_dirs
        assert "_opencode-data" in mount_dirs
        assert "_opencode-state" in mount_dirs
        assert "_toad-config" in mount_dirs

    def test_mounts_deduplicated(self) -> None:
        reg = load_registry()
        host_dirs = [m.host_dir for m in reg.mounts]
        assert len(host_dirs) == len(set(host_dirs))

    def test_get_provider_resolves(self) -> None:
        reg = load_registry()
        p = reg.get_provider("codex")
        assert p.name == "codex"

    def test_get_provider_fallback(self) -> None:
        reg = load_registry()
        p = reg.get_provider(None)
        assert p.name == "claude"

    def test_get_provider_unknown_exits(self) -> None:
        reg = load_registry()
        with pytest.raises(SystemExit, match="Unknown headless provider"):
            reg.get_provider("nonexistent")

    def test_get_auth_provider_unknown_exits(self) -> None:
        reg = load_registry()
        with pytest.raises(SystemExit, match="Unknown auth provider"):
            reg.get_auth_provider("nonexistent")

    def test_collect_all_auto_approve_env(self) -> None:
        reg = load_registry()
        env = reg.collect_all_auto_approve_env()
        assert "COPILOT_ALLOW_ALL" in env
        assert "VIBE_AUTO_APPROVE" in env
        assert "OPENCODE_PERMISSION" in env

    def test_collect_opencode_provider_env(self) -> None:
        reg = load_registry()
        env = reg.collect_opencode_provider_env()
        assert any(k.startswith("TEROK_OC_BLABLADOR_") for k in env)
        assert any(k.startswith("TEROK_OC_KISSKI_") for k in env)


# ---------------------------------------------------------------------------
# User override merging
# ---------------------------------------------------------------------------


class TestUserOverrides:
    """Verify user extension YAML files are deep-merged correctly."""

    def test_user_override_field(self, tmp_path: Path) -> None:
        """A user file can override a single field of a bundled agent."""
        user_dir = tmp_path / "agents"
        user_dir.mkdir()
        (user_dir / "claude.yaml").write_text("tier: 99\n")

        with patch("terok_agent.registry._user_agents_dir", return_value=user_dir):
            reg = load_registry()

        # Provider still loads correctly, tier is just metadata
        p = reg.get_provider("claude")
        assert p.name == "claude"
        assert p.label == "Claude"  # unchanged

    def test_user_new_agent(self, tmp_path: Path) -> None:
        """A user can add an entirely new agent."""
        user_dir = tmp_path / "agents"
        user_dir.mkdir()
        (user_dir / "custom.yaml").write_text(
            "kind: agent\nlabel: Custom Agent\nbinary: custom\n"
            "git_identity:\n  name: Custom\n  email: a@b.c\n"
            "headless:\n  prompt_flag: '-p'\n"
            "session:\n  supports_resume: false\n"
            "capabilities:\n  log_format: plain\n"
        )

        with patch("terok_agent.registry._user_agents_dir", return_value=user_dir):
            reg = load_registry()

        assert "custom" in reg.agent_names
        p = reg.get_provider("custom")
        assert p.label == "Custom Agent"

    def test_user_new_tool(self, tmp_path: Path) -> None:
        """A user can add a new tool."""
        user_dir = tmp_path / "agents"
        user_dir.mkdir()
        (user_dir / "mytool.yaml").write_text(
            "kind: tool\nlabel: My Tool\nbinary: mytool\n"
            "auth:\n  host_dir: _mytool-config\n"
            "  container_mount: /home/dev/.mytool\n"
            "  command: ['mytool', 'auth']\n"
            "  banner_hint: Authenticate.\n"
        )

        with patch("terok_agent.registry._user_agents_dir", return_value=user_dir):
            reg = load_registry()

        assert "mytool" in reg.all_names
        assert "mytool" not in reg.agent_names  # it's a tool, not an agent
        ap = reg.get_auth_provider("mytool")
        assert ap.command == ["mytool", "auth"]

    def test_no_user_dir_ok(self, tmp_path: Path) -> None:
        """Missing user dir is fine — only bundled agents are loaded."""
        with patch("terok_agent.registry._user_agents_dir", return_value=tmp_path / "nonexistent"):
            reg = load_registry()

        assert "claude" in reg.agent_names


# ---------------------------------------------------------------------------
# Behavioral validation — registry produces usable dataclasses
# ---------------------------------------------------------------------------


class TestRegistryBehavior:
    """Verify the registry produces well-formed, usable provider dataclasses."""

    def test_every_agent_has_valid_headless_provider(self) -> None:
        """Each agent deserializes into a HeadlessProvider with required fields."""
        reg = load_registry()
        for name in reg.agent_names:
            p = reg.get_provider(name)
            assert isinstance(p, HeadlessProvider)
            assert p.name == name
            assert p.binary  # non-empty binary
            assert p.label  # non-empty label
            assert p.git_author_name
            assert p.git_author_email
            assert p.log_format in {"plain", "claude-stream-json"}

    def test_every_auth_provider_has_valid_config(self) -> None:
        """Each auth provider has mount paths and at least one auth mode."""
        reg = load_registry()
        for name, ap in reg.auth_providers.items():
            assert isinstance(ap, AuthProvider)
            assert ap.host_dir_name, f"{name}: empty host_dir"
            assert ap.container_mount, f"{name}: empty container_mount"
            assert ap.modes, f"{name}: no auth modes"
            # OAuth providers must have a container command
            if ap.supports_oauth:
                assert ap.command, f"{name}: oauth mode but no command"

    def test_opencode_providers_have_complete_config(self) -> None:
        """Providers with opencode config have all required fields populated."""
        reg = load_registry()
        for name, p in reg.providers.items():
            if p.opencode_config is None:
                continue
            oc = p.opencode_config
            assert oc.display_name, f"{name}: empty display_name"
            assert oc.base_url.startswith("https://"), f"{name}: invalid base_url"
            assert oc.preferred_model, f"{name}: empty preferred_model"
            assert oc.fallback_model, f"{name}: empty fallback_model"
            assert oc.env_var_prefix, f"{name}: empty env_var_prefix"
            assert oc.config_dir, f"{name}: empty config_dir"

    def test_auto_approve_env_values_are_strings(self) -> None:
        """Auto-approve env values must be strings (injected into container env)."""
        reg = load_registry()
        for name, p in reg.providers.items():
            for k, v in p.auto_approve_env.items():
                assert isinstance(k, str), f"{name}: env key {k!r} not str"
                assert isinstance(v, str), f"{name}: env value {v!r} not str"

    def test_session_resume_consistency(self) -> None:
        """Providers with session resume must have a resume_flag."""
        reg = load_registry()
        for name, p in reg.providers.items():
            if p.supports_session_resume:
                assert p.resume_flag, f"{name}: supports_resume but no resume_flag"
