# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the YAML agent roster loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from terok_executor.credentials.auth import AuthProvider
from terok_executor.provider.providers import AgentProvider
from terok_executor.roster import (
    SidecarSpec,
    load_roster,
)
from terok_executor.roster.loader import (
    _load_bundled_agents,
    _to_agent_provider,
    _to_auth_provider,
    _to_sidecar_spec,
)


@pytest.fixture(autouse=True)
def _isolate_user_agents_dir(tmp_path: Path) -> None:
    """Prevent real ~/.config/terok/agent/agents/ from leaking into tests."""
    isolated = tmp_path / "empty-agents"
    with patch("terok_executor.roster.loader._user_agents_dir", return_value=isolated):
        yield


# ---------------------------------------------------------------------------
# Bundled YAML loading
# ---------------------------------------------------------------------------


class TestLoadBundledAgents:
    """Verify bundled agent YAML files are loadable and well-formed."""

    def test_loads_all_bundled_agents(self) -> None:
        agents = _load_bundled_agents()
        expected = {
            "caddy",
            "claude",
            "coderabbit",
            "codex",
            "copilot",
            "gh",
            "glab",
            "blablador",
            "kisski",
            "opencode",
            "sonar",
            "toad",
            "vibe",
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


class TestRosterVersion:
    """Roster-schema versioning lets in-container contracts evolve without
    silently breaking existing user overrides."""

    def test_every_bundled_yaml_declares_version(self) -> None:
        """Every shipped YAML declares ``roster_version: 1`` so the file
        is self-describing — readers (including sessions that pull raw
        YAML without the loader) know which contract it targets."""
        import importlib.resources

        from terok_executor.roster.loader import ROSTER_VERSION, _load_yaml

        pkg = importlib.resources.files("terok_executor.resources.agents")
        missing = []
        for item in pkg.iterdir():
            if not hasattr(item, "name") or not item.name.endswith(".yaml"):
                continue
            data = _load_yaml(item.read_text(encoding="utf-8"))
            if data.get("roster_version") != ROSTER_VERSION:
                missing.append(item.name)
        assert not missing, f"bundled YAMLs lacking roster_version={ROSTER_VERSION}: {missing}"

    def test_loader_accepts_current_version(self, capsys) -> None:
        """Version matching ``ROSTER_VERSION`` loads silently."""
        from terok_executor.roster.loader import ROSTER_VERSION, _check_roster_version

        _check_roster_version("x", {"roster_version": ROSTER_VERSION}, source="test")
        assert capsys.readouterr().err == ""

    def test_loader_accepts_missing_version(self, capsys) -> None:
        """Missing version is treated as the current version — existing
        user overrides written before the marker existed must keep working."""
        from terok_executor.roster.loader import _check_roster_version

        _check_roster_version("x", {"kind": "native"}, source="test")
        assert capsys.readouterr().err == ""

    def test_loader_accepts_older_version_silently(self, capsys) -> None:
        """A declared version *older* than the current loader is the
        backward-compat path — the loader knows how to read it, no warning."""
        from terok_executor.roster.loader import _check_roster_version

        _check_roster_version("x", {"roster_version": 0}, source="old.yaml")
        assert capsys.readouterr().err == ""

    def test_loader_warns_on_future_version(self, capsys) -> None:
        """A future version still loads but surfaces a warning — the host
        may not speak every field in the file."""
        from terok_executor.roster.loader import _check_roster_version

        _check_roster_version("x", {"roster_version": 99}, source="user.yaml")
        stderr = capsys.readouterr().err
        assert "roster_version=99" in stderr
        assert "user.yaml" in stderr

    def test_loader_warns_on_non_integer_version(self, capsys) -> None:
        """A non-integer version is malformed metadata; warn and treat
        as current so the file still loads."""
        from terok_executor.roster.loader import _check_roster_version

        _check_roster_version("x", {"roster_version": "one-point-oh"}, source="weird.yaml")
        stderr = capsys.readouterr().err
        assert "not a valid integer" in stderr

    def test_loader_strips_version_from_data(self) -> None:
        """``roster_version`` is metadata, not a field the deserializer
        should see — removed from the dict before it flows downstream."""
        from terok_executor.roster.loader import _check_roster_version

        data = {"roster_version": 1, "kind": "native"}
        _check_roster_version("x", data, source="test")
        assert "roster_version" not in data

    def test_metadata_only_file_is_skipped(self, capsys) -> None:
        """A YAML containing only ``roster_version`` has no agent to
        register — don't add an empty dict to the roster that would
        later confuse the deserializer."""
        from terok_executor.roster.loader import _add_agent

        agents: dict[str, dict] = {}
        _add_agent(agents, "meta_only", {"roster_version": 1}, source="meta.yaml")
        assert agents == {}
        assert "metadata-only" in capsys.readouterr().err

    def test_add_agent_preserves_real_content(self) -> None:
        """Non-empty agent definitions land in the map with their version
        marker stripped out."""
        from terok_executor.roster.loader import _add_agent

        agents: dict[str, dict] = {}
        _add_agent(
            agents,
            "claude",
            {"roster_version": 1, "kind": "native", "label": "Claude"},
            source="test.yaml",
        )
        assert "claude" in agents
        assert "roster_version" not in agents["claude"]
        assert agents["claude"]["label"] == "Claude"

    def test_add_agent_ignores_empty_input(self) -> None:
        """Empty or ``None`` data (parse error / blank file) is a no-op."""
        from terok_executor.roster.loader import _add_agent

        agents: dict[str, dict] = {}
        _add_agent(agents, "empty", None, source="blank.yaml")
        _add_agent(agents, "also_empty", {}, source="blank2.yaml")
        assert agents == {}


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


class TestDeserializeProvider:
    """Verify YAML → AgentProvider conversion."""

    def test_claude_full_fidelity(self) -> None:
        agents = _load_bundled_agents()
        p = _to_agent_provider("claude", agents["claude"])

        assert isinstance(p, AgentProvider)
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
        p = _to_agent_provider("codex", agents["codex"])

        assert p.headless_subcommand == "exec"
        assert p.prompt_flag == ""
        assert p.auto_approve_flags == ("--yolo",)
        assert p.supports_session_resume is False

    def test_blablador_opencode_config(self) -> None:
        agents = _load_bundled_agents()
        p = _to_agent_provider("blablador", agents["blablador"])

        assert p.opencode_config is not None
        assert p.opencode_config.display_name == "Helmholtz Blablador"
        assert p.opencode_config.env_var_prefix == "BLABLADOR"
        assert p.opencode_config.config_dir == ".blablador"

    def test_vibe_session_support(self) -> None:
        agents = _load_bundled_agents()
        p = _to_agent_provider("vibe", agents["vibe"])

        assert p.supports_session_resume is True
        assert p.resume_flag == "--resume"
        assert p.continue_flag == "--continue"
        assert p.session_file == "vibe-session.txt"
        assert p.model_flag == "--agent"

    def test_defaults_for_omitted_fields(self) -> None:
        """Omitted optional fields get sensible defaults."""
        p = _to_agent_provider("minimal", {"label": "Test", "binary": "test"})

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

    def test_claude_post_capture_state(self) -> None:
        """Claude YAML declares post_capture_state for onboarding."""
        agents = _load_bundled_agents()
        ap = _to_auth_provider("claude", agents["claude"])
        assert ap.post_capture_state == {".claude.json": {"hasCompletedOnboarding": True}}

    def test_post_capture_state_rejects_non_dict_root(self) -> None:
        """Loader rejects post_capture_state that is not a mapping."""
        import pytest

        data = {
            "auth": {
                "host_dir": "_x",
                "container_mount": "/x",
                "post_capture_state": "invalid",
            },
        }
        with pytest.raises(ValueError, match="must be a mapping"):
            _to_auth_provider("test", data)

    def test_post_capture_state_rejects_non_dict_value(self) -> None:
        """Loader rejects post_capture_state with a non-dict value."""
        import pytest

        data = {
            "auth": {
                "host_dir": "_x",
                "container_mount": "/x",
                "post_capture_state": {".foo.json": "not-a-dict"},
            },
        }
        with pytest.raises(ValueError, match="must map filename -> mapping"):
            _to_auth_provider("test", data)

    def test_post_capture_state_none_coerced_to_empty(self) -> None:
        """YAML null for post_capture_state is coerced to empty dict."""
        data = {
            "auth": {
                "host_dir": "_x",
                "container_mount": "/x",
                "post_capture_state": None,
            },
        }
        ap = _to_auth_provider("test", data)
        assert ap.post_capture_state == {}


# ---------------------------------------------------------------------------
# Full registry
# ---------------------------------------------------------------------------


class TestLoadRegistry:
    """Integration tests for the complete registry load cycle."""

    def test_loads_all_agents(self) -> None:
        reg = load_roster()
        expected_agents = {"claude", "codex", "copilot", "vibe", "blablador", "kisski", "opencode"}
        assert set(reg.agent_names) == expected_agents

    def test_all_names_includes_tools(self) -> None:
        reg = load_roster()
        assert "gh" in reg.all_names
        assert "glab" in reg.all_names
        assert "claude" in reg.all_names

    def test_providers_only_agents(self) -> None:
        reg = load_roster()
        assert "gh" not in reg.providers
        assert "glab" not in reg.providers
        assert "claude" in reg.providers

    def test_auth_includes_tools(self) -> None:
        reg = load_roster()
        assert "gh" in reg.auth_providers
        assert "glab" in reg.auth_providers

    def test_auth_includes_opencode_derived(self) -> None:
        reg = load_roster()
        # blablador has no explicit auth section but has opencode config → auto-derived
        assert "blablador" in reg.auth_providers
        assert "kisski" in reg.auth_providers

    def test_mounts_include_auth_dirs(self) -> None:
        reg = load_roster()
        mount_dirs = {m.host_dir for m in reg.mounts}
        assert "_claude-config" in mount_dirs
        assert "_codex-config" in mount_dirs
        assert "_gh-config" in mount_dirs
        assert "_glab-config" in mount_dirs

    def test_mounts_include_extra_dirs(self) -> None:
        reg = load_roster()
        mount_dirs = {m.host_dir for m in reg.mounts}
        assert "_opencode-config" in mount_dirs
        assert "_opencode-data" in mount_dirs
        assert "_opencode-state" in mount_dirs
        assert "_toad-config" in mount_dirs

    def test_mounts_deduplicated(self) -> None:
        reg = load_roster()
        host_dirs = [m.host_dir for m in reg.mounts]
        assert len(host_dirs) == len(set(host_dirs))

    def test_get_provider_resolves(self) -> None:
        reg = load_roster()
        p = reg.get_provider("codex")
        assert p.name == "codex"

    def test_get_provider_fallback(self) -> None:
        reg = load_roster()
        p = reg.get_provider(None)
        assert p.name == "claude"

    def test_get_provider_unknown_exits(self) -> None:
        reg = load_roster()
        with pytest.raises(SystemExit, match="Unknown provider"):
            reg.get_provider("nonexistent")

    def test_get_auth_provider_unknown_exits(self) -> None:
        reg = load_roster()
        with pytest.raises(SystemExit, match="Unknown auth provider"):
            reg.get_auth_provider("nonexistent")

    def test_collect_all_auto_approve_env(self) -> None:
        reg = load_roster()
        env = reg.collect_all_auto_approve_env()
        assert "COPILOT_ALLOW_ALL" in env
        assert "VIBE_AUTO_APPROVE" in env
        assert "OPENCODE_PERMISSION" in env

    def test_collect_opencode_provider_env(self) -> None:
        reg = load_roster()
        env = reg.collect_opencode_provider_env()
        assert any(k.startswith("TEROK_OC_BLABLADOR_") for k in env)
        assert any(k.startswith("TEROK_OC_KISSKI_") for k in env)


# ---------------------------------------------------------------------------
# Sidecar spec deserialization
# ---------------------------------------------------------------------------


class TestDeserializeSidecar:
    """Verify YAML → SidecarSpec conversion."""

    def test_coderabbit_sidecar_spec(self) -> None:
        agents = _load_bundled_agents()
        spec = _to_sidecar_spec("coderabbit", agents["coderabbit"])

        assert isinstance(spec, SidecarSpec)
        assert spec.tool_name == "coderabbit"
        assert spec.env_map == {"CODERABBIT_API_KEY": "key"}

    def test_no_sidecar_returns_none(self) -> None:
        result = _to_sidecar_spec("claude", {"label": "Claude", "binary": "claude"})
        assert result is None

    def test_roster_exposes_sidecar_specs(self) -> None:
        reg = load_roster()
        assert "coderabbit" in reg.sidecar_specs
        assert reg.sidecar_specs["coderabbit"].tool_name == "coderabbit"

    def test_get_sidecar_spec_resolves(self) -> None:
        reg = load_roster()
        spec = reg.get_sidecar_spec("coderabbit")
        assert spec.tool_name == "coderabbit"

    def test_get_sidecar_spec_unknown_exits(self) -> None:
        reg = load_roster()
        with pytest.raises(SystemExit, match="No sidecar config"):
            reg.get_sidecar_spec("nonexistent")


class TestWebIngress:
    """Verify the ``web_ingress`` flag is surfaced via the roster."""

    def test_toad_is_web_ingress(self) -> None:
        reg = load_roster()
        assert "toad" in reg.web_ingress

    def test_non_web_agents_absent(self) -> None:
        reg = load_roster()
        assert "claude" not in reg.web_ingress
        assert "caddy" not in reg.web_ingress

    def test_toad_depends_on_caddy(self) -> None:
        reg = load_roster()
        # resolve_selection returns the canonical alphabetical tuple.
        # Asserting the exact set guards against accidental extras creeping in.
        assert reg.resolve_selection(("toad",)) == ("caddy", "toad")


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

        with patch("terok_executor.roster.loader._user_agents_dir", return_value=user_dir):
            reg = load_roster()

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

        with patch("terok_executor.roster.loader._user_agents_dir", return_value=user_dir):
            reg = load_roster()

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

        with patch("terok_executor.roster.loader._user_agents_dir", return_value=user_dir):
            reg = load_roster()

        assert "mytool" in reg.all_names
        assert "mytool" not in reg.agent_names  # it's a tool, not an agent
        ap = reg.get_auth_provider("mytool")
        assert ap.command == ["mytool", "auth"]

    def test_no_user_dir_ok(self, tmp_path: Path) -> None:
        """Missing user dir is fine — only bundled agents are loaded."""
        with patch(
            "terok_executor.roster.loader._user_agents_dir", return_value=tmp_path / "nonexistent"
        ):
            reg = load_roster()

        assert "claude" in reg.agent_names


# ---------------------------------------------------------------------------
# Behavioral validation — registry produces usable dataclasses
# ---------------------------------------------------------------------------


class TestRegistryBehavior:
    """Verify the registry produces well-formed, usable provider dataclasses."""

    def test_every_agent_has_valid_headless_provider(self) -> None:
        """Each agent deserializes into a AgentProvider with required fields."""
        reg = load_roster()
        for name in reg.agent_names:
            p = reg.get_provider(name)
            assert isinstance(p, AgentProvider)
            assert p.name == name
            assert p.binary  # non-empty binary
            assert p.label  # non-empty label
            assert p.git_author_name
            assert p.git_author_email
            assert p.log_format in {"plain", "claude-stream-json"}

    def test_every_auth_provider_has_valid_config(self) -> None:
        """Each auth provider has mount paths and at least one auth mode."""
        reg = load_roster()
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
        reg = load_roster()
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
        reg = load_roster()
        for name, p in reg.providers.items():
            for k, v in p.auto_approve_env.items():
                assert isinstance(k, str), f"{name}: env key {k!r} not str"
                assert isinstance(v, str), f"{name}: env value {v!r} not str"

    def test_session_resume_consistency(self) -> None:
        """Providers with session resume must have a resume_flag."""
        reg = load_roster()
        for name, p in reg.providers.items():
            if p.supports_session_resume:
                assert p.resume_flag, f"{name}: supports_resume but no resume_flag"
