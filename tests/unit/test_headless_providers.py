# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for headless provider registry and dispatch functions."""

from dataclasses import FrozenInstanceError

import pytest

from terok_agent.agent_config import resolve_provider_value
from terok_agent.agents import _generate_claude_wrapper
from terok_agent.headless_providers import (
    HEADLESS_PROVIDERS,
    PROVIDER_NAMES,
    apply_provider_config,
    build_headless_command,
    collect_all_auto_approve_env,
    generate_agent_wrapper,
    generate_all_wrappers,
    get_provider,
)
from tests.constants import CONTAINER_TEROK_DIR


def _provider_wrapper(
    name: str,
    *,
    has_agents: bool = False,
) -> str:
    """Generate a wrapper for a provider under test."""
    kwargs = {"claude_wrapper_fn": _generate_claude_wrapper} if name == "claude" else {}
    return generate_agent_wrapper(
        HEADLESS_PROVIDERS[name],
        has_agents=has_agents,
        **kwargs,
    )


def _all_wrappers(*, has_agents: bool = False) -> str:
    """Generate the combined multi-provider wrapper file."""
    return generate_all_wrappers(
        has_agents=has_agents,
        claude_wrapper_fn=_generate_claude_wrapper,
    )


class TestHeadlessProviderRegistry:
    """Tests for the HEADLESS_PROVIDERS registry."""

    def test_all_seven_providers_exist(self) -> None:
        """Registry contains exactly the seven expected providers."""
        expected = {"claude", "codex", "copilot", "vibe", "blablador", "opencode", "kisski"}
        assert set(HEADLESS_PROVIDERS.keys()) == expected

    def test_provider_names_tuple(self) -> None:
        """PROVIDER_NAMES is a tuple matching registry keys."""
        assert isinstance(PROVIDER_NAMES, tuple)
        assert set(PROVIDER_NAMES) == set(HEADLESS_PROVIDERS.keys())

    def test_providers_are_frozen(self) -> None:
        """HeadlessProvider instances are immutable."""
        provider = HEADLESS_PROVIDERS["claude"]
        with pytest.raises(FrozenInstanceError):
            provider.name = "changed"  # type: ignore[misc]


class TestGetProvider:
    """Tests for get_provider() resolution."""

    def test_explicit_name(self) -> None:
        """Explicit provider name resolves correctly."""
        p = get_provider("codex")
        assert p.name == "codex"

    def test_none_falls_back_to_default(self) -> None:
        """None name uses default_agent."""
        p = get_provider(None, default_agent="copilot")
        assert p.name == "copilot"

    def test_none_falls_back_to_claude(self) -> None:
        """None name with no default resolves to claude."""
        p = get_provider(None, default_agent=None)
        assert p.name == "claude"

    def test_invalid_name_raises_system_exit(self) -> None:
        """Unknown provider name raises SystemExit."""
        with pytest.raises(SystemExit) as ctx:
            get_provider("nonexistent")
        assert "nonexistent" in str(ctx.value)


class TestBuildHeadlessCommand:
    """Tests for build_headless_command() per provider."""

    def test_all_commands_start_with_init(self) -> None:
        """All provider commands start with init-ssh-and-repo.sh."""
        for _name, p in HEADLESS_PROVIDERS.items():
            cmd = build_headless_command(p, timeout=1800)
            assert cmd.startswith("init-ssh-and-repo.sh")


class TestGenerateAgentWrapper:
    """Tests for generate_agent_wrapper() per provider."""

    def test_claude_wrapper_uses_claude_function(self) -> None:
        """Claude wrapper defines a claude() function with --add-dir."""
        wrapper = _provider_wrapper("claude")
        assert "claude()" in wrapper
        assert "--add-dir" in wrapper

    def test_claude_wrapper_requires_fn(self) -> None:
        """Claude provider without claude_wrapper_fn raises ValueError."""
        p = HEADLESS_PROVIDERS["claude"]
        with pytest.raises(ValueError):
            generate_agent_wrapper(p, has_agents=False)

    def test_generic_wrapper_has_timeout_support(self) -> None:
        """All non-Claude wrappers support --terok-timeout."""
        for name in HEADLESS_PROVIDERS:
            if name == "claude":
                continue
            wrapper = _provider_wrapper(name)
            assert "--terok-timeout" in wrapper, f"{name} missing timeout support"

    def test_session_resume_uses_explicit_id(self) -> None:
        """Providers with session_file use --session/--resume with explicit ID."""
        for name in ("vibe", "opencode", "blablador", "kisski"):
            p = HEADLESS_PROVIDERS[name]
            wrapper = _provider_wrapper(name)
            assert p.resume_flag in wrapper, f"{name} missing resume flag"
            assert f"cat {CONTAINER_TEROK_DIR}/{p.session_file}" in wrapper


class TestResolveProviderValue:
    """Tests for resolve_provider_value() config resolution."""

    def test_flat_string_value(self) -> None:
        """Flat string value is returned for any provider."""
        config = {"model": "opus"}
        assert resolve_provider_value("model", config, "claude") == "opus"
        assert resolve_provider_value("model", config, "codex") == "opus"

    def test_per_provider_dict(self) -> None:
        """Per-provider dict returns provider-specific value."""
        config = {"model": {"claude": "opus", "codex": "o3"}}
        assert resolve_provider_value("model", config, "claude") == "opus"
        assert resolve_provider_value("model", config, "codex") == "o3"

    def test_per_provider_dict_with_default(self) -> None:
        """Per-provider dict falls back to _default for unlisted providers."""
        config = {"model": {"claude": "opus", "_default": "fast"}}
        assert resolve_provider_value("model", config, "codex") == "fast"

    def test_missing_key_returns_none(self) -> None:
        """Missing key returns None."""
        assert resolve_provider_value("model", {}, "claude") is None


class TestApplyProviderConfig:
    """Tests for apply_provider_config() best-effort feature mapping."""

    def test_model_from_config(self) -> None:
        """Model value is read from config when no CLI override."""
        p = HEADLESS_PROVIDERS["claude"]
        pcfg = apply_provider_config(p, {"model": "opus"})
        assert pcfg.model == "opus"

    def test_timeout_default(self) -> None:
        """Missing timeout defaults to 1800."""
        p = HEADLESS_PROVIDERS["claude"]
        pcfg = apply_provider_config(p, {})
        assert pcfg.timeout == 1800

    def test_max_turns_unsupported_injects_prompt(self) -> None:
        """Provider without max_turns_flag gets prompt injection + warning."""
        p = HEADLESS_PROVIDERS["codex"]
        pcfg = apply_provider_config(p, {"max_turns": 30})
        assert pcfg.max_turns is None
        assert "30 steps" in pcfg.prompt_extra


class TestGenerateAllWrappers:
    """Tests for generate_all_wrappers() multi-provider file."""

    def test_all_providers_in_output(self) -> None:
        """Output contains wrapper functions for all providers."""
        wrapper = _all_wrappers()
        for name, p in HEADLESS_PROVIDERS.items():
            assert f"{p.binary}()" in wrapper, f"Missing wrapper for {name}"

    def test_all_wrappers_valid_bash_syntax(self) -> None:
        """Combined wrapper output passes bash -n syntax check."""
        import shutil
        import subprocess

        if shutil.which("bash") is None:
            pytest.skip("bash is required for wrapper syntax validation")
        wrapper = _all_wrappers(has_agents=True)
        result = subprocess.run(["bash", "-n"], input=wrapper, capture_output=True, text=True)
        assert result.returncode == 0, f"bash syntax error:\n{result.stderr}"

    def test_collect_all_auto_approve_env(self) -> None:
        """The merged auto-approve env map contains all provider env vars."""
        merged = collect_all_auto_approve_env()
        assert merged["OPENCODE_PERMISSION"] == '{"*":"allow"}'
        assert merged["VIBE_AUTO_APPROVE"] == "true"
        assert merged["COPILOT_ALLOW_ALL"] == "true"
