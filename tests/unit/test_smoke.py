# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests — verify the package is importable and wired correctly."""

import importlib


class TestPackageImport:
    """Basic import checks."""

    def test_import_terok_agent(self) -> None:
        """Package root is importable."""
        mod = importlib.import_module("terok_agent")
        assert hasattr(mod, "__version__")

    def test_import_cli(self) -> None:
        """CLI module is importable."""
        mod = importlib.import_module("terok_agent.cli")
        assert hasattr(mod, "main")

    def test_version_is_string(self) -> None:
        """__version__ is a non-empty string."""
        from terok_agent import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_headless_providers_registry(self) -> None:
        """HEADLESS_PROVIDERS registry contains expected agents."""
        from terok_agent import HEADLESS_PROVIDERS

        assert len(HEADLESS_PROVIDERS) >= 7
        assert "claude" in HEADLESS_PROVIDERS
        assert "codex" in HEADLESS_PROVIDERS

    def test_auth_providers_registry(self) -> None:
        """AUTH_PROVIDERS registry contains expected providers."""
        from terok_agent import AUTH_PROVIDERS

        assert "claude" in AUTH_PROVIDERS
        assert "codex" in AUTH_PROVIDERS

    def test_get_provider_resolves(self) -> None:
        """get_provider resolves explicit name."""
        from terok_agent import get_provider

        p = get_provider("claude")
        assert p.name == "claude"
        assert p.binary == "claude"

    def test_get_provider_default_fallback(self) -> None:
        """get_provider falls back to claude when no default set."""
        from terok_agent import get_provider

        p = get_provider(None)
        assert p.name == "claude"

    def test_config_stack_resolve(self) -> None:
        """ConfigStack merges scopes correctly."""
        from terok_agent import ConfigScope, ConfigStack

        stack = ConfigStack()
        stack.push(ConfigScope("base", None, {"model": "haiku", "timeout": 60}))
        stack.push(ConfigScope("override", None, {"model": "opus"}))
        result = stack.resolve()
        assert result["model"] == "opus"
        assert result["timeout"] == 60

    def test_resolve_provider_value_flat(self) -> None:
        """resolve_provider_value returns flat values."""
        from terok_agent import resolve_provider_value

        assert resolve_provider_value("model", {"model": "opus"}, "claude") == "opus"

    def test_resolve_provider_value_per_provider(self) -> None:
        """resolve_provider_value picks provider-specific values."""
        from terok_agent import resolve_provider_value

        config = {"model": {"claude": "opus", "codex": "o3"}}
        assert resolve_provider_value("model", config, "claude") == "opus"
        assert resolve_provider_value("model", config, "codex") == "o3"

    def test_bundled_instructions(self) -> None:
        """Bundled default instructions are loadable and non-empty."""
        from terok_agent import bundled_default_instructions

        text = bundled_default_instructions()
        assert len(text) > 100
        assert "container" in text.lower()

    def test_resolve_instructions_default(self) -> None:
        """resolve_instructions returns bundled default when config is empty."""
        from terok_agent import bundled_default_instructions, resolve_instructions

        result = resolve_instructions({}, "claude")
        assert result == bundled_default_instructions()

    def test_import_util(self) -> None:
        """Vendored _util module is importable."""
        mod = importlib.import_module("terok_agent._util")
        assert hasattr(mod, "ensure_dir")
        assert hasattr(mod, "yaml_load")
