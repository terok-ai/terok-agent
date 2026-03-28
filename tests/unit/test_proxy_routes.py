# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for credential proxy route parsing and routes.json generation."""

from __future__ import annotations

import json

from terok_agent.registry import get_registry


class TestProxyRoutesParsed:
    """Verify credential_proxy YAML sections are parsed into the registry."""

    def test_claude_route_exists(self) -> None:
        """Claude has a proxy route with Anthropic upstream."""
        reg = get_registry()
        route = reg.proxy_routes.get("claude")
        assert route is not None
        assert route.route_prefix == "claude"
        assert route.upstream == "https://api.anthropic.com"
        assert route.auth_header == "dynamic"
        assert "ANTHROPIC_API_KEY" in route.phantom_env
        assert route.base_url_env == "ANTHROPIC_BASE_URL"

    def test_codex_route_exists(self) -> None:
        """Codex has a proxy route with OpenAI upstream."""
        route = get_registry().proxy_routes.get("codex")
        assert route is not None
        assert route.upstream == "https://api.openai.com"
        assert "OPENAI_API_KEY" in route.phantom_env

    def test_gh_route_exists(self) -> None:
        """GitHub CLI has a proxy route with token-style auth."""
        route = get_registry().proxy_routes.get("gh")
        assert route is not None
        assert route.auth_prefix == "token "
        assert route.upstream == "https://api.github.com"

    def test_glab_route_exists(self) -> None:
        """GitLab CLI has a proxy route with PRIVATE-TOKEN header."""
        route = get_registry().proxy_routes.get("glab")
        assert route is not None
        assert route.auth_header == "PRIVATE-TOKEN"
        assert route.auth_prefix == ""
        assert route.route_prefix == "gl"

    def test_opencode_agents_have_routes(self) -> None:
        """Blablador and KISSKI have proxy routes."""
        reg = get_registry()
        for name in ("blablador", "kisski"):
            route = reg.proxy_routes.get(name)
            assert route is not None, f"{name} missing proxy route"
            assert route.credential_type == "api_key"

    def test_copilot_has_no_route(self) -> None:
        """Copilot has no credential_proxy section (tier-3, no base URL support)."""
        assert get_registry().proxy_routes.get("copilot") is None

    def test_claude_has_oauth_refresh(self) -> None:
        """Claude has oauth_refresh config for proactive token refresh."""
        route = get_registry().proxy_routes.get("claude")
        assert route is not None
        assert route.oauth_refresh is not None
        assert "token_url" in route.oauth_refresh
        assert "client_id" in route.oauth_refresh

    def test_codex_has_no_oauth_refresh(self) -> None:
        """Codex does not yet have oauth_refresh configured."""
        route = get_registry().proxy_routes.get("codex")
        assert route is not None
        assert route.oauth_refresh is None


class TestGenerateRoutesJson:
    """Verify routes.json generation."""

    def test_generates_valid_json(self) -> None:
        """generate_routes_json() produces parseable JSON with expected keys."""
        routes_json = get_registry().generate_routes_json()
        routes = json.loads(routes_json)
        assert "claude" in routes
        assert routes["claude"]["upstream"] == "https://api.anthropic.com"
        assert routes["claude"]["auth_header"] == "dynamic"

    def test_all_routes_have_upstream(self) -> None:
        """Every route in the JSON has an upstream field."""
        routes = json.loads(get_registry().generate_routes_json())
        for prefix, cfg in routes.items():
            assert "upstream" in cfg, f"Route '{prefix}' missing upstream"

    def test_glab_keyed_by_provider_name(self) -> None:
        """GitLab route is keyed by provider name 'glab'."""
        routes = json.loads(get_registry().generate_routes_json())
        assert "glab" in routes

    def test_claude_routes_json_includes_oauth_refresh(self) -> None:
        """Claude's routes.json entry includes oauth_refresh config."""
        routes = json.loads(get_registry().generate_routes_json())
        assert "oauth_refresh" in routes["claude"]
        assert routes["claude"]["oauth_refresh"]["client_id"]

    def test_gh_routes_json_omits_oauth_refresh(self) -> None:
        """Providers without oauth_refresh omit it from routes.json."""
        routes = json.loads(get_registry().generate_routes_json())
        assert "oauth_refresh" not in routes["gh"]


class TestScanLeakedCredentials:
    """Verify scan_leaked_credentials detects real secrets in shared mounts."""

    def test_empty_when_no_files(self, tmp_path) -> None:
        """Returns empty list when no credential files exist."""
        from terok_agent.proxy_commands import scan_leaked_credentials

        assert scan_leaked_credentials(tmp_path) == []

    def test_detects_nonempty_credential_file(self, tmp_path) -> None:
        """Returns (provider, path) when a credential file is present and non-empty."""
        from terok_agent import get_registry
        from terok_agent.proxy_commands import scan_leaked_credentials

        registry = get_registry()
        auth = registry.auth_providers.get("claude")
        route = registry.proxy_routes.get("claude")
        assert auth is not None and route is not None

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred_file = cred_dir / route.credential_file
        cred_file.write_text('{"claudeAiOauth": {"accessToken": "sk-leaked"}}')

        leaked = scan_leaked_credentials(tmp_path)
        providers = [p for p, _ in leaked]
        assert "claude" in providers

    def test_skips_empty_files(self, tmp_path) -> None:
        """Empty credential files are not flagged."""
        from terok_agent import get_registry
        from terok_agent.proxy_commands import scan_leaked_credentials

        registry = get_registry()
        auth = registry.auth_providers["claude"]
        route = registry.proxy_routes["claude"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        (cred_dir / route.credential_file).write_text("")

        assert scan_leaked_credentials(tmp_path) == []

    def test_skips_providers_without_credential_file(self, tmp_path, monkeypatch) -> None:
        """Providers whose proxy route has no credential_file are skipped."""
        from unittest.mock import MagicMock

        from terok_agent.proxy_commands import scan_leaked_credentials

        # Mock a registry with a provider that has a proxy route but no credential_file
        mock_registry = MagicMock()
        mock_route = MagicMock()
        mock_route.credential_file = ""
        mock_registry.proxy_routes = {"fake-provider": mock_route}
        mock_registry.auth_providers = {"fake-provider": MagicMock(host_dir_name="_fake")}
        monkeypatch.setattr("terok_agent.registry.get_registry", lambda: mock_registry)

        assert scan_leaked_credentials(tmp_path) == []

    def test_clean_removes_leaked_files(self, tmp_path) -> None:
        """The clean handler removes detected credential files."""
        from unittest.mock import patch

        from terok_agent import get_registry
        from terok_agent.proxy_commands import _handle_clean

        registry = get_registry()
        auth = registry.auth_providers["claude"]
        route = registry.proxy_routes["claude"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred_file = cred_dir / route.credential_file
        cred_file.write_text('{"secret": true}')

        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.effective_envs_dir = tmp_path
            _handle_clean()

        assert not cred_file.exists()

    def test_clean_no_files(self, capsys) -> None:
        """Clean reports nothing when no leaked files found."""
        from pathlib import Path
        from unittest.mock import patch

        from terok_agent.proxy_commands import _handle_clean

        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.effective_envs_dir = Path("/nonexistent")
            _handle_clean()

        assert "No leaked" in capsys.readouterr().out


class TestEnsureProxyRoutes:
    """Verify ensure_proxy_routes writes routes.json to disk."""

    def test_writes_routes_json(self, tmp_path, monkeypatch):
        """ensure_proxy_routes() creates a valid routes.json file."""
        from unittest.mock import MagicMock

        import terok_sandbox

        mock_cfg = MagicMock()
        mock_cfg.proxy_routes_path = tmp_path / "proxy" / "routes.json"
        monkeypatch.setattr(terok_sandbox, "SandboxConfig", lambda: mock_cfg)

        from terok_agent.registry import ensure_proxy_routes

        path = ensure_proxy_routes()

        assert path == mock_cfg.proxy_routes_path
        assert path.is_file()
        routes = json.loads(path.read_text())
        # Should have at least claude route from the YAML registry
        assert "claude" in routes
        assert "upstream" in routes["claude"]
