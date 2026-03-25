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

    def test_glab_uses_gl_prefix(self) -> None:
        """GitLab route uses 'gl' as the path prefix (not 'glab')."""
        routes = json.loads(get_registry().generate_routes_json())
        assert "gl" in routes
        assert "glab" not in routes
