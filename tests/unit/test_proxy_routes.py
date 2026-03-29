# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for credential proxy route parsing, routes.json, and CLI handlers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_agent.roster import get_roster


class TestProxyRoutesParsed:
    """Verify credential_proxy YAML sections are parsed into the roster."""

    def test_claude_route_exists(self) -> None:
        """Claude has a proxy route with Anthropic upstream."""
        reg = get_roster()
        route = reg.proxy_routes.get("claude")
        assert route is not None
        assert route.route_prefix == "claude"
        assert route.upstream == "https://api.anthropic.com"
        assert route.auth_header == "dynamic"
        assert "ANTHROPIC_API_KEY" in route.phantom_env
        assert route.base_url_env == "ANTHROPIC_BASE_URL"

    def test_codex_route_exists(self) -> None:
        """Codex has a proxy route with OpenAI upstream."""
        route = get_roster().proxy_routes.get("codex")
        assert route is not None
        assert route.upstream == "https://api.openai.com"
        assert "OPENAI_API_KEY" in route.phantom_env

    def test_gh_route_exists(self) -> None:
        """GitHub CLI has a proxy route with token-style auth."""
        route = get_roster().proxy_routes.get("gh")
        assert route is not None
        assert route.auth_prefix == "token "
        assert route.upstream == "https://api.github.com"

    def test_glab_route_exists(self) -> None:
        """GitLab CLI has a proxy route with PRIVATE-TOKEN header."""
        route = get_roster().proxy_routes.get("glab")
        assert route is not None
        assert route.auth_header == "PRIVATE-TOKEN"
        assert route.auth_prefix == ""
        assert route.route_prefix == "gl"

    def test_opencode_agents_have_routes(self) -> None:
        """Blablador and KISSKI have proxy routes."""
        reg = get_roster()
        for name in ("blablador", "kisski"):
            route = reg.proxy_routes.get(name)
            assert route is not None, f"{name} missing proxy route"
            assert route.credential_type == "api_key"

    def test_copilot_has_no_route(self) -> None:
        """Copilot has no credential_proxy section (tier-3, no base URL support)."""
        assert get_roster().proxy_routes.get("copilot") is None

    def test_claude_has_oauth_refresh(self) -> None:
        """Claude has oauth_refresh config for proactive token refresh."""
        route = get_roster().proxy_routes.get("claude")
        assert route is not None
        assert route.oauth_refresh is not None
        assert "token_url" in route.oauth_refresh
        assert "client_id" in route.oauth_refresh

    def test_codex_has_no_oauth_refresh(self) -> None:
        """Codex does not yet have oauth_refresh configured."""
        route = get_roster().proxy_routes.get("codex")
        assert route is not None
        assert route.oauth_refresh is None


class TestGenerateRoutesJson:
    """Verify routes.json generation."""

    def test_generates_valid_json(self) -> None:
        """generate_routes_json() produces parseable JSON with expected keys."""
        routes_json = get_roster().generate_routes_json()
        routes = json.loads(routes_json)
        assert "claude" in routes
        assert routes["claude"]["upstream"] == "https://api.anthropic.com"
        assert routes["claude"]["auth_header"] == "dynamic"

    def test_all_routes_have_upstream(self) -> None:
        """Every route in the JSON has an upstream field."""
        routes = json.loads(get_roster().generate_routes_json())
        for prefix, cfg in routes.items():
            assert "upstream" in cfg, f"Route '{prefix}' missing upstream"

    def test_glab_keyed_by_provider_name(self) -> None:
        """GitLab route is keyed by provider name 'glab'."""
        routes = json.loads(get_roster().generate_routes_json())
        assert "glab" in routes

    def test_claude_routes_json_includes_oauth_refresh(self) -> None:
        """Claude's routes.json entry includes oauth_refresh config."""
        routes = json.loads(get_roster().generate_routes_json())
        assert "oauth_refresh" in routes["claude"]
        assert routes["claude"]["oauth_refresh"]["client_id"]

    def test_gh_routes_json_omits_oauth_refresh(self) -> None:
        """Providers without oauth_refresh omit it from routes.json."""
        routes = json.loads(get_roster().generate_routes_json())
        assert "oauth_refresh" not in routes["gh"]


class TestScanLeakedCredentials:
    """Verify scan_leaked_credentials detects real secrets in shared mounts."""

    def test_empty_when_no_files(self, tmp_path) -> None:
        """Returns empty list when no credential files exist."""
        from terok_agent.proxy_commands import scan_leaked_credentials

        assert scan_leaked_credentials(tmp_path) == []

    def test_detects_nonempty_credential_file(self, tmp_path) -> None:
        """Returns (provider, path) when a credential file is present and non-empty."""
        from terok_agent import get_roster
        from terok_agent.proxy_commands import scan_leaked_credentials

        roster = get_roster()
        auth = roster.auth_providers.get("claude")
        route = roster.proxy_routes.get("claude")
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
        from terok_agent import get_roster
        from terok_agent.proxy_commands import scan_leaked_credentials

        roster = get_roster()
        auth = roster.auth_providers["claude"]
        route = roster.proxy_routes["claude"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        (cred_dir / route.credential_file).write_text("")

        assert scan_leaked_credentials(tmp_path) == []

    def test_skips_providers_without_credential_file(self, tmp_path, monkeypatch) -> None:
        """Providers whose proxy route has no credential_file are skipped."""
        from unittest.mock import MagicMock

        from terok_agent.proxy_commands import scan_leaked_credentials

        # Mock a roster with a provider that has a proxy route but no credential_file
        mock_roster = MagicMock()
        mock_route = MagicMock()
        mock_route.credential_file = ""
        mock_roster.proxy_routes = {"fake-provider": mock_route}
        mock_roster.auth_providers = {"fake-provider": MagicMock(host_dir_name="_fake")}
        monkeypatch.setattr("terok_agent.roster.get_roster", lambda: mock_roster)

        assert scan_leaked_credentials(tmp_path) == []

    def test_clean_removes_leaked_files(self, tmp_path) -> None:
        """The clean handler removes detected credential files."""
        from unittest.mock import patch

        from terok_agent import get_roster
        from terok_agent.proxy_commands import _handle_clean

        roster = get_roster()
        auth = roster.auth_providers["claude"]
        route = roster.proxy_routes["claude"]

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


class TestProxyCommandHandlers:
    """Verify proxy CLI command handlers."""

    @patch("terok_sandbox.start_proxy")
    @patch("terok_agent.proxy_commands._ensure_routes")
    @patch("terok_sandbox.is_proxy_running", return_value=False)
    def test_start_generates_routes_and_starts(self, _running, _routes, _start, capsys) -> None:
        """start generates routes then starts the daemon."""
        from terok_agent.proxy_commands import _handle_start

        _handle_start()
        _routes.assert_called_once()
        _start.assert_called_once()
        assert "started" in capsys.readouterr().out

    @patch("terok_sandbox.is_proxy_running", return_value=True)
    def test_start_already_running_exits(self, _running) -> None:
        """start exits if proxy is already running."""
        from terok_agent.proxy_commands import _handle_start

        with pytest.raises(SystemExit):
            _handle_start()

    @patch("terok_sandbox.stop_proxy")
    @patch("terok_sandbox.is_proxy_running", return_value=True)
    def test_stop_stops_daemon(self, _running, _stop, capsys) -> None:
        """stop calls stop_proxy when running."""
        from terok_agent.proxy_commands import _handle_stop

        _handle_stop()
        _stop.assert_called_once()
        assert "stopped" in capsys.readouterr().out

    @patch("terok_sandbox.is_proxy_running", return_value=False)
    def test_stop_not_running(self, _running, capsys) -> None:
        """stop prints info when not running."""
        from terok_agent.proxy_commands import _handle_stop

        _handle_stop()
        assert "not running" in capsys.readouterr().out

    @patch("terok_agent.proxy_commands.scan_leaked_credentials", return_value=[])
    @patch("terok_sandbox.SandboxConfig")
    @patch("terok_sandbox.is_proxy_systemd_available", return_value=False)
    @patch("terok_sandbox.get_proxy_status")
    def test_status_prints_info(self, mock_status, _sd, _cfg, _scan, capsys) -> None:
        """status prints formatted proxy info."""
        mock_status.return_value = MagicMock(
            mode="daemon",
            running=True,
            socket_path="/run/proxy.sock",
            db_path="/data/creds.db",
            routes_path="/data/routes.json",
            routes_configured=3,
            credentials_stored=("claude", "gh"),
        )
        from terok_agent.proxy_commands import _handle_status

        _handle_status()
        out = capsys.readouterr().out
        assert "running" in out
        assert "claude" in out

    @patch("terok_sandbox.install_proxy_systemd")
    @patch("terok_agent.proxy_commands._ensure_routes")
    @patch("terok_sandbox.is_proxy_systemd_available", return_value=True)
    def test_install_generates_routes_and_installs(self, _sd, _routes, _install, capsys) -> None:
        """install generates routes then installs systemd units."""
        from terok_agent.proxy_commands import _handle_install

        _handle_install()
        _routes.assert_called_once()
        _install.assert_called_once()
        assert "installed" in capsys.readouterr().out

    @patch("terok_sandbox.is_proxy_systemd_available", return_value=False)
    def test_install_no_systemd_exits(self, _sd) -> None:
        """install exits when systemd unavailable."""
        from terok_agent.proxy_commands import _handle_install

        with pytest.raises(SystemExit):
            _handle_install()

    @patch("terok_sandbox.uninstall_proxy_systemd")
    @patch("terok_sandbox.is_proxy_systemd_available", return_value=True)
    def test_uninstall_removes_units(self, _sd, _uninstall, capsys) -> None:
        """uninstall removes systemd units."""
        from terok_agent.proxy_commands import _handle_uninstall

        _handle_uninstall()
        _uninstall.assert_called_once()
        assert "removed" in capsys.readouterr().out

    @patch("terok_sandbox.is_proxy_systemd_available", return_value=False)
    def test_uninstall_no_systemd_exits(self, _sd) -> None:
        """uninstall exits when systemd unavailable."""
        from terok_agent.proxy_commands import _handle_uninstall

        with pytest.raises(SystemExit):
            _handle_uninstall()

    @patch("terok_agent.proxy_commands._ensure_routes", return_value=Path("/tmp/routes.json"))
    def test_routes_prints_path(self, _routes, capsys) -> None:
        """routes prints the written path."""
        from terok_agent.proxy_commands import _handle_routes

        _handle_routes()
        assert "routes.json" in capsys.readouterr().out

    @patch(
        "terok_agent.proxy_commands.scan_leaked_credentials",
        return_value=[("claude", Path("/envs/_claude-config/.credentials.json"))],
    )
    @patch("terok_sandbox.SandboxConfig")
    @patch("terok_sandbox.is_proxy_systemd_available", return_value=False)
    @patch("terok_sandbox.get_proxy_status")
    def test_status_shows_leak_warning(self, mock_status, _sd, _cfg, _scan, capsys) -> None:
        """status shows WARNING when leaked credentials detected."""
        mock_status.return_value = MagicMock(
            mode="daemon",
            running=True,
            socket_path="/run/proxy.sock",
            db_path="/data/creds.db",
            routes_path="/data/routes.json",
            routes_configured=3,
            credentials_stored=("claude",),
        )
        from terok_agent.proxy_commands import _handle_status

        _handle_status()
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "claude" in out
        assert "clean" in out


class TestEnsureProxyRoutes:
    """Verify ensure_proxy_routes writes routes.json to disk."""

    def test_writes_routes_json(self, tmp_path, monkeypatch):
        """ensure_proxy_routes() creates a valid routes.json file."""
        from unittest.mock import MagicMock

        import terok_sandbox

        mock_cfg = MagicMock()
        mock_cfg.proxy_routes_path = tmp_path / "proxy" / "routes.json"
        monkeypatch.setattr(terok_sandbox, "SandboxConfig", lambda: mock_cfg)

        from terok_agent.roster import ensure_proxy_routes

        path = ensure_proxy_routes()

        assert path == mock_cfg.proxy_routes_path
        assert path.is_file()
        routes = json.loads(path.read_text())
        # Should have at least claude route from the YAML roster
        assert "claude" in routes
        assert "upstream" in routes["claude"]
