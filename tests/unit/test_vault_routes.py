# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for vault route parsing, routes.json, and CLI handlers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from terok_executor.roster import AgentRoster


class TestVaultRoutesParsed:
    """Verify provider endpoints + agent bindings project into provider-keyed routes."""

    def test_anthropic_route_exists(self) -> None:
        """The anthropic provider (claude's default) routes with OAuth + dynamic header."""
        reg = AgentRoster.shared()
        route = reg.vault_routes.get("anthropic")
        assert route is not None
        assert route.upstream == "https://api.anthropic.com"
        assert route.auth_header == "dynamic"
        assert route.oauth_extra_headers == {"anthropic-beta": "oauth-2025-04-20"}
        assert route.token_env["_default"] == "ANTHROPIC_API_KEY"
        assert route.token_env["oauth"] == "CLAUDE_CODE_OAUTH_TOKEN"
        assert route.base_url_env == "ANTHROPIC_BASE_URL"
        assert route.socket_env == "ANTHROPIC_UNIX_SOCKET"

    def test_openai_route_exists(self) -> None:
        """The openai provider (codex's default) routes with OpenAI + ChatGPT upstreams."""
        route = AgentRoster.shared().vault_routes.get("openai")
        assert route is not None
        assert route.upstream == "https://api.openai.com"
        assert route.path_upstreams == {"/backend-api/": "https://chatgpt.com"}
        assert route.oauth_extra_headers == {}
        assert route.shared_config_patch is not None
        assert route.shared_config_patch["file"] == "config.toml"

    def test_github_route_exists(self) -> None:
        """The github provider (gh's default) routes with token-style auth."""
        route = AgentRoster.shared().vault_routes.get("github")
        assert route is not None
        assert route.auth_prefix == "token "
        assert route.upstream == "https://api.github.com"

    def test_gitlab_route_exists(self) -> None:
        """The gitlab provider (glab's default) routes with a PRIVATE-TOKEN header."""
        route = AgentRoster.shared().vault_routes.get("gitlab")
        assert route is not None
        assert route.auth_header == "PRIVATE-TOKEN"
        assert route.auth_prefix == ""

    def test_api_key_only_providers_use_default_token_env(self) -> None:
        """Providers without OAuth support map only ``_default`` in token_env."""
        for name in ("mistral", "blablador", "kisski"):
            route = AgentRoster.shared().vault_routes.get(name)
            assert route is not None, f"{name} missing vault route"
            assert list(route.token_env) == ["_default"], f"{name} should only map _default"
            assert route.socket_env == "", f"{name} should have no socket_env"

    def test_opencode_provider_routes(self) -> None:
        """Blablador and KISSKI providers have routes with api_key credentials."""
        reg = AgentRoster.shared()
        for name in ("blablador", "kisski"):
            route = reg.vault_routes.get(name)
            assert route is not None, f"{name} missing vault route"
            assert route.credential_type == "api_key"

    def test_copilot_has_no_route(self) -> None:
        """Copilot binds no provider (tier-3, no base URL support)."""
        # No route is keyed under any copilot-ish name, and copilot has no binding.
        assert AgentRoster.shared().agents["copilot"].provider_binding is None

    def test_anthropic_has_oauth_refresh(self) -> None:
        """The anthropic provider carries oauth_refresh for proactive token refresh."""
        route = AgentRoster.shared().vault_routes.get("anthropic")
        assert route is not None
        assert route.oauth_refresh is not None
        assert "token_url" in route.oauth_refresh
        assert "client_id" in route.oauth_refresh

    def test_openai_has_oauth_refresh(self) -> None:
        """The openai provider has an oauth_refresh block so the vault rotates tokens."""
        route = AgentRoster.shared().vault_routes.get("openai")
        assert route is not None
        assert route.oauth_refresh is not None
        assert route.oauth_refresh["token_url"] == "https://auth.openai.com/oauth/token"
        assert route.oauth_refresh["client_id"] == "app_EMoamEEZ73f0CkXaXp7hrann"


class TestSharedDomain:
    """Verify the provider ``shared_domain`` flag flows through to the route."""

    def test_default_is_false(self) -> None:
        """API-only upstreams leave the flag unset."""
        roster = AgentRoster.shared()
        for name in (
            "anthropic",
            "openai",
            "github",
            "mistral",
            "blablador",
            "kisski",
            "openrouter",
        ):
            route = roster.vault_routes[name]
            assert route.shared_domain is False, f"{name} should not be shared_domain"

    def test_gitlab_is_shared_domain(self) -> None:
        """gitlab.com hosts both API and ``git push`` traffic."""
        assert AgentRoster.shared().vault_routes["gitlab"].shared_domain is True

    def test_sonarcloud_is_shared_domain(self) -> None:
        """sonarcloud.io hosts API + project pages + docs + badges."""
        assert AgentRoster.shared().vault_routes["sonarcloud"].shared_domain is True


class TestGenerateRoutesJson:
    """Verify routes.json generation (now keyed by clean provider names)."""

    def test_generates_valid_json(self) -> None:
        """generate_routes_json() produces parseable JSON with expected keys."""
        routes_json = AgentRoster.shared().generate_routes_json()
        routes = json.loads(routes_json)
        assert "anthropic" in routes
        assert routes["anthropic"]["upstream"] == "https://api.anthropic.com"
        assert routes["anthropic"]["auth_header"] == "dynamic"
        assert routes["anthropic"]["oauth_extra_headers"] == {"anthropic-beta": "oauth-2025-04-20"}
        assert routes["openai"]["path_upstreams"] == {"/backend-api/": "https://chatgpt.com"}
        assert routes["openai"]["oauth_credential_headers"] == {"ChatGPT-Account-ID": "account_id"}
        assert "oauth_extra_headers" not in routes["openai"]

    def test_all_routes_have_upstream(self) -> None:
        """Every route in the JSON has an upstream field."""
        routes = json.loads(AgentRoster.shared().generate_routes_json())
        for prefix, cfg in routes.items():
            assert "upstream" in cfg, f"Route '{prefix}' missing upstream"

    def test_gitlab_keyed_by_provider_name(self) -> None:
        """The GitLab route is keyed by the clean provider name 'gitlab'."""
        routes = json.loads(AgentRoster.shared().generate_routes_json())
        assert "gitlab" in routes
        assert "glab" not in routes

    def test_anthropic_routes_json_includes_oauth_refresh(self) -> None:
        """The anthropic routes.json entry includes oauth_refresh config."""
        routes = json.loads(AgentRoster.shared().generate_routes_json())
        assert "oauth_refresh" in routes["anthropic"]
        assert routes["anthropic"]["oauth_refresh"]["client_id"]

    def test_github_routes_json_omits_oauth_refresh(self) -> None:
        """Providers without oauth_refresh omit it from routes.json."""
        routes = json.loads(AgentRoster.shared().generate_routes_json())
        assert "oauth_refresh" not in routes["github"]


class TestScanLeakedCredentials:
    """Verify scan_leaked_credentials detects real secrets in shared mounts."""

    def test_empty_when_no_files(self, tmp_path) -> None:
        """Returns empty list when no credential files exist."""
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        assert scan_leaked_credentials(tmp_path) == []

    def test_detects_nonempty_credential_file(self, tmp_path) -> None:
        """Returns (provider, path) when a credential file is present and non-empty."""
        from terok_executor import AgentRoster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = AgentRoster.shared()
        auth = roster.auth_providers.get("claude")
        route = roster.vault_routes.get("anthropic")
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
        from terok_executor import AgentRoster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = AgentRoster.shared()
        auth = roster.auth_providers["claude"]
        route = roster.vault_routes["anthropic"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        (cred_dir / route.credential_file).write_text("")

        assert scan_leaked_credentials(tmp_path) == []

    def test_skips_mounts_without_credential_file(self, tmp_path, monkeypatch) -> None:
        """Mounts with no credential_file (opencode state dirs, …) are skipped."""
        from unittest.mock import MagicMock

        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        # Mock a roster whose only mount carries no credential file.
        mock_roster = MagicMock()
        mock_mount = MagicMock()
        mock_mount.credential_file = ""
        mock_roster.mounts = [mock_mount]
        monkeypatch.setattr("terok_executor.roster.loader._shared_roster", lambda: mock_roster)

        assert scan_leaked_credentials(tmp_path) == []

    def test_clean_removes_leaked_files(self, tmp_path) -> None:
        """The clean handler removes detected credential files."""
        from unittest.mock import patch

        from terok_executor import AgentRoster
        from terok_executor.credentials.vault_commands import _handle_clean

        roster = AgentRoster.shared()
        auth = roster.auth_providers["claude"]
        route = roster.vault_routes["anthropic"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred_file = cred_dir / route.credential_file
        cred_file.write_text('{"secret": true}')

        with patch("terok_executor.paths.mounts_dir", return_value=tmp_path):
            _handle_clean()

        assert not cred_file.exists()

    def test_clean_no_files(self, capsys) -> None:
        """Clean reports nothing when no leaked files found."""
        from pathlib import Path
        from unittest.mock import patch

        from terok_executor.credentials.vault_commands import _handle_clean

        with patch("terok_executor.paths.mounts_dir", return_value=Path("/nonexistent")):
            _handle_clean()

        assert "No leaked" in capsys.readouterr().out


class TestGlabConfigScan:
    """glab's ``config.yml`` mixes settings and credentials — only a token is a leak.

    The shared mount is writable by design (glab rewrites the file on
    startup), so a non-empty settings-only file is the expected steady
    state and must not be reported as a leaked credential.
    """

    def _write_glab_config(self, tmp_path: Path, content: str) -> Path:
        """Materialise *content* as glab's credential file in a mock mount tree."""
        from terok_executor import AgentRoster

        roster = AgentRoster.shared()
        mount = next(m for m in roster.mounts if m.provider == "glab" and m.credential_file)
        cred_dir = tmp_path / mount.host_dir
        cred_dir.mkdir(parents=True)
        cred_file = cred_dir / mount.credential_file
        cred_file.write_text(content)
        return cred_file

    def test_settings_only_config_not_flagged(self, tmp_path: Path) -> None:
        """A config.yml holding only glab settings is benign."""
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        self._write_glab_config(
            tmp_path,
            'last_seen_version: v1.107.0\nlast_update_check_timestamp: "2026-07-13T22:05:22+02:00"\n',
        )

        assert "glab" not in [p for p, _ in scan_leaked_credentials(tmp_path)]

    def test_config_with_token_flagged(self, tmp_path: Path) -> None:
        """A config.yml carrying a hosts.<host>.token entry is a real leak."""
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        cred_file = self._write_glab_config(
            tmp_path,
            "git_protocol: ssh\nhosts:\n  gitlab.com:\n    token: glpat-real-secret\n",
        )

        assert ("glab", cred_file) in scan_leaked_credentials(tmp_path)

    def test_unparseable_config_flagged(self, tmp_path: Path) -> None:
        """A config.yml the scanner cannot parse is treated as a leak."""
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        cred_file = self._write_glab_config(tmp_path, "hosts: [not, a, mapping]\n")

        assert ("glab", cred_file) in scan_leaked_credentials(tmp_path)


class TestTokenlessGlabConfigFile:
    """Verify _is_tokenless_glab_config_file classifies vendor files correctly."""

    def test_tokenless_config_is_benign(self, tmp_path: Path) -> None:
        """Settings-only config (no hosts section) reads as tokenless."""
        from terok_executor.credentials.vault_commands import _is_tokenless_glab_config_file

        cred_file = tmp_path / "config.yml"
        cred_file.write_text("check_update: true\n")

        assert _is_tokenless_glab_config_file(cred_file) is True

    def test_empty_host_block_is_benign(self, tmp_path: Path) -> None:
        """A hosts entry without a token (e.g. api_protocol only) reads as tokenless."""
        from terok_executor.credentials.vault_commands import _is_tokenless_glab_config_file

        cred_file = tmp_path / "config.yml"
        cred_file.write_text("hosts:\n  gitlab.com:\n    api_protocol: https\n")

        assert _is_tokenless_glab_config_file(cred_file) is True

    def test_token_is_not_benign(self, tmp_path: Path) -> None:
        """Any non-empty hosts.<host>.token makes the file a credential."""
        from terok_executor.credentials.vault_commands import _is_tokenless_glab_config_file

        cred_file = tmp_path / "config.yml"
        cred_file.write_text("hosts:\n  gitlab.example.com:\n    token: glpat-abc\n")

        assert _is_tokenless_glab_config_file(cred_file) is False

    def test_missing_file_is_not_benign(self, tmp_path: Path) -> None:
        """A missing file cannot be vouched for."""
        from terok_executor.credentials.vault_commands import _is_tokenless_glab_config_file

        assert _is_tokenless_glab_config_file(tmp_path / "config.yml") is False


class TestVaultCommandHandlers:
    """Verify executor's vault CLI command handlers.

    After the per-container-supervisor refactor the vault is no
    longer a host-side daemon — there is no ``start`` / ``stop`` /
    ``install`` / ``uninstall`` / ``status`` lifecycle.  Sandbox
    keeps only ``unlock`` / ``lock`` / ``passphrase``, and executor
    contributes two file-level verbs (``routes`` and ``clean``).
    """

    @patch(
        "terok_executor.credentials.vault_commands._ensure_routes",
        return_value=Path("/tmp/routes.json"),
    )
    def test_routes_prints_path(self, _routes, capsys) -> None:
        """routes prints the written path."""
        from terok_executor.credentials.vault_commands import _handle_routes

        _handle_routes()
        assert "routes.json" in capsys.readouterr().out

    @patch(
        "terok_executor.credentials.vault_commands.scan_leaked_credentials",
        return_value=[],
    )
    def test_clean_reports_when_no_leaks(self, _scan, capsys) -> None:
        """clean prints a friendly no-op message when the scan is empty."""
        from terok_executor.credentials.vault_commands import _handle_clean

        _handle_clean()
        assert "No leaked" in capsys.readouterr().out


class TestInjectedCredentialsFile:
    """Verify _is_injected_credentials_file detects phantom vs real credentials."""

    def test_recognises_injected_file(self, tmp_path: Path) -> None:
        """Correctly identifies a terok-injected .credentials.json."""
        from terok_executor.credentials.auth import PHANTOM_CREDENTIALS_MARKER
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        cred = {
            "claudeAiOauth": {
                "accessToken": PHANTOM_CREDENTIALS_MARKER,
                "refreshToken": "",
                "scopes": "user:inference user:profile",
                "subscriptionType": "max",
            }
        }
        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text(json.dumps(cred))
        assert _is_injected_credentials_file(cred_file) is True

    def test_rejects_real_credentials(self, tmp_path: Path) -> None:
        """Real OAuth tokens are NOT identified as injected."""
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        cred = {"claudeAiOauth": {"accessToken": "sk-ant-real-token", "refreshToken": "rt-real"}}
        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text(json.dumps(cred))
        assert _is_injected_credentials_file(cred_file) is False

    def test_rejects_phantom_token_with_refresh(self, tmp_path: Path) -> None:
        """Phantom accessToken with a non-empty refreshToken is suspicious -- flag it."""
        from terok_executor.credentials.auth import PHANTOM_CREDENTIALS_MARKER
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        cred = {
            "claudeAiOauth": {
                "accessToken": PHANTOM_CREDENTIALS_MARKER,
                "refreshToken": "rt-leaked-somehow",
            }
        }
        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text(json.dumps(cred))
        assert _is_injected_credentials_file(cred_file) is False

    def test_handles_malformed_json(self, tmp_path: Path) -> None:
        """Malformed JSON falls through to False (treat as potential leak)."""
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text("{not valid json")
        assert _is_injected_credentials_file(cred_file) is False

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        """Missing file returns False."""
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        assert _is_injected_credentials_file(tmp_path / "nonexistent.json") is False

    def test_handles_non_dict_oauth_section(self, tmp_path: Path) -> None:
        """Non-dict claudeAiOauth returns False."""
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text(json.dumps({"claudeAiOauth": "not a dict"}))
        assert _is_injected_credentials_file(cred_file) is False


class TestScanSkipsInjectedFile:
    """Verify scan_leaked_credentials skips terok-injected .credentials.json."""

    def test_skips_injected_credentials(self, tmp_path: Path) -> None:
        """Injected phantom credentials are NOT flagged as leaked."""
        from terok_executor import AgentRoster
        from terok_executor.credentials.auth import PHANTOM_CREDENTIALS_MARKER
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = AgentRoster.shared()
        auth = roster.auth_providers["claude"]
        route = roster.vault_routes["anthropic"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred = {
            "claudeAiOauth": {
                "accessToken": PHANTOM_CREDENTIALS_MARKER,
                "refreshToken": "",
                "subscriptionType": "max",
            }
        }
        (cred_dir / route.credential_file).write_text(json.dumps(cred))

        assert scan_leaked_credentials(tmp_path) == []

    def test_still_detects_real_credentials(self, tmp_path: Path) -> None:
        """Real OAuth tokens are still flagged even when file structure matches."""
        from terok_executor import AgentRoster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = AgentRoster.shared()
        auth = roster.auth_providers["claude"]
        route = roster.vault_routes["anthropic"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred = {"claudeAiOauth": {"accessToken": "sk-ant-real", "refreshToken": "rt-real"}}
        (cred_dir / route.credential_file).write_text(json.dumps(cred))

        leaked = scan_leaked_credentials(tmp_path)
        assert len(leaked) == 1
        assert leaked[0][0] == "claude"

    def test_skips_injected_codex_auth_json(self, tmp_path: Path) -> None:
        """Injected shared Codex auth.json is NOT flagged as leaked."""
        from terok_sandbox import CODEX_SHARED_OAUTH_MARKER

        from terok_executor import AgentRoster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = AgentRoster.shared()
        auth = roster.auth_providers["codex"]
        route = roster.vault_routes["openai"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred = {
            "tokens": {
                "access_token": CODEX_SHARED_OAUTH_MARKER,
                "refresh_token": CODEX_SHARED_OAUTH_MARKER,
                "id_token": "dummy.dummy.dummy",
            }
        }
        (cred_dir / route.credential_file).write_text(json.dumps(cred))

        assert scan_leaked_credentials(tmp_path) == []

    def test_codex_auth_json_with_live_api_key_is_still_leaked(self, tmp_path: Path) -> None:
        """Marker tokens do not hide a live top-level OPENAI_API_KEY."""
        from terok_sandbox import CODEX_SHARED_OAUTH_MARKER

        from terok_executor import AgentRoster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = AgentRoster.shared()
        auth = roster.auth_providers["codex"]
        route = roster.vault_routes["openai"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred = {
            "OPENAI_API_KEY": "sk-real-leak",
            "tokens": {
                "access_token": CODEX_SHARED_OAUTH_MARKER,
                "refresh_token": CODEX_SHARED_OAUTH_MARKER,
                "id_token": "dummy.dummy.dummy",
            },
        }
        (cred_dir / route.credential_file).write_text(json.dumps(cred))

        assert scan_leaked_credentials(tmp_path) == [("codex", cred_dir / route.credential_file)]

    def test_malformed_codex_auth_json_is_suspicious_not_crashing(self, tmp_path: Path) -> None:
        """Non-object auth.json roots are treated as leaks, not parser crashes."""
        from terok_executor import AgentRoster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = AgentRoster.shared()
        auth = roster.auth_providers["codex"]
        route = roster.vault_routes["openai"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        (cred_dir / route.credential_file).write_text(json.dumps(["not", "an", "object"]))

        assert scan_leaked_credentials(tmp_path) == [("codex", cred_dir / route.credential_file)]


class TestCleanSkipsInjectedFile:
    """Verify the clean handler preserves injected .credentials.json."""

    def test_clean_preserves_injected_file(self, tmp_path: Path) -> None:
        """Clean removes real leaks but preserves injected phantom credentials."""
        from unittest.mock import patch

        from terok_executor import AgentRoster
        from terok_executor.credentials.auth import PHANTOM_CREDENTIALS_MARKER
        from terok_executor.credentials.vault_commands import _handle_clean

        roster = AgentRoster.shared()
        auth = roster.auth_providers["claude"]
        route = roster.vault_routes["anthropic"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred = {
            "claudeAiOauth": {
                "accessToken": PHANTOM_CREDENTIALS_MARKER,
                "refreshToken": "",
            }
        }
        cred_file = cred_dir / route.credential_file
        cred_file.write_text(json.dumps(cred))

        with patch("terok_executor.paths.mounts_dir", return_value=tmp_path):
            _handle_clean()

        # Injected file should still exist
        assert cred_file.is_file()


class TestEnsureVaultRoutes:
    """Verify AgentRoster.ensure_vault_routes writes routes.json to disk."""

    def test_writes_routes_json(self, tmp_path):
        """ensure_vault_routes() creates a valid routes.json file."""
        mock_cfg = MagicMock()
        mock_cfg.routes_path = tmp_path / "proxy" / "routes.json"

        path = AgentRoster.shared().ensure_vault_routes(cfg=mock_cfg)

        assert path == mock_cfg.routes_path
        assert path.is_file()
        routes = json.loads(path.read_text())
        # Should have at least the anthropic route from the provider roster
        assert "anthropic" in routes
        assert "upstream" in routes["anthropic"]

    def test_falls_back_to_default_config(self, tmp_path, monkeypatch):
        """ensure_vault_routes(cfg=None) creates a SandboxConfig with standalone defaults."""
        import terok_sandbox

        mock_cfg = MagicMock()
        mock_cfg.routes_path = tmp_path / "proxy" / "routes.json"
        monkeypatch.setattr(terok_sandbox, "SandboxConfig", lambda: mock_cfg)

        path = AgentRoster.shared().ensure_vault_routes()
        assert path.is_file()


class TestVaultHandlerCfgSignatures:
    """All vault command handlers accept a ``cfg`` keyword argument."""

    def test_all_leaf_handlers_accept_cfg(self) -> None:
        import inspect

        from terok_util import LazyHandler

        from terok_executor.credentials.vault_commands import VAULT_COMMANDS

        vault_group = VAULT_COMMANDS[0]
        for cmd in vault_group.children:
            # Skip the nested ``passphrase`` group; its leaves don't take cfg.
            if cmd.children:
                continue
            # Handlers are wired as opaque LazyHandler("mod:fn"); resolve to
            # the real callable before reading its signature.
            handler = cmd.handler.resolve() if isinstance(cmd.handler, LazyHandler) else cmd.handler
            sig = inspect.signature(handler)
            assert "cfg" in sig.parameters, f"{getattr(handler, '__name__', handler)} missing cfg"


class TestVaultCommandsOverlay:
    """Executor's ``VAULT_COMMANDS`` extends sandbox's vault subtree.

    Sandbox owns the verb registry and argparse schema (``unlock`` /
    ``lock`` / ``passphrase {seal,to-keyring,reveal,acknowledge,
    destroy}``).  Executor appends two file-level verbs (``routes`` /
    ``clean``); every sandbox verb flows through unchanged so new
    sandbox commands reach ``terok-executor vault …`` zero-edit.

    Post-supervisor-refactor: there is no host-side daemon lifecycle
    (no ``start`` / ``stop`` / ``status`` / ``install`` / ``uninstall``
    on either side) — the per-container supervisor handles its own
    spawn-on-start via the terok-sandbox OCI hook.
    """

    def test_sandbox_verbs_pass_through(self) -> None:
        """``unlock`` / ``lock`` and the ``passphrase`` subgroup pass through unchanged."""
        from terok_sandbox.commands import COMMANDS as SANDBOX_COMMANDS

        from terok_executor.credentials.vault_commands import VAULT_COMMANDS

        sandbox_vault = SANDBOX_COMMANDS.find_at(("vault",))
        executor_vault = VAULT_COMMANDS[0]
        sandbox_by_name = {c.name: c for c in sandbox_vault.children}
        executor_by_name = {c.name: c for c in executor_vault.children}
        for verb in ("unlock", "lock"):
            assert executor_by_name[verb].handler is sandbox_by_name[verb].handler
            assert executor_by_name[verb].args == sandbox_by_name[verb].args
        # Nested ``passphrase`` subgroup survives identically — every leaf
        # routes to the sandbox handler.
        sandbox_passphrase = {c.name: c for c in sandbox_by_name["passphrase"].children}
        executor_passphrase = {c.name: c for c in executor_by_name["passphrase"].children}
        for verb in sandbox_passphrase:
            assert executor_passphrase[verb].handler is sandbox_passphrase[verb].handler

    def test_executor_only_verbs_appended(self) -> None:
        """``routes`` and ``clean`` exist in executor's vault group but not sandbox's."""
        from terok_sandbox.commands import COMMANDS as SANDBOX_COMMANDS

        from terok_executor.credentials.vault_commands import VAULT_COMMANDS

        sandbox_names = {c.name for c in SANDBOX_COMMANDS.find_at(("vault",)).children}
        executor_names = {c.name for c in VAULT_COMMANDS[0].children}
        executor_only = executor_names - sandbox_names
        assert executor_only == {"routes", "clean"}

    def test_deep_path_shares_identity_with_shortcut(self) -> None:
        """``terok-executor sandbox vault X`` and ``terok-executor vault X`` resolve to
        the same ``CommandDef`` — the load-bearing property for wraps to apply uniformly."""
        from terok_executor.cli import COMMANDS

        # ``sandbox`` and ``vault`` are lazy top-level roots (name+help
        # placeholders until invoked); resolve them to reach the shared
        # subtree — both source off the one ``SANDBOX_TREE`` instance.
        sandbox_group = COMMANDS.find_at(("sandbox",)).resolve()
        deep = next(child for child in sandbox_group.children if child.name == "vault")
        shortcut = COMMANDS.find_at(("vault",)).resolve()
        assert deep is shortcut

    def test_argparse_wires_both_paths_to_the_same_handler(self) -> None:
        """The argparse parser must reach the same handler from both paths.

        ``find_at`` proves the registry is consistent; argparse wiring
        could still regress (e.g. someone duplicating a CommandDef
        when constructing a parent group).  Build the actual parser
        and confirm both ``vault unlock`` and ``sandbox vault unlock``
        dispatch to the same handler object.
        """
        import argparse

        from terok_executor.cli import COMMANDS

        parser = argparse.ArgumentParser()
        COMMANDS.wire(parser)

        deep_args = parser.parse_args(["sandbox", "vault", "unlock"])
        short_args = parser.parse_args(["vault", "unlock"])
        assert deep_args._cmd is short_args._cmd
        assert deep_args._cmd.handler is short_args._cmd.handler
