# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the credential capture path in the auth interceptor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from terok_agent.auth import (
    PHANTOM_CREDENTIALS_MARKER,
    _capture_credentials,
    _write_claude_credentials_file,
    store_api_key,
)


class TestCaptureCredentials:
    """Verify _capture_credentials stores extracted credentials in the DB."""

    def test_captures_claude_credentials(self, tmp_path: Path) -> None:
        """Successful extraction stores credentials in the DB."""
        # Create a fake Claude credential file
        cred = {"claudeAiOauth": {"accessToken": "sk-test-123"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))

        db_path = tmp_path / "proxy" / "credentials.db"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials("claude", tmp_path, "default")

        # Verify it's in the DB
        from terok_sandbox import CredentialDB

        db = CredentialDB(db_path)
        stored = db.load_credential("default", "claude")
        db.close()
        assert stored is not None
        assert stored["access_token"] == "sk-test-123"

    def test_captures_json_api_key(self, tmp_path: Path) -> None:
        """API key extraction works for JSON-based providers."""
        (tmp_path / "config.json").write_text(json.dumps({"api_key": "blab-key"}))

        db_path = tmp_path / "proxy" / "credentials.db"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials("blablador", tmp_path, "default")

        from terok_sandbox import CredentialDB

        db = CredentialDB(db_path)
        stored = db.load_credential("default", "blablador")
        db.close()
        assert stored["key"] == "blab-key"

    def test_extraction_failure_prints_warning(self, tmp_path: Path, capsys) -> None:
        """Failed extraction prints a warning mentioning the provider."""
        # Empty dir — no credential file to extract
        _capture_credentials("claude", tmp_path, "default")

        err = capsys.readouterr().err
        assert "Warning" in err
        assert "claude" in err
        assert "not captured" in err

    def test_unknown_provider_prints_warning(self, tmp_path: Path, capsys) -> None:
        """Unknown provider prints a warning mentioning the provider name."""
        _capture_credentials("unknown-agent", tmp_path, "default")

        err = capsys.readouterr().err
        assert "Warning" in err
        assert "unknown-agent" in err

    def test_db_failure_prints_warning(self, tmp_path: Path, capsys) -> None:
        """If DB storage fails, prints warning but doesn't raise."""
        cred = {"claudeAiOauth": {"accessToken": "sk-test"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))

        with patch("terok_sandbox.SandboxConfig", side_effect=RuntimeError("DB broken")):
            _capture_credentials("claude", tmp_path, "default")

        err = capsys.readouterr().err
        assert "Warning" in err
        assert "not saved" in err

    def test_custom_credential_set(self, tmp_path: Path) -> None:
        """Credentials can be stored under a custom credential set."""
        (tmp_path / "config.json").write_text(json.dumps({"api_key": "work-key"}))

        db_path = tmp_path / "proxy" / "credentials.db"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials("kisski", tmp_path, "work-project")

        from terok_sandbox import CredentialDB

        db = CredentialDB(db_path)
        stored = db.load_credential("work-project", "kisski")
        db.close()
        assert stored["key"] == "work-key"


class TestWriteClaudeCredentialsFile:
    """Verify _write_claude_credentials_file produces the correct static file."""

    def test_writes_phantom_token(self, tmp_path: Path) -> None:
        """Written file has the phantom marker as accessToken, not real credentials."""
        cred_data = {
            "type": "oauth",
            "scopes": "user:inference user:profile",
            "subscription_type": "max",
            "rate_limit_tier": "max_5x",
        }
        _write_claude_credentials_file(cred_data, tmp_path)

        cred_file = tmp_path / "_claude-config" / ".credentials.json"
        assert cred_file.is_file()
        data = json.loads(cred_file.read_text())
        oauth = data["claudeAiOauth"]
        assert oauth["accessToken"] == PHANTOM_CREDENTIALS_MARKER
        assert oauth["refreshToken"] == ""
        assert oauth["expiresAt"] is None

    def test_includes_subscription_metadata(self, tmp_path: Path) -> None:
        """Written file preserves scopes, subscriptionType, and rateLimitTier."""
        cred_data = {
            "scopes": "user:inference user:profile",
            "subscription_type": "max",
            "rate_limit_tier": "max_5x",
        }
        _write_claude_credentials_file(cred_data, tmp_path)

        data = json.loads((tmp_path / "_claude-config" / ".credentials.json").read_text())
        oauth = data["claudeAiOauth"]
        assert oauth["scopes"] == "user:inference user:profile"
        assert oauth["subscriptionType"] == "max"
        assert oauth["rateLimitTier"] == "max_5x"

    def test_missing_metadata_defaults(self, tmp_path: Path) -> None:
        """Missing subscription fields default to empty/None in the written file."""
        _write_claude_credentials_file({"type": "oauth"}, tmp_path)

        data = json.loads((tmp_path / "_claude-config" / ".credentials.json").read_text())
        oauth = data["claudeAiOauth"]
        assert oauth["scopes"] == ""
        assert oauth["subscriptionType"] is None
        assert oauth["rateLimitTier"] is None

    def test_creates_directory_if_absent(self, tmp_path: Path) -> None:
        """Creates the _claude-config directory if it doesn't exist."""
        target = tmp_path / "nested" / "mounts"
        _write_claude_credentials_file({"type": "oauth"}, target)
        assert (target / "_claude-config" / ".credentials.json").is_file()


class TestCaptureWritesCredentialsFile:
    """Verify _capture_credentials writes .credentials.json for Claude OAuth."""

    def test_capture_claude_oauth_writes_credentials_file(self, tmp_path: Path) -> None:
        """Capturing Claude OAuth triggers .credentials.json creation."""
        cred = {
            "claudeAiOauth": {
                "accessToken": "sk-test-oauth",
                "refreshToken": "rt-test",
                "scopes": "user:inference",
                "subscriptionType": "pro",
            }
        }
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))

        db_path = tmp_path / "proxy" / "credentials.db"
        mounts = tmp_path / "mounts"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials("claude", tmp_path, "default", mounts_base=mounts)

        cred_file = mounts / "_claude-config" / ".credentials.json"
        assert cred_file.is_file()
        data = json.loads(cred_file.read_text())
        assert data["claudeAiOauth"]["accessToken"] == PHANTOM_CREDENTIALS_MARKER
        assert data["claudeAiOauth"]["subscriptionType"] == "pro"

    def test_capture_claude_api_key_skips_credentials_file(self, tmp_path: Path) -> None:
        """API key auth does NOT write .credentials.json (only OAuth needs it)."""
        (tmp_path / "config.json").write_text(json.dumps({"api_key": "sk-ant-key"}))

        db_path = tmp_path / "proxy" / "credentials.db"
        mounts = tmp_path / "mounts"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials("claude", tmp_path, "default", mounts_base=mounts)

        assert not (mounts / "_claude-config" / ".credentials.json").exists()

    def test_capture_non_claude_skips_credentials_file(self, tmp_path: Path) -> None:
        """Non-Claude providers don't write .credentials.json even with OAuth."""
        (tmp_path / "auth.json").write_text(
            json.dumps({"tokens": {"access_token": "sk-oai", "refresh_token": "rt"}})
        )

        db_path = tmp_path / "proxy" / "credentials.db"
        mounts = tmp_path / "mounts"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials("codex", tmp_path, "default", mounts_base=mounts)

        assert not (mounts / "_claude-config").exists()


class TestStoreApiKey:
    """Verify direct API key storage (--api-key flag)."""

    def test_stores_key(self, tmp_path: Path) -> None:
        """store_api_key writes to the DB without a container."""
        db_path = tmp_path / "proxy" / "credentials.db"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            store_api_key("vibe", "sk-test-key-123")

        from terok_sandbox import CredentialDB

        db = CredentialDB(db_path)
        stored = db.load_credential("default", "vibe")
        db.close()
        assert stored == {"type": "api_key", "key": "sk-test-key-123"}

    def test_custom_credential_set(self, tmp_path: Path) -> None:
        """store_api_key supports custom credential sets."""
        db_path = tmp_path / "proxy" / "credentials.db"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            store_api_key("claude", "sk-ant-key", credential_set="work")

        from terok_sandbox import CredentialDB

        db = CredentialDB(db_path)
        stored = db.load_credential("work", "claude")
        db.close()
        assert stored["key"] == "sk-ant-key"
