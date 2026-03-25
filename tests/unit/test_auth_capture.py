# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the credential capture path in the auth interceptor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from terok_agent.auth import _capture_credentials, store_api_key


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

        out = capsys.readouterr().out
        assert "Warning" in out
        assert "claude" in out
        assert "not captured" in out

    def test_unknown_provider_prints_warning(self, tmp_path: Path, capsys) -> None:
        """Unknown provider prints a warning mentioning the provider name."""
        _capture_credentials("unknown-agent", tmp_path, "default")

        out = capsys.readouterr().out
        assert "Warning" in out
        assert "unknown-agent" in out

    def test_db_failure_prints_warning(self, tmp_path: Path, capsys) -> None:
        """If DB storage fails, prints warning but doesn't raise."""
        cred = {"claudeAiOauth": {"accessToken": "sk-test"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))

        with patch("terok_sandbox.SandboxConfig", side_effect=RuntimeError("DB broken")):
            _capture_credentials("claude", tmp_path, "default")

        out = capsys.readouterr().out
        assert "Warning" in out
        assert "not saved" in out

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
