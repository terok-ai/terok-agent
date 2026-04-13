# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the credential capture path in the auth interceptor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from terok_executor.credentials.auth import (
    PHANTOM_CREDENTIALS_MARKER,
    _apply_post_capture_state,
    _capture_credentials,
    _copy_real_credentials,
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


class TestApplyPostCaptureState:
    """Verify _apply_post_capture_state writes declarative JSON state files."""

    def test_writes_state(self, tmp_path: Path) -> None:
        """post_capture_state creates the declared JSON file."""
        _apply_post_capture_state(
            "_test-config",
            {".state.json": {"setupDone": True}},
            tmp_path,
        )
        state_path = tmp_path / "_test-config" / ".state.json"
        assert state_path.is_file()
        assert json.loads(state_path.read_text()) == {"setupDone": True}

    def test_merges_with_existing_state(self, tmp_path: Path) -> None:
        """Existing keys are preserved when merging post-capture state."""
        target_dir = tmp_path / "_test-config"
        target_dir.mkdir(parents=True)
        (target_dir / ".state.json").write_text(json.dumps({"theme": "dark"}))

        _apply_post_capture_state(
            "_test-config",
            {".state.json": {"setupDone": True}},
            tmp_path,
        )
        state = json.loads((target_dir / ".state.json").read_text())
        assert state == {"theme": "dark", "setupDone": True}

    def test_skips_when_already_current(self, tmp_path: Path) -> None:
        """Does not rewrite file when state already matches."""
        target_dir = tmp_path / "_test-config"
        target_dir.mkdir(parents=True)
        state_path = target_dir / ".state.json"
        original = json.dumps({"setupDone": True, "extra": "keep"})
        state_path.write_text(original)
        _apply_post_capture_state(
            "_test-config",
            {".state.json": {"setupDone": True}},
            tmp_path,
        )
        assert state_path.read_text() == original

    def test_recovers_from_corrupt_json(self, tmp_path: Path) -> None:
        """Corrupt JSON in existing file is discarded; patch is applied fresh."""
        target_dir = tmp_path / "_test-config"
        target_dir.mkdir(parents=True)
        (target_dir / ".state.json").write_text("{corrupt!!!")

        _apply_post_capture_state(
            "_test-config",
            {".state.json": {"setupDone": True}},
            tmp_path,
        )
        state = json.loads((target_dir / ".state.json").read_text())
        assert state == {"setupDone": True}

    def test_replaces_non_dict_json(self, tmp_path: Path) -> None:
        """Non-dict JSON (e.g. a list) in existing file is discarded."""
        target_dir = tmp_path / "_test-config"
        target_dir.mkdir(parents=True)
        (target_dir / ".state.json").write_text("[1, 2, 3]")

        _apply_post_capture_state(
            "_test-config",
            {".state.json": {"setupDone": True}},
            tmp_path,
        )
        state = json.loads((target_dir / ".state.json").read_text())
        assert state == {"setupDone": True}

    def test_rejects_traversal_in_host_dir_name(self, tmp_path: Path) -> None:
        """Path traversal in host_dir_name is rejected."""
        import pytest

        with pytest.raises(ValueError, match="Invalid host_dir_name"):
            _apply_post_capture_state("../../etc", {".x": {"a": 1}}, tmp_path)

    def test_rejects_traversal_in_filename(self, tmp_path: Path) -> None:
        """Path traversal in a patch filename is rejected."""
        import pytest

        with pytest.raises(ValueError, match="Invalid post_capture_state filename"):
            _apply_post_capture_state("_ok", {"../escape.json": {"a": 1}}, tmp_path)

    def test_rejects_absolute_host_dir_name(self, tmp_path: Path) -> None:
        """Absolute host_dir_name is rejected."""
        import pytest

        with pytest.raises(ValueError, match="Invalid host_dir_name"):
            _apply_post_capture_state("/etc/shadow", {".x": {"a": 1}}, tmp_path)

    def test_rejects_absolute_filename(self, tmp_path: Path) -> None:
        """Absolute patch filename is rejected."""
        import pytest

        with pytest.raises(ValueError, match="Invalid post_capture_state filename"):
            _apply_post_capture_state("_ok", {"/etc/shadow": {"a": 1}}, tmp_path)


class TestCaptureAppliesPostCaptureState:
    """Verify _capture_credentials invokes post-capture state when provider is given."""

    def test_capture_triggers_post_capture_state(self, tmp_path: Path) -> None:
        """When auth_provider has post_capture_state, it is applied after capture."""
        from terok_executor.credentials.auth import AuthProvider

        provider = AuthProvider(
            name="claude",
            label="Claude",
            host_dir_name="_claude-config",
            container_mount="/home/dev/.claude",
            command=["claude"],
            banner_hint="",
            modes=("api_key",),
            post_capture_state={".claude.json": {"hasCompletedOnboarding": True}},
        )

        # Set up a valid credential file so capture succeeds
        cred = {"claudeAiOauth": {"accessToken": "sk-test"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))

        mounts = tmp_path / "mounts"
        db_path = tmp_path / "proxy" / "credentials.db"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials(
                "claude", tmp_path, "default", mounts_base=mounts, auth_provider=provider
            )

        state_path = mounts / "_claude-config" / ".claude.json"
        assert state_path.is_file()
        assert json.loads(state_path.read_text()) == {"hasCompletedOnboarding": True}

    def test_capture_skips_post_capture_when_empty(self, tmp_path: Path) -> None:
        """No post_capture_state means no extra files are written."""
        from terok_executor.credentials.auth import AuthProvider

        provider = AuthProvider(
            name="claude",
            label="Claude",
            host_dir_name="_claude-config",
            container_mount="/home/dev/.claude",
            command=["claude"],
            banner_hint="",
            modes=("api_key",),
        )

        cred = {"claudeAiOauth": {"accessToken": "sk-test"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))

        mounts = tmp_path / "mounts"
        db_path = tmp_path / "proxy" / "credentials.db"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials(
                "claude", tmp_path, "default", mounts_base=mounts, auth_provider=provider
            )

        # No .claude.json should exist — post_capture_state is empty
        assert not (mounts / "_claude-config" / ".claude.json").exists()

    def test_capture_degrades_to_warning_on_post_capture_error(
        self, tmp_path: Path, capsys
    ) -> None:
        """Post-capture state failure logs a warning but doesn't abort capture."""
        from terok_executor.credentials.auth import AuthProvider

        provider = AuthProvider(
            name="claude",
            label="Claude",
            host_dir_name="_claude-config",
            container_mount="/home/dev/.claude",
            command=["claude"],
            banner_hint="",
            modes=("api_key",),
            # Target file is a directory — write will fail
            post_capture_state={".claude.json": {"hasCompletedOnboarding": True}},
        )

        cred = {"claudeAiOauth": {"accessToken": "sk-test"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))

        mounts = tmp_path / "mounts"
        # Pre-create the target as a directory so the JSON write fails
        blocker = mounts / "_claude-config" / ".claude.json"
        blocker.mkdir(parents=True)

        db_path = tmp_path / "proxy" / "credentials.db"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            # Should NOT raise — error is caught and printed
            _capture_credentials(
                "claude", tmp_path, "default", mounts_base=mounts, auth_provider=provider
            )

        err = capsys.readouterr().err
        assert "Warning" in err
        assert "post_capture_state" in err

        # Verify credentials were still stored in the DB
        from terok_sandbox import CredentialDB

        db = CredentialDB(db_path)
        stored = db.load_credential("default", "claude")
        db.close()
        assert stored is not None


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


class TestCopyRealCredentials:
    """Verify _copy_real_credentials preserves the real token file."""

    def test_copies_real_file(self, tmp_path: Path) -> None:
        """Real .credentials.json is copied verbatim to the shared mount."""
        auth_dir = tmp_path / "auth"
        auth_dir.mkdir()
        real_creds = {"claudeAiOauth": {"accessToken": "real-token-abc"}}
        (auth_dir / ".credentials.json").write_text(json.dumps(real_creds))

        mounts = tmp_path / "mounts"
        _copy_real_credentials(auth_dir, mounts)

        dest = mounts / "_claude-config" / ".credentials.json"
        assert dest.is_file()
        assert json.loads(dest.read_text()) == real_creds

    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when auth dir has no .credentials.json."""
        import pytest

        with pytest.raises(FileNotFoundError):
            _copy_real_credentials(tmp_path, tmp_path / "mounts")


class TestCaptureWithExposeToken:
    """Verify _capture_credentials with expose_token=True copies the real file."""

    def test_expose_token_copies_real_credentials(self, tmp_path: Path) -> None:
        """expose_token=True copies the real .credentials.json instead of phantom."""
        real_creds = {"claudeAiOauth": {"accessToken": "real-oauth-token", "scopes": "all"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(real_creds))

        db_path = tmp_path / "proxy" / "credentials.db"
        mounts = tmp_path / "mounts"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials(
                "claude", tmp_path, "default", mounts_base=mounts, expose_token=True
            )

        dest = mounts / "_claude-config" / ".credentials.json"
        assert dest.is_file()
        data = json.loads(dest.read_text())
        # Real token, NOT the phantom marker
        assert data["claudeAiOauth"]["accessToken"] == "real-oauth-token"

    def test_expose_token_false_writes_phantom(self, tmp_path: Path) -> None:
        """expose_token=False (default) writes the phantom marker as before."""
        real_creds = {"claudeAiOauth": {"accessToken": "real-token", "scopes": "all"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(real_creds))

        db_path = tmp_path / "proxy" / "credentials.db"
        mounts = tmp_path / "mounts"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials(
                "claude", tmp_path, "default", mounts_base=mounts, expose_token=False
            )

        data = json.loads((mounts / "_claude-config" / ".credentials.json").read_text())
        assert data["claudeAiOauth"]["accessToken"] == PHANTOM_CREDENTIALS_MARKER

    def test_expose_token_prints_warning(self, tmp_path: Path, capsys) -> None:
        """expose_token=True prints an EXPOSED warning."""
        real_creds = {"claudeAiOauth": {"accessToken": "tok"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(real_creds))

        db_path = tmp_path / "proxy" / "credentials.db"
        mounts = tmp_path / "mounts"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials(
                "claude", tmp_path, "default", mounts_base=mounts, expose_token=True
            )

        out = capsys.readouterr().out
        assert "EXPOSED" in out

    def test_expose_token_skips_proxy_db(self, tmp_path: Path) -> None:
        """expose_token=True does NOT store in proxy DB — avoids refresh conflict."""
        real_creds = {"claudeAiOauth": {"accessToken": "real-tok"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(real_creds))

        db_path = tmp_path / "proxy" / "credentials.db"
        mounts = tmp_path / "mounts"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.proxy_db_path = db_path
            _capture_credentials(
                "claude", tmp_path, "default", mounts_base=mounts, expose_token=True
            )

        from terok_sandbox import CredentialDB

        db = CredentialDB(db_path)
        stored = db.load_credential("default", "claude")
        db.close()
        assert stored is None


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
