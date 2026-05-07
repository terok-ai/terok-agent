# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the credential capture path in the auth interceptor."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

from terok_sandbox import CODEX_SHARED_OAUTH_MARKER, PHANTOM_CREDENTIALS_MARKER

from terok_executor.credentials.auth import (
    _apply_post_capture_state,
    _capture_credentials,
    _claude_oauth_mount_writer,
    _codex_oauth_mount_writer,
    _write_claude_credentials_file,
    store_api_key,
)


def _fake_jwt(payload: dict | None = None) -> str:
    """Return a syntactically valid unsigned JWT for tests."""
    header = {"alg": "none", "typ": "JWT"}
    data = payload or {"email": "codex@example.com"}

    def _b64url(obj: dict) -> str:
        encoded = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")

    return ".".join((_b64url(header), _b64url(data), "test"))


def _jwt_payload(token: str) -> dict:
    """Decode the unsigned JWT payload used in tests."""
    _header, payload, _sig = token.split(".", 2)
    padded = payload + "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))


class TestCaptureCredentials:
    """Verify _capture_credentials stores extracted credentials in the DB."""

    def test_captures_claude_credentials(self, tmp_path: Path) -> None:
        """Successful extraction stores credentials in the DB."""
        # Create a fake Claude credential file
        cred = {"claudeAiOauth": {"accessToken": "sk-test-123"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))

        db_path = tmp_path / "proxy" / "credentials.db"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.db_path = db_path
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
            mock_cfg_cls.return_value.db_path = db_path
            _capture_credentials("blablador", tmp_path, "default")

        from terok_sandbox import CredentialDB

        db = CredentialDB(db_path)
        stored = db.load_credential("default", "blablador")
        db.close()
        assert stored["key"] == "blab-key"

    def test_extraction_failure_prints_error(self, tmp_path: Path, capsys) -> None:
        """Failed extraction prints an error mentioning the provider."""
        # Empty dir — no credential file to extract
        _capture_credentials("claude", tmp_path, "default")

        err = capsys.readouterr().err
        assert "Error" in err
        assert "claude" in err
        assert "not captured" in err

    def test_unknown_provider_prints_error(self, tmp_path: Path, capsys) -> None:
        """Unknown provider prints an error mentioning the provider name."""
        _capture_credentials("unknown-agent", tmp_path, "default")

        err = capsys.readouterr().err
        assert "Error" in err
        assert "unknown-agent" in err

    def test_db_failure_prints_error(self, tmp_path: Path, capsys) -> None:
        """If DB storage fails, prints error but doesn't raise."""
        cred = {"claudeAiOauth": {"accessToken": "sk-test"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))

        with patch("terok_sandbox.SandboxConfig", side_effect=RuntimeError("DB broken")):
            _capture_credentials("claude", tmp_path, "default")

        err = capsys.readouterr().err
        assert "Error" in err
        assert "not saved" in err

    def test_custom_credential_set(self, tmp_path: Path) -> None:
        """Credentials can be stored under a custom credential set."""
        (tmp_path / "config.json").write_text(json.dumps({"api_key": "work-key"}))

        db_path = tmp_path / "proxy" / "credentials.db"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.db_path = db_path
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

    def test_replaces_symlink_without_touching_target(self, tmp_path: Path) -> None:
        """No-follow writes replace a hostile mount symlink instead of following it."""
        victim = tmp_path / "victim.json"
        victim.write_text("do not overwrite")
        cred_dir = tmp_path / "_claude-config"
        cred_dir.mkdir()
        dest = cred_dir / ".credentials.json"
        dest.symlink_to(victim)

        _write_claude_credentials_file({"type": "oauth"}, tmp_path)

        assert victim.read_text() == "do not overwrite"
        assert not dest.is_symlink()
        assert json.loads(dest.read_text())["claudeAiOauth"]["accessToken"] == (
            PHANTOM_CREDENTIALS_MARKER
        )


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
            mock_cfg_cls.return_value.db_path = db_path
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
            mock_cfg_cls.return_value.db_path = db_path
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
            mock_cfg_cls.return_value.db_path = db_path
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
            mock_cfg_cls.return_value.db_path = db_path
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
            mock_cfg_cls.return_value.db_path = db_path
            _capture_credentials("claude", tmp_path, "default", mounts_base=mounts)

        assert not (mounts / "_claude-config" / ".credentials.json").exists()

    def test_capture_codex_default_writes_phantom(self, tmp_path: Path) -> None:
        """Codex OAuth capture without expose writes a phantom auth.json."""
        real_id_token = _fake_jwt(
            {
                "email": "coder@example.com",
                "https://api.openai.com/auth": {
                    "chatgpt_plan_type": "pro",
                    "chatgpt_account_id": "org-42",
                },
            }
        )
        (tmp_path / "auth.json").write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": "sk-oai",
                        "refresh_token": "rt",
                        "id_token": real_id_token,
                    }
                }
            )
        )

        db_path = tmp_path / "proxy" / "credentials.db"
        mounts = tmp_path / "mounts"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.db_path = db_path
            _capture_credentials("codex", tmp_path, "default", mounts_base=mounts)

        phantom = mounts / "_codex-config" / "auth.json"
        assert phantom.is_file()
        data = json.loads(phantom.read_text())
        assert data["tokens"]["access_token"] == CODEX_SHARED_OAUTH_MARKER
        assert data["tokens"]["refresh_token"] == CODEX_SHARED_OAUTH_MARKER
        # The file carries a synthetic JWT, not the original opaque id_token.
        assert data["tokens"]["id_token"] != real_id_token


class TestClaudeOAuthMountWriter:
    """Verify the Claude mount reconciler preserves or phantomises creds."""

    def test_expose_copies_real_file(self, tmp_path: Path) -> None:
        """Real .credentials.json is copied verbatim when exposed."""
        auth_dir = tmp_path / "auth"
        auth_dir.mkdir()
        real_creds = {"claudeAiOauth": {"accessToken": "real-token-abc"}}
        (auth_dir / ".credentials.json").write_text(json.dumps(real_creds))

        mounts = tmp_path / "mounts"
        _claude_oauth_mount_writer(auth_dir, mounts, real_creds, expose_token=True)

        dest = mounts / "_claude-config" / ".credentials.json"
        assert dest.is_file()
        assert json.loads(dest.read_text()) == real_creds

    def test_expose_raises_when_file_missing(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when exposed and no .credentials.json."""
        import pytest

        with pytest.raises(FileNotFoundError):
            _claude_oauth_mount_writer(tmp_path, tmp_path / "mounts", {}, expose_token=True)


class TestCodexOAuthMountWriter:
    """Verify the Codex mount reconciler copies-or-wipes auth.json."""

    def test_expose_copies_real_auth_json(self, tmp_path: Path) -> None:
        """expose_token=True copies auth.json into _codex-config."""
        auth_dir = tmp_path / "auth"
        auth_dir.mkdir()
        payload = {"tokens": {"access_token": "sk-oai", "refresh_token": "rt"}}
        (auth_dir / "auth.json").write_text(json.dumps(payload))

        mounts = tmp_path / "mounts"
        _codex_oauth_mount_writer(auth_dir, mounts, payload, expose_token=True)

        dest = mounts / "_codex-config" / "auth.json"
        assert dest.is_file()
        assert json.loads(dest.read_text()) == payload

    def test_default_writes_phantom_preserving_id_token(self, tmp_path: Path) -> None:
        """expose_token=False writes a phantom auth.json with synthetic id_token claims.

        Codex's TUI bootstrap calls ``account/read`` (an internal
        JSON-RPC), which fails unless both ``email`` AND
        ``chatgpt_plan_type`` parse out of the id_token JWT — see
        ``codex-rs/login/src/token_data.rs:139`` and
        ``model-provider/src/provider.rs::account_state``.  The
        synthetic JWT must therefore preserve the top-level ``email``
        claim alongside the ``chatgpt_*`` namespace; everything else
        (PII outside that contract, opaque internal claims) is dropped.
        """
        mounts = tmp_path / "mounts"
        real_id_token = _fake_jwt(
            {
                "email": "coder@example.com",
                "extra_internal_claim": "dropped",
                "https://api.openai.com/auth": {
                    "chatgpt_plan_type": "pro",
                    "chatgpt_account_id": "org-42",
                },
            }
        )
        cred = {"id_token": real_id_token, "account_id": "org-42"}

        _codex_oauth_mount_writer(tmp_path, mounts, cred, expose_token=False)

        data = json.loads((mounts / "_codex-config" / "auth.json").read_text())
        tokens = data["tokens"]
        assert tokens["id_token"] != real_id_token
        assert len(tokens["id_token"].split(".")) == 3
        synthetic_claims = _jwt_payload(tokens["id_token"])
        # Email survives — required by Codex's account/read.
        assert synthetic_claims["email"] == "coder@example.com"
        # The ``auth`` namespace stays minimal: only the well-known keys.
        assert "https://api.openai.com/profile" not in synthetic_claims
        assert synthetic_claims["https://api.openai.com/auth"]["chatgpt_account_id"] == "org-42"
        assert synthetic_claims["https://api.openai.com/auth"]["chatgpt_plan_type"] == "pro"
        # Anything outside the documented surface is dropped.
        assert "extra_internal_claim" not in synthetic_claims
        assert tokens["account_id"] == "org-42"
        assert tokens["access_token"] == CODEX_SHARED_OAUTH_MARKER
        assert tokens["refresh_token"] == CODEX_SHARED_OAUTH_MARKER
        assert data["OPENAI_API_KEY"] is None
        # last_refresh is pinned far in the future so the CLI's 8-day
        # staleness check never fires an in-container refresh attempt.
        assert data["last_refresh"] == "9999-01-01T00:00:00Z"

    def test_default_omits_email_when_upstream_lacks_it(self, tmp_path: Path) -> None:
        """If the upstream JWT has no email claim, the synthetic JWT also omits it.

        Forwarding ``email: null`` would fail Codex's ``Option<String>``
        deserializer; just leave the field out entirely.
        """
        mounts = tmp_path / "mounts"
        real_id_token = _fake_jwt(
            {
                "https://api.openai.com/auth": {"chatgpt_plan_type": "pro"},
            }
        )
        cred = {"id_token": real_id_token}

        _codex_oauth_mount_writer(tmp_path, mounts, cred, expose_token=False)

        synthetic_claims = _jwt_payload(
            json.loads((mounts / "_codex-config" / "auth.json").read_text())["tokens"]["id_token"]
        )
        assert "email" not in synthetic_claims

    def test_default_overwrites_stale_real_auth_json(self, tmp_path: Path) -> None:
        """expose_token=False replaces a prior real auth.json with the phantom."""
        mounts = tmp_path / "mounts"
        codex_dir = mounts / "_codex-config"
        codex_dir.mkdir(parents=True)
        (codex_dir / "auth.json").write_text(
            json.dumps({"tokens": {"access_token": "leaked-real"}})
        )

        _codex_oauth_mount_writer(tmp_path, mounts, {}, expose_token=False)

        data = json.loads((codex_dir / "auth.json").read_text())
        assert data["tokens"]["access_token"] == CODEX_SHARED_OAUTH_MARKER

    def test_expose_replaces_symlink_without_touching_target(self, tmp_path: Path) -> None:
        """Exposed copy mode writes through the same no-follow destination helper."""
        auth_dir = tmp_path / "auth"
        auth_dir.mkdir()
        payload = {"tokens": {"access_token": "sk-oai", "refresh_token": "rt"}}
        (auth_dir / "auth.json").write_text(json.dumps(payload))

        mounts = tmp_path / "mounts"
        codex_dir = mounts / "_codex-config"
        codex_dir.mkdir(parents=True)
        victim = tmp_path / "victim.json"
        victim.write_text("do not overwrite")
        dest = codex_dir / "auth.json"
        dest.symlink_to(victim)

        _codex_oauth_mount_writer(auth_dir, mounts, payload, expose_token=True)

        assert victim.read_text() == "do not overwrite"
        assert not dest.is_symlink()
        assert json.loads(dest.read_text()) == payload

    def test_expose_raises_when_file_missing(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when exposed and no auth.json."""
        import pytest

        with pytest.raises(FileNotFoundError):
            _codex_oauth_mount_writer(tmp_path, tmp_path / "mounts", {}, expose_token=True)


class TestCaptureWithExposeToken:
    """Verify _capture_credentials with expose_token=True copies the real file."""

    def test_expose_token_copies_real_credentials(self, tmp_path: Path) -> None:
        """expose_token=True copies the real .credentials.json instead of phantom."""
        real_creds = {"claudeAiOauth": {"accessToken": "real-oauth-token", "scopes": "all"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(real_creds))

        db_path = tmp_path / "proxy" / "credentials.db"
        mounts = tmp_path / "mounts"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.db_path = db_path
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
            mock_cfg_cls.return_value.db_path = db_path
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
            mock_cfg_cls.return_value.db_path = db_path
            _capture_credentials(
                "claude", tmp_path, "default", mounts_base=mounts, expose_token=True
            )

        out = capsys.readouterr().out
        assert "EXPOSED" in out

    def test_expose_token_skips_vault_db(self, tmp_path: Path) -> None:
        """expose_token=True does NOT store in vault DB -- avoids refresh conflict."""
        real_creds = {"claudeAiOauth": {"accessToken": "real-tok"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(real_creds))

        db_path = tmp_path / "proxy" / "credentials.db"
        mounts = tmp_path / "mounts"
        with patch("terok_sandbox.SandboxConfig") as mock_cfg_cls:
            mock_cfg_cls.return_value.db_path = db_path
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
            mock_cfg_cls.return_value.db_path = db_path
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
            mock_cfg_cls.return_value.db_path = db_path
            store_api_key("claude", "sk-ant-key", credential_set="work")

        from terok_sandbox import CredentialDB

        db = CredentialDB(db_path)
        stored = db.load_credential("work", "claude")
        db.close()
        assert stored["key"] == "sk-ant-key"


class TestAuthenticateImageLaziness:
    """Verify ``authenticate(image=callable)`` defers L1 build to the OAuth path.

    Picking API key from the OAuth-or-API-key prompt must short-circuit
    before the lazy resolver fires — otherwise users who only ever use
    API keys still pay for an L1 image build they don't need.
    """

    def test_api_key_choice_skips_image_resolution(self, tmp_path: Path) -> None:
        """User picks ``2`` (API key) → resolver is never called."""
        from unittest.mock import MagicMock, patch

        from terok_executor.credentials.auth import AuthProvider, authenticate

        provider = AuthProvider(
            name="claude",
            label="Claude",
            host_dir_name="_claude-config",
            container_mount="/home/dev/.claude",
            command=["claude"],
            banner_hint="",
            modes=("oauth", "api_key"),
            api_key_hint="hint",
        )
        resolver = MagicMock()
        with (
            patch.dict(
                "terok_executor.credentials.auth.AUTH_PROVIDERS",
                {"claude": provider},
                clear=True,
            ),
            patch("builtins.input", return_value="2"),
            patch(
                "terok_executor.credentials.auth._prompt_api_key",
                return_value="sk-ant-test",
            ),
            patch("terok_executor.credentials.auth.store_api_key") as mock_store,
        ):
            authenticate(None, "claude", mounts_dir=tmp_path, image=resolver)

        resolver.assert_not_called()
        mock_store.assert_called_once_with("claude", "sk-ant-test")

    def test_oauth_choice_resolves_image(self, tmp_path: Path) -> None:
        """User picks ``1`` (OAuth) → resolver is called exactly once."""
        from unittest.mock import MagicMock, patch

        from terok_executor.credentials.auth import AuthProvider, authenticate

        provider = AuthProvider(
            name="claude",
            label="Claude",
            host_dir_name="_claude-config",
            container_mount="/home/dev/.claude",
            command=["claude"],
            banner_hint="",
            modes=("oauth", "api_key"),
        )
        resolver = MagicMock(return_value="terok-l1-cli:ubuntu-24.04")
        with (
            patch.dict(
                "terok_executor.credentials.auth.AUTH_PROVIDERS",
                {"claude": provider},
                clear=True,
            ),
            patch("builtins.input", return_value="1"),
            patch("terok_executor.credentials.auth._run_auth_container") as mock_run,
        ):
            authenticate(None, "claude", mounts_dir=tmp_path, image=resolver)

        resolver.assert_called_once()
        # The resolved tag is what reaches _run_auth_container, not the callable.
        assert mock_run.call_args.kwargs["image"] == "terok-l1-cli:ubuntu-24.04"

    def test_oauth_only_provider_resolves_image(self, tmp_path: Path) -> None:
        """OAuth-only providers don't show the prompt — image resolves immediately."""
        from unittest.mock import MagicMock, patch

        from terok_executor.credentials.auth import AuthProvider, authenticate

        provider = AuthProvider(
            name="codex",
            label="Codex",
            host_dir_name="_codex-config",
            container_mount="/home/dev/.codex",
            command=["setup-codex-auth.sh"],
            banner_hint="",
            modes=("oauth",),
        )
        resolver = MagicMock(return_value="terok-l1-cli:ubuntu-24.04")
        with (
            patch.dict(
                "terok_executor.credentials.auth.AUTH_PROVIDERS",
                {"codex": provider},
                clear=True,
            ),
            patch("terok_executor.credentials.auth._run_auth_container"),
        ):
            authenticate(None, "codex", mounts_dir=tmp_path, image=resolver)

        resolver.assert_called_once()

    def test_api_key_only_provider_ignores_image(self, tmp_path: Path) -> None:
        """API-key-only providers never resolve the image, even if one is given."""
        from unittest.mock import MagicMock, patch

        from terok_executor.credentials.auth import AuthProvider, authenticate

        provider = AuthProvider(
            name="blablador",
            label="Blablador",
            host_dir_name="_blablador-config",
            container_mount="/home/dev/.blablador",
            command=[],
            banner_hint="",
            modes=("api_key",),
            api_key_hint="hint",
        )
        resolver = MagicMock()
        with (
            patch.dict(
                "terok_executor.credentials.auth.AUTH_PROVIDERS",
                {"blablador": provider},
                clear=True,
            ),
            patch(
                "terok_executor.credentials.auth._prompt_api_key",
                return_value="sk-bbl-test",
            ),
            patch("terok_executor.credentials.auth.store_api_key"),
        ):
            authenticate(None, "blablador", mounts_dir=tmp_path, image=resolver)

        resolver.assert_not_called()

    def test_eager_string_image_still_works(self, tmp_path: Path) -> None:
        """Backwards compatibility: a plain string image is accepted and used as-is."""
        from unittest.mock import patch

        from terok_executor.credentials.auth import AuthProvider, authenticate

        provider = AuthProvider(
            name="codex",
            label="Codex",
            host_dir_name="_codex-config",
            container_mount="/home/dev/.codex",
            command=["setup-codex-auth.sh"],
            banner_hint="",
            modes=("oauth",),
        )
        with (
            patch.dict(
                "terok_executor.credentials.auth.AUTH_PROVIDERS",
                {"codex": provider},
                clear=True,
            ),
            patch("terok_executor.credentials.auth._run_auth_container") as mock_run,
        ):
            authenticate(None, "codex", mounts_dir=tmp_path, image="my-l1:tag")

        assert mock_run.call_args.kwargs["image"] == "my-l1:tag"

    def test_oauth_path_with_no_image_raises(self, tmp_path: Path) -> None:
        """OAuth path without an image (or callable) is a programming error."""
        from unittest.mock import patch

        import pytest

        from terok_executor.credentials.auth import AuthProvider, authenticate

        provider = AuthProvider(
            name="codex",
            label="Codex",
            host_dir_name="_codex-config",
            container_mount="/home/dev/.codex",
            command=["setup-codex-auth.sh"],
            banner_hint="",
            modes=("oauth",),
        )
        with patch.dict(
            "terok_executor.credentials.auth.AUTH_PROVIDERS",
            {"codex": provider},
            clear=True,
        ):
            with pytest.raises(ValueError, match="needs an L1 image"):
                authenticate(None, "codex", mounts_dir=tmp_path, image=None)


class TestAuthenticateOauthGate:
    """Verify ``oauth_enabled=False`` collapses dual-mode providers to API-key.

    The roster declares which modes a provider supports; deployments
    sometimes have to clamp that down (terok's ``experimental`` +
    ``allow_oauth`` gating for Codex/Claude).  When the gate is closed,
    the OAuth prompt must not be offered — even though the roster says
    OAuth is supported.
    """

    def test_oauth_disabled_skips_prompt_and_goes_to_api_key(self, tmp_path: Path) -> None:
        """Dual-mode provider with ``oauth_enabled=False`` short-circuits to API key."""
        from unittest.mock import MagicMock, patch

        from terok_executor.credentials.auth import AuthProvider, authenticate

        provider = AuthProvider(
            name="claude",
            label="Claude",
            host_dir_name="_claude-config",
            container_mount="/home/dev/.claude",
            command=["claude"],
            banner_hint="",
            modes=("oauth", "api_key"),  # roster says both
        )
        # No ``input`` patch — if the OAuth-or-API-key prompt fires the
        # test will hang (catching that regression too).
        with (
            patch.dict(
                "terok_executor.credentials.auth.AUTH_PROVIDERS",
                {"claude": provider},
                clear=True,
            ),
            patch(
                "terok_executor.credentials.auth._prompt_api_key",
                return_value="sk-ant-test",
            ),
            patch("terok_executor.credentials.auth.store_api_key") as mock_store,
            patch("terok_executor.credentials.auth._run_auth_container") as mock_run,
        ):
            authenticate(
                None,
                "claude",
                mounts_dir=tmp_path,
                image=MagicMock(),
                oauth_enabled=False,
            )

        mock_store.assert_called_once_with("claude", "sk-ant-test")
        mock_run.assert_not_called()

    def test_oauth_enabled_default_keeps_dual_prompt(self, tmp_path: Path) -> None:
        """``oauth_enabled`` defaults to True so existing callers see the prompt."""
        from unittest.mock import patch

        from terok_executor.credentials.auth import AuthProvider, authenticate

        provider = AuthProvider(
            name="claude",
            label="Claude",
            host_dir_name="_claude-config",
            container_mount="/home/dev/.claude",
            command=["claude"],
            banner_hint="",
            modes=("oauth", "api_key"),
        )
        with (
            patch.dict(
                "terok_executor.credentials.auth.AUTH_PROVIDERS",
                {"claude": provider},
                clear=True,
            ),
            patch("builtins.input", return_value="2") as mock_input,  # picks API key
            patch(
                "terok_executor.credentials.auth._prompt_api_key",
                return_value="sk-ant",
            ),
            patch("terok_executor.credentials.auth.store_api_key"),
        ):
            authenticate(None, "claude", mounts_dir=tmp_path, image="img:tag")
        # Default ``oauth_enabled=True`` ⇒ user is asked to choose.
        mock_input.assert_called_once()

    def test_oauth_only_provider_with_gate_closed_raises(self, tmp_path: Path) -> None:
        """A provider that only declares OAuth raises when the gate forbids it."""
        from unittest.mock import patch

        import pytest

        from terok_executor.credentials.auth import AuthProvider, authenticate

        provider = AuthProvider(
            name="some-future-oauth-only",
            label="Future",
            host_dir_name="_future-config",
            container_mount="/home/dev/.future",
            command=["future"],
            banner_hint="",
            modes=("oauth",),  # no api_key fallback
        )
        with patch.dict(
            "terok_executor.credentials.auth.AUTH_PROVIDERS",
            {"some-future-oauth-only": provider},
            clear=True,
        ):
            with pytest.raises(SystemExit, match="OAuth.*disabled"):
                authenticate(
                    None,
                    "some-future-oauth-only",
                    mounts_dir=tmp_path,
                    image="img:tag",
                    oauth_enabled=False,
                )
