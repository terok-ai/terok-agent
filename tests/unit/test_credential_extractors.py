# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-provider credential extractors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terok_agent.credential_extractors import (
    extract_api_key_env,
    extract_claude_oauth,
    extract_codex_oauth,
    extract_credential,
    extract_gh_token,
    extract_glab_token,
    extract_json_api_key,
)


class TestClaudeOAuth:
    """Verify Claude credential extraction (OAuth + API key fallback)."""

    def test_extracts_oauth_token(self, tmp_path: Path) -> None:
        """Extracts accessToken from .credentials.json (OAuth path)."""
        cred = {
            "claudeAiOauth": {
                "accessToken": "sk-ant-test-123",
                "refreshToken": "rt-test-456",
                "expiresAt": 1700000000,
            }
        }
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))
        result = extract_claude_oauth(tmp_path)
        assert result["access_token"] == "sk-ant-test-123"
        assert result["refresh_token"] == "rt-test-456"
        assert result["type"] == "oauth"
        assert result["expires_at"] == 1700000000  # seconds value kept as-is

    def test_converts_ms_expires_at_to_seconds(self, tmp_path: Path) -> None:
        """expiresAt from Claude Code (JS ms timestamp) is converted to POSIX seconds."""
        expires_at_ms = 1_700_000_000_000  # realistic Claude Code value (ms)
        cred = {
            "claudeAiOauth": {
                "accessToken": "sk-ant-ms-test",
                "refreshToken": "rt-ms",
                "expiresAt": expires_at_ms,
            }
        }
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))
        result = extract_claude_oauth(tmp_path)
        assert result["expires_at"] == pytest.approx(expires_at_ms / 1000)

    def test_missing_expires_at_stored_as_none(self, tmp_path: Path) -> None:
        """Missing expiresAt field results in expires_at=None (triggers proactive refresh)."""
        cred = {"claudeAiOauth": {"accessToken": "sk-ant-no-exp", "refreshToken": "rt-no-exp"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))
        result = extract_claude_oauth(tmp_path)
        assert result["expires_at"] is None

    @pytest.mark.parametrize("bad_value", ["1700000000000", True, False, None])
    def test_non_numeric_expires_at_stored_as_none(self, tmp_path: Path, bad_value: object) -> None:
        """Non-numeric expiresAt (string, bool, null) is ignored; expires_at falls back to None."""
        cred = {
            "claudeAiOauth": {
                "accessToken": "sk-ant-bad-exp",
                "refreshToken": "rt-bad",
                "expiresAt": bad_value,
            }
        }
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))
        result = extract_claude_oauth(tmp_path)
        assert result["expires_at"] is None

    def test_extracts_api_key_fallback(self, tmp_path: Path) -> None:
        """Falls back to config.json API key when no OAuth credentials."""
        (tmp_path / "config.json").write_text(json.dumps({"api_key": "sk-ant-key-test"}))
        result = extract_claude_oauth(tmp_path)
        assert result["key"] == "sk-ant-key-test"
        assert result["type"] == "api_key"

    def test_oauth_takes_precedence(self, tmp_path: Path) -> None:
        """OAuth credentials win when both files exist."""
        cred = {"claudeAiOauth": {"accessToken": "sk-oauth"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))
        (tmp_path / "config.json").write_text(json.dumps({"api_key": "sk-apikey"}))
        result = extract_claude_oauth(tmp_path)
        assert result["type"] == "oauth"
        assert result["access_token"] == "sk-oauth"

    def test_no_credentials_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when neither file exists."""
        with pytest.raises(ValueError, match="No Claude credentials"):
            extract_claude_oauth(tmp_path)

    def test_empty_oauth_falls_back_to_api_key(self, tmp_path: Path) -> None:
        """Empty OAuth token falls back to API key."""
        (tmp_path / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {}}))
        (tmp_path / "config.json").write_text(json.dumps({"api_key": "sk-fallback"}))
        result = extract_claude_oauth(tmp_path)
        assert result["type"] == "api_key"

    def test_empty_everything_raises(self, tmp_path: Path) -> None:
        """Raises when OAuth has no token AND config.json has no api_key."""
        (tmp_path / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {}}))
        with pytest.raises(ValueError, match="No Claude credentials"):
            extract_claude_oauth(tmp_path)


class TestCodexOAuth:
    """Verify Codex credential extraction."""

    def test_extracts_access_token(self, tmp_path: Path) -> None:
        """Extracts access_token from auth.json."""
        (tmp_path / "auth.json").write_text(
            json.dumps({"tokens": {"access_token": "sk-openai-test", "refresh_token": "rt-oai"}})
        )
        result = extract_codex_oauth(tmp_path)
        assert result["access_token"] == "sk-openai-test"
        assert result["type"] == "oauth"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when file is missing."""
        with pytest.raises(ValueError, match="not found"):
            extract_codex_oauth(tmp_path)

    def test_missing_tokens_field_raises(self, tmp_path: Path) -> None:
        """Raises when tokens field has no access_token."""
        (tmp_path / "auth.json").write_text(json.dumps({"tokens": {}}))
        with pytest.raises(ValueError, match="no access_token"):
            extract_codex_oauth(tmp_path)


class TestApiKeyEnv:
    """Verify dotenv-style API key extraction."""

    def test_extracts_named_var(self, tmp_path: Path) -> None:
        """Extracts a specific named variable."""
        (tmp_path / ".env").write_text("MISTRAL_API_KEY=test-key-123\n")
        result = extract_api_key_env(tmp_path, ".env", "MISTRAL_API_KEY")
        assert result["key"] == "test-key-123"
        assert result["type"] == "api_key"

    def test_strips_single_quotes(self, tmp_path: Path) -> None:
        """Strips single quotes from values."""
        (tmp_path / ".env").write_text("MISTRAL_API_KEY='quoted-key'\n")
        result = extract_api_key_env(tmp_path, ".env", "MISTRAL_API_KEY")
        assert result["key"] == "quoted-key"

    def test_strips_double_quotes(self, tmp_path: Path) -> None:
        """Strips double quotes from values."""
        (tmp_path / ".env").write_text('MISTRAL_API_KEY="dquoted-key"\n')
        result = extract_api_key_env(tmp_path, ".env", "MISTRAL_API_KEY")
        assert result["key"] == "dquoted-key"

    def test_skips_comments_and_blanks(self, tmp_path: Path) -> None:
        """Skips comment lines and empty lines."""
        (tmp_path / ".env").write_text("# comment\n\nMISTRAL_API_KEY=real-key\n")
        result = extract_api_key_env(tmp_path, ".env", "MISTRAL_API_KEY")
        assert result["key"] == "real-key"

    def test_wrong_var_name_raises(self, tmp_path: Path) -> None:
        """Raises when the requested var name is not found."""
        (tmp_path / ".env").write_text("OTHER_VAR=something\n")
        with pytest.raises(ValueError, match="No API key"):
            extract_api_key_env(tmp_path, ".env", "MISTRAL_API_KEY")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when file is missing."""
        with pytest.raises(ValueError, match="not found"):
            extract_api_key_env(tmp_path, ".env", "X")


class TestJsonApiKey:
    """Verify JSON config API key extraction."""

    def test_extracts_api_key(self, tmp_path: Path) -> None:
        """Extracts api_key from config.json."""
        (tmp_path / "config.json").write_text(json.dumps({"api_key": "blab-key-123"}))
        result = extract_json_api_key(tmp_path)
        assert result["key"] == "blab-key-123"

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when api_key field is absent."""
        (tmp_path / "config.json").write_text(json.dumps({"other": "value"}))
        with pytest.raises(ValueError, match="No api_key"):
            extract_json_api_key(tmp_path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when config.json is missing."""
        with pytest.raises(ValueError, match="not found"):
            extract_json_api_key(tmp_path)


class TestGhToken:
    """Verify GitHub CLI token extraction."""

    def test_extracts_oauth_token(self, tmp_path: Path) -> None:
        """Extracts oauth_token from hosts.yml."""
        (tmp_path / "hosts.yml").write_text(
            "github.com:\n  oauth_token: ghp_test123\n  user: testuser\n"
        )
        result = extract_gh_token(tmp_path)
        assert result["token"] == "ghp_test123"
        assert result["host"] == "github.com"

    def test_no_token_raises(self, tmp_path: Path) -> None:
        """Raises when no host has an oauth_token."""
        (tmp_path / "hosts.yml").write_text("github.com:\n  user: test\n")
        with pytest.raises(ValueError, match="No oauth_token"):
            extract_gh_token(tmp_path)

    def test_non_dict_root_raises(self, tmp_path: Path) -> None:
        """Raises for non-dict YAML root."""
        (tmp_path / "hosts.yml").write_text("- item\n")
        with pytest.raises(ValueError, match="Unexpected"):
            extract_gh_token(tmp_path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when file is missing."""
        with pytest.raises(ValueError, match="not found"):
            extract_gh_token(tmp_path)


class TestGlabToken:
    """Verify GitLab CLI token extraction."""

    def test_extracts_token(self, tmp_path: Path) -> None:
        """Extracts token from config.yml hosts section."""
        (tmp_path / "config.yml").write_text(
            "hosts:\n  gitlab.com:\n    token: glpat-test456\n    api_host: gitlab.com\n"
        )
        result = extract_glab_token(tmp_path)
        assert result["token"] == "glpat-test456"
        assert result["host"] == "gitlab.com"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when config.yml is missing."""
        with pytest.raises(ValueError, match="not found"):
            extract_glab_token(tmp_path)


class TestGlabTokenEdgeCases:
    """Additional glab edge cases."""

    def test_no_token_raises(self, tmp_path: Path) -> None:
        """Raises when no host has a token."""
        (tmp_path / "config.yml").write_text("hosts:\n  gitlab.com:\n    api_host: gitlab.com\n")
        with pytest.raises(ValueError, match="No token"):
            extract_glab_token(tmp_path)

    def test_non_dict_root_raises(self, tmp_path: Path) -> None:
        """Raises for non-dict YAML root."""
        (tmp_path / "config.yml").write_text("- item\n")
        with pytest.raises(ValueError, match="Unexpected"):
            extract_glab_token(tmp_path)


class TestMalformedPayloads:
    """Verify extractors reject non-mapping JSON/YAML."""

    def test_claude_array_root_raises(self, tmp_path: Path) -> None:
        """Claude extractor rejects JSON array root."""
        (tmp_path / ".credentials.json").write_text("[]")
        with pytest.raises(ValueError, match="No Claude credentials"):
            extract_claude_oauth(tmp_path)

    def test_codex_array_root_raises(self, tmp_path: Path) -> None:
        """Codex extractor rejects JSON array root."""
        (tmp_path / "auth.json").write_text("[]")
        with pytest.raises(ValueError, match="not found or unreadable"):
            extract_codex_oauth(tmp_path)

    def test_json_api_key_array_root_raises(self, tmp_path: Path) -> None:
        """JSON API key extractor rejects non-mapping root."""
        (tmp_path / "config.json").write_text('"just a string"')
        with pytest.raises(ValueError, match="not found or unreadable"):
            extract_json_api_key(tmp_path)

    def test_glab_non_mapping_hosts_raises(self, tmp_path: Path) -> None:
        """GitLab extractor rejects non-mapping hosts section."""
        (tmp_path / "config.yml").write_text("hosts:\n  - item1\n  - item2\n")
        with pytest.raises(ValueError, match="Expected mapping"):
            extract_glab_token(tmp_path)


class TestExtractCredential:
    """Verify the dispatch function."""

    def test_dispatches_to_claude(self, tmp_path: Path) -> None:
        """extract_credential('claude', ...) calls extract_claude_oauth."""
        cred = {"claudeAiOauth": {"accessToken": "sk-test"}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))
        result = extract_credential("claude", tmp_path)
        assert result["access_token"] == "sk-test"

    def test_unknown_provider_raises(self, tmp_path: Path) -> None:
        """Unknown provider name raises ValueError."""
        with pytest.raises(ValueError, match="No credential extractor"):
            extract_credential("unknown-agent", tmp_path)
