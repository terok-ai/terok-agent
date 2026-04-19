# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for warning paths in vault_config and roster.

Verifies that parse errors in TOML/YAML config files and agent definition
files are surfaced as warnings on stderr rather than silently swallowed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# vault_config: _apply_toml_patch warns on invalid TOML
# ---------------------------------------------------------------------------


class TestApplyTomlPatchWarning:
    """_apply_toml_patch warns on TOML parse errors and falls back to empty dict."""

    def test_warns_on_invalid_toml(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Invalid TOML triggers a warning on stderr and is treated as empty."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("this is not valid toml {{{")

        patch_spec = {
            "file": "config.toml",
            "toml_table": "servers",
            "toml_match": {"name": "proxy"},
            "toml_set": {"api_base": "{vault_url}/v1"},
        }

        from terok_executor.credentials.vault_config import VaultLocation, _apply_toml_patch

        _apply_toml_patch(
            config_path, patch_spec, VaultLocation(url="http://localhost:9999", socket="")
        )

        captured = capsys.readouterr()
        assert "Warning [vault-config]" in captured.err
        assert str(config_path) in captured.err

        # The file should still be written with the new entry (fallback to empty dict)
        import tomllib

        result = tomllib.loads(config_path.read_text())
        assert "servers" in result
        assert result["servers"][0]["name"] == "proxy"
        assert result["servers"][0]["api_base"] == "http://localhost:9999/v1"

    def test_no_warning_on_missing_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A missing file is not a parse error — no warning emitted."""
        config_path = tmp_path / "nonexistent.toml"
        patch_spec = {
            "file": "nonexistent.toml",
            "toml_table": "servers",
            "toml_match": {"name": "proxy"},
            "toml_set": {"api_base": "{vault_url}/v1"},
        }

        from terok_executor.credentials.vault_config import VaultLocation, _apply_toml_patch

        _apply_toml_patch(
            config_path, patch_spec, VaultLocation(url="http://localhost:9999", socket="")
        )

        captured = capsys.readouterr()
        assert "Warning" not in captured.err

    def test_no_warning_on_valid_toml(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Valid TOML is parsed without warning."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[[servers]]\nname = "existing"\nport = 80\n')

        patch_spec = {
            "file": "config.toml",
            "toml_table": "servers",
            "toml_match": {"name": "proxy"},
            "toml_set": {"api_base": "{vault_url}/v1"},
        }

        from terok_executor.credentials.vault_config import VaultLocation, _apply_toml_patch

        _apply_toml_patch(
            config_path, patch_spec, VaultLocation(url="http://localhost:9999", socket="")
        )

        captured = capsys.readouterr()
        assert "Warning" not in captured.err


# ---------------------------------------------------------------------------
# vault_config: _apply_yaml_patch warns on invalid YAML
# ---------------------------------------------------------------------------


class TestApplyYamlPatchWarning:
    """_apply_yaml_patch warns on YAML parse errors and falls back to empty dict."""

    def test_warns_on_invalid_yaml(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Invalid YAML triggers a warning on stderr and is treated as empty."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(":\n  - :\n  bad: [unterminated")

        patch_spec = {
            "file": "config.yaml",
            "yaml_set": {"api_base": "{vault_url}/v1"},
        }

        from terok_executor.credentials.vault_config import VaultLocation, _apply_yaml_patch

        _apply_yaml_patch(
            config_path, patch_spec, VaultLocation(url="http://localhost:9999", socket="")
        )

        captured = capsys.readouterr()
        assert "Warning [vault-config]" in captured.err
        assert str(config_path) in captured.err

        # The file should be written with only the new key (fallback to empty)
        from ruamel.yaml import YAML

        yaml = YAML()
        result = yaml.load(config_path)
        assert result["api_base"] == "http://localhost:9999/v1"

    def test_no_warning_on_missing_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A missing file is not a parse error — no warning emitted."""
        config_path = tmp_path / "nonexistent.yaml"
        patch_spec = {
            "file": "nonexistent.yaml",
            "yaml_set": {"api_base": "{vault_url}/v1"},
        }

        from terok_executor.credentials.vault_config import VaultLocation, _apply_yaml_patch

        _apply_yaml_patch(
            config_path, patch_spec, VaultLocation(url="http://localhost:9999", socket="")
        )

        captured = capsys.readouterr()
        assert "Warning" not in captured.err

    def test_no_warning_on_valid_yaml(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Valid YAML is parsed without warning."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("existing_key: hello\n")

        patch_spec = {
            "file": "config.yaml",
            "yaml_set": {"api_base": "{vault_url}/v1"},
        }

        from terok_executor.credentials.vault_config import VaultLocation, _apply_yaml_patch

        _apply_yaml_patch(
            config_path, patch_spec, VaultLocation(url="http://localhost:9999", socket="")
        )

        captured = capsys.readouterr()
        assert "Warning" not in captured.err


# ---------------------------------------------------------------------------
# roster: _load_bundled_agents warns on parse failure
# ---------------------------------------------------------------------------


class TestLoadBundledAgentsWarning:
    """_load_bundled_agents warns when a bundled agent YAML fails to parse."""

    def test_warns_on_bundled_parse_failure(self, capsys: pytest.CaptureFixture) -> None:
        """A bundled agent that raises on read_text triggers a warning and is skipped."""
        good_item = MagicMock()
        good_item.name = "good.yaml"
        good_item.read_text.return_value = "kind: agent\nlabel: Good\nbinary: good\n"

        bad_item = MagicMock()
        bad_item.name = "broken.yaml"
        bad_item.read_text.side_effect = OSError("permission denied")

        mock_pkg = MagicMock()
        mock_pkg.iterdir.return_value = [good_item, bad_item]

        with patch("importlib.resources.files", return_value=mock_pkg):
            from terok_executor.roster.loader import _load_bundled_agents

            agents = _load_bundled_agents()

        captured = capsys.readouterr()
        assert "Warning [roster]" in captured.err
        assert "broken" in captured.err
        assert "OSError" in captured.err

        # Good agent still loaded, broken one skipped
        assert "good" in agents
        assert "broken" not in agents

    def test_warns_on_bundled_yaml_syntax_error(self, capsys: pytest.CaptureFixture) -> None:
        """A bundled YAML file with syntax errors triggers a warning and is skipped."""
        bad_yaml_item = MagicMock()
        bad_yaml_item.name = "garbled.yaml"
        bad_yaml_item.read_text.return_value = ":\n  - :\n  bad: [unterminated"

        mock_pkg = MagicMock()
        mock_pkg.iterdir.return_value = [bad_yaml_item]

        with patch("importlib.resources.files", return_value=mock_pkg):
            from terok_executor.roster.loader import _load_bundled_agents

            agents = _load_bundled_agents()

        captured = capsys.readouterr()
        assert "Warning [roster]" in captured.err
        assert "garbled" in captured.err

        assert "garbled" not in agents


# ---------------------------------------------------------------------------
# roster: _load_user_agents warns on parse failure
# ---------------------------------------------------------------------------


class TestLoadUserAgentsWarning:
    """_load_user_agents warns when a user agent YAML file fails to parse."""

    def test_warns_on_user_agent_parse_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A user agent YAML with invalid content triggers a warning and is skipped."""
        user_dir = tmp_path / "agents"
        user_dir.mkdir()
        (user_dir / "good.yaml").write_text("kind: agent\nlabel: Good\nbinary: good\n")
        (user_dir / "broken.yaml").write_text(":\n  - :\n  bad: [unterminated")

        with patch("terok_executor.roster.loader._user_agents_dir", return_value=user_dir):
            from terok_executor.roster.loader import _load_user_agents

            agents = _load_user_agents()

        captured = capsys.readouterr()
        assert "Warning [roster]" in captured.err
        assert str(user_dir / "broken.yaml") in captured.err

        # Good agent loaded, broken one skipped
        assert "good" in agents
        assert "broken" not in agents

    def test_warns_on_user_agent_read_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A user agent file that raises on read triggers a warning and is skipped."""
        user_dir = tmp_path / "agents"
        user_dir.mkdir()
        bad_file = user_dir / "unreadable.yaml"
        bad_file.write_text("kind: agent\n")

        with (
            patch("terok_executor.roster.loader._user_agents_dir", return_value=user_dir),
            patch.object(Path, "read_text", side_effect=PermissionError("denied")),
        ):
            from terok_executor.roster.loader import _load_user_agents

            agents = _load_user_agents()

        captured = capsys.readouterr()
        assert "Warning [roster]" in captured.err
        assert "PermissionError" in captured.err

        assert "unreadable" not in agents

    def test_no_warning_when_dir_missing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Missing user agents directory produces no warning (expected case)."""
        with patch(
            "terok_executor.roster.loader._user_agents_dir", return_value=tmp_path / "nonexistent"
        ):
            from terok_executor.roster.loader import _load_user_agents

            agents = _load_user_agents()

        captured = capsys.readouterr()
        assert "Warning" not in captured.err
        assert agents == {}
