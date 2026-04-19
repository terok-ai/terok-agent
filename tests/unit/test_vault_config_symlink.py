# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the TOCTOU symlink hardening in vault_config.

The shared mount dir is bind-mounted read-write into task containers, so a
compromised container can plant a symlink between the ``_safe_config_path``
check and the write — without ``O_NOFOLLOW`` that would silently redirect
the write to any file writable by the executor's user.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_executor.credentials.vault_config import (
    ConfigPatchError,
    VaultLocation,
    _apply_toml_patch,  # noqa: PLC2701
    _apply_yaml_patch,  # noqa: PLC2701
    _read_nofollow,  # noqa: PLC2701
    _write_nofollow,  # noqa: PLC2701
)

_TEST_LOCATION = VaultLocation(url="http://vault", socket="/tmp/terok-testing/vault.sock")


class TestWriteNofollow:
    """_write_nofollow refuses to follow symlinks."""

    def test_writes_regular_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        _write_nofollow(path, b"hello")
        assert path.read_bytes() == b"hello"

    def test_overwrites_existing_regular_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_bytes(b"old")
        _write_nofollow(path, b"new")
        assert path.read_bytes() == b"new"

    def test_refuses_to_follow_symlink_at_target(self, tmp_path: Path) -> None:
        victim = tmp_path / "victim"
        victim.write_bytes(b"UNCHANGED")
        path = tmp_path / "config.toml"
        path.symlink_to(victim)

        with pytest.raises(ConfigPatchError, match="refusing to write"):
            _write_nofollow(path, b"ATTACKER")

        # Victim file was not touched.
        assert victim.read_bytes() == b"UNCHANGED"


class TestReadNofollow:
    """_read_nofollow refuses to follow symlinks."""

    def test_reads_regular_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_bytes(b"hello")
        assert _read_nofollow(path) == b"hello"

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        assert _read_nofollow(tmp_path / "missing") is None

    def test_refuses_to_follow_symlink(self, tmp_path: Path) -> None:
        secret = tmp_path / "secret"
        secret.write_bytes(b"SECRET")
        link = tmp_path / "config.yml"
        link.symlink_to(secret)
        with pytest.raises(ConfigPatchError, match="refusing to read"):
            _read_nofollow(link)


class TestApplyPatchesSymlinkSafety:
    """End-to-end: TOML / YAML patch helpers refuse planted symlinks."""

    def test_toml_patch_rejects_symlinked_config(self, tmp_path: Path) -> None:
        victim = tmp_path / "victim"
        victim.write_bytes(b"UNCHANGED")
        config = tmp_path / "config.toml"
        config.symlink_to(victim)

        patch = {
            "file": "config.toml",
            "toml_table": "providers",
            "toml_match": {"name": "mistral"},
            "toml_set": {"base_url": "{vault_url}"},
        }
        with pytest.raises(ConfigPatchError, match="symlink"):
            _apply_toml_patch(config, patch, _TEST_LOCATION)
        assert victim.read_bytes() == b"UNCHANGED"

    def test_yaml_patch_rejects_symlinked_config(self, tmp_path: Path) -> None:
        victim = tmp_path / "victim"
        victim.write_bytes(b"UNCHANGED")
        config = tmp_path / "config.yml"
        config.symlink_to(victim)

        patch = {
            "file": "config.yml",
            "yaml_set": {"http_unix_socket": "{vault_socket}"},
        }
        with pytest.raises(ConfigPatchError, match="symlink"):
            _apply_yaml_patch(config, patch, _TEST_LOCATION)
        assert victim.read_bytes() == b"UNCHANGED"

    def test_toml_patch_writes_when_no_symlink(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        patch = {
            "file": "config.toml",
            "toml_table": "providers",
            "toml_match": {"name": "mistral"},
            "toml_set": {"base_url": "{vault_url}/v1"},
        }
        _apply_toml_patch(config, patch, _TEST_LOCATION)
        assert config.is_file() and not config.is_symlink()
        text = config.read_text(encoding="utf-8")
        assert "http://vault/v1" in text
