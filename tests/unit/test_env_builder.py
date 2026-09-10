# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the container environment assembly function."""

from __future__ import annotations

import dataclasses
import json
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
from terok_sandbox import SandboxConfig, VolumeSpec

from terok_executor.container.env import (
    ContainerEnvResult,
    ContainerEnvSpec,
    _resolve_git_identity,
    _shared_config_mounts,
    assemble_container_env,
)
from terok_executor.integrations.sandbox import (
    CONTAINER_SSH_SIGNER_SOCKET,
    CONTAINER_VAULT_SOCKET,
)
from terok_executor.roster import AgentRoster
from terok_executor.roster.types import Provider, ProviderAuth, ProviderModel, VaultRoute
from terok_executor.vault_addr import LOOPBACK_BRIDGE_SOCKET
from tests.constants import EXAMPLE_PROVIDER_HOST, EXAMPLE_PROVIDER_UPSTREAM
from tests.unit.conftest import TEST_VAULT_PASSPHRASE

_EXAMPLE_MODEL = "example-chat"


def _find_vol(volumes: tuple[VolumeSpec, ...], container_path: str) -> VolumeSpec | None:
    """Find a VolumeSpec by container_path prefix."""
    return next((v for v in volumes if container_path in v.container_path), None)


def _make_vault_db(
    tmp_path: Path,
    cred_name: str = "anthropic",
    cred_data: dict | None = None,
    *,
    credential_set: str = "default",
):
    """Return a ``SandboxConfig`` with one credential pre-stored in its ``CredentialDB``.

    The DB is created, populated, and closed internally.  Credentials are keyed
    by *provider* name (``"anthropic"``) — the post-v3-migration vault contract
    that ``routed = stored & roster.providers`` intersects against.
    """
    from terok_sandbox import CredentialDB, SandboxConfig

    cfg = SandboxConfig(state_dir=tmp_path, vault_dir=tmp_path / "credentials")
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = CredentialDB(cfg.db_path, passphrase=TEST_VAULT_PASSPHRASE)
    db.store_credential(
        credential_set,
        cred_name,
        {"type": "api_key", "key": "sk-test"} if cred_data is None else cred_data,
    )
    db.close()
    return cfg


def _make_vault_db_with_ssh_keys(tmp_path: Path, scope: str = "myproj"):
    """Return a SandboxConfig with credential DB seeded with a key assigned to *scope*."""
    from terok_sandbox.vault.ssh.keypair import generate_keypair
    from terok_sandbox.vault.store.db import CredentialDB

    cfg = _make_vault_db(tmp_path)
    cfg.vault_dir.mkdir(parents=True, exist_ok=True)
    db = CredentialDB(cfg.db_path, passphrase=TEST_VAULT_PASSPHRASE)
    try:
        kp = generate_keypair("ed25519", comment=f"tk-main:{scope}")
        key_id = db.store_ssh_key(
            key_type=kp.key_type,
            private_der=kp.private_der,
            public_blob=kp.public_blob,
            comment=kp.comment,
            fingerprint=kp.fingerprint,
        )
        db.assign_ssh_key(scope, key_id)
    finally:
        db.close()
    return cfg


@pytest.fixture
def roster():
    """Return the live agent roster (loaded from bundled YAML)."""
    return AgentRoster.shared()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Return a pre-created workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def envs_dir(tmp_path: Path) -> Path:
    """Temp-backed envs directory — prevents mutating real home."""
    d = tmp_path / "envs"
    d.mkdir()
    return d


@pytest.fixture
def base_spec(workspace: Path, envs_dir: Path) -> ContainerEnvSpec:
    """Minimal spec with only required fields (all dirs tmp-backed)."""
    return ContainerEnvSpec(
        task_id="test-123",
        agent_name="claude",
        workspace_host_path=workspace,
        envs_dir=envs_dir,
    )


def _spec(workspace: Path, envs_dir: Path, **overrides) -> ContainerEnvSpec:
    """Shorthand for a tmp-backed spec with overrides."""
    defaults = {
        "task_id": "t1",
        "agent_name": "claude",
        "workspace_host_path": workspace,
        "envs_dir": envs_dir,
    }
    return ContainerEnvSpec(**(defaults | overrides))


# ---------------------------------------------------------------------------
# assemble_container_env — base env
# ---------------------------------------------------------------------------


class TestBaseEnv:
    """Verify base environment variables are always set."""

    def test_task_id(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert result.env["TASK_ID"] == "test-123"

    def test_repo_root(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert result.env["REPO_ROOT"] == "/workspace"

    def test_git_reset_mode(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert result.env["GIT_RESET_MODE"] == "none"

    def test_claude_config_dir(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert result.env["CLAUDE_CONFIG_DIR"] == "/home/dev/.claude"

    def test_returns_frozen_result(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert isinstance(result, ContainerEnvResult)

    def test_container_protocol_marker(self, base_spec, roster):
        """Every container is stamped with the contract version it was created against.

        The number is a record, not a gate — it tells an operator which
        host↔container contract a long-lived container predates.  Protocol 3
        marks terok-sandbox #491, which moved the ``/run/terok`` sockets into
        per-service subdirectories and so changed the socket env vars.  The
        equality is a tripwire: bumping it is a deliberate act.
        """
        from terok_executor.container.env import CONTAINER_PROTOCOL

        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert CONTAINER_PROTOCOL == 3
        assert result.env["TEROK_CONTAINER_PROTOCOL"] == str(CONTAINER_PROTOCOL)


class TestProviderSelection:
    """Verify the selected provider is injected as ``TEROK_PROVIDER``."""

    def test_no_provider_unset(self, base_spec, roster):
        """No provider selected → ``TEROK_PROVIDER`` is absent."""
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert "TEROK_PROVIDER" not in result.env

    def test_provider_injected(self, workspace, envs_dir, roster):
        """A selected provider lands in ``TEROK_PROVIDER`` for the wrappers."""
        spec = _spec(workspace, envs_dir, provider="openrouter")
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["TEROK_PROVIDER"] == "openrouter"


# ---------------------------------------------------------------------------
# Git identity
# ---------------------------------------------------------------------------


class TestGitIdentity:
    """Verify git identity resolution from spec fields or roster fallback."""

    def test_identity_from_roster_provider(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir)
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["GIT_AUTHOR_NAME"] == "Claude"
        assert result.env["GIT_AUTHOR_EMAIL"] == "noreply@anthropic.com"
        assert result.env["GIT_COMMITTER_NAME"] == "Claude"

    def test_explicit_identity_overrides_roster(self, workspace, envs_dir, roster):
        spec = _spec(
            workspace,
            envs_dir,
            git_author_name="Human Author",
            git_author_email="human@example.com",
            git_committer_name="AI Committer",
            git_committer_email="ai@example.com",
        )
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["GIT_AUTHOR_NAME"] == "Human Author"
        assert result.env["GIT_AUTHOR_EMAIL"] == "human@example.com"
        assert result.env["GIT_COMMITTER_NAME"] == "AI Committer"
        assert result.env["GIT_COMMITTER_EMAIL"] == "ai@example.com"

    def test_committer_defaults_to_author(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir, git_author_name="Custom", git_author_email="custom@t.com")
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["GIT_COMMITTER_NAME"] == "Custom"
        assert result.env["GIT_COMMITTER_EMAIL"] == "custom@t.com"

    def test_unknown_provider_uses_fallback(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir, agent_name="nonexistent")
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["GIT_AUTHOR_NAME"] == "AI Agent"


# ---------------------------------------------------------------------------
# Authorship env
# ---------------------------------------------------------------------------


class TestAuthorship:
    """Verify authorship mode and human identity env vars."""

    def test_defaults(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert result.env["TEROK_GIT_AUTHORSHIP"] == "agent"
        assert result.env["HUMAN_GIT_NAME"] == "Nobody"
        assert result.env["HUMAN_GIT_EMAIL"] == "nobody@localhost"

    def test_custom_authorship(self, workspace, envs_dir, roster):
        spec = _spec(
            workspace,
            envs_dir,
            authorship="agent-human",
            human_name="Jane Doe",
            human_email="jane@example.com",
        )
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["TEROK_GIT_AUTHORSHIP"] == "agent-human"
        assert result.env["HUMAN_GIT_NAME"] == "Jane Doe"
        assert result.env["HUMAN_GIT_EMAIL"] == "jane@example.com"


# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------


class TestTimezone:
    """Verify TZ propagation: explicit override wins, otherwise follow the host."""

    def test_explicit_override(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir, timezone="Europe/Prague")
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["TZ"] == "Europe/Prague"

    def test_explicit_utc_pins_container(self, workspace, envs_dir, roster):
        """Passing ``"UTC"`` explicitly is how callers opt out of host-follow."""
        spec = _spec(workspace, envs_dir, timezone="UTC")
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["TZ"] == "UTC"

    def test_detects_host_when_unset(self, base_spec, roster):
        with patch("terok_executor.container.env.detect_host_timezone") as mock_detect:
            mock_detect.return_value = "America/Los_Angeles"
            result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert result.env["TZ"] == "America/Los_Angeles"

    def test_explicit_wins_over_detection(self, workspace, envs_dir, roster):
        """Override takes precedence — detection is never even consulted."""
        spec = _spec(workspace, envs_dir, timezone="Asia/Tokyo")
        with patch("terok_executor.container.env.detect_host_timezone") as mock_detect:
            mock_detect.return_value = "Europe/Berlin"
            result = assemble_container_env(spec, roster, caller_manages_vault=True)
            mock_detect.assert_not_called()
        assert result.env["TZ"] == "Asia/Tokyo"

    def test_undetectable_host_omits_tz(self, base_spec, roster):
        """No override + no detectable host TZ → leave TZ unset (use image default)."""
        with patch("terok_executor.container.env.detect_host_timezone") as mock_detect:
            mock_detect.return_value = None
            result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert "TZ" not in result.env


# ---------------------------------------------------------------------------
# Repository setup
# ---------------------------------------------------------------------------


class TestRepoSetup:
    """Verify repository env vars and branch."""

    def test_code_repo(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir, code_repo="http://gate@host:9418/repo")
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["CODE_REPO"] == "http://gate@host:9418/repo"

    def test_clone_from(self, workspace, envs_dir, roster):
        spec = _spec(
            workspace,
            envs_dir,
            clone_from="http://gate@host:9418/mirror",
            code_repo="https://github.com/user/repo",
        )
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["CLONE_FROM"] == "http://gate@host:9418/mirror"
        assert result.env["CODE_REPO"] == "https://github.com/user/repo"

    def test_branch(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir, branch="feat/my-branch")
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["GIT_BRANCH"] == "feat/my-branch"

    def test_no_branch_omits_key(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert "GIT_BRANCH" not in result.env

    def test_no_code_repo_omits_key(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert "CODE_REPO" not in result.env


# ---------------------------------------------------------------------------
# Workspace volume
# ---------------------------------------------------------------------------


class TestWorkspaceVolume:
    """Verify workspace volume mount."""

    def test_workspace_mounted_with_exclusive_label(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        ws = _find_vol(result.volumes, "/workspace")
        assert ws is not None
        assert ws.host_path == base_spec.workspace_host_path
        assert ws.sharing == "private"

    def test_in_container_workspace_mounts_nothing(self, envs_dir, roster):
        """No workspace_host_path → no /workspace volume (writable-layer cell)."""
        spec = ContainerEnvSpec(task_id="t1", agent_name="claude", envs_dir=envs_dir)
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert _find_vol(result.volumes, "/workspace") is None
        assert result.env["REPO_ROOT"] == "/workspace"


# ---------------------------------------------------------------------------
# Shared config mounts
# ---------------------------------------------------------------------------


class TestSharedConfigMounts:
    """Verify roster-derived shared config mounts."""

    def test_claude_config_mounted(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir)
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        claude = _find_vol(result.volumes, "/home/dev/.claude")
        assert claude is not None
        assert "_claude-config" in str(claude.host_path)
        assert claude.sharing == "shared"

    def test_shared_mounts_use_lowercase_z(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir)
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        shared = [v for v in result.volumes if v.sharing == "shared"]
        assert len(shared) > 0

    def test_host_dirs_created(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir)
        assemble_container_env(spec, roster, caller_manages_vault=True)
        assert (envs_dir / "_claude-config").is_dir()


# ---------------------------------------------------------------------------
# Agent config mount
# ---------------------------------------------------------------------------


class TestAgentConfigMount:
    """Verify agent config directory mount."""

    def test_agent_config_mounted_when_set(self, workspace, envs_dir, roster, tmp_path):
        cfg_dir = tmp_path / "agent-config"
        cfg_dir.mkdir()
        spec = _spec(workspace, envs_dir, agent_config_dir=cfg_dir)
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        vol = _find_vol(result.volumes, "/home/dev/.terok")
        assert vol is not None
        assert vol.host_path == cfg_dir
        assert vol.sharing == "private"

    def test_no_agent_config_when_none(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert _find_vol(result.volumes, "/home/dev/.terok") is None


# ---------------------------------------------------------------------------
# Unrestricted mode
# ---------------------------------------------------------------------------


class TestUnrestrictedMode:
    """Verify unrestricted/auto-approve env injection."""

    def test_unrestricted_sets_env(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir, unrestricted=True)
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.env["TEROK_UNRESTRICTED"] == "1"

    def test_restricted_omits_env(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir, unrestricted=False)
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert "TEROK_UNRESTRICTED" not in result.env


# ---------------------------------------------------------------------------
# Shared task directory
# ---------------------------------------------------------------------------


class TestSharedTaskDir:
    """Verify shared task directory mount and env var."""

    def test_shared_dir_mounted_when_set(self, workspace, envs_dir, roster, tmp_path):
        shared = tmp_path / "shared"
        spec = _spec(workspace, envs_dir, shared_dir=shared)
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        vol = _find_vol(result.volumes, "/shared")
        assert vol is not None and vol.host_path == shared and vol.sharing == "shared"
        assert result.env["TEROK_SHARED_DIR"] == "/shared"

    def test_shared_dir_custom_mount(self, workspace, envs_dir, roster, tmp_path):
        shared = tmp_path / "data"
        spec = _spec(workspace, envs_dir, shared_dir=shared, shared_mount="/data/ipc")
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        vol = _find_vol(result.volumes, "/data/ipc")
        assert vol is not None and vol.host_path == shared
        assert result.env["TEROK_SHARED_DIR"] == "/data/ipc"

    def test_shared_dir_created(self, workspace, envs_dir, roster, tmp_path):
        shared = tmp_path / "new-shared"
        spec = _spec(workspace, envs_dir, shared_dir=shared)
        assemble_container_env(spec, roster, caller_manages_vault=True)
        assert shared.is_dir()

    def test_no_shared_dir_by_default(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert "TEROK_SHARED_DIR" not in result.env
        assert _find_vol(result.volumes, "/shared") is None


# ---------------------------------------------------------------------------
# Extra volumes
# ---------------------------------------------------------------------------


class TestExtraVolumes:
    """Verify caller-provided extra volumes are appended."""

    def test_extra_volumes_appended(self, workspace, envs_dir, roster):
        extra = VolumeSpec(Path("/host/ssh"), "/home/dev/.ssh")
        spec = _spec(workspace, envs_dir, extra_volumes=(extra,))
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        vol = _find_vol(result.volumes, "/home/dev/.ssh")
        assert vol is not None and vol.host_path == Path("/host/ssh")


# ---------------------------------------------------------------------------
# Vault token injection
# ---------------------------------------------------------------------------


class TestVaultTokenInjection:
    """Verify vault token injection."""

    def test_caller_manages_vault_skips_injection(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_running_injects_tokens(self, workspace, envs_dir, roster, tmp_path):
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="test-project")

        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "ANTHROPIC_API_KEY" in result.env
        assert result.env["ANTHROPIC_API_KEY"].startswith("terok-p-")

    def test_no_routed_providers_returns_empty(self, workspace, envs_dir, roster, tmp_path):
        """Stored credentials that don't match any vault route produce no tokens."""
        cfg = _make_vault_db(tmp_path, "nonexistent-provider")
        spec = _spec(workspace, envs_dir)
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_db_error_returns_empty(self, base_spec, roster):
        """DB open failure returns empty env gracefully."""
        with patch(
            "terok_sandbox.config.SandboxConfig.open_credential_db",
            side_effect=OSError("corrupt"),
        ):
            result = assemble_container_env(base_spec, roster, caller_manages_vault=False)
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_oauth_credential_uses_oauth_token_env(
        self, workspace, envs_dir, roster, tmp_path
    ):
        """OAuth credential selects the oauth ``token_env`` entry (CLAUDE_CODE_OAUTH_TOKEN)."""
        cfg = _make_vault_db(tmp_path, cred_data={"type": "oauth", "access_token": "oa-tok"})
        spec = _spec(workspace, envs_dir, credential_scope="test-project")
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "CLAUDE_CODE_OAUTH_TOKEN" in result.env
        assert result.env["CLAUDE_CODE_OAUTH_TOKEN"].startswith("terok-p-")
        # API key env var must NOT be set when OAuth credential is stored
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_codex_apps_receives_same_phantom_token(self, workspace, envs_dir, roster, tmp_path):
        """Codex's Apps MCP bearer aliases the protected OpenAI handle."""
        cfg = _make_vault_db(
            tmp_path,
            "openai",
            {"type": "oauth", "access_token": "oauth-token", "account_id": "account-a"},
        )
        spec = _spec(workspace, envs_dir, agent_name="codex")

        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert result.env["CODEX_CONNECTORS_TOKEN"] == result.env["OPENAI_API_KEY"]
        assert result.env["CODEX_CONNECTORS_TOKEN"].startswith("terok-p-")

    def test_vault_api_key_uses_default_token_env(self, workspace, envs_dir, roster, tmp_path):
        """API-key credential falls back to the ``_default`` ``token_env`` entry."""
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="test-project")
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "ANTHROPIC_API_KEY" in result.env
        assert result.env["ANTHROPIC_API_KEY"].startswith("terok-p-")
        # OAuth env var must NOT be set for API key credentials
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in result.env

    def test_vault_token_creation_error_returns_empty(self, workspace, envs_dir, roster, tmp_path):
        """Token creation failure returns empty env gracefully."""
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir)

        with (
            patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg),
            patch(
                "terok_executor.integrations.sandbox.CredentialDB.create_token",
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_socket_transport_injects_socket_env(self, workspace, envs_dir, roster, tmp_path):
        """Socket transport ignores a stale configured TCP broker port."""
        cfg = dataclasses.replace(_make_vault_db(tmp_path), token_broker_port=18731)
        spec = _spec(workspace, envs_dir, credential_scope="proj", vault_transport="socket")
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert result.env["ANTHROPIC_UNIX_SOCKET"] == LOOPBACK_BRIDGE_SOCKET
        assert result.env["TEROK_VAULT_SOCKET"] == CONTAINER_VAULT_SOCKET
        assert "TEROK_TOKEN_BROKER_PORT" not in result.env

    def test_vault_direct_transport_points_socket_at_local_bridge(
        self, workspace, envs_dir, roster, tmp_path
    ):
        """Direct (TCP) transport still sets socket_env — now to the local socat bridge."""
        cfg = dataclasses.replace(_make_vault_db(tmp_path), token_broker_port=18731)
        spec = _spec(workspace, envs_dir, credential_scope="proj", vault_transport="direct")
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert result.env["ANTHROPIC_UNIX_SOCKET"] == "/tmp/terok-vault.sock"
        assert "TEROK_VAULT_SOCKET" not in result.env

    def test_vault_socket_transport_omits_tcp_broker_env_when_port_none(
        self, workspace, envs_dir, roster, tmp_path
    ):
        """Socket-only deployments have no TCP broker port; the env must reflect that.

        Under socket transport the broker has no TCP listener — the
        port field on the config is ``None``.  The assembled env must
        omit the TCP broker port variable rather than interpolate the
        literal string ``"None"`` — that string would otherwise trip
        bridge scripts and silently break credential routing.
        """
        cfg = dataclasses.replace(_make_vault_db(tmp_path), token_broker_port=None)
        spec = _spec(workspace, envs_dir, credential_scope="proj", vault_transport="socket")
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "TEROK_TOKEN_BROKER_PORT" not in result.env
        # GITLAB_API_HOST is now always set for glab — but never with "None".
        assert not any("None" in v for v in result.env.values())
        # Socket transport uses the mounted host socket directly.
        assert result.env.get("ANTHROPIC_UNIX_SOCKET") == LOOPBACK_BRIDGE_SOCKET
        assert result.env.get("TEROK_VAULT_SOCKET") == CONTAINER_VAULT_SOCKET
        # The in-container loopback port is advertised so ensure-bridges.sh
        # stands up its TCP→UNIX bridge.
        assert result.env.get("TEROK_VAULT_LOOPBACK_PORT") == "9419"

    def test_vault_injects_ssh_signer_token(self, workspace, envs_dir, roster, tmp_path):
        """SSH signer token injected when scope has valid keys in ssh-keys.json."""
        cfg = dataclasses.replace(_make_vault_db_with_ssh_keys(tmp_path), ssh_signer_port=18732)
        spec = _spec(workspace, envs_dir, credential_scope="myproj")
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "TEROK_SSH_SIGNER_TOKEN" in result.env
        assert result.env["TEROK_SSH_SIGNER_TOKEN"].startswith("terok-p-")
        assert result.env["TEROK_SSH_SIGNER_PORT"] == "18732"
        assert "TEROK_SSH_SIGNER_SOCKET" not in result.env

    def test_per_container_broker_port_wins_over_cfg_singleton(
        self, workspace, envs_dir, roster, tmp_path
    ):
        """Production path: when the runner passes ``per_container``, the broker
        port comes from *that* allocation, not cfg's host-singleton field.

        Concurrent containers each get a distinct broker port; reading the
        singleton ``cfg.token_broker_port`` would collide them.  The cfg
        field is deliberately set to a *different* value so a regression
        that ignored ``per_container`` would surface here.
        """
        from terok_sandbox import PerContainerResources

        cfg = dataclasses.replace(_make_vault_db(tmp_path), token_broker_port=19000)
        spec = _spec(workspace, envs_dir, credential_scope="test-project", vault_transport="direct")
        per_container = PerContainerResources(
            container_runtime_dir=Path("/run/terok/sandbox/run/c"),
            token_broker_port=20555,
            ssh_signer_port=20556,
            gate_port=20557,
        )
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(
                spec, roster, caller_manages_vault=False, per_container=per_container
            )

        assert result.env["TEROK_TOKEN_BROKER_PORT"] == "20555"

    def test_per_container_ssh_signer_port_wins_over_cfg_singleton(
        self, workspace, envs_dir, roster, tmp_path
    ):
        """Same per-container precedence for the SSH signer port (TCP/direct)."""
        from terok_sandbox import PerContainerResources

        cfg = dataclasses.replace(_make_vault_db_with_ssh_keys(tmp_path), ssh_signer_port=18732)
        spec = _spec(workspace, envs_dir, credential_scope="myproj", vault_transport="direct")
        per_container = PerContainerResources(
            container_runtime_dir=Path("/run/terok/sandbox/run/c"),
            token_broker_port=20555,
            ssh_signer_port=20999,
            gate_port=20557,
        )
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(
                spec, roster, caller_manages_vault=False, per_container=per_container
            )

        assert result.env["TEROK_SSH_SIGNER_TOKEN"].startswith("terok-p-")
        assert result.env["TEROK_SSH_SIGNER_PORT"] == "20999"

    def test_vault_ssh_signer_socket_transport(self, workspace, envs_dir, roster, tmp_path):
        """Socket transport injects TEROK_SSH_SIGNER_SOCKET instead of _PORT."""
        cfg = _make_vault_db_with_ssh_keys(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="myproj", vault_transport="socket")
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "TEROK_SSH_SIGNER_TOKEN" in result.env
        assert result.env["TEROK_SSH_SIGNER_SOCKET"] == CONTAINER_SSH_SIGNER_SOCKET
        assert "TEROK_SSH_SIGNER_PORT" not in result.env

    def test_vault_no_ssh_keys_omits_token(self, workspace, envs_dir, roster, tmp_path):
        """No SSH signer token when ssh-keys.json has no entry for scope."""
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="no-keys-project")
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "TEROK_SSH_SIGNER_TOKEN" not in result.env
        assert "TEROK_SSH_SIGNER_PORT" not in result.env

    def test_vault_credential_set_isolates_lookups(self, workspace, envs_dir, roster, tmp_path):
        """``credential_set`` selects which vault DB namespace gets read.

        A credential stored under set ``other`` must be invisible when the
        spec asks for the default set, and vice versa — otherwise per-project
        opt-in would silently leak shared tokens into project-scoped tasks.
        """
        cfg = _make_vault_db(tmp_path, credential_set="my-proj")
        # spec without credential_set override → reads from "default" → no creds visible
        default_spec = _spec(workspace, envs_dir, credential_scope="my-proj")
        # spec with matching credential_set → reads from "my-proj" → token issued
        scoped_spec = _spec(
            workspace, envs_dir, credential_scope="my-proj", credential_set="my-proj"
        )
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            default_result = assemble_container_env(
                default_spec, roster, caller_manages_vault=False
            )
            scoped_result = assemble_container_env(scoped_spec, roster, caller_manages_vault=False)

        assert "ANTHROPIC_API_KEY" not in default_result.env
        assert "ANTHROPIC_API_KEY" in scoped_result.env
        assert scoped_result.env["ANTHROPIC_API_KEY"].startswith("terok-p-")

    def test_vault_materializes_generic_provider_handle(
        self, workspace, envs_dir, roster, tmp_path
    ):
        """An authenticated provider gets a generic ``TEROK_PROVIDER_*`` handle so a
        harness can select it at runtime, and OpenRouter keeps its mandatory ``/api``."""
        cfg = _make_vault_db(tmp_path, cred_name="openrouter")
        spec = _spec(workspace, envs_dir)
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            env = assemble_container_env(spec, roster, caller_manages_vault=False).env

        assert env["TEROK_PROVIDER_OPENROUTER_TOKEN"].startswith("terok-p-")
        # OpenRouter serves openai-chat at /api/v1 and its Anthropic skin at /api.
        assert env["TEROK_PROVIDER_OPENROUTER_BASE_OPENAI_CHAT"].endswith("/api/v1")
        assert env["TEROK_PROVIDER_OPENROUTER_BASE_ANTHROPIC_MESSAGES"].endswith("/api")
        # The curated OpenCode base URL picks up the same /api/v1, not a bare /v1.
        assert env["TEROK_OC_OPENROUTER_BASE_URL"].endswith("/api/v1")

    def test_new_provider_refreshes_routes_and_projects_model_metadata(
        self, workspace, envs_dir, roster, tmp_path
    ):
        """Token assembly makes a new provider routable without an image rebuild.

        The same materialized handle gives every harness the provider label,
        default, and provider-neutral model limits.
        """
        provider = Provider(
            name="example",
            label="Example",
            upstream=EXAMPLE_PROVIDER_UPSTREAM,
            api_key_auth=ProviderAuth(header="Authorization", prefix="Bearer "),
            serves={"openai-chat": "/v1"},
            default_model=_EXAMPLE_MODEL,
            models={
                _EXAMPLE_MODEL: ProviderModel(
                    name="Example Chat",
                    context_limit=120_000,
                )
            },
        )
        route = VaultRoute(
            provider="example",
            route_prefix="example",
            upstream=EXAMPLE_PROVIDER_UPSTREAM,
            token_env={"_default": "EXAMPLE_API_KEY"},
        )
        live_roster = dataclasses.replace(
            roster,
            _providers=roster.providers | {"example": provider},
            _vault_routes=roster.vault_routes | {"example": route},
        )
        cfg = _make_vault_db(tmp_path, cred_name="example")
        cfg.routes_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.routes_path.write_text("{}\n", encoding="utf-8")
        spec = _spec(workspace, envs_dir)

        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, live_roster, caller_manages_vault=False)
        env = result.env

        assert "example" in json.loads(cfg.routes_path.read_text(encoding="utf-8"))
        assert EXAMPLE_PROVIDER_HOST in result.egress.deny_to_vault
        assert env["TEROK_PROVIDER_EXAMPLE_LABEL"] == "Example"
        assert env["TEROK_PROVIDER_EXAMPLE_DEFAULT_MODEL"] == _EXAMPLE_MODEL
        assert json.loads(env["TEROK_PROVIDER_EXAMPLE_MODELS"]) == {
            _EXAMPLE_MODEL: {
                "context_limit": 120_000,
                "name": "Example Chat",
            }
        }

    def test_vault_ssh_only_no_provider_creds(self, workspace, envs_dir, roster, tmp_path):
        """SSH signer token injected even when no provider credentials are stored."""
        from terok_sandbox import CredentialDB, SandboxConfig
        from terok_sandbox.vault.ssh.keypair import generate_keypair

        cfg = SandboxConfig(state_dir=tmp_path, vault_dir=tmp_path / "credentials")
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_dir.mkdir(parents=True, exist_ok=True)
        # DB exists with NO provider credentials — only SSH keys.
        db = CredentialDB(cfg.db_path, passphrase=TEST_VAULT_PASSPHRASE)
        try:
            kp = generate_keypair("ed25519", comment="tk-main:sshonly")
            key_id = db.store_ssh_key(
                key_type=kp.key_type,
                private_der=kp.private_der,
                public_blob=kp.public_blob,
                comment=kp.comment,
                fingerprint=kp.fingerprint,
            )
            db.assign_ssh_key("sshonly", key_id)
        finally:
            db.close()

        spec = _spec(workspace, envs_dir, credential_scope="sshonly")
        with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "TEROK_SSH_SIGNER_TOKEN" in result.env
        assert result.env["TEROK_SSH_SIGNER_TOKEN"].startswith("terok-p-")
        # No provider tokens
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_required_hard_fails_on_db_error(self, workspace, envs_dir, roster):
        """vault_required=True raises SystemExit on credential-DB open failure.

        Post-supervisor-refactor the daemon-liveness probe is gone, but
        a DB open failure is still a hard stop in project-required mode
        because token creation cannot proceed without a usable DB.
        """
        spec = _spec(workspace, envs_dir, vault_required=True)
        with (
            patch(
                "terok_sandbox.config.SandboxConfig.open_credential_db",
                side_effect=OSError("corrupt"),
            ),
            pytest.raises(SystemExit, match="DB unavailable"),
        ):
            assemble_container_env(spec, roster, caller_manages_vault=False)

    def test_vault_required_hard_fails_on_token_error(self, workspace, envs_dir, roster, tmp_path):
        """vault_required=True raises SystemExit on token creation failure."""
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir, vault_required=True)
        with (
            patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=cfg),
            patch(
                "terok_executor.integrations.sandbox.CredentialDB.create_token",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(SystemExit, match="injection failed"),
        ):
            assemble_container_env(spec, roster, caller_manages_vault=False)

    def test_scan_leaked_creds_emits_warning(self, workspace, envs_dir, roster, caplog):
        """scan_leaked_creds=True logs warnings for leaked files."""
        spec = _spec(workspace, envs_dir, scan_leaked_creds=True)
        with patch(
            "terok_executor.credentials.vault_commands.scan_leaked_credentials",
            return_value=[
                ("claude", Path("/tmp/terok-testing/mounts/_claude-config/.credentials.json"))
            ],
        ):
            assemble_container_env(spec, roster, caller_manages_vault=True)
        assert any("claude" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Task dir
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OpenCode provider env
# ---------------------------------------------------------------------------


class TestOpenCodeEnv:
    """Verify OpenCode provider env vars from roster."""

    def test_opencode_vars_present(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        oc_vars = [k for k in result.env if k.startswith("TEROK_OC_")]
        assert len(oc_vars) > 0


# ---------------------------------------------------------------------------
# _resolve_git_identity (unit)
# ---------------------------------------------------------------------------


class TestResolveGitIdentityUnit:
    """Unit tests for the internal git identity resolver."""

    def test_spec_fields_take_precedence(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir, git_author_name="Override")
        identity = _resolve_git_identity(spec, roster)
        assert identity["GIT_AUTHOR_NAME"] == "Override"
        assert identity["GIT_COMMITTER_NAME"] == "Override"

    def test_roster_fallback(self, workspace, envs_dir, roster):
        spec = _spec(workspace, envs_dir)
        identity = _resolve_git_identity(spec, roster)
        assert identity["GIT_AUTHOR_NAME"] == "Claude"


# ---------------------------------------------------------------------------
# _shared_config_mounts (unit)
# ---------------------------------------------------------------------------


class TestSharedConfigPatches:
    """Verify vault config patches are applied during env assembly."""

    def test_apply_patches_writes_toml(self, roster, tmp_path):
        """Config patches in the roster must produce patched TOML files."""
        from terok_executor.credentials.vault_config import apply_shared_config_patches

        # Create the host mount dir that _shared_config_mounts would create.
        vibe_dir = tmp_path / "_vibe-config"
        vibe_dir.mkdir()
        codex_dir = tmp_path / "_codex-config"
        codex_dir.mkdir()

        with patch(
            "terok_executor.integrations.sandbox.SandboxConfig",
            return_value=SandboxConfig(token_broker_port=18731),
        ):
            apply_shared_config_patches(roster, tmp_path)

        config_path = vibe_dir / "config.toml"
        assert config_path.exists(), "apply_shared_config_patches must create config.toml"

        import tomllib

        data = tomllib.loads(config_path.read_text())
        providers = data.get("providers", [])
        mistral = next((p for p in providers if p.get("name") == "mistral"), None)
        assert mistral is not None, "config.toml must contain a mistral provider entry"
        # The container-side vault URL is uniform across transports: an
        # in-container bridge re-targets ``localhost:9419`` to either a unix
        # socket (socket mode) or the host's per-container TCP port (tcp mode),
        # so config patches always pin to the well-known loopback endpoint.
        assert "localhost:9419" in mistral["api_base"]

        codex_cfg = tomllib.loads((codex_dir / "config.toml").read_text())
        # No stored credential → falls back to the route's declared type (oauth),
        # so codex routes to the ChatGPT backend.
        assert codex_cfg["openai_base_url"] == "http://localhost:9419/backend-api/codex"
        assert codex_cfg["chatgpt_base_url"] == "http://localhost:9419/backend-api/"

    def test_codex_api_key_credential_routes_to_v1(self, roster, tmp_path):
        """An API-key codex credential routes inference to /v1, not the ChatGPT backend."""
        from terok_executor.credentials import vault_config
        from terok_executor.credentials.vault_config import apply_shared_config_patches

        codex_dir = tmp_path / "_codex-config"
        codex_dir.mkdir()
        with (
            patch(
                "terok_executor.integrations.sandbox.SandboxConfig",
                return_value=SandboxConfig(token_broker_port=18731),
            ),
            patch.object(vault_config, "_stored_credential_type", return_value="api_key"),
        ):
            apply_shared_config_patches(roster, tmp_path, providers=frozenset({"codex"}))

        import tomllib

        cfg = tomllib.loads((codex_dir / "config.toml").read_text())
        assert cfg["openai_base_url"] == "http://localhost:9419/v1"
        assert cfg["chatgpt_base_url"] == "http://localhost:9419/backend-api/"

    def test_credential_type_overlay_folds_per_type_keys(self):
        """The overlay merges by_credential_type[type] over toml_set; unknowns fall through."""
        from terok_executor.credentials.vault_config import _credential_type_overlay

        spec = {
            "file": "config.toml",
            "toml_set": {"chatgpt_base_url": "X"},
            "by_credential_type": {
                "oauth": {"openai_base_url": "BACKEND"},
                "api_key": {"openai_base_url": "V1"},
            },
        }
        assert _credential_type_overlay(spec, "oauth")["toml_set"] == {
            "chatgpt_base_url": "X",
            "openai_base_url": "BACKEND",
        }
        assert _credential_type_overlay(spec, "api_key")["toml_set"] == {
            "chatgpt_base_url": "X",
            "openai_base_url": "V1",
        }
        # Unknown type → base toml_set only (caller's declared-default fallback handles this).
        assert _credential_type_overlay(spec, "weird")["toml_set"] == {"chatgpt_base_url": "X"}
        # No overlay → patch returned unchanged.
        bare = {"toml_set": {"a": "b"}}
        assert _credential_type_overlay(bare, "oauth") is bare

    def test_applies_tool_config_patch(self, roster, tmp_path):
        """A *tool* (gh) — not in roster.agents — still gets its config patch.

        Regression guard: patch application must iterate auth providers (agents
        AND tools), not roster.agents, or gh's ``http_unix_socket`` vault
        routing is silently skipped.
        """
        from terok_executor.credentials.vault_config import apply_shared_config_patches

        with patch(
            "terok_executor.integrations.sandbox.SandboxConfig",
            return_value=SandboxConfig(token_broker_port=18731),
        ):
            apply_shared_config_patches(roster, tmp_path, providers=frozenset({"gh"}))

        gh_cfg = (tmp_path / "_gh-config" / "config.yml").read_text()
        assert "http_unix_socket" in gh_cfg  # vault socket wired in for the gh tool

    def test_disabled_provider_not_repatched_when_providers_none(self, roster, tmp_path):
        """Disabling a config patch must win even when ``providers=None`` (apply-all).

        Regression: with ``providers=None`` the entry landed in both the patched
        and disabled sets, so the removal ran and then the same patch was written
        back in the same call — disabling had no effect.
        """
        from terok_executor.credentials.vault_config import apply_shared_config_patches

        codex_dir = tmp_path / "_codex-config"
        codex_dir.mkdir()
        with patch(
            "terok_executor.integrations.sandbox.SandboxConfig",
            return_value=SandboxConfig(token_broker_port=18731),
        ):
            apply_shared_config_patches(
                roster, tmp_path, providers=None, disabled_providers=frozenset({"codex"})
            )

        cfg = codex_dir / "config.toml"
        assert not cfg.exists() or "openai_base_url" not in cfg.read_text()

    def test_assemble_env_calls_patches_with_and_without_bypass(self, workspace, envs_dir, roster):
        """assemble_container_env invokes patches regardless of caller_manages_vault."""
        for bypass in (True, False):
            with patch(
                "terok_executor.credentials.vault_config.apply_shared_config_patches"
            ) as m_patches:
                assemble_container_env(
                    spec=_spec(workspace, envs_dir), roster=roster, caller_manages_vault=bypass
                )

            m_patches.assert_called_once_with(
                roster,
                envs_dir,
                providers=None,
                disabled_providers=None,
                credential_set="default",
            )

    def test_patches_idempotent(self, roster, tmp_path):
        """Calling apply_shared_config_patches twice must not duplicate entries."""
        from terok_executor.credentials.vault_config import apply_shared_config_patches

        vibe_dir = tmp_path / "_vibe-config"
        vibe_dir.mkdir()

        with patch(
            "terok_executor.integrations.sandbox.SandboxConfig",
            return_value=SandboxConfig(token_broker_port=18731),
        ):
            apply_shared_config_patches(roster, tmp_path)
            apply_shared_config_patches(roster, tmp_path)

        config_path = vibe_dir / "config.toml"
        assert config_path.exists(), "config.toml must exist after double apply"

        import tomllib

        data = tomllib.loads(config_path.read_text())
        providers = data.get("providers", [])
        mistral_entries = [p for p in providers if p.get("name") == "mistral"]
        assert len(mistral_entries) == 1, "idempotent: must have exactly one mistral entry"

    def test_patches_can_be_limited_to_selected_providers(self, roster, tmp_path):
        """Provider filtering can skip Codex while still patching others."""
        from terok_executor.credentials.vault_config import apply_shared_config_patches

        (tmp_path / "_vibe-config").mkdir()
        (tmp_path / "_codex-config").mkdir()

        with patch(
            "terok_executor.integrations.sandbox.SandboxConfig",
            return_value=SandboxConfig(token_broker_port=18731),
        ):
            apply_shared_config_patches(roster, tmp_path, providers=frozenset({"vibe"}))

        assert (tmp_path / "_vibe-config" / "config.toml").exists()
        assert not (tmp_path / "_codex-config" / "config.toml").exists()

    def test_patches_write_managed_sidecar(self, roster, tmp_path):
        """Applied patches record the exact values terok owns."""
        from terok_executor.credentials.vault_config import apply_shared_config_patches

        (tmp_path / "_codex-config").mkdir()

        with patch(
            "terok_executor.integrations.sandbox.SandboxConfig",
            return_value=SandboxConfig(token_broker_port=18731),
        ):
            apply_shared_config_patches(roster, tmp_path, providers=frozenset({"codex"}))

        import json

        sidecar = json.loads(
            (tmp_path / "_codex-config" / ".terok-managed-config.json").read_text()
        )
        codex_records = sidecar["files"]["config.toml"]["providers"]["codex"]
        assert codex_records == [
            {
                "kind": "toml_top",
                "values": {
                    "openai_base_url": "http://localhost:9419/backend-api/codex",
                    "chatgpt_base_url": "http://localhost:9419/backend-api/",
                },
            }
        ]

    def test_disabled_provider_removes_owned_top_level_toml(self, roster, tmp_path):
        """Disabled providers remove stale top-level TOML keys that terok owns."""
        from terok_executor.credentials.vault_config import apply_shared_config_patches

        (tmp_path / "_codex-config").mkdir()

        with patch(
            "terok_executor.integrations.sandbox.SandboxConfig",
            return_value=SandboxConfig(token_broker_port=18731),
        ):
            apply_shared_config_patches(roster, tmp_path, providers=frozenset({"codex"}))

        apply_shared_config_patches(
            roster,
            tmp_path,
            providers=frozenset(),
            disabled_providers=frozenset({"codex"}),
        )

        import tomllib

        config = tomllib.loads((tmp_path / "_codex-config" / "config.toml").read_text())
        assert "openai_base_url" not in config
        assert "chatgpt_base_url" not in config
        assert not (tmp_path / "_codex-config" / ".terok-managed-config.json").exists()

    def test_disabled_provider_preserves_user_modified_value(self, roster, tmp_path):
        """A value changed after terok wrote it becomes user-owned and survives removal."""
        from terok_executor.credentials.vault_config import apply_shared_config_patches

        codex_dir = tmp_path / "_codex-config"
        codex_dir.mkdir()

        with patch(
            "terok_executor.integrations.sandbox.SandboxConfig",
            return_value=SandboxConfig(token_broker_port=18731),
        ):
            apply_shared_config_patches(roster, tmp_path, providers=frozenset({"codex"}))

        config_path = codex_dir / "config.toml"
        config_path.write_text(
            'openai_base_url = "https://user.example/v1"\n'
            'chatgpt_base_url = "http://localhost:9419/backend-api/"\n'
        )

        apply_shared_config_patches(
            roster,
            tmp_path,
            providers=frozenset(),
            disabled_providers=frozenset({"codex"}),
        )

        import tomllib

        config = tomllib.loads(config_path.read_text())
        assert config == {"openai_base_url": "https://user.example/v1"}
        assert not (codex_dir / ".terok-managed-config.json").exists()

    def test_disabled_provider_does_not_resolve_vault_location(self, roster, tmp_path):
        """Pure removal does not require the vault to have an address.

        ``resolve_vault_location`` is the only path that reads the
        broker port — confirm a removal-only call never touches it
        by failing if the helper is invoked at all.
        """
        from terok_executor.credentials.vault_config import apply_shared_config_patches

        (tmp_path / "_codex-config").mkdir()

        with patch(
            "terok_executor.credentials.vault_config.resolve_vault_location",
            side_effect=AssertionError("vault location should not be resolved"),
        ):
            apply_shared_config_patches(
                roster,
                tmp_path,
                providers=frozenset(),
                disabled_providers=frozenset({"codex"}),
            )


class TestConfigPatchSecurity:
    """Security constraints on vault config patching."""

    def test_path_traversal_rejected(self, tmp_path):
        """Patch file paths with '..' must be rejected."""
        from terok_executor.credentials.vault_config import ConfigPatchError, _safe_config_path

        shared = tmp_path / "mount"
        shared.mkdir()
        with pytest.raises(ConfigPatchError, match="invalid patch file path"):
            _safe_config_path(shared, "../../../etc/passwd")

    def test_absolute_path_rejected(self, tmp_path):
        """Absolute patch file paths must be rejected."""
        from terok_executor.credentials.vault_config import ConfigPatchError, _safe_config_path

        shared = tmp_path / "mount"
        shared.mkdir()
        with pytest.raises(ConfigPatchError, match="invalid patch file path"):
            _safe_config_path(shared, "/etc/passwd")

    def test_safe_relative_path_accepted(self, tmp_path):
        """A plain filename stays within the shared dir."""
        from terok_executor.credentials.vault_config import _safe_config_path

        shared = tmp_path / "mount"
        shared.mkdir()
        result = _safe_config_path(shared, "config.toml")
        assert result == (shared / "config.toml").resolve()

    def test_patch_failure_raises_not_swallows(self, roster, tmp_path):
        """Patch errors must propagate as ConfigPatchError, not be silently logged."""
        from terok_executor.credentials.vault_config import (
            ConfigPatchError,
            apply_shared_config_patches,
        )

        vibe_dir = tmp_path / "_vibe-config"
        vibe_dir.mkdir()
        # Make the config path a directory so write_bytes fails
        (vibe_dir / "config.toml").mkdir()

        with (
            patch(
                "terok_executor.integrations.sandbox.SandboxConfig",
                return_value=SandboxConfig(token_broker_port=18731),
            ),
            pytest.raises(ConfigPatchError, match="Failed to apply"),
        ):
            apply_shared_config_patches(roster, tmp_path)


class TestSharedConfigMountsUnit:
    """Unit tests for the internal shared mount builder."""

    def test_creates_host_dirs(self, roster, tmp_path):
        mounts = _shared_config_mounts(roster, tmp_path)
        assert len(mounts) > 0
        assert (tmp_path / "_claude-config").is_dir()

    def test_deduplicates_by_host_dir(self, roster, tmp_path):
        mounts = _shared_config_mounts(roster, tmp_path)
        # Directory mounts (rw shared dir) must be unique by host path; ro
        # credential shadows are layered on top and reuse the parent.
        dir_paths = [str(m.host_path) for m in mounts if not m.read_only]
        assert len(dir_paths) == len(set(dir_paths))

    def test_all_use_shared_label(self, roster, tmp_path):
        mounts = _shared_config_mounts(roster, tmp_path)
        for m in mounts:
            assert m.sharing == "shared", f"Expected sharing='shared', got: {m.sharing}"

    def test_credential_file_layered_ro_when_host_has_phantom(self, roster, tmp_path):
        """Host has the phantom credential — ro bind sources from it."""
        host_cred = tmp_path / "_claude-config" / ".credentials.json"
        host_cred.parent.mkdir(parents=True)
        host_cred.write_text('{"phantom": true}')

        mounts = _shared_config_mounts(roster, tmp_path)
        cred = next(
            (m for m in mounts if m.container_path == "/home/dev/.claude/.credentials.json"),
            None,
        )
        assert cred is not None
        assert cred.read_only is True
        assert cred.host_path == host_cred

    def test_credential_file_touched_when_host_absent(self, roster, tmp_path):
        """Host has no credential — touch an empty file at the natural path so
        podman doesn't materialise the bind target as root with mode 0700."""
        mounts = _shared_config_mounts(roster, tmp_path)
        cred = next(
            (m for m in mounts if m.container_path == "/home/dev/.claude/.credentials.json"),
            None,
        )
        assert cred is not None
        assert cred.read_only is True
        assert cred.host_path == tmp_path / "_claude-config" / ".credentials.json"
        assert cred.host_path.exists()
        assert cred.host_path.stat().st_size == 0

    def test_credential_file_skipped_when_provider_exposed(self, roster, tmp_path):
        """Providers in expose set keep the writable mount — no ro shadow."""
        mounts = _shared_config_mounts(
            roster,
            tmp_path,
            expose_credential_providers=frozenset({"claude"}),
        )
        cred = [m for m in mounts if m.container_path == "/home/dev/.claude/.credentials.json"]
        assert cred == []

    def test_credential_file_created_owner_only(self, roster, tmp_path):
        """A freshly touched credential file must be 0600 — glab aborts on looser."""
        _shared_config_mounts(roster, tmp_path)
        cred = tmp_path / "_claude-config" / ".credentials.json"
        assert stat.S_IMODE(cred.stat().st_mode) == 0o600

    def test_existing_loose_credential_file_reclamped(self, roster, tmp_path):
        """A 0644 credential file left behind by an older release is healed to 0600.

        Releases up to 0.2.x touched credential files with the umask default;
        the stale file persists in the shared mount across upgrades and makes
        glab refuse to start.  Content must survive the re-clamp.
        """
        host_cred = tmp_path / "_glab-config" / "config.yml"
        host_cred.parent.mkdir(parents=True)
        host_cred.write_text("hosts:\n  gitlab.com:\n    token: t\n")
        host_cred.chmod(0o644)

        _shared_config_mounts(roster, tmp_path)

        assert stat.S_IMODE(host_cred.stat().st_mode) == 0o600
        assert "gitlab.com" in host_cred.read_text()

    def test_stricter_credential_file_left_alone(self, roster, tmp_path):
        """Re-clamping only strips group/other bits — a 0400 file stays 0400."""
        host_cred = tmp_path / "_glab-config" / "config.yml"
        host_cred.parent.mkdir(parents=True)
        host_cred.write_text("hosts: {}\n")
        host_cred.chmod(0o400)

        _shared_config_mounts(roster, tmp_path)

        assert stat.S_IMODE(host_cred.stat().st_mode) == 0o400

    def test_writable_credential_file_skips_ro_shadow(self, roster, tmp_path):
        """A ``credential_file_writable`` provider (glab) gets no ro shadow.

        glab's config.yml is both credentials and settings, rewritten on
        startup, so the shadow would abort it — only the rw dir mount remains.
        """
        mounts = _shared_config_mounts(roster, tmp_path)
        shadow = [m for m in mounts if m.container_path == "/home/dev/.config/glab-cli/config.yml"]
        assert shadow == []
        dir_mount = [m for m in mounts if m.container_path == "/home/dev/.config/glab-cli"]
        assert len(dir_mount) == 1
        assert dir_mount[0].read_only is False


class TestExposedProviderMaterialization:
    """`_materialize_exposed_providers` surfaces exposed creds as TEROK_PROVIDER_* handles."""

    def test_emits_real_token_and_direct_upstream_base(self, roster, tmp_path):
        """An exposed provider gets the same handle shape as a routed one — but the
        real token and the upstream base, so a harness reaches it directly."""
        import json

        from terok_executor.container.env import _materialize_exposed_providers

        claude_dir = tmp_path / "_claude-config"
        claude_dir.mkdir()
        (claude_dir / ".credentials.json").write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "sk-ant-oat-real",
                        "refreshToken": "r",
                        "expiresAt": 1,
                    }
                }
            )
        )
        env = _materialize_exposed_providers(roster, tmp_path, frozenset({"claude"}))
        assert env["TEROK_PROVIDER_ANTHROPIC_TOKEN"] == "sk-ant-oat-real"
        assert (
            env["TEROK_PROVIDER_ANTHROPIC_BASE_ANTHROPIC_MESSAGES"] == "https://api.anthropic.com"
        )

    def test_skips_provider_without_credential_file(self, roster, tmp_path):
        """A missing/unreadable credential file is skipped, not fatal."""
        from terok_executor.container.env import _materialize_exposed_providers

        assert _materialize_exposed_providers(roster, tmp_path, frozenset({"claude"})) == {}


class TestEgressProjection:
    """assemble_container_env computes the roster egress projection into result.egress."""

    def test_result_carries_projection(self, base_spec, roster):
        """Dedicated provider hosts are denied; shared-domain apexes are not."""
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert "api.anthropic.com" in result.egress.deny_to_vault
        assert "gitlab.com" not in result.egress.deny_to_vault

    def test_exposed_agent_frees_its_provider(self, workspace, envs_dir, roster):
        """A spec exposing 'claude' frees anthropic's host (real cred in-container)."""
        spec = _spec(workspace, envs_dir, expose_credential_providers=frozenset({"claude"}))
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert "api.anthropic.com" not in result.egress.deny_to_vault
        assert "api.openai.com" in result.egress.deny_to_vault

    def test_default_result_has_empty_projection(self):
        """A bare ContainerEnvResult defaults to an empty projection."""
        from terok_executor.container.env import ContainerEnvResult
        from terok_executor.roster.types import EgressProjection

        assert ContainerEnvResult(env={}, volumes=()).egress == EgressProjection()
