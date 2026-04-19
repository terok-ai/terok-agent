# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the container environment assembly function."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from terok_sandbox import VolumeSpec

from terok_executor.container.env import (
    ContainerEnvResult,
    ContainerEnvSpec,
    _resolve_git_identity,
    _shared_config_mounts,
    assemble_container_env,
)
from terok_executor.roster import get_roster


def _find_vol(volumes: tuple[VolumeSpec, ...], container_path: str) -> VolumeSpec | None:
    """Find a VolumeSpec by container_path prefix."""
    return next((v for v in volumes if container_path in v.container_path), None)


def _make_vault_db(tmp_path: Path, cred_name: str = "claude", cred_data: dict | None = None):
    """Return a ``SandboxConfig`` with one credential pre-stored in its ``CredentialDB``.

    The DB is created, populated, and closed internally.
    """
    from terok_sandbox import CredentialDB, SandboxConfig

    cfg = SandboxConfig(state_dir=tmp_path, vault_dir=tmp_path / "credentials")
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = CredentialDB(cfg.db_path)
    db.store_credential(
        "default",
        cred_name,
        {"type": "api_key", "key": "sk-test"} if cred_data is None else cred_data,
    )
    db.close()
    return cfg


def _make_vault_db_with_ssh_keys(tmp_path: Path, scope: str = "myproj"):
    """Return a SandboxConfig with credential DB seeded with a key assigned to *scope*."""
    from terok_sandbox.credentials.db import CredentialDB
    from terok_sandbox.credentials.ssh_keypair import generate_keypair

    cfg = _make_vault_db(tmp_path)
    cfg.vault_dir.mkdir(parents=True, exist_ok=True)
    db = CredentialDB(cfg.db_path)
    try:
        kp = generate_keypair("ed25519", comment=f"tk-main:{scope}")
        key_id = db.store_ssh_key(
            key_type=kp.key_type,
            private_pem=kp.private_pem,
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
    return get_roster()


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
        provider_name="claude",
        workspace_host_path=workspace,
        envs_dir=envs_dir,
    )


def _spec(workspace: Path, envs_dir: Path, **overrides) -> ContainerEnvSpec:
    """Shorthand for a tmp-backed spec with overrides."""
    defaults = {
        "task_id": "t1",
        "provider_name": "claude",
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
        """Every container receives the host↔container contract version so
        in-container scripts can adapt on an update without guessing."""
        from terok_executor.container.env import CONTAINER_PROTOCOL

        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert result.env["TEROK_CONTAINER_PROTOCOL"] == str(CONTAINER_PROTOCOL)


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
        spec = _spec(workspace, envs_dir, provider_name="nonexistent")
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

    def test_vault_not_running_returns_no_tokens(self, base_spec, roster):
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=False),
            patch("terok_sandbox.is_vault_running", return_value=False),
        ):
            result = assemble_container_env(base_spec, roster, caller_manages_vault=False)
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_running_injects_tokens(self, workspace, envs_dir, roster, tmp_path):
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="test-project")

        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=False),
            patch("terok_sandbox.is_vault_running", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "ANTHROPIC_API_KEY" in result.env
        assert result.env["ANTHROPIC_API_KEY"].startswith("terok-p-")

    def test_no_routed_providers_returns_empty(self, workspace, envs_dir, roster, tmp_path):
        """Stored credentials that don't match any vault route produce no tokens."""
        cfg = _make_vault_db(tmp_path, "nonexistent-provider")
        spec = _spec(workspace, envs_dir)
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_db_error_returns_empty(self, base_spec, roster):
        """DB open failure returns empty env gracefully."""
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.CredentialDB", side_effect=OSError("corrupt")),
        ):
            result = assemble_container_env(base_spec, roster, caller_manages_vault=False)
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_oauth_credential_uses_oauth_phantom_env(
        self, workspace, envs_dir, roster, tmp_path
    ):
        """OAuth credential selects oauth_phantom_env (e.g. CLAUDE_CODE_OAUTH_TOKEN)."""
        cfg = _make_vault_db(tmp_path, cred_data={"type": "oauth", "access_token": "oa-tok"})
        spec = _spec(workspace, envs_dir, credential_scope="test-project")
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "CLAUDE_CODE_OAUTH_TOKEN" in result.env
        assert result.env["CLAUDE_CODE_OAUTH_TOKEN"].startswith("terok-p-")
        # API key env var must NOT be set when OAuth credential is stored
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_api_key_falls_back_to_phantom_env(self, workspace, envs_dir, roster, tmp_path):
        """API-key credential uses phantom_env even when oauth_phantom_env exists."""
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="test-project")
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
        ):
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
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
            patch("terok_sandbox.CredentialDB.create_token", side_effect=RuntimeError("boom")),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_socket_transport_injects_socket_env(self, workspace, envs_dir, roster, tmp_path):
        """Socket transport points socket_env at the mounted host vault socket."""
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="proj", vault_transport="socket")
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
            patch("terok_sandbox.get_token_broker_port", return_value=None),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert result.env["ANTHROPIC_UNIX_SOCKET"] == "/run/terok/vault.sock"

    def test_vault_direct_transport_points_socket_at_local_bridge(
        self, workspace, envs_dir, roster, tmp_path
    ):
        """Direct (TCP) transport still sets socket_env — now to the local socat bridge."""
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="proj", vault_transport="direct")
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
            patch("terok_sandbox.get_token_broker_port", return_value=18731),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert result.env["ANTHROPIC_UNIX_SOCKET"] == "/tmp/terok-vault.sock"

    def test_vault_socket_transport_omits_tcp_broker_env_when_port_none(
        self, workspace, envs_dir, roster, tmp_path
    ):
        """Socket-only deployments have no TCP broker port; the env must reflect that.

        Under socket transport the broker listens only on the mounted Unix socket,
        and ``get_token_broker_port`` returns ``None``.  The assembled env must
        omit the TCP broker port variable rather than interpolate the literal
        string ``"None"`` — that string would otherwise trip bridge scripts and
        silently break credential routing.
        """
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="proj", vault_transport="socket")
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
            patch("terok_sandbox.get_token_broker_port", return_value=None),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "TEROK_TOKEN_BROKER_PORT" not in result.env
        # GITLAB_API_HOST is now always set for glab — but never with "None".
        assert not any("None" in v for v in result.env.values())
        # Socket transport uses the mounted host socket directly.
        assert result.env.get("ANTHROPIC_UNIX_SOCKET") == "/run/terok/vault.sock"
        # The in-container loopback port is advertised so ensure-bridges.sh
        # stands up its TCP→UNIX bridge.
        assert result.env.get("TEROK_VAULT_LOOPBACK_PORT") == "9419"

    def test_vault_required_raises_when_unreachable(self, workspace, envs_dir, roster):
        """vault_required=True raises SystemExit when vault is not running."""
        spec = _spec(workspace, envs_dir, vault_required=True)
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=False),
            patch("terok_sandbox.is_vault_running", return_value=False),
            pytest.raises(SystemExit, match="Vault is not running"),
        ):
            assemble_container_env(spec, roster, caller_manages_vault=False)

    def test_vault_not_required_soft_fails(self, workspace, envs_dir, roster):
        """vault_required=False (default) returns empty env when vault is down."""
        spec = _spec(workspace, envs_dir, vault_required=False)
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=False),
            patch("terok_sandbox.is_vault_running", return_value=False),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_injects_ssh_signer_token(self, workspace, envs_dir, roster, tmp_path):
        """SSH signer token injected when scope has valid keys in ssh-keys.json."""
        cfg = _make_vault_db_with_ssh_keys(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="myproj")
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "TEROK_SSH_SIGNER_TOKEN" in result.env
        assert result.env["TEROK_SSH_SIGNER_TOKEN"].startswith("terok-p-")
        assert "TEROK_SSH_SIGNER_PORT" in result.env
        assert "TEROK_SSH_SIGNER_SOCKET" not in result.env

    def test_vault_ssh_signer_socket_transport(self, workspace, envs_dir, roster, tmp_path):
        """Socket transport injects TEROK_SSH_SIGNER_SOCKET instead of _PORT."""
        cfg = _make_vault_db_with_ssh_keys(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="myproj", vault_transport="socket")
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "TEROK_SSH_SIGNER_TOKEN" in result.env
        assert result.env["TEROK_SSH_SIGNER_SOCKET"] == "/run/terok/ssh-agent.sock"
        assert "TEROK_SSH_SIGNER_PORT" not in result.env

    def test_vault_no_ssh_keys_omits_token(self, workspace, envs_dir, roster, tmp_path):
        """No SSH signer token when ssh-keys.json has no entry for scope."""
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir, credential_scope="no-keys-project")
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "TEROK_SSH_SIGNER_TOKEN" not in result.env
        assert "TEROK_SSH_SIGNER_PORT" not in result.env

    def test_vault_ssh_only_no_provider_creds(self, workspace, envs_dir, roster, tmp_path):
        """SSH signer token injected even when no provider credentials are stored."""
        from terok_sandbox import CredentialDB, SandboxConfig
        from terok_sandbox.credentials.ssh_keypair import generate_keypair

        cfg = SandboxConfig(state_dir=tmp_path, vault_dir=tmp_path / "credentials")
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_dir.mkdir(parents=True, exist_ok=True)
        # DB exists with NO provider credentials — only SSH keys.
        db = CredentialDB(cfg.db_path)
        try:
            kp = generate_keypair("ed25519", comment="tk-main:sshonly")
            key_id = db.store_ssh_key(
                key_type=kp.key_type,
                private_pem=kp.private_pem,
                public_blob=kp.public_blob,
                comment=kp.comment,
                fingerprint=kp.fingerprint,
            )
            db.assign_ssh_key("sshonly", key_id)
        finally:
            db.close()

        spec = _spec(workspace, envs_dir, credential_scope="sshonly")
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
        ):
            result = assemble_container_env(spec, roster, caller_manages_vault=False)

        assert "TEROK_SSH_SIGNER_TOKEN" in result.env
        assert result.env["TEROK_SSH_SIGNER_TOKEN"].startswith("terok-p-")
        # No provider tokens
        assert "ANTHROPIC_API_KEY" not in result.env

    def test_vault_required_hard_fails_on_db_error(self, workspace, envs_dir, roster):
        """vault_required=True raises SystemExit on CredentialDB construction failure."""
        spec = _spec(workspace, envs_dir, vault_required=True)
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.CredentialDB", side_effect=OSError("corrupt")),
            pytest.raises(SystemExit, match="DB unavailable.*Check logs"),
        ):
            assemble_container_env(spec, roster, caller_manages_vault=False)

    def test_vault_required_hard_fails_on_token_error(self, workspace, envs_dir, roster, tmp_path):
        """vault_required=True raises SystemExit on token creation failure."""
        cfg = _make_vault_db(tmp_path)
        spec = _spec(workspace, envs_dir, vault_required=True)
        with (
            patch("terok_sandbox.is_vault_socket_active", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
            patch("terok_sandbox.CredentialDB.create_token", side_effect=RuntimeError("boom")),
            pytest.raises(SystemExit, match="injection failed.*Check logs"),
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


class TestTaskDir:
    """Verify task_dir resolution."""

    def test_explicit_task_dir(self, workspace, envs_dir, roster, tmp_path):
        td = tmp_path / "my-task"
        td.mkdir()
        spec = _spec(workspace, envs_dir, task_dir=td)
        result = assemble_container_env(spec, roster, caller_manages_vault=True)
        assert result.task_dir == td

    def test_auto_creates_temp_dir(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, caller_manages_vault=True)
        assert result.task_dir.exists()
        assert "terok-executor-test-123" in str(result.task_dir)


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

        with (
            patch("terok_sandbox.SandboxConfig"),
            patch("terok_sandbox.get_token_broker_port", return_value=18731),
        ):
            apply_shared_config_patches(roster, tmp_path)

        config_path = vibe_dir / "config.toml"
        assert config_path.exists(), "apply_shared_config_patches must create config.toml"

        import tomllib

        data = tomllib.loads(config_path.read_text())
        providers = data.get("providers", [])
        mistral = next((p for p in providers if p.get("name") == "mistral"), None)
        assert mistral is not None, "config.toml must contain a mistral provider entry"
        assert "host.containers.internal:18731" in mistral["api_base"]

    def test_assemble_env_calls_patches_with_and_without_bypass(self, workspace, envs_dir, roster):
        """assemble_container_env invokes patches regardless of caller_manages_vault."""
        for bypass in (True, False):
            with patch(
                "terok_executor.credentials.vault_config.apply_shared_config_patches"
            ) as m_patches:
                assemble_container_env(
                    spec=_spec(workspace, envs_dir), roster=roster, caller_manages_vault=bypass
                )

            m_patches.assert_called_once_with(roster, envs_dir)

    def test_patches_idempotent(self, roster, tmp_path):
        """Calling apply_shared_config_patches twice must not duplicate entries."""
        from terok_executor.credentials.vault_config import apply_shared_config_patches

        vibe_dir = tmp_path / "_vibe-config"
        vibe_dir.mkdir()

        with (
            patch("terok_sandbox.SandboxConfig"),
            patch("terok_sandbox.get_token_broker_port", return_value=18731),
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
            patch("terok_sandbox.SandboxConfig"),
            patch("terok_sandbox.get_token_broker_port", return_value=18731),
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
        host_dirs = [str(m.host_path) for m in mounts]
        assert len(host_dirs) == len(set(host_dirs))

    def test_all_use_shared_label(self, roster, tmp_path):
        mounts = _shared_config_mounts(roster, tmp_path)
        for m in mounts:
            assert m.sharing == "shared", f"Expected sharing='shared', got: {m.sharing}"
