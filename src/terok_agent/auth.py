# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Andreas Knüpfer
# SPDX-License-Identifier: Apache-2.0

"""Authentication workflows for AI coding agents.

Provides a data-driven registry of auth providers (``AUTH_PROVIDERS``) and a
single entry point ``authenticate(project_id, provider)`` that runs the
appropriate flow inside a temporary L2 CLI container.

The shared helper ``_run_auth_container`` handles the common lifecycle:
check podman, load project, ensure host dir, cleanup old container, run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from terok_sandbox import PHANTOM_CREDENTIALS_MARKER

from ._util import podman_userns_args

# ---------------------------------------------------------------------------
# Provider descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthProvider:
    """Describes how to authenticate one tool/agent."""

    name: str
    """Short key used in CLI and TUI dispatch (e.g. ``"codex"``)."""

    label: str
    """Human-readable display name (e.g. ``"Codex"``)."""

    host_dir_name: str
    """Directory name under ``mounts_dir()`` (e.g. ``"_codex-config"``)."""

    container_mount: str
    """Mount point inside the container (e.g. ``"/home/dev/.codex"``)."""

    command: list[str]
    """Command to execute inside the container (OAuth mode only)."""

    banner_hint: str
    """Provider-specific help text shown before the container runs."""

    extra_run_args: tuple[str, ...] = field(default_factory=tuple)
    """Additional ``podman run`` arguments (e.g. port forwarding)."""

    modes: tuple[str, ...] = ("api_key",)
    """Supported auth modes: ``"oauth"`` (container), ``"api_key"`` (fast path)."""

    api_key_hint: str = ""
    """Hint shown when prompting for an API key (URL to get one)."""

    @property
    def supports_oauth(self) -> bool:
        """Whether this provider supports OAuth (container-based) auth."""
        return "oauth" in self.modes

    @property
    def supports_api_key(self) -> bool:
        """Whether this provider supports direct API key entry."""
        return "api_key" in self.modes


# ---------------------------------------------------------------------------
# Helper for API-key-style providers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthKeyConfig:
    """Describes how to prompt for and store an API key."""

    label: str
    """Human name shown in the prompt (e.g. ``"Claude"``)."""

    key_url: str
    """URL where the user can obtain the key."""

    env_var: str
    """Name shown in the ``read -p`` prompt (e.g. ``"ANTHROPIC_API_KEY"``)."""

    config_path: str
    """Destination inside the container (e.g. ``"~/.claude/config.json"``)."""

    printf_template: str
    """``printf`` format string (e.g. ``'{\"api_key\": \"%s\"}'``)."""

    tool_name: str
    """Name shown in the success message (e.g. ``"claude"``)."""


def _api_key_command(cfg: AuthKeyConfig) -> list[str]:
    """Build a bash command that prompts for an API key and writes it to a config file."""
    config_dir = cfg.config_path.rsplit("/", 1)[0]
    parts = [
        f"echo 'Enter your {cfg.label} API key (get one at {cfg.key_url}):'",
        f"read -r -p '{cfg.env_var}=' api_key",
        f"mkdir -p {config_dir}",
        f"printf '{cfg.printf_template}\\n' \"$api_key\" > {cfg.config_path}",
        "echo",
        f"echo 'API key saved to {cfg.config_path}'",
        f"echo 'You can now use {cfg.tool_name} in task containers.'",
    ]
    return ["bash", "-c", " && ".join(parts)]


# ---------------------------------------------------------------------------
# Provider registry — populated from YAML by __init__.py at package load time
# ---------------------------------------------------------------------------

AUTH_PROVIDERS: dict[str, AuthProvider] = {}
"""All known auth providers (agents + tools), keyed by name.  Loaded from ``resources/agents/*.yaml``."""


# ---------------------------------------------------------------------------
# Shared container lifecycle
# ---------------------------------------------------------------------------


def _check_podman() -> None:
    """Verify podman is available."""
    if shutil.which("podman") is None:
        raise SystemExit("podman not found; please install podman")


def _cleanup_existing_container(container_name: str) -> None:
    """Remove an existing container if it exists."""
    result = subprocess.run(
        ["podman", "container", "exists", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        print(f"Removing existing auth container: {container_name}")
        subprocess.run(
            ["podman", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _run_auth_container(
    project_id: str,
    provider: AuthProvider,
    *,
    mounts_dir: Path,
    image: str,
    credential_set: str = "default",
) -> None:
    """Run an auth container, capture credentials to the DB.

    Uses a **temporary directory** as the container mount target so the
    vendor auth flow writes credential files into a disposable location.
    After the container exits, per-provider extractors parse the credential
    files and store them in the credential DB.  The shared config mount
    (settings, memories) is untouched — no real secrets land there.
    """
    import tempfile

    _check_podman()

    # Use an empty temp dir as the mount target.  The auth tool starts with
    # a clean slate (no existing config, sessions, or cached auth) which
    # forces a full re-authentication flow.  After the container exits, the
    # extractor reads the credential file from the temp dir and stores it
    # in the DB.  The shared config mount is never written to.
    with tempfile.TemporaryDirectory(prefix=f"terok-auth-{provider.name}-") as tmpdir:
        host_dir = Path(tmpdir)

        container_name = f"{project_id}-auth-{provider.name}"
        _cleanup_existing_container(container_name)

        cmd = ["podman", "run", "--rm"]
        cmd.extend(podman_userns_args())
        cmd.append("-it")
        if provider.extra_run_args:
            cmd.extend(provider.extra_run_args)
        cmd.extend(["-v", f"{host_dir}:{provider.container_mount}:Z"])
        cmd.extend(["--name", container_name])
        cmd.append(image)
        cmd.extend(provider.command)

        # Banner
        print(f"Authenticating {provider.label} for project: {project_id}")
        print()
        for line in provider.banner_hint.splitlines():
            print(line)
        print()
        print("$", " ".join(map(str, cmd)))
        print()

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            if e.returncode == 130:
                print("\nAuthentication container stopped.")
                return  # user cancelled — don't capture stale pre-seeded files
            raise SystemExit(f"Auth failed: {e}")
        except KeyboardInterrupt:
            print("\nAuthentication interrupted.")
            subprocess.run(
                ["podman", "rm", "-f", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        # Extract credentials from the temp dir and store in DB
        _capture_credentials(provider.name, host_dir, credential_set)


def _write_claude_credentials_file(cred_data: dict, mounts_base: Path) -> None:
    """Write a static ``.credentials.json`` with subscription metadata.

    The file lets Claude Code determine the subscription tier locally
    (``subscriptionType``, ``scopes``, ``rateLimitTier``) without
    exposing the real OAuth token.  ``accessToken`` is set to a dummy
    marker — actual API auth uses the per-task phantom token from the
    ``CLAUDE_CODE_OAUTH_TOKEN`` env var.
    """
    import json

    claude_dir = mounts_base / "_claude-config"
    claude_dir.mkdir(parents=True, exist_ok=True)

    creds = {
        "claudeAiOauth": {
            "accessToken": PHANTOM_CREDENTIALS_MARKER,
            "refreshToken": "",
            "expiresAt": None,
            "scopes": cred_data.get("scopes", ""),
            "subscriptionType": cred_data.get("subscription_type"),
            "rateLimitTier": cred_data.get("rate_limit_tier"),
        }
    }
    (claude_dir / ".credentials.json").write_text(
        json.dumps(creds, indent=2) + "\n", encoding="utf-8"
    )


def _capture_credentials(
    provider_name: str,
    auth_dir: Path,
    credential_set: str,
    mounts_base: Path | None = None,
) -> None:
    """Extract credentials from *auth_dir* and store in the credential DB.

    Uses the per-provider extractors from :mod:`credential_extractors`.
    If extraction fails (no credential file, malformed), prints a warning
    but does not raise — the auth flow succeeded, the user can retry.
    """
    from .credential_extractors import extract_credential

    try:
        cred_data = extract_credential(provider_name, auth_dir)
    except Exception as exc:
        print(
            f"\nWarning [auth]: could not extract credentials for {provider_name} "
            f"from {auth_dir}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("The auth flow completed but credentials were not captured.", file=sys.stderr)
        print(
            "You may need to re-authenticate or check the credential file format.",
            file=sys.stderr,
        )
        # List files in the auth dir to aid debugging
        files = sorted(p.relative_to(auth_dir) for p in auth_dir.rglob("*") if p.is_file())
        if files:
            print(f"\nFiles found in auth dir ({len(files)}):")
            for f in files[:20]:
                print(f"  {f}")
        return

    try:
        from terok_sandbox import CredentialDB, SandboxConfig

        cfg = SandboxConfig()
        db = CredentialDB(cfg.proxy_db_path)
        try:
            db.store_credential(credential_set, provider_name, cred_data)
            print(f"\nCredentials captured for {provider_name} (set: {credential_set})")
        finally:
            db.close()
    except Exception as exc:
        print(
            f"\nWarning [auth]: failed to store credentials for {provider_name} "
            f"in proxy DB: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "The auth flow completed but credentials were not saved to the proxy DB.",
            file=sys.stderr,
        )
        return

    # Write static .credentials.json for OAuth subscription mode detection
    if provider_name == "claude" and cred_data.get("type") == "oauth":
        if mounts_base is None:
            from .paths import mounts_dir

            mounts_base = mounts_dir()
        try:
            _write_claude_credentials_file(cred_data, mounts_base)
            print("Subscription metadata written to shared Claude config mount.")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not write .credentials.json: {exc}")
        print(
            "\nNote: Claude OAuth uses a shared credential for all tasks."
            "\n      API calls are routed through the credential proxy — the real"
            "\n      token never enters any container."
        )


def store_api_key(
    provider: str,
    api_key: str,
    credential_set: str = "default",
) -> None:
    """Store an API key directly in the credential DB (no container needed).

    This is the non-interactive fast path for automated workflows and CI.
    The key is stored as ``{"type": "api_key", "key": "<value>"}``.
    """
    from terok_sandbox import CredentialDB, SandboxConfig

    cfg = SandboxConfig()
    db = CredentialDB(cfg.proxy_db_path)
    try:
        db.store_credential(credential_set, provider, {"type": "api_key", "key": api_key})
        print(f"API key stored for {provider} (set: {credential_set})")
    finally:
        db.close()


def _prompt_api_key(info: AuthProvider) -> str:
    """Interactively prompt for an API key (input is hidden)."""
    import getpass

    if info.api_key_hint:
        print(info.api_key_hint)
    key = getpass.getpass(f"{info.label} API key: ").strip()
    if not key:
        raise SystemExit("No API key entered.")
    return key


def authenticate(
    project_id: str,
    provider: str,
    *,
    mounts_dir: Path,
    image: str,
) -> None:
    """Run the auth flow for *provider* against *project_id*.

    Dispatches based on the provider's ``modes`` field:

    - **api_key only**: prompt for key, store directly (no container)
    - **oauth only**: launch container with vendor CLI
    - **both**: ask user to choose, then dispatch accordingly

    Args:
        project_id: Project identifier (for container naming).
        provider: Auth provider name (e.g. ``"claude"``).
        mounts_dir: Base directory for shared config bind-mounts.
        image: Container image to use for the auth container.

    Raises ``SystemExit`` if the provider name is unknown.
    """
    info = AUTH_PROVIDERS.get(provider)
    if not info:
        available = ", ".join(AUTH_PROVIDERS)
        raise SystemExit(f"Unknown auth provider: {provider}. Available: {available}")

    if info.supports_oauth and info.supports_api_key:
        # Both modes — let the user choose
        print(f"Authenticate {info.label}:\n")
        print("  1. OAuth / interactive login (launches container)")
        print("  2. API key (paste key, no container needed)")
        print()
        choice = input("Choose [1/2]: ").strip()
        if choice == "2":
            key = _prompt_api_key(info)
            store_api_key(provider, key)
            return
        # choice == "1" or anything else → OAuth
        _run_auth_container(project_id, info, mounts_dir=mounts_dir, image=image)

    elif info.supports_api_key:
        # API key only — fast path, no container
        key = _prompt_api_key(info)
        store_api_key(provider, key)

    else:
        # OAuth only
        _run_auth_container(project_id, info, mounts_dir=mounts_dir, image=image)
