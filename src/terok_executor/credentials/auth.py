# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Andreas Knüpfer
# SPDX-License-Identifier: Apache-2.0

"""Authenticates AI coding agents via OAuth or API key.

Two public entry points:

- ``authenticate(project_id, provider, *, mounts_dir, image)`` — dispatches
  based on the provider's ``modes`` field: prompts for an API key (no
  container) or launches an auth container with the vendor CLI.
- ``store_api_key(provider, api_key)`` — stores an API key directly in the
  credential DB (non-interactive fast path for CI).

``AUTH_PROVIDERS`` is a registry dict populated from the YAML roster at
package load time; ``authenticate`` looks up the provider by name and
delegates to the matching flow.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from terok_sandbox import CODEX_SHARED_OAUTH_MARKER, PHANTOM_CREDENTIALS_MARKER

from terok_executor._util import podman_userns_args

# ── Vocabulary ──


@dataclass(frozen=True)
class AuthProvider:
    """Describes how to authenticate one tool/agent."""

    name: str
    """Short key used in CLI and TUI dispatch (e.g. ``"codex"``)."""

    label: str
    """Human-readable display name (e.g. ``"Codex"``)."""

    host_dir_name: str
    """Single-segment directory name under ``mounts_dir()`` (e.g. ``"_codex-config"``)."""

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

    post_capture_state: dict[str, dict] = field(default_factory=dict)
    """JSON state files to write after credential capture.

    Maps filename → key-value dict to merge into a JSON file in the auth
    mount directory.  Example: ``{".claude.json": {"hasCompletedOnboarding": true}}``
    marks Claude Code onboarding as complete so the first-run wizard is skipped.
    """

    def __post_init__(self) -> None:
        """Validate fields that become filesystem paths."""
        p = Path(self.host_dir_name)
        if p.is_absolute() or ".." in p.parts or len(p.parts) != 1:
            raise ValueError(
                f"host_dir_name must be a single directory segment, got {self.host_dir_name!r}"
            )

    @property
    def supports_oauth(self) -> bool:
        """Whether this provider supports OAuth (container-based) auth."""
        return "oauth" in self.modes

    @property
    def supports_api_key(self) -> bool:
        """Whether this provider supports direct API key entry."""
        return "api_key" in self.modes


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


AUTH_PROVIDERS: dict[str, AuthProvider] = {}
"""All known auth providers (agents + tools), keyed by name.  Loaded from ``resources/agents/*.yaml``."""


# ── Public API ──


def authenticate(
    project_id: str | None,
    provider: str,
    *,
    mounts_dir: Path,
    image: str,
    expose_token: bool = False,
) -> None:
    """Run the auth flow for *provider*, optionally scoped to a project.

    Dispatches based on the provider's ``modes`` field:

    - **api_key only**: prompt for key, store directly (no container)
    - **oauth only**: launch container with vendor CLI
    - **both**: ask user to choose, then dispatch accordingly

    Args:
        project_id: Project identifier used for container naming and the
            banner line.  Pass ``None`` for host-wide auth — the banner
            drops the project reference and the container gets a neutral
            ``host-auth-<provider>`` name.
        provider: Auth provider name (e.g. ``"claude"``).
        mounts_dir: Base directory for shared config bind-mounts.
        image: Container image to use for the auth container.
        expose_token: When True, copy the real credential files into
            the shared mount instead of writing a phantom marker.  Used
            by tier 3 (``expose_oauth_token``) where containers need
            the actual token.

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
        _run_auth_container(
            project_id, info, mounts_dir=mounts_dir, image=image, expose_token=expose_token
        )

    elif info.supports_api_key:
        # API key only — fast path, no container
        key = _prompt_api_key(info)
        store_api_key(provider, key)

    else:
        # OAuth only
        _run_auth_container(
            project_id, info, mounts_dir=mounts_dir, image=image, expose_token=expose_token
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
    db = CredentialDB(cfg.db_path)
    try:
        db.store_credential(credential_set, provider, {"type": "api_key", "key": api_key})
        print(f"API key stored for {provider} (set: {credential_set})")
    finally:
        db.close()


# ── Private helpers ──


def _prompt_api_key(info: AuthProvider) -> str:
    """Interactively prompt for an API key (echoes ``*`` per character).

    pwinput only guards against a *replaced* ``sys.stdin``, not against a
    non-TTY one — shell redirection (``< keyfile.txt``) leaves ``sys.stdin``
    identical to ``sys.__stdin__`` with a non-TTY fd, which would crash
    ``tty.setraw`` inside pwinput.  Gate the call on ``isatty()`` and read
    a line deterministically otherwise so pipe-fed automation still works.
    """
    import sys

    if info.api_key_hint:
        print(info.api_key_hint)
    prompt = f"{info.label} API key: "
    if sys.stdin.isatty():
        import pwinput

        key = pwinput.pwinput(prompt=prompt, mask="*").strip()
    else:
        print(prompt, end="", flush=True)
        key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("No API key entered.")
    return key


def _run_auth_container(
    project_id: str | None,
    provider: AuthProvider,
    *,
    mounts_dir: Path,
    image: str,
    credential_set: str = "default",
    expose_token: bool = False,
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

        # ``project_id`` must lead the container name; Podman rejects names
        # starting with ``_`` or other non-alphanumeric chars, so the
        # host-wide caller passes ``None`` and we fall back to ``host``.
        name_prefix = project_id or "host"
        container_name = f"{name_prefix}-auth-{provider.name}"
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
        if project_id:
            print(f"Authenticating {provider.label} for project: {project_id}")
        else:
            print(f"Authenticating {provider.label} (host-wide)")
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
        _capture_credentials(
            provider.name,
            host_dir,
            credential_set,
            mounts_base=mounts_dir,
            auth_provider=provider,
            expose_token=expose_token,
        )


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


def _capture_credentials(
    provider_name: str,
    auth_dir: Path,
    credential_set: str,
    mounts_base: Path | None = None,
    auth_provider: AuthProvider | None = None,
    *,
    expose_token: bool = False,
) -> None:
    """Extract credentials from *auth_dir* and store in the credential DB.

    Uses the per-provider extractors from `credential_extractors`.
    If extraction fails (no credential file, malformed), prints a warning
    but does not raise — the auth flow succeeded, the user can retry.

    When *expose_token* is ``True`` for Claude OAuth (tier 3), the credential
    is **not** stored in the vault DB — Claude manages its own token lifecycle
    directly, and vault-side refresh would invalidate the exposed token.
    """
    from .extractors import extract_credential

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

    is_oauth = cred_data.get("type") == "oauth"
    post_capture = _OAUTH_MOUNT_WRITERS.get(provider_name) if is_oauth else None
    # Tier 3 bypass: the host-side vault never refreshes the token, so the
    # container must own its lifecycle.  Storing in the DB would let the
    # background refresher rotate a token nobody brokers back to the mount.
    exposed_directly = expose_token and post_capture is not None

    if exposed_directly:
        print(f"\nCredentials for {provider_name} bypassing vault DB (exposed directly)")
    else:
        try:
            from terok_sandbox import CredentialDB, SandboxConfig

            cfg = SandboxConfig()
            db = CredentialDB(cfg.db_path)
            try:
                db.store_credential(credential_set, provider_name, cred_data)
                print(f"\nCredentials captured for {provider_name} (set: {credential_set})")
            finally:
                db.close()
        except Exception as exc:
            print(
                f"\nWarning [auth]: failed to store credentials for {provider_name} "
                f"in vault DB: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            print(
                "The auth flow completed but credentials were not saved to the vault DB.",
                file=sys.stderr,
            )
            return

    # Reconcile the shared mount with the captured credential.  Provider-specific
    # writers drop a phantom marker (proxied mode) or copy the real file (exposed).
    if post_capture is not None:
        if mounts_base is None:
            from terok_executor.paths import mounts_dir

            mounts_base = mounts_dir()
        try:
            post_capture(auth_dir, mounts_base, cred_data, expose_token)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not reconcile {provider_name} mount: {exc}")

    # Apply declarative post-capture state from roster YAML
    if auth_provider and auth_provider.post_capture_state:
        try:
            _apply_post_capture_state(
                auth_provider.host_dir_name,
                auth_provider.post_capture_state,
                mounts_base,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: could not apply post_capture_state for {provider_name}: {exc}",
                file=sys.stderr,
            )


def _claude_oauth_mount_writer(
    auth_dir: Path, mounts_base: Path, cred_data: dict, expose_token: bool
) -> None:
    """Reconcile Claude's shared mount after an OAuth capture.

    Default path writes a phantom ``.credentials.json`` with subscription
    metadata only; tier 3 copies the real credential file so Claude Code
    can reach ``api.anthropic.com`` directly (its hardcoded OAuth host).
    """
    if expose_token:
        src = auth_dir / ".credentials.json"
        if not src.is_file():
            raise FileNotFoundError(f"No .credentials.json in {auth_dir}")
        dest_dir = mounts_base / "_claude-config"
        dest_dir.mkdir(parents=True, exist_ok=True)
        _write_bytes_nofollow(dest_dir / ".credentials.json", src.read_bytes())
        print("Real .credentials.json copied to shared Claude config mount.")
        print(
            "\nNote: Claude OAuth token is EXPOSED in the shared mount."
            "\n      Every task container can read the real token."
            "\n      The vault does NOT protect it."
        )
    else:
        _write_claude_credentials_file(cred_data, mounts_base)
        print("Subscription metadata written to shared Claude config mount.")
        print(
            "\nNote: Claude OAuth credential is shared across all task containers."
            "\n      API calls are routed through the vault — the real"
            "\n      token stays on the host."
        )


def _codex_oauth_mount_writer(
    auth_dir: Path, mounts_base: Path, cred_data: dict, expose_token: bool
) -> None:
    """Reconcile Codex's shared mount after an OAuth capture.

    Two modes:

    - **Exposed** (``expose_token=True``): copy the real ``auth.json``
      verbatim so the in-container Codex reads the live OAuth token
      (tier 3, unsafe — every task container can read it).
    - **Default**: drop a shared synthetic ``auth.json`` — the real
      ``id_token`` JWT (for plan-tier + workspace UI, public claims only)
      and ``account_id`` survive, but ``access_token`` and ``refresh_token``
      are replaced with `CODEX_SHARED_OAUTH_MARKER`.  The vault translates
      the marker back to the real token on inference requests; the CLI
      itself never sees the live bearer.  This is the fallback for tier 2
      (proxied) and also a no-harm default for tier 1 (vault stores creds
      but nothing wires the proxy yet).
    """
    dest_dir = mounts_base / "_codex-config"
    dest_file = dest_dir / "auth.json"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if expose_token:
        src = auth_dir / "auth.json"
        if not src.is_file():
            raise FileNotFoundError(f"No auth.json in {auth_dir}")
        _write_bytes_nofollow(dest_file, src.read_bytes())
        print("Real auth.json copied to shared Codex config mount.")
        print(
            "\nNote: Codex OAuth token is EXPOSED in the shared mount."
            "\n      Every task container can read the real token."
            "\n      The vault does NOT protect it."
        )
    else:
        _write_codex_phantom_auth_json(cred_data, dest_file)
        print("Phantom auth.json written to shared Codex config mount.")
        print(
            "\nNote: Codex OAuth credential is shared across all task containers."
            "\n      API calls are routed through the vault — the real"
            "\n      OAuth tokens stay on the host.  Standalone executor"
            "\n      uses the shared Codex config rewrite directly;"
            "\n      terok project mode gates that rewrite via"
            "\n      agent.codex.allow_oauth."
        )


#: Static far-future timestamp for the phantom ``auth.json``'s
#: ``last_refresh`` field.  Codex's CLI fires a client-side token
#: refresh against the hardcoded ``auth.openai.com`` whenever
#: ``now - last_refresh > 8 days`` (manager.rs:1743).  Those attempts
#: would arrive at the upstream with the phantom refresh token and come
#: back as ``refresh_token_invalidated``, prompting the user to re-login
#: even though vault-side refresh keeps the real credential fresh.
#: Pinning ``last_refresh`` to year 9999 keeps the check permanently
#: satisfied — the same trick Claude's phantom file uses with its
#: ``"expiresAt": null`` sentinel.
_CODEX_PHANTOM_LAST_REFRESH = "9999-01-01T00:00:00Z"


def _write_codex_phantom_auth_json(cred_data: dict, dest: Path) -> None:
    """Write a shared synthetic ``auth.json`` without real bearer tokens.

    Codex's ``TokenData`` serde contract (codex-rs/login/src/token_data.rs)
    requires ``id_token`` to be a parseable JWT string.  We synthesize a
    minimal JWT that preserves non-PII account metadata while dropping the
    original opaque token body entirely.

    The access/refresh tokens are the actual bearer secrets; both are
    replaced with the Codex-specific shared vault marker.  Inference
    rides through the vault, which substitutes the live access token.
    """
    import json

    tokens: dict = {
        "id_token": _build_codex_shared_id_token(cred_data.get("id_token", "")),
        "access_token": CODEX_SHARED_OAUTH_MARKER,
        "refresh_token": CODEX_SHARED_OAUTH_MARKER,
    }
    account_id = cred_data.get("account_id")
    if account_id:
        tokens["account_id"] = account_id

    payload = {
        "OPENAI_API_KEY": None,
        "tokens": tokens,
        "last_refresh": _CODEX_PHANTOM_LAST_REFRESH,
    }
    _write_bytes_nofollow(dest, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _build_codex_shared_id_token(raw_jwt: str) -> str:
    """Return a synthetic JWT with only the Codex-local claims we need."""
    import base64
    import json

    def _b64url(data: dict) -> str:
        encoded = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")

    def _jwt_payload(token: str) -> dict:
        try:
            _header, payload, _sig = token.split(".", 2)
            padded = payload + "=" * (-len(payload) % 4)
            decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
            parsed = json.loads(decoded)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError, base64.binascii.Error):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    payload = _jwt_payload(raw_jwt)
    auth = payload.get("https://api.openai.com/auth")

    safe_payload: dict[str, object] = {"exp": 253402300799}
    if isinstance(auth, dict):
        safe_auth = {
            key: auth[key]
            for key in (
                "chatgpt_plan_type",
                "chatgpt_user_id",
                "user_id",
                "chatgpt_account_id",
                "chatgpt_account_is_fedramp",
            )
            if key in auth
        }
        if safe_auth:
            safe_payload["https://api.openai.com/auth"] = safe_auth

    return ".".join((_b64url({"alg": "none", "typ": "JWT"}), _b64url(safe_payload), "terok"))


def _write_bytes_nofollow(path: Path, data: bytes) -> None:
    """Atomically write *data* without following a pre-existing target symlink."""
    import os
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(tmp, flags, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


#: Maps provider name → post-capture mount reconciler.  OAuth-capable
#: providers register here to share the expose/proxy dispatch in
#: `_capture_credentials`.
_OAUTH_MOUNT_WRITERS: dict[str, Callable[[Path, Path, dict, bool], None]] = {
    "claude": _claude_oauth_mount_writer,
    "codex": _codex_oauth_mount_writer,
}


def _write_claude_credentials_file(cred_data: dict, mounts_base: Path) -> None:
    """Write a static ``.credentials.json`` with subscription metadata.

    The file lets Claude Code determine the subscription tier locally
    (``subscriptionType``, ``scopes``, ``rateLimitTier``) without
    exposing the real OAuth token.  ``accessToken`` is set to a dummy
    marker — actual API auth uses the per-task phantom token from the
    ``CLAUDE_CODE_OAUTH_TOKEN`` env var.

    Onboarding state (``.claude.json`` / ``hasCompletedOnboarding``) is
    applied separately via ``_apply_post_capture_state`` after capture.
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
    _write_bytes_nofollow(
        claude_dir / ".credentials.json",
        (json.dumps(creds, indent=2) + "\n").encode("utf-8"),
    )


def _apply_post_capture_state(
    host_dir_name: str,
    patches: dict[str, dict],
    mounts_base: Path | None,
) -> None:
    """Apply ``post_capture_state`` after credential capture.

    Merges key-value pairs into JSON files in the provider's auth mount
    directory.  Declared in ``auth.post_capture_state`` in the agent YAML.
    Takes resolved data directly — no roster lookup (avoids circular dep).
    """
    import json

    if mounts_base is None:
        from terok_executor.paths import mounts_dir

        mounts_base = mounts_dir()

    mounts_root = mounts_base.resolve()
    host_rel = Path(host_dir_name)
    if host_rel.is_absolute() or ".." in host_rel.parts:
        raise ValueError(f"Invalid host_dir_name: {host_dir_name!r}")
    target_dir = (mounts_root / host_rel).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, patch in patches.items():
        rel = Path(filename)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Invalid post_capture_state filename: {filename!r}")
        path = (target_dir / rel).resolve()
        state: dict = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                state = loaded if isinstance(loaded, dict) else {}
            except (json.JSONDecodeError, OSError):
                state = {}

        if all(state.get(k) == v for k, v in patch.items()):
            continue  # already up to date

        state.update(patch)
        _write_bytes_nofollow(path, (json.dumps(state, indent=2) + "\n").encode("utf-8"))


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
