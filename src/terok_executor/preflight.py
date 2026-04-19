# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Pre-launch prerequisite checks for ``terok-executor run``.

Detects missing infrastructure (podman, vault, agent credentials,
container images) and — in interactive mode — offers to fix each issue before
the run starts.  In non-interactive mode, reports all problems and exits.

The :func:`run_preflight` entry point is called from :mod:`commands` before
:class:`~terok_executor.container.runner.AgentRunner` is created.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single prerequisite check."""

    name: str
    ok: bool
    message: str


# ── Individual checks ──────────────────────────────────────────────────


def check_podman() -> CheckResult:
    """Verify that podman is installed and responds to ``podman version``."""
    if not shutil.which("podman"):
        return CheckResult("podman", False, "not found on PATH")
    try:
        subprocess.run(
            ["podman", "version", "--format", "{{.Client.Version}}"],
            capture_output=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return CheckResult("podman", False, f"found but not responding: {exc}")
    return CheckResult("podman", True, "ok")


def check_vault() -> CheckResult:
    """Check whether the vault is reachable."""
    from terok_sandbox import SandboxConfig, is_vault_running, is_vault_socket_active

    if is_vault_socket_active() or is_vault_running(cfg=SandboxConfig()):
        return CheckResult("vault", True, "running")
    return CheckResult("vault", False, "not running")


def check_credentials(provider: str) -> CheckResult:
    """Check whether credentials are stored for *provider*."""
    from terok_sandbox import CredentialDB, SandboxConfig

    cfg = SandboxConfig()
    try:
        db = CredentialDB(cfg.db_path)
    except Exception:  # noqa: BLE001
        return CheckResult(f"{provider} credentials", False, "credential database unavailable")
    try:
        cred = db.load_credential("default", provider)
    finally:
        db.close()
    if cred:
        return CheckResult(f"{provider} credentials", True, "stored")
    return CheckResult(f"{provider} credentials", False, "not found")


def check_images(base_image: str) -> CheckResult:
    """Check whether L0+L1 container images exist."""
    from terok_executor.container.build import l1_image_tag

    tag = l1_image_tag(base_image)
    try:
        result = subprocess.run(
            ["podman", "image", "exists", tag],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return CheckResult("container images", True, "ready")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return CheckResult("container images", False, "not built")


def check_shield() -> CheckResult:
    """Check whether shield OCI hooks are installed (informational only)."""
    from terok_sandbox import check_environment

    ec = check_environment()
    if ec.health == "ok":
        return CheckResult("shield", True, "active")
    return CheckResult("shield", False, "not installed (containers have unrestricted network)")


# ── Interactive fixers ─────────────────────────────────────────────────


def _confirm(prompt: str) -> bool:
    """Ask a yes/no question, default yes."""
    try:
        answer = input(f"  {prompt} [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("", "y", "yes")


def _fix_vault() -> bool:
    """Start the vault, installing systemd units if needed."""
    from terok_sandbox import (
        SandboxConfig,
        install_vault_systemd,
        is_vault_running,
        is_vault_systemd_available,
        start_vault,
    )

    from terok_executor.roster.loader import ensure_vault_routes

    cfg = SandboxConfig()
    ensure_vault_routes(cfg=cfg)

    if is_vault_systemd_available():
        install_vault_systemd(cfg=cfg)
        # systemd socket activation will start on first connection
        return True

    start_vault(cfg=cfg)
    return is_vault_running(cfg=cfg)


def _fix_credentials(provider: str) -> bool:
    """Run the interactive authentication flow for *provider*."""
    from terok_executor.container.build import l1_image_tag
    from terok_executor.credentials.auth import authenticate
    from terok_executor.paths import mounts_dir

    image = l1_image_tag("ubuntu:24.04")
    try:
        authenticate("standalone", provider, mounts_dir=mounts_dir(), image=image)
    except SystemExit:
        return False

    # Write vault config patches for the authenticated provider
    from terok_executor.credentials.vault_config import write_vault_config

    write_vault_config(provider)
    return True


def _fix_images(base_image: str, family: str | None = None) -> bool:
    """Build L0+L1 container images."""
    from terok_executor.container.build import BuildError, build_base_images

    try:
        build_base_images(base_image, family=family)
        return True
    except BuildError as exc:
        print(f"  Build failed: {exc}", file=sys.stderr)
        return False


# ── Orchestrator ───────────────────────────────────────────────────────


def _print_step(step: int, total: int, result: CheckResult) -> None:
    """Print a preflight check result."""
    marker = "ok" if result.ok else "FAIL"
    print(f"  [{step}/{total}] {result.name}... {marker}")


def _provider_hints(current_provider: str) -> None:
    """Print a hint about authenticating additional tools."""
    from terok_executor.roster.loader import get_roster

    roster = get_roster()
    others = sorted(name for name in roster.all_names if name != current_provider)
    if others:
        print("\n  Hint: authenticate additional tools with: terok-executor auth <name>")
        print(f"        Available: {', '.join(others)}")


def run_preflight(
    provider: str,
    *,
    interactive: bool = True,
    base_image: str = "ubuntu:24.04",
    family: str | None = None,
) -> bool:
    """Run all prerequisite checks; fix interactively if possible.

    Returns ``True`` if all checks pass (or were fixed), ``False`` otherwise.
    """
    print()
    total = 4
    all_ok = True

    # 1. Podman
    r = check_podman()
    _print_step(1, total, r)
    if not r.ok:
        print("      Install podman first: https://podman.io/docs/installation", file=sys.stderr)
        return False

    # 2. Vault
    r = check_vault()
    if not r.ok and interactive:
        print(f"  [{2}/{total}] {r.name}... {r.message}")
        if _confirm("Start vault?"):
            fixed = _fix_vault()
            r = CheckResult(r.name, fixed, "started" if fixed else "failed to start")
    _print_step(2, total, r)
    if not r.ok:
        print("      Start with: terok-executor vault start", file=sys.stderr)
        all_ok = False

    # 3. Credentials for the requested provider
    r = check_credentials(provider)
    if not r.ok and interactive:
        print(f"  [{3}/{total}] {r.name}... {r.message}")
        if _confirm(f"Authenticate {provider} now?"):
            fixed = _fix_credentials(provider)
            r = CheckResult(r.name, fixed, "authenticated" if fixed else "authentication failed")
    _print_step(3, total, r)
    if not r.ok:
        print(f"      Run: terok-executor auth {provider}", file=sys.stderr)
        all_ok = False

    # 4. Container images
    r = check_images(base_image)
    if not r.ok and interactive:
        print(f"  [{4}/{total}] {r.name}... {r.message}")
        print("      Building agent images (this may take a few minutes)...")
        fixed = _fix_images(base_image, family=family)
        r = CheckResult(r.name, fixed, "built" if fixed else "build failed")
    _print_step(4, total, r)
    if not r.ok:
        print("      Run: terok-executor build", file=sys.stderr)
        all_ok = False

    # Shield check (informational)
    r = check_shield()
    if not r.ok:
        print(f"\n  Note: {r.message}")

    # Hint about other providers
    if all_ok and interactive:
        _provider_hints(provider)

    print()
    return all_ok
