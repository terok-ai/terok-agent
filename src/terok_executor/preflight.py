# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""First-run readiness gate for ``terok-executor run``.

Mandatory prerequisites (podman, sandbox services, container images)
block the launch if unmet after interactive remediation; optional
prerequisites (SSH key, per-agent credentials) print the consequence
of skipping and let the launch proceed.
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


# ── Orchestrator ───────────────────────────────────────────────────────


def run_preflight(
    provider: str,
    *,
    interactive: bool = True,
    assume_yes: bool = False,
    base_image: str = "ubuntu:24.04",
    family: str | None = None,
) -> bool:
    """Run every prerequisite check and return ``True`` iff mandatory items pass.

    In non-interactive mode, missing mandatory prerequisites are
    reported once and the return is ``False``; in interactive mode
    each one is offered up as a y/N fix before counting against
    readiness.  Optional items never turn the return into ``False`` —
    their consequence is printed and the launch proceeds.
    """
    print()
    all_ready = True

    if not _require_podman():
        return False

    if not _require_sandbox_services(interactive=interactive, assume_yes=assume_yes):
        all_ready = False

    if not _require_images(base_image, family, interactive=interactive, assume_yes=assume_yes):
        all_ready = False

    _offer_ssh_key(interactive=interactive, assume_yes=assume_yes)
    _offer_credentials(provider, interactive=interactive, assume_yes=assume_yes)
    _note_shield_bypass()

    if all_ready and interactive:
        _provider_hints(provider)

    print()
    return all_ready


# ── Mandatory gates ────────────────────────────────────────────────────


def _require_podman() -> bool:
    """Hard-stop when podman is missing — nothing terok can install it with."""
    r = check_podman()
    _print_step(r)
    if not r.ok:
        print("      Install podman first: https://podman.io/docs/installation", file=sys.stderr)
        return False
    return True


def _require_sandbox_services(*, interactive: bool, assume_yes: bool) -> bool:
    """Install shield+vault+gate if needed; report remaining gap if not."""
    r = check_sandbox_services()
    if not r.ok and interactive:
        print(f"  {r.name}... {r.message}")
        if _confirm("Install shield + vault + gate now?", assume_yes=assume_yes) and (
            _fix_sandbox_services()
        ):
            r = check_sandbox_services()
    _print_step(r)
    if not r.ok:
        print("      Run: terok-executor setup", file=sys.stderr)
    return r.ok


def _require_images(
    base_image: str, family: str | None, *, interactive: bool, assume_yes: bool
) -> bool:
    """Build L0+L1 images if missing — mandatory, first-run-heavy."""
    r = check_images(base_image)
    if not r.ok and interactive:
        print(f"  {r.name}... {r.message}")
        if _confirm("Build container images now?", assume_yes=assume_yes) and (
            _fix_images(base_image, family=family)
        ):
            r = check_images(base_image)
    _print_step(r)
    if not r.ok:
        print("      Run: terok-executor build", file=sys.stderr)
    return r.ok


# ── Optional offers ────────────────────────────────────────────────────


def _offer_ssh_key(*, interactive: bool, assume_yes: bool) -> None:
    """Generate a gate-signing SSH key when missing; gate push is the consequence."""
    r = check_ssh_key()
    if not r.ok and interactive:
        print(f"  {r.name}... {r.message}")
        if _confirm("Generate an SSH key for gate signing?", assume_yes=assume_yes) and (
            _fix_ssh_key()
        ):
            r = check_ssh_key()
    _print_step(r)
    if not r.ok:
        print("      Without a gate SSH key, git push via the gate won't work.")


def _offer_credentials(provider: str, *, interactive: bool, assume_yes: bool) -> None:
    """Authenticate *provider* when missing; login-on-first-turn is the consequence."""
    r = check_credentials(provider)
    if not r.ok and interactive:
        print(f"  {r.name}... {r.message}")
        if _confirm(f"Authenticate {provider} now?", assume_yes=assume_yes) and (
            _fix_credentials(provider)
        ):
            r = check_credentials(provider)
    _print_step(r)
    if not r.ok:
        print(f"      Without credentials, {provider} will prompt for login on first turn.")


def _note_shield_bypass() -> None:
    """Surface the bypass override when set — regular shield state is in sandbox-services."""
    from terok_sandbox import check_environment

    if check_environment().health == "bypass":
        print("\n  Note: shield is in bypass mode — containers have unrestricted network")


# ── Prerequisite probes ────────────────────────────────────────────────


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


def check_sandbox_services() -> CheckResult:
    """Roll vault + shield-hooks + gate into a single readiness verdict.

    Treated as a unit because all three are installed by the sandbox
    aggregator and fail the same way on a fresh host — reporting each
    individually would just clutter the first-run summary.
    """
    from terok_sandbox import (
        SandboxConfig,
        check_environment,
        get_server_status,
        is_vault_running,
        is_vault_socket_active,
    )

    # One SandboxConfig read covers every downstream probe — each of the
    # helpers below would otherwise rebuild it from layered YAML.
    cfg = SandboxConfig()
    missing: list[str] = []
    if not (is_vault_socket_active() or is_vault_running(cfg=cfg)):
        missing.append("vault")
    if check_environment(cfg).health != "ok":
        missing.append("shield hooks")
    if get_server_status(cfg).mode not in ("systemd", "daemon"):
        missing.append("gate")

    if missing:
        return CheckResult("sandbox services", False, f"missing: {', '.join(missing)}")
    return CheckResult("sandbox services", True, "shield + vault + gate ready")


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


def check_credentials(provider: str) -> CheckResult:
    """Check whether credentials are stored for *provider*."""
    from terok_sandbox import CredentialDB, SandboxConfig

    try:
        db = CredentialDB(SandboxConfig().db_path)
    except Exception:  # noqa: BLE001
        return CheckResult(f"{provider} credentials", False, "credential database unavailable")
    try:
        cred = db.load_credential("default", provider)
    finally:
        db.close()
    if cred:
        return CheckResult(f"{provider} credentials", True, "stored")
    return CheckResult(f"{provider} credentials", False, "not found")


def check_ssh_key(scope: str = "standalone") -> CheckResult:
    """Check whether a gate-signing SSH key exists for *scope*."""
    from terok_sandbox import CredentialDB, SandboxConfig

    try:
        db = CredentialDB(SandboxConfig().db_path)
    except Exception:  # noqa: BLE001
        return CheckResult("ssh key", False, "credential database unavailable")
    try:
        keys = db.list_ssh_keys_for_scope(scope)
    finally:
        db.close()
    if keys:
        return CheckResult("ssh key", True, f"{len(keys)} key(s) registered for '{scope}'")
    return CheckResult("ssh key", False, f"none registered for '{scope}'")


def check_shield() -> CheckResult:
    """Check whether shield OCI hooks are installed (informational)."""
    from terok_sandbox import check_environment

    ec = check_environment()
    if ec.health == "ok":
        return CheckResult("shield", True, "active")
    return CheckResult("shield", False, "not installed (containers have unrestricted network)")


# ── Interactive remediation ────────────────────────────────────────────


def _confirm(prompt: str, *, assume_yes: bool = False) -> bool:
    """Ask a yes/no question; *assume_yes* short-circuits with True."""
    if assume_yes:
        print(f"  {prompt} [Y/n] y")
        return True
    try:
        answer = input(f"  {prompt} [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("", "y", "yes")


def _fix_sandbox_services() -> bool:
    """Run the executor setup path to install + wire up shield+vault+gate+clearance.

    Always installs into the per-user scope.  System-wide installation
    is an explicit operator choice, exposed via ``terok-executor setup
    --root``; the interactive preflight path never escalates to sudo
    behind the user's back.  Goes through
    :func:`terok_executor.sandbox.ensure_sandbox_ready` so the vault
    picks up the roster-derived ``routes.json`` on first start — a
    bare sandbox aggregator call would leave the vault running with
    no authorization config, which breaks credential fetch on the
    next agent run.
    """
    from terok_executor.sandbox import ensure_sandbox_ready

    try:
        ensure_sandbox_ready()
    except (SystemExit, Exception) as exc:  # noqa: BLE001
        print(f"  sandbox setup failed: {exc}", file=sys.stderr)
        return False
    return True


def _fix_images(base_image: str, family: str | None = None) -> bool:
    """Build L0+L1 container images with a friendly first-run banner."""
    from terok_executor.container.build import BuildError, build_base_images

    _print_first_build_preamble()
    try:
        build_base_images(base_image, family=family)
    except BuildError as exc:
        print(f"  Build failed: {exc}", file=sys.stderr)
        return False
    _print_first_build_postamble()
    return True


def _fix_ssh_key(scope: str = "standalone") -> bool:
    """Generate a gate-signing SSH key for *scope* in the credential DB."""
    from terok_sandbox import SandboxConfig, SSHManager

    try:
        with SSHManager.open(scope=scope, db_path=SandboxConfig().db_path) as mgr:
            result = mgr.init()
    except Exception as exc:  # noqa: BLE001
        print(f"  SSH key generation failed: {exc}", file=sys.stderr)
        return False
    print(f"  Generated {result['key_type']} key (fingerprint SHA256:{result['fingerprint']}).")
    print(f"  Public line: {result['public_line']}")
    return True


def _fix_credentials(provider: str) -> bool:
    """Run the interactive authentication flow for *provider*."""
    from terok_executor.container.build import l1_image_tag
    from terok_executor.credentials.auth import authenticate
    from terok_executor.credentials.vault_config import write_vault_config
    from terok_executor.paths import mounts_dir

    image = l1_image_tag("ubuntu:24.04")
    try:
        authenticate(None, provider, mounts_dir=mounts_dir(), image=image)
    except SystemExit:
        return False

    write_vault_config(provider)
    return True


# ── Printing ───────────────────────────────────────────────────────────


def _print_step(result: CheckResult) -> None:
    """Print a preflight check result."""
    marker = "ok" if result.ok else "FAIL"
    print(f"  {result.name:<22} {marker} ({result.message})")


def _print_first_build_preamble() -> None:
    """Announce the first-run image build so the wait doesn't look like a hang."""
    print()
    print("  ─ Building agent images ────────────────────────────────────")
    print("  This is a first-run step and usually takes a few minutes.")
    print("  Subsequent runs reuse the cached layers and start instantly.")
    print("  ────────────────────────────────────────────────────────────")


def _print_first_build_postamble() -> None:
    """Close the build banner once the images are ready."""
    print("  ────────────────────────────────────────────────────────────")
    print("  Images ready.  Next run will skip this step.")
    print()


def _provider_hints(current_provider: str) -> None:
    """Print a hint about authenticating additional tools."""
    from terok_executor.roster.loader import get_roster

    roster = get_roster()
    others = sorted(name for name in roster.all_names if name != current_provider)
    if others:
        print("\n  Hint: authenticate additional tools with: terok-executor auth <name>")
        print(f"        Available: {', '.join(others)}")
