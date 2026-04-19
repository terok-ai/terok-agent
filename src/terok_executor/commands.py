# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Registers the subcommands that terok-executor exposes to users.

Each subcommand is a :class:`CommandDef` built from :class:`ArgDef` pieces.
``COMMANDS`` at module bottom is the authoritative catalog — higher-level
consumers (terok) import it to build their own CLI frontends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


# ── Vocabulary ──


@dataclass(frozen=True)
class ArgDef:
    """Definition of a single CLI argument."""

    name: str
    help: str = ""
    type: Callable[[str], Any] | None = None
    default: Any = None
    action: str | None = None
    dest: str | None = None
    nargs: int | str | None = None


@dataclass(frozen=True)
class CommandDef:
    """Definition of a terok-executor subcommand."""

    name: str
    help: str = ""
    handler: Callable[..., None] | None = None
    args: tuple[ArgDef, ...] = ()
    group: str = ""


# ── Handlers ──


def _resolve_host_git_identity() -> tuple[str | None, str | None]:
    """Read git user.name / user.email from the host's global config."""
    import subprocess

    name = email = None
    for key, target in (("user.name", "name"), ("user.email", "email")):
        try:
            result = subprocess.run(
                ["git", "config", "--global", key],
                capture_output=True,
                timeout=5,
            )
            val = result.stdout.decode().strip() if result.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            val = None
        if target == "name":
            name = val
        else:
            email = val
    return name, email


def _handle_run(
    *,
    agent: str,
    repo: str = ".",
    prompt: str | None = None,
    model: str | None = None,
    max_turns: int | None = None,
    timeout: int = 1800,
    interactive: bool = False,
    web: bool = False,
    port: int | None = None,
    gate: bool = True,
    no_gate: bool = False,
    branch: str | None = None,
    name: str | None = None,
    restricted: bool = False,
    gpu: bool = False,
    memory: str | None = None,
    cpus: str | None = None,
    git_identity_from_host: bool = False,
    shared_dir: str | None = None,
    shared_mount: str = "/shared",
    base: str = "ubuntu:24.04",
    family: str | None = None,
) -> None:
    """Run an agent in a hardened container."""
    import sys

    from .preflight import run_preflight

    if not run_preflight(agent, interactive=sys.stdin.isatty(), base_image=base, family=family):
        raise SystemExit(1)

    from .container.runner import AgentRunner

    # Resolve human identity from host git config if requested
    human_name = human_email = authorship = None
    if git_identity_from_host:
        human_name, human_email = _resolve_host_git_identity()
        if human_name:
            authorship = "agent-human"
            print(f"Git identity from host: {human_name} <{human_email or 'nobody@localhost'}>")
        else:
            print("Warning: --git-identity-from-host: git config user.name not set, skipping")

    effective_gate = gate and not no_gate
    runner = AgentRunner(base_image=base, family=family)
    resolved_shared_dir = Path(shared_dir) if shared_dir else None
    common: dict = {
        "gate": effective_gate,
        "name": name,
        "branch": branch,
        "unrestricted": not restricted,
        "gpu": gpu,
        "memory": memory,
        "cpus": cpus,
        "human_name": human_name,
        "human_email": human_email,
        "authorship": authorship,
        "shared_dir": resolved_shared_dir,
    }
    if resolved_shared_dir:
        common["shared_mount"] = shared_mount

    if web:
        cname = runner.run_web(repo, port=port, **common)
    elif interactive:
        cname = runner.run_interactive(agent, repo, **common)
    elif prompt:
        cname = runner.run_headless(
            agent,
            repo,
            prompt=prompt,
            model=model,
            max_turns=max_turns,
            timeout=timeout,
            follow=True,
            **common,
        )
    else:
        raise SystemExit(
            "Specify --prompt for headless mode, --interactive for CLI mode, or --web for toad mode."
        )

    print(f"Container: {cname}")


def _handle_run_tool(
    *,
    tool: str,
    repo: str = ".",
    branch: str | None = None,
    gate: bool = True,
    no_gate: bool = False,
    name: str | None = None,
    timeout: int = 600,
    tool_args: list[str] | None = None,
    base: str = "ubuntu:24.04",
    family: str | None = None,
) -> None:
    """Run a tool in a sidecar container."""
    from .container.runner import AgentRunner

    effective_gate = gate and not no_gate
    runner = AgentRunner(base_image=base, family=family)
    cname = runner.run_tool(
        tool,
        repo,
        tool_args=tuple(tool_args or ()),
        branch=branch,
        gate=effective_gate,
        name=name,
        timeout=timeout,
    )
    print(f"Container: {cname}")


def _handle_auth(*, agent: str, api_key: str | None = None) -> None:
    """Run auth flow for an agent."""
    from .credentials.auth import AUTH_PROVIDERS, authenticate, store_api_key

    if api_key is not None:
        if not api_key.strip():
            raise SystemExit("API key cannot be empty.")
        if agent not in AUTH_PROVIDERS:
            available = ", ".join(AUTH_PROVIDERS)
            raise SystemExit(f"Unknown provider: {agent}. Available: {available}")
        store_api_key(agent, api_key.strip())
    else:
        from .container.build import l1_image_tag

        image = l1_image_tag("ubuntu:24.04")
        from .paths import mounts_dir

        authenticate("standalone", agent, mounts_dir=mounts_dir(), image=image)

    # Write vault URLs to shared config files (e.g. Vibe config.toml, gh config.yml)
    from .credentials.vault_config import write_vault_config

    write_vault_config(agent)


def _handle_agents(*, show_all: bool = False) -> None:
    """List registered agents."""
    import sys

    from .roster.loader import _load_bundled_agents, _load_user_agents, get_roster

    roster = get_roster()
    names = roster.all_names if show_all else roster.agent_names

    if not names:
        print("No agents registered.", file=sys.stderr)
        return

    raw = _load_bundled_agents()
    raw.update(_load_user_agents())

    rows: list[tuple[str, str, str]] = []
    for name in sorted(names):
        p = roster.providers.get(name)
        auth = roster.auth_providers.get(name)
        label = p.label if p else (auth.label if auth else name)
        kind = raw.get(name, {}).get("kind", "native")
        rows.append((name, label, kind))

    w_name = max(len("NAME"), max(len(r[0]) for r in rows))
    w_label = max(len("LABEL"), max(len(r[1]) for r in rows))

    print(f"{'NAME':<{w_name}}  {'LABEL':<{w_label}}  TYPE")
    for name, label, kind in rows:
        print(f"{name:<{w_name}}  {label:<{w_label}}  {kind}")


def _handle_build(
    *,
    base: str = "ubuntu:24.04",
    family: str | None = None,
    agents: str = "all",
    rebuild: bool = False,
    full_rebuild: bool = False,
    sidecar: bool = False,
) -> None:
    """Build L0+L1 container images (optionally include sidecar L1)."""
    from .container.build import BuildError, build_base_images, build_sidecar_image
    from .roster.loader import parse_agent_selection

    selection = parse_agent_selection(agents)

    try:
        images = build_base_images(
            base, family=family, agents=selection, rebuild=rebuild, full_rebuild=full_rebuild
        )
    except (BuildError, ValueError) as e:
        # ValueError is raised by resolve_selection() for unknown agent names
        # — surface it as a clean CLI message rather than a traceback.
        raise SystemExit(str(e)) from e
    print(f"\nL0: {images.l0}")
    print(f"L1: {images.l1}")

    if sidecar:
        try:
            tag = build_sidecar_image(
                base, family=family, rebuild=rebuild, full_rebuild=full_rebuild
            )
        except BuildError as e:
            raise SystemExit(str(e)) from e
        print(f"L1 (sidecar): {tag}")


def _handle_list() -> None:
    """List running terok-executor containers."""
    from terok_sandbox import PodmanRuntime

    states = PodmanRuntime().container_states("terok-executor")
    if not states:
        print("No running containers.")
        return
    for name, state in sorted(states.items()):
        print(f"{name}  {state}")


def _handle_stop(*, name: str) -> None:
    """Stop a running container (best-effort)."""
    from terok_sandbox import PodmanRuntime

    runtime = PodmanRuntime()
    container = runtime.container(name)
    if container.state is None:
        print(f"Container not found: {name}")
        return
    runtime.force_remove([container])
    print(f"Stopped: {name}")


def _handle_setup(*, check: bool = False) -> None:
    """Check prerequisites and optionally fix them interactively."""

    from .preflight import (
        check_credentials,
        check_images,
        check_podman,
        check_proxy,
        check_shield,
    )

    checks = [
        ("podman", check_podman()),
        ("vault", check_proxy()),
        ("shield", check_shield()),
        ("images", check_images("ubuntu:24.04")),
    ]

    print("\nterok-executor status:\n")
    all_ok = True
    for label, result in checks:
        marker = "ok" if result.ok else ("info" if label == "shield" else "FAIL")
        print(f"  {result.name:<22} {marker} ({result.message})")
        if not result.ok and label != "shield":
            all_ok = False

    # Credential summary for known agent providers
    from .roster.loader import get_roster

    roster = get_roster()
    print()
    for name in sorted(roster.agent_names):
        cr = check_credentials(name)
        marker = "ok" if cr.ok else "--"
        print(f"  {name:<22} {marker}")

    if all_ok:
        print("\nAll prerequisites met. Run: terok-executor run <agent> <repo>")
    else:
        if not check:
            print("\nFix issues above, or just run: terok-executor run <agent> <repo>")
            print("The run command will offer to fix missing prerequisites interactively.")
        else:
            print("\nSome prerequisites are missing (see above).")
            raise SystemExit(1)
    print()


# ── Command definitions ──

RUN_COMMAND = CommandDef(
    name="run",
    help="Run an agent in a hardened container",
    handler=_handle_run,
    args=(
        ArgDef(name="agent", help="Agent name (claude, codex, vibe, ...)"),
        ArgDef(name="repo", nargs="?", default=".", help="Local path or git URL (default: .)"),
        ArgDef(name="-p", dest="prompt", help="Prompt for headless mode"),
        ArgDef(name="-m", dest="model", help="Model override"),
        ArgDef(name="--max-turns", type=int, help="Maximum agent turns"),
        ArgDef(name="--timeout", type=int, default=1800, help="Timeout in seconds (default: 1800)"),
        ArgDef(name="--interactive", action="store_true", help="CLI mode (user logs in)"),
        ArgDef(name="--web", action="store_true", help="Toad web mode"),
        ArgDef(name="--port", type=int, help="Port for web mode (auto-allocated if omitted)"),
        ArgDef(name="--gate", action="store_true", default=True, help="Use gate (default)"),
        ArgDef(name="--no-gate", action="store_true", help="Disable gate (direct network)"),
        ArgDef(name="--branch", help="Git branch to check out"),
        ArgDef(name="--name", help="Container name override"),
        ArgDef(
            name="--restricted",
            action="store_true",
            help="Restrict agent permissions (no auto-approve, no-new-privileges)",
        ),
        ArgDef(name="--gpu", action="store_true", help="Enable GPU passthrough"),
        ArgDef(name="--memory", help="Container memory limit (e.g. 4g, 512m)"),
        ArgDef(name="--cpus", help="Container CPU limit (e.g. 2.0, 0.5)"),
        ArgDef(
            name="--git-identity-from-host",
            action="store_true",
            help="Use host git config user.name/email as human committer identity",
        ),
        ArgDef(name="--shared-dir", help="Host directory to mount as shared IPC space"),
        ArgDef(
            name="--shared-mount",
            default="/shared",
            help="Container mount point for shared dir (default: /shared)",
        ),
        ArgDef(name="--base", default="ubuntu:24.04", help="Base OS image (default: ubuntu:24.04)"),
        ArgDef(
            name="--family",
            default=None,
            help="Override package family for unknown base images (deb or rpm)",
        ),
    ),
)

RUN_TOOL_COMMAND = CommandDef(
    name="run-tool",
    help="Run a tool in a sidecar container (separate L1, real API key)",
    handler=_handle_run_tool,
    args=(
        ArgDef(name="tool", help="Tool name (coderabbit)"),
        ArgDef(name="repo", nargs="?", default=".", help="Local path or git URL (default: .)"),
        ArgDef(name="--branch", help="Git branch to check out"),
        ArgDef(name="--gate", action="store_true", default=True, help="Use gate (default)"),
        ArgDef(name="--no-gate", action="store_true", help="Disable gate"),
        ArgDef(name="--name", help="Container name override"),
        ArgDef(name="--timeout", type=int, default=600, help="Timeout in seconds (default: 600)"),
        ArgDef(name="tool_args", nargs="*", help="Extra args passed to the tool (after --)"),
        ArgDef(name="--base", default="ubuntu:24.04", help="Base OS image (default: ubuntu:24.04)"),
        ArgDef(
            name="--family",
            default=None,
            help="Override package family for unknown base images (deb or rpm)",
        ),
    ),
)

AUTH_COMMAND = CommandDef(
    name="auth",
    help="Authenticate an agent",
    handler=_handle_auth,
    args=(
        ArgDef(name="agent", help="Agent or tool name (claude, codex, gh, ...)"),
        ArgDef(name="--api-key", help="Store an API key directly (skip interactive auth)"),
    ),
)

AGENTS_COMMAND = CommandDef(
    name="agents",
    help="List registered agents (use --all to include tools like gh, glab)",
    handler=_handle_agents,
    args=(
        ArgDef(name="--all", action="store_true", dest="show_all", help="Include tools (gh, glab)"),
    ),
)

BUILD_COMMAND = CommandDef(
    name="build",
    help="Build L0+L1 container images",
    handler=_handle_build,
    args=(
        ArgDef(name="--base", default="ubuntu:24.04", help="Base OS image (default: ubuntu:24.04)"),
        ArgDef(
            name="--family",
            default=None,
            help="Override package family for unknown base images (deb or rpm)",
        ),
        ArgDef(
            name="--agents",
            default="all",
            help='Comma-separated roster entries to install, or "all" (default).',
        ),
        ArgDef(name="--rebuild", action="store_true", help="Force rebuild (cache bust)"),
        ArgDef(name="--full-rebuild", action="store_true", help="Force --no-cache --pull=always"),
        ArgDef(name="--sidecar", action="store_true", help="Also build sidecar L1 (CodeRabbit)"),
    ),
)

LIST_COMMAND = CommandDef(name="list", help="List running containers", handler=_handle_list)

STOP_COMMAND = CommandDef(
    name="stop",
    help="Stop a running container",
    handler=_handle_stop,
    args=(ArgDef(name="name", help="Container name"),),
)

SETUP_COMMAND = CommandDef(
    name="setup",
    help="Check prerequisites (vault, images, credentials)",
    handler=_handle_setup,
    args=(
        ArgDef(
            name="--check",
            action="store_true",
            help="Report status and exit non-zero if prerequisites are missing",
        ),
    ),
)

#: All terok-executor commands.
COMMANDS: tuple[CommandDef, ...] = (
    RUN_COMMAND,
    RUN_TOOL_COMMAND,
    AUTH_COMMAND,
    AGENTS_COMMAND,
    BUILD_COMMAND,
    SETUP_COMMAND,
    LIST_COMMAND,
    STOP_COMMAND,
)
