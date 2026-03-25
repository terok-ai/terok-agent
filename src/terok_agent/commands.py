# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Command registry for terok-agent.

Follows the same :class:`CommandDef` / :class:`ArgDef` pattern as
``terok_sandbox.commands``.  Higher-level consumers (terok) can import
``COMMANDS`` to build their own CLI frontends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


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
    """Definition of a terok-agent subcommand."""

    name: str
    help: str = ""
    handler: Callable[..., None] | None = None
    args: tuple[ArgDef, ...] = ()
    group: str = ""


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_agents(*, show_all: bool = False) -> None:
    """List registered agents."""
    import sys

    from .registry import _load_bundled_agents, _load_user_agents, get_registry

    reg = get_registry()
    names = reg.all_names if show_all else reg.agent_names

    if not names:
        print("No agents registered.", file=sys.stderr)
        return

    raw = _load_bundled_agents()
    raw.update(_load_user_agents())

    rows: list[tuple[str, str, str]] = []
    for name in sorted(names):
        p = reg.providers.get(name)
        auth = reg.auth_providers.get(name)
        label = p.label if p else (auth.label if auth else name)
        kind = raw.get(name, {}).get("kind", "native")
        rows.append((name, label, kind))

    w_name = max(len("NAME"), max(len(r[0]) for r in rows))
    w_label = max(len("LABEL"), max(len(r[1]) for r in rows))

    print(f"{'NAME':<{w_name}}  {'LABEL':<{w_label}}  TYPE")
    for name, label, kind in rows:
        print(f"{name:<{w_name}}  {label:<{w_label}}  {kind}")


def _handle_build(
    *, base: str = "ubuntu:24.04", rebuild: bool = False, full_rebuild: bool = False
) -> None:
    """Build L0+L1 container images."""
    from .build import BuildError, build_base_images

    try:
        images = build_base_images(base, rebuild=rebuild, full_rebuild=full_rebuild)
    except BuildError as e:
        raise SystemExit(str(e)) from e
    print(f"\nL0: {images.l0}")
    print(f"L1: {images.l1}")


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
    bypass_shield_no_protection: bool = False,
) -> None:
    """Run an agent in a hardened container."""
    from .runner import AgentRunner

    effective_gate = gate and not no_gate
    runner = AgentRunner()
    common = {
        "gate": effective_gate,
        "name": name,
        "branch": branch,
        "unrestricted": not restricted,
        "gpu": gpu,
        "bypass_shield": bypass_shield_no_protection,
    }

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


def _handle_auth(*, agent: str, api_key: str | None = None) -> None:
    """Run auth flow for an agent."""
    from .auth import AUTH_PROVIDERS, authenticate, store_api_key

    if api_key is not None:
        if not api_key.strip():
            raise SystemExit("API key cannot be empty.")
        if agent not in AUTH_PROVIDERS:
            available = ", ".join(AUTH_PROVIDERS)
            raise SystemExit(f"Unknown provider: {agent}. Available: {available}")
        store_api_key(agent, api_key.strip())
        return

    from .build import l1_image_tag

    # Need an L1 image for the auth container
    image = l1_image_tag("ubuntu:24.04")
    from terok_sandbox import SandboxConfig

    cfg = SandboxConfig()
    authenticate("standalone", agent, envs_base_dir=cfg.effective_envs_dir, image=image)


def _handle_ls() -> None:
    """List running terok-agent containers."""
    from terok_sandbox import get_project_container_states

    states = get_project_container_states("terok-agent-")
    if not states:
        print("No running containers.")
        return
    for name, state in sorted(states.items()):
        print(f"{name}  {state}")


def _handle_stop(*, name: str) -> None:
    """Stop a running container (best-effort)."""
    from terok_sandbox import get_container_state, stop_task_containers

    state = get_container_state(name)
    if state is None:
        print(f"Container not found: {name}")
        return
    stop_task_containers([name])
    print(f"Stopped: {name}")


# ---------------------------------------------------------------------------
# Command definitions
# ---------------------------------------------------------------------------

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
        ArgDef(
            name="--bypass-shield-no-protection",
            action="store_true",
            help="DANGEROUS: disable egress firewall entirely",
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
        ArgDef(name="--rebuild", action="store_true", help="Force rebuild (cache bust)"),
        ArgDef(name="--full-rebuild", action="store_true", help="Force --no-cache --pull=always"),
    ),
)

LS_COMMAND = CommandDef(name="ls", help="List running containers", handler=_handle_ls)

STOP_COMMAND = CommandDef(
    name="stop",
    help="Stop a running container",
    handler=_handle_stop,
    args=(ArgDef(name="name", help="Container name"),),
)

#: All terok-agent commands.
COMMANDS: tuple[CommandDef, ...] = (
    RUN_COMMAND,
    AUTH_COMMAND,
    AGENTS_COMMAND,
    BUILD_COMMAND,
    LS_COMMAND,
    STOP_COMMAND,
)
