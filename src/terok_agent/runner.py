# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""High-level agent runner composing sandbox + agent config + container launch.

This is the core of ``terok-agent run`` — it builds the environment,
prepares agent config, and launches a hardened Podman container with
the requested AI agent.  Three launch modes:

- **Headless**: fire-and-forget with a prompt (``run_headless``)
- **Interactive**: user logs in, agent is ready (``run_interactive``)
- **Web**: toad served over HTTP (``run_web``)

All user config is runtime (env vars + volumes) — no L2 image build needed.
Gate is on by default (safe-by-default egress control).
"""

from __future__ import annotations

import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .build import BuildError, build_base_images

if TYPE_CHECKING:
    from terok_sandbox import Sandbox

    from .registry import AgentRegistry


def _generate_task_id() -> str:
    """Generate a short unique task identifier."""
    return uuid.uuid4().hex[:12]


def _resolve_repo(repo: str) -> tuple[str | None, Path | None]:
    """Classify *repo* as a git URL or local path.

    Returns ``(code_repo, local_path)`` — exactly one is non-None.
    Raises ``SystemExit`` for ambiguous local paths (look like paths but
    don't exist).
    """
    # Heuristic: if it looks like a local path (starts with /, ./, ~, or has
    # no : before /), check existence
    p = Path(repo).expanduser()
    if p.is_dir():
        return None, p.resolve()
    # If it looks like a local path but doesn't exist, fail early
    if repo.startswith(("/", "./", "../", "~")) or (
        not repo.startswith("git@") and "://" not in repo and ":" not in repo
    ):
        raise SystemExit(f"Local path not found: {repo}")
    # Treat as git URL (SSH, HTTPS, or file://)
    return repo, None


class AgentRunner:
    """Composes sandbox + agent config into a single container launch.

    All three run methods follow the same flow:

    1. Ensure L0+L1 images exist (build if missing)
    2. Prepare agent-config directory (wrapper, instructions, prompt)
    3. Assemble environment variables and volume mounts
    4. Optionally set up gate (mirror repo, create token)
    5. Launch container via podman
    """

    def __init__(
        self,
        *,
        sandbox: Sandbox | None = None,
        registry: AgentRegistry | None = None,
        base_image: str = "ubuntu:24.04",
    ) -> None:
        self._base_image = base_image
        self._sandbox: Sandbox | None = sandbox
        self._registry: AgentRegistry | None = registry

    @property
    def sandbox(self) -> Sandbox:
        """Lazy-init sandbox facade."""
        if self._sandbox is None:
            from terok_sandbox import Sandbox

            self._sandbox = Sandbox()
        return self._sandbox

    @property
    def registry(self) -> AgentRegistry:
        """Lazy-init agent registry."""
        if self._registry is None:
            from .registry import get_registry

            self._registry = get_registry()
        return self._registry

    def _ensure_images(self) -> str:
        """Ensure L0+L1 images exist, return L1 tag."""
        images = build_base_images(self._base_image)
        return images.l1

    def _shared_mounts(self, envs_dir: Path) -> list[str]:
        """Derive shared config volume mounts from the agent registry.

        Each registry entry with an ``auth.host_dir`` and ``auth.container_mount``
        becomes a bind mount so that auth credentials persist across runs.
        """
        mounts = []
        for _name, ap in sorted(self.registry.auth_providers.items()):
            host_dir = envs_dir / ap.host_dir_name
            host_dir.mkdir(parents=True, exist_ok=True)
            mounts.append(f"{host_dir}:{ap.container_mount}:z")
        return mounts

    def _base_env(self, task_id: str, provider_name: str) -> dict[str, str]:
        """Assemble the base environment variables for a container."""
        from .headless_providers import HEADLESS_PROVIDERS

        env: dict[str, str] = {
            "TASK_ID": task_id,
            "REPO_ROOT": "/workspace",
            "GIT_RESET_MODE": "none",
        }

        # OpenCode provider env vars (TEROK_OC_* for Blablador, KISSKI, etc.)
        env.update(self.registry.collect_opencode_provider_env())

        # Git identity — use the agent's configured identity
        provider = HEADLESS_PROVIDERS.get(provider_name)
        if provider:
            env["GIT_AUTHOR_NAME"] = provider.git_author_name
            env["GIT_AUTHOR_EMAIL"] = provider.git_author_email
            env["GIT_COMMITTER_NAME"] = provider.git_author_name
            env["GIT_COMMITTER_EMAIL"] = provider.git_author_email

        return env

    def _prepare_agent_config(
        self,
        task_dir: Path,
        task_id: str,
        provider: str,
        *,
        prompt: str | None = None,
        instructions: str | None = None,
        envs_dir: Path,
    ) -> Path:
        """Prepare the agent-config directory for a task."""
        from .agents import AgentConfigSpec, prepare_agent_config_dir
        from .instructions import resolve_instructions

        resolved_instructions = instructions or resolve_instructions({}, provider)

        spec = AgentConfigSpec(
            tasks_root=task_dir.parent,
            task_id=task_id,
            subagents=(),
            prompt=prompt,
            provider=provider,
            instructions=resolved_instructions,
            envs_base_dir=envs_dir,
        )
        return prepare_agent_config_dir(spec)

    def _launch(
        self,
        *,
        image: str,
        task_id: str,
        env: dict[str, str],
        volumes: list[str],
        command: list[str],
        task_dir: Path,
        name: str | None = None,
        extra_args: list[str] | None = None,
    ) -> str:
        """Assemble and execute the podman run command. Returns container name."""
        from terok_agent._util import podman_userns_args

        cname = name or f"terok-agent-{task_id}"

        cmd = ["podman", "run", "-d"]
        cmd += podman_userns_args()

        # Shield integration — best-effort; runs without if hooks aren't installed
        try:
            shield_args = self.sandbox.pre_start_args(cname, task_dir)
            cmd += shield_args
        except (OSError, FileNotFoundError, SystemExit):
            pass

        if extra_args:
            cmd += extra_args

        for vol in volumes:
            cmd += ["-v", vol]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]

        cmd += ["--name", cname, "-w", "/workspace", image]
        cmd += command

        print("$", shlex.join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise BuildError(f"Container launch failed: {e}") from e

        return cname

    def run_headless(
        self,
        provider: str,
        repo: str,
        *,
        prompt: str,
        branch: str | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        timeout: int = 1800,
        gate: bool = True,
        name: str | None = None,
        follow: bool = True,
    ) -> str:
        """Launch a headless agent run. Returns container name.

        The agent executes the *prompt* against *repo* (local path or git URL)
        and exits when done or when *timeout* is reached.
        """
        return self._run(
            provider=provider,
            repo=repo,
            prompt=prompt,
            branch=branch,
            model=model,
            max_turns=max_turns,
            timeout=timeout,
            gate=gate,
            name=name,
            follow=follow,
            mode="headless",
        )

    def run_interactive(
        self,
        provider: str,
        repo: str,
        *,
        branch: str | None = None,
        gate: bool = True,
        name: str | None = None,
    ) -> str:
        """Launch an interactive container. Returns container name.

        The container stays up after init; user logs in via ``podman exec``.
        """
        return self._run(
            provider=provider,
            repo=repo,
            branch=branch,
            gate=gate,
            name=name,
            mode="interactive",
        )

    def run_web(
        self,
        repo: str,
        *,
        port: int,
        branch: str | None = None,
        gate: bool = True,
        name: str | None = None,
        public_url: str | None = None,
    ) -> str:
        """Launch a toad web container. Returns container name."""
        return self._run(
            provider="claude",  # toad uses claude as default
            repo=repo,
            branch=branch,
            gate=gate,
            name=name,
            mode="web",
            port=port,
            public_url=public_url,
        )

    def _run(
        self,
        *,
        provider: str,
        repo: str,
        mode: str,
        prompt: str | None = None,
        branch: str | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        timeout: int = 1800,
        gate: bool = True,
        name: str | None = None,
        follow: bool = False,
        port: int | None = None,
        public_url: str | None = None,
    ) -> str:
        """Unified launch flow for all three modes."""
        from .headless_providers import build_headless_command, get_provider

        agent = get_provider(provider)
        task_id = _generate_task_id()
        code_repo, local_path = _resolve_repo(repo)

        # Ensure images
        l1_tag = self._ensure_images()

        # Task directory (ephemeral for standalone runs)
        task_dir = Path(tempfile.mkdtemp(prefix=f"terok-agent-{task_id}-"))

        # Env base for shared auth mounts
        envs_dir = self.sandbox.config.effective_envs_dir

        # Prepare agent config
        agent_config_dir = self._prepare_agent_config(
            task_dir,
            task_id,
            provider,
            prompt=prompt,
            envs_dir=envs_dir,
        )

        # Assemble environment
        env = self._base_env(task_id, provider)
        if branch:
            env["GIT_BRANCH"] = branch

        # Repo access: local bind-mount or git clone
        # Gate integration for standalone terok-agent is not yet implemented.
        # When called from terok, the gate flow runs in terok's environment.py
        # (ensure_server_reachable, GitGate.sync, create_token, gate_url)
        # which is independent of AgentRunner.
        if gate and code_repo:
            import warnings

            warnings.warn(
                "Gate mode is not yet implemented in standalone terok-agent. "
                "Running with direct network access.",
                stacklevel=2,
            )

        volumes: list[str] = []
        if local_path:
            volumes.append(f"{local_path}:/workspace:Z")
        elif code_repo:
            env["CODE_REPO"] = code_repo
            workspace = task_dir / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            volumes.append(f"{workspace}:/workspace:Z")

        # Shared auth mounts (derived from registry)
        volumes += self._shared_mounts(envs_dir)

        # Agent config mount
        volumes.append(f"{agent_config_dir}:/home/dev/.terok:Z")

        # Unrestricted mode (auto-approve all agents)
        env["TEROK_UNRESTRICTED"] = "1"
        env.update(self.registry.collect_all_auto_approve_env())

        # Build command based on mode
        extra_args: list[str] = []
        if mode == "headless":
            cmd_str = build_headless_command(
                agent, timeout=timeout, model=model, max_turns=max_turns
            )
            command = ["bash", "-lc", cmd_str]
        elif mode == "interactive":
            command = ["bash", "-lc", "init-ssh-and-repo.sh && echo __CLI_READY__; exec bash -l"]
        elif mode == "web":
            toad_cmd = "init-ssh-and-repo.sh && toad --serve -H 0.0.0.0 -p 8080"
            if public_url:
                toad_cmd += f" --public-url {shlex.quote(public_url)}"
            toad_cmd += " /workspace"
            command = ["bash", "-lc", toad_cmd]
            extra_args += ["-p", f"127.0.0.1:{port}:8080"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Launch
        cname = self._launch(
            image=l1_tag,
            task_id=task_id,
            env=env,
            volumes=volumes,
            command=command,
            task_dir=task_dir,
            name=name,
            extra_args=extra_args or None,
        )

        # Follow output if requested
        if follow and mode == "headless":
            exit_code = self.sandbox.wait_for_exit(cname, timeout=float(timeout + 60))
            if exit_code != 0:
                print(f"Agent exited with code {exit_code}")
        elif mode == "interactive":
            from terok_sandbox import READY_MARKER

            ready = self.sandbox.stream_logs(
                cname,
                timeout=120.0,
                ready_check=lambda line: "__CLI_READY__" in line or READY_MARKER in line,
            )
            if ready:
                print(f"\nContainer ready. Login with:\n  podman exec -it {cname} bash -l")
        elif mode == "web" and port:
            from terok_sandbox import READY_MARKER

            self.sandbox.stream_logs(cname, timeout=120.0)
            url = public_url or f"http://127.0.0.1:{port}"
            print(f"\nToad available at: {url}")

        return cname
