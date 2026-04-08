# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Launches AI agents in hardened Podman containers.

Builds the environment, prepares agent config, and launches a hardened
Podman container with the requested AI agent.  Three launch modes:

- **Headless**: fire-and-forget with a prompt (``run_headless``)
- **Interactive**: user logs in, agent is ready (``run_interactive``)
- **Web**: toad served over HTTP (``run_web``)

All user config is runtime (env vars + volumes) — no L2 image build needed.
Gate is on by default (safe-by-default egress control).
"""

from __future__ import annotations

import logging
import shlex
import sys
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .build import BuildError, build_base_images

if TYPE_CHECKING:
    from terok_sandbox import LifecycleHooks, Sandbox

    from terok_agent.roster.loader import AgentRoster

_logger = logging.getLogger(__name__)


class AgentRunner:
    """Composes sandbox + agent config into a single container launch.

    All three run methods follow the same flow:

    1. Ensure L0+L1 images exist (build if missing)
    2. Prepare agent-config directory (wrapper, instructions, prompt)
    3. Assemble environment variables and volume mounts
    4. Optionally set up gate (mirror repo, create token)
    5. Launch container via podman
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        sandbox: Sandbox | None = None,
        roster: AgentRoster | None = None,
        base_image: str = "ubuntu:24.04",
    ) -> None:
        self._base_image = base_image
        self._sandbox: Sandbox | None = sandbox
        self._roster: AgentRoster | None = roster

    # ------------------------------------------------------------------
    # Properties (lazy init)
    # ------------------------------------------------------------------

    @property
    def sandbox(self) -> Sandbox:
        """Lazy-init sandbox facade."""
        if self._sandbox is None:
            from terok_sandbox import Sandbox

            self._sandbox = Sandbox()
        return self._sandbox

    @property
    def roster(self) -> AgentRoster:
        """Lazy-init agent roster."""
        if self._roster is None:
            from terok_agent.roster.loader import get_roster

            self._roster = get_roster()
        return self._roster

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

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
        follow: bool = False,
        unrestricted: bool = True,
        gpu: bool = False,
        hooks: LifecycleHooks | None = None,
        human_name: str | None = None,
        human_email: str | None = None,
        authorship: str | None = None,
        shared_dir: Path | None = None,
        shared_mount: str = "/shared",
    ) -> str:
        """Launch a headless agent run. Returns container name.

        The agent executes the *prompt* against *repo* (local path or git URL)
        and exits when done or when *timeout* is reached.  Set *follow=True*
        to block until the agent finishes (the CLI does this by default).
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
            unrestricted=unrestricted,
            gpu=gpu,
            hooks=hooks,
            human_name=human_name,
            human_email=human_email,
            authorship=authorship,
            shared_dir=shared_dir,
            shared_mount=shared_mount,
        )

    def run_interactive(
        self,
        provider: str,
        repo: str,
        *,
        branch: str | None = None,
        gate: bool = True,
        name: str | None = None,
        unrestricted: bool = True,
        gpu: bool = False,
        hooks: LifecycleHooks | None = None,
        human_name: str | None = None,
        human_email: str | None = None,
        authorship: str | None = None,
        shared_dir: Path | None = None,
        shared_mount: str = "/shared",
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
            unrestricted=unrestricted,
            gpu=gpu,
            hooks=hooks,
            human_name=human_name,
            human_email=human_email,
            authorship=authorship,
            shared_dir=shared_dir,
            shared_mount=shared_mount,
        )

    def run_web(
        self,
        repo: str,
        *,
        port: int | None = None,
        branch: str | None = None,
        gate: bool = True,
        name: str | None = None,
        public_url: str | None = None,
        unrestricted: bool = True,
        gpu: bool = False,
        hooks: LifecycleHooks | None = None,
        human_name: str | None = None,
        human_email: str | None = None,
        authorship: str | None = None,
        shared_dir: Path | None = None,
        shared_mount: str = "/shared",
    ) -> str:
        """Launch a toad web container. Returns container name.

        If *port* is None, an available port is auto-allocated.
        """
        if port is None:
            from terok_sandbox import find_free_port

            port = find_free_port()
        return self._run(
            provider="claude",  # toad uses claude as default
            repo=repo,
            branch=branch,
            gate=gate,
            name=name,
            mode="web",
            port=port,
            public_url=public_url,
            unrestricted=unrestricted,
            gpu=gpu,
            hooks=hooks,
            human_name=human_name,
            human_email=human_email,
            authorship=authorship,
            shared_dir=shared_dir,
            shared_mount=shared_mount,
        )

    def run_tool(
        self,
        tool: str,
        repo: str,
        *,
        tool_args: tuple[str, ...] = (),
        branch: str | None = None,
        gate: bool = True,
        name: str | None = None,
        follow: bool = True,
        timeout: int = 600,
    ) -> str:
        """Launch a sidecar tool container. Returns container name.

        Runs the named tool in a lightweight sidecar L1 image (no agent
        CLIs).  The tool receives the real API key from the credential
        store — not a phantom token.
        """
        return self._run(
            provider=tool,
            repo=repo,
            mode="tool",
            gate=gate,
            name=name,
            follow=follow,
            timeout=timeout,
            tool_args=tool_args,
            branch=branch,
        )

    # ------------------------------------------------------------------
    # Internal orchestrator (all public entry points delegate here)
    # ------------------------------------------------------------------

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
        unrestricted: bool = True,
        gpu: bool = False,
        hooks: LifecycleHooks | None = None,
        tool_args: tuple[str, ...] = (),
        human_name: str | None = None,
        human_email: str | None = None,
        authorship: str | None = None,
        shared_dir: Path | None = None,
        shared_mount: str = "/shared",
    ) -> str:
        """Unified launch flow for all modes (headless, interactive, web, tool)."""
        from terok_agent.paths import mounts_dir

        from .env import ContainerEnvSpec, assemble_container_env

        is_tool = mode == "tool"
        task_id = _generate_task_id()
        code_repo, local_path = _resolve_repo(repo)

        # Ensure images — sidecar L1 for tools, agent L1 for everything else
        if is_tool:
            sidecar_spec = self.roster.get_sidecar_spec(provider)
            image_tag = self._ensure_sidecar_image(sidecar_spec.tool_name)
        else:
            from terok_agent.provider.headless import build_headless_command

            agent = self.roster.get_provider(provider)
            image_tag = self._ensure_images()

        # Task directory (ephemeral for standalone runs)
        task_dir = Path(tempfile.mkdtemp(prefix=f"terok-agent-{task_id}-"))

        mounts_base = mounts_dir()

        if is_tool:
            # Sidecar tools: minimal env, no shared mounts, real API key
            env: dict[str, str] = {
                "TASK_ID": task_id,
                "REPO_ROOT": "/workspace",
                "GIT_RESET_MODE": "none",
            }
            if branch:
                env["GIT_BRANCH"] = branch
            env.update(self._direct_credential_env(provider))

            volumes: list[str] = []
            if local_path:
                volumes.append(f"{local_path}:/workspace:Z")
            elif code_repo:
                effective_repo = self._setup_gate(code_repo, task_id) if gate else code_repo
                env["CODE_REPO"] = effective_repo
                workspace = task_dir / "workspace"
                workspace.mkdir(parents=True, exist_ok=True)
                volumes.append(f"{workspace}:/workspace:Z")
        else:
            # Agent modes: full env assembly via canonical builder
            agent_config_dir = self._prepare_agent_config(
                task_dir,
                task_id,
                provider,
                prompt=prompt,
                mounts_base=mounts_base,
                project_root=local_path,
            )

            # Resolve workspace and gate URL
            if local_path:
                ws_host = local_path
                resolved_code_repo = None
            elif code_repo:
                effective_repo = self._setup_gate(code_repo, task_id) if gate else code_repo
                ws_host = task_dir / "workspace"
                ws_host.mkdir(parents=True, exist_ok=True)
                resolved_code_repo = effective_repo
            else:
                ws_host = task_dir / "workspace"
                ws_host.mkdir(parents=True, exist_ok=True)
                resolved_code_repo = None

            spec_kwargs: dict = {
                "task_id": task_id,
                "provider_name": provider,
                "workspace_host_path": ws_host,
                "code_repo": resolved_code_repo,
                "branch": branch,
                "unrestricted": unrestricted,
                "agent_config_dir": agent_config_dir,
                "task_dir": task_dir,
                "envs_dir": mounts_base,
            }
            if human_name:
                spec_kwargs["human_name"] = human_name
            if human_email:
                spec_kwargs["human_email"] = human_email
            if authorship:
                spec_kwargs["authorship"] = authorship
            if shared_dir:
                if not shared_mount.startswith("/") or ":" in shared_mount:
                    raise SystemExit(
                        f"--shared-mount must be an absolute path without ':', got: {shared_mount!r}"
                    )
                if shared_dir.is_file():
                    raise SystemExit(f"--shared-dir exists as a file: {shared_dir}")
                spec_kwargs["shared_dir"] = shared_dir
                spec_kwargs["shared_mount"] = shared_mount

            result = assemble_container_env(
                ContainerEnvSpec(**spec_kwargs),
                self.roster,
            )
            env = dict(result.env)
            volumes = list(result.volumes)

        # Build command based on mode
        extra_args: list[str] = []
        if mode == "tool":
            tool_cmd = f"init-ssh-and-repo.sh && {shlex.quote(provider)}"
            if tool_args:
                tool_cmd += " " + " ".join(shlex.quote(a) for a in tool_args)
            command = ["bash", "-lc", tool_cmd]
        elif mode == "headless":
            cmd_str = build_headless_command(
                agent, timeout=timeout, model=model, max_turns=max_turns
            )
            command = ["bash", "-lc", cmd_str]
        elif mode == "interactive":
            command = [
                "bash",
                "-lc",
                "init-ssh-and-repo.sh && echo __CLI_READY__; tail -f /dev/null",
            ]
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
            image=image_tag,
            task_id=task_id,
            env=env,
            volumes=volumes,
            command=command,
            task_dir=task_dir,
            name=name,
            extra_args=extra_args or None,
            unrestricted=unrestricted,
            gpu=gpu,
            hooks=hooks,
        )

        # Follow output if requested
        if follow and mode in ("headless", "tool"):
            self._stream_headless(cname, timeout=float(timeout + 60))
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

    # ------------------------------------------------------------------
    # Private helpers (in call order from _run)
    # ------------------------------------------------------------------

    def _ensure_images(self) -> str:
        """Ensure L0+L1 images exist, return L1 tag."""
        images = build_base_images(self._base_image)
        return images.l1

    def _ensure_sidecar_image(self, tool_name: str) -> str:
        """Ensure sidecar L1 exists for *tool_name*, return its tag."""
        from .build import build_sidecar_image

        return build_sidecar_image(self._base_image, tool_name=tool_name)

    def _setup_gate(self, repo_url: str, task_id: str) -> str:
        """Mirror a repo via the sandbox gate and return the gate HTTP URL.

        Steps:
        1. Create a bare git mirror under the gate base path
        2. Ensure the gate server is running
        3. Create a task-scoped access token
        4. Construct the HTTP gate URL

        The container will clone from this URL — shield blocks all other egress.
        """
        from terok_sandbox import GitGate

        cfg = self.sandbox.config
        gate_base = cfg.gate_base_path
        gate_base.mkdir(parents=True, exist_ok=True)

        # Derive a collision-free repo key from the full URL
        import hashlib

        url_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
        basename = repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        repo_key = f"{basename}-{url_hash}"
        gate_path = gate_base / repo_key

        # Sync (creates bare mirror if missing, fetches if exists)
        gate = GitGate(
            scope=repo_key,
            gate_path=gate_path,
            upstream_url=repo_url,
        )
        gate.sync()

        # Ensure gate server is running, create token, build URL
        self.sandbox.ensure_gate()
        token = self.sandbox.create_token(repo_key, task_id)
        return self.sandbox.gate_url(gate_path, token)

    def _direct_credential_env(self, tool_name: str) -> dict[str, str]:
        """Load the real API key for a sidecar tool and return as env dict.

        Unlike :meth:`_credential_proxy_env` (which creates phantom tokens),
        this injects the actual credential.  Safe because sidecar containers
        have no agent code that could leak it.
        """
        from terok_sandbox import CredentialDB

        spec = self.roster.get_sidecar_spec(tool_name)
        cfg = self.sandbox.config
        try:
            db = CredentialDB(cfg.proxy_db_path)
        except Exception as exc:
            print(
                f"Warning [runner]: credential DB unavailable: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return {}
        try:
            cred = db.load_credential("default", tool_name)
        finally:
            db.close()

        if cred is None:
            print(
                f"Warning [runner]: no credentials stored for {tool_name!r}. "
                f"Run: terok-agent auth {tool_name} --api-key <key>",
                file=sys.stderr,
            )
            return {}

        return {
            env_var: str(cred.get(cred_key, ""))
            for env_var, cred_key in spec.env_map.items()
            if cred.get(cred_key)
        }

    def _prepare_agent_config(
        self,
        task_dir: Path,
        task_id: str,
        provider: str,
        *,
        prompt: str | None = None,
        instructions: str | None = None,
        mounts_base: Path,
        project_root: Path | None = None,
    ) -> Path:
        """Prepare the agent-config directory for a task.

        *project_root* is passed to :func:`resolve_instructions` so that
        ``<repo>/instructions.md`` is appended when present.
        """
        from terok_agent.provider.agents import AgentConfigSpec, prepare_agent_config_dir
        from terok_agent.provider.instructions import resolve_instructions

        resolved_instructions = instructions or resolve_instructions(
            {}, provider, project_root=project_root
        )

        spec = AgentConfigSpec(
            tasks_root=task_dir.parent,
            task_id=task_id,
            subagents=(),
            prompt=prompt,
            provider=provider,
            instructions=resolved_instructions,
            mounts_base=mounts_base,
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
        unrestricted: bool = True,
        gpu: bool = False,
        hooks: LifecycleHooks | None = None,
    ) -> str:
        """Delegate container launch to :meth:`Sandbox.run`. Returns container name.

        Builds a :class:`~terok_sandbox.RunSpec` from the parameters and
        passes it to the sandbox executor.  All podman command assembly,
        shield integration, GPU args, and error handling live in
        :mod:`terok_sandbox` — this method is a thin mapping layer.
        """
        from terok_sandbox import GpuConfigError, RunSpec

        cname = name or f"terok-agent-{task_id}"

        spec = RunSpec(
            container_name=cname,
            image=image,
            env=env,
            volumes=tuple(volumes),
            command=tuple(command),
            task_dir=task_dir,
            gpu_enabled=gpu,
            extra_args=tuple(extra_args or ()),
            unrestricted=unrestricted,
        )

        try:
            self.sandbox.run(spec, hooks=hooks)
        except GpuConfigError as exc:
            raise BuildError(str(exc)) from exc

        return cname

    @staticmethod
    def _stream_headless(cname: str, timeout: float) -> None:
        """Stream container logs to stdout and print exit code when done."""
        import subprocess
        import sys

        try:
            proc = subprocess.Popen(
                ["podman", "logs", "-f", cname],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            print("Agent timed out", file=sys.stderr)
        except (FileNotFoundError, OSError) as exc:
            print(
                f"Warning [runner]: failed to stream logs for {cname}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

        # Retrieve exit code from the container itself
        try:
            result = subprocess.run(["podman", "wait", cname], capture_output=True, timeout=10)
            exit_code = int(result.stdout.decode().strip()) if result.stdout else 1
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError) as exc:
            print(
                f"Warning [runner]: failed to retrieve exit code for {cname}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            exit_code = 1

        if exit_code != 0:
            print(f"Agent exited with code {exit_code}")


# ── Module-level helpers ────────────────────────────────────────────────


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
