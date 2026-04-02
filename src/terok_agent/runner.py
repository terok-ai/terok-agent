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

    from .roster import AgentRoster

_logger = logging.getLogger(__name__)


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
        roster: AgentRoster | None = None,
        base_image: str = "ubuntu:24.04",
    ) -> None:
        self._base_image = base_image
        self._sandbox: Sandbox | None = sandbox
        self._roster: AgentRoster | None = roster

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
            from .roster import get_roster

            self._roster = get_roster()
        return self._roster

    def _ensure_images(self) -> str:
        """Ensure L0+L1 images exist, return L1 tag."""
        images = build_base_images(self._base_image)
        return images.l1

    def _shared_mounts(self, mounts_base: Path) -> list[str]:
        """Derive shared volume mounts from the agent roster.

        Includes both auth config mounts (per-provider) and general mounts
        (e.g. OpenCode runtime dirs) from the ``mounts:`` YAML section.
        """
        seen: set[str] = set()
        mounts = []
        for _name, ap in sorted(self.roster.auth_providers.items()):
            host_dir = mounts_base / ap.host_dir_name
            host_dir.mkdir(parents=True, exist_ok=True)
            mount = f"{host_dir}:{ap.container_mount}:z"
            if ap.host_dir_name not in seen:
                mounts.append(mount)
                seen.add(ap.host_dir_name)
        for m in self.roster.mounts:
            if m.host_dir not in seen:
                host_dir = mounts_base / m.host_dir
                host_dir.mkdir(parents=True, exist_ok=True)
                mounts.append(f"{host_dir}:{m.container_path}:z")
                seen.add(m.host_dir)
        return mounts

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
            project_id=repo_key,
            gate_path=gate_path,
            upstream_url=repo_url,
        )
        gate.sync()

        # Ensure gate server is running, create token, build URL
        self.sandbox.ensure_gate()
        token = self.sandbox.create_token(repo_key, task_id)
        return self.sandbox.gate_url(gate_path, token)

    def _credential_proxy_env(self, task_id: str) -> dict[str, str]:
        """Inject phantom tokens and base URL overrides if the proxy is running.

        Unlike terok (which hard-fails when the proxy is down), the standalone
        runner silently skips proxy integration — credentials may come from
        shared config mounts instead.

        The proxy listens on a TCP port on the host; containers reach it via
        ``host.containers.internal:<port>`` (same pattern as the gate server).
        """
        from terok_sandbox import (
            CredentialDB,
            get_proxy_port,
            is_proxy_running,
            is_proxy_socket_active,
        )

        cfg = self.sandbox.config
        if not (is_proxy_socket_active() or is_proxy_running(cfg)):
            return {}

        proxy_routes = self.roster.proxy_routes
        try:
            db = CredentialDB(cfg.proxy_db_path)
        except Exception as exc:
            print(
                f"Warning [runner]: credential proxy is running but DB at "
                f"{cfg.proxy_db_path} is unavailable: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return {}
        try:
            credential_set = "default"
            stored_providers = set(db.list_credentials(credential_set))
            routed = stored_providers & proxy_routes.keys()
            if not routed:
                return {}
            # Per-provider phantom tokens — each token encodes the provider
            tokens = {
                name: db.create_proxy_token("standalone", task_id, credential_set, name)
                for name in routed
            }
        finally:
            db.close()

        port = get_proxy_port(cfg)
        proxy_base = f"http://host.containers.internal:{port}"
        env: dict[str, str] = {}

        for name, route in proxy_routes.items():
            if name not in routed:
                continue
            for env_var in route.phantom_env:
                env[env_var] = tokens[name]
            if route.base_url_env:
                env[route.base_url_env] = proxy_base
            # Override OpenCode base URL for proxied providers
            provider = self.roster.providers.get(name)
            if provider and provider.opencode_config:
                oc_base_key = f"TEROK_OC_{name.upper()}_BASE_URL"
                env[oc_base_key] = f"{proxy_base}/v1"
            # glab: redirect API to proxy via env vars
            if name == "glab":
                env["GITLAB_API_HOST"] = f"host.containers.internal:{port}"
                env["API_PROTOCOL"] = "http"

        # Signal for init script to start socat bridges
        if routed:
            env["TEROK_PROXY_PORT"] = str(port)

        _logger.debug("Credential proxy: injected %d env vars for %s", len(env), stored_providers)
        return env

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

    def _base_env(self, task_id: str, provider_name: str) -> dict[str, str]:
        """Assemble the base environment variables for a container."""
        env: dict[str, str] = {
            "TASK_ID": task_id,
            "REPO_ROOT": "/workspace",
            "GIT_RESET_MODE": "none",
        }

        # OpenCode provider env vars (TEROK_OC_* for Blablador, KISSKI, etc.)
        env.update(self.roster.collect_opencode_provider_env())

        # Git identity — use the agent's configured identity from roster
        provider = self.roster.providers.get(provider_name)
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
        mounts_base: Path,
        project_root: Path | None = None,
    ) -> Path:
        """Prepare the agent-config directory for a task.

        *project_root* is passed to :func:`resolve_instructions` so that
        ``<repo>/instructions.md`` is appended when present.
        """
        from .agents import AgentConfigSpec, prepare_agent_config_dir
        from .instructions import resolve_instructions

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
        bypass_shield: bool = False,
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
            bypass_shield=bypass_shield,
        )

        try:
            self.sandbox.run(spec, hooks=hooks)
        except GpuConfigError as exc:
            raise BuildError(str(exc)) from exc

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
        follow: bool = False,
        unrestricted: bool = True,
        gpu: bool = False,
        bypass_shield: bool = False,
        hooks: LifecycleHooks | None = None,
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
            bypass_shield=bypass_shield,
            hooks=hooks,
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
        bypass_shield: bool = False,
        hooks: LifecycleHooks | None = None,
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
            bypass_shield=bypass_shield,
            hooks=hooks,
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
        bypass_shield: bool = False,
        hooks: LifecycleHooks | None = None,
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
            bypass_shield=bypass_shield,
            hooks=hooks,
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
        unrestricted: bool = True,
        gpu: bool = False,
        bypass_shield: bool = False,
        hooks: LifecycleHooks | None = None,
    ) -> str:
        """Unified launch flow for all three modes."""
        from .headless_providers import build_headless_command

        agent = self.roster.get_provider(provider)
        task_id = _generate_task_id()
        code_repo, local_path = _resolve_repo(repo)

        # Ensure images
        l1_tag = self._ensure_images()

        # Task directory (ephemeral for standalone runs)
        task_dir = Path(tempfile.mkdtemp(prefix=f"terok-agent-{task_id}-"))

        # Env base for shared auth mounts
        from .paths import mounts_dir

        mounts_base = mounts_dir()

        # Prepare agent config (pass local_path so repo instructions.md is found)
        agent_config_dir = self._prepare_agent_config(
            task_dir,
            task_id,
            provider,
            prompt=prompt,
            mounts_base=mounts_base,
            project_root=local_path,
        )

        # Assemble environment
        env = self._base_env(task_id, provider)
        if branch:
            env["GIT_BRANCH"] = branch

        # Repo access: local bind-mount or git clone (with optional gate)
        volumes: list[str] = []
        if local_path:
            volumes.append(f"{local_path}:/workspace:Z")
        elif code_repo:
            if gate:
                # Gate mode: mirror repo, serve via HTTP, block other egress
                effective_repo = self._setup_gate(code_repo, task_id)
            else:
                effective_repo = code_repo
            env["CODE_REPO"] = effective_repo
            workspace = task_dir / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            volumes.append(f"{workspace}:/workspace:Z")

        # Shared auth mounts (derived from roster)
        volumes += self._shared_mounts(mounts_base)

        # Credential proxy: inject phantom tokens and base URL overrides
        env.update(self._credential_proxy_env(task_id))

        # Agent config mount
        volumes.append(f"{agent_config_dir}:/home/dev/.terok:Z")

        # Permission mode — unrestricted enables auto-approve for all agents
        if unrestricted:
            env["TEROK_UNRESTRICTED"] = "1"
            env.update(self.roster.collect_all_auto_approve_env())

        # Build command based on mode
        extra_args: list[str] = []
        if mode == "headless":
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
            image=l1_tag,
            task_id=task_id,
            env=env,
            volumes=volumes,
            command=command,
            task_dir=task_dir,
            name=name,
            extra_args=extra_args or None,
            unrestricted=unrestricted,
            gpu=gpu,
            bypass_shield=bypass_shield,
            hooks=hooks,
        )

        # Follow output if requested
        if follow and mode == "headless":
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
