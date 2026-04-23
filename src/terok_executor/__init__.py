# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-executor: single-agent task runner for hardened Podman containers.

Builds agent images, launches instrumented containers, and manages the
lifecycle of one AI coding agent at a time.  Designed for standalone use
(``terok-executor run claude .``) and as a library for terok orchestration.

The public surface is ``__all__`` below.  Key entry points:

- :class:`AgentRunner` — launch agents in containers
- :func:`authenticate` / :func:`store_api_key` — credential flows
- :func:`build_base_images` — image construction
- :func:`get_roster` — YAML agent registry
"""

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version

try:
    __version__ = _meta_version("terok-executor")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata

# -- terok-sandbox protocol types (re-exported for convenience) ----------------
from terok_sandbox import ConfigScope, ConfigStack
from terok_sandbox.doctor import CheckVerdict, DoctorCheck

# -- Commands + CLI surface ----------------------------------------------------
from .commands import COMMANDS as AGENT_COMMANDS, CommandDef

# -- Config schema (executor-owned slice of the shared config.yml) -----------
from .config_schema import ExecutorConfigView, RawImageSection

# -- Container (build, env assembly, runner) -----------------------------------
from .container.build import (
    AGENTS_LABEL,
    DEFAULT_BASE_IMAGE,
    INSTALLED_ENV_PATH,
    BuildError,
    ImageSet,
    build_base_images,
    build_project_image,
    build_sidecar_image,
    detect_family,
    l0_image_tag,
    l1_image_tag,
    l1_sidecar_image_tag,
    render_l1_sidecar,
    stage_scripts,
    stage_tmux_config,
    stage_toad_agents,
)
from .container.cache import seed_workspace_from_clone_cache
from .container.env import ContainerEnvResult, ContainerEnvSpec, assemble_container_env
from .container.inject import inject_agent_config, inject_prompt
from .container.runner import AgentRunner

# -- Credentials (auth flows, extractors, vault commands) ----------------------
from .credentials.auth import (
    AUTH_PROVIDERS,
    PHANTOM_CREDENTIALS_MARKER,
    AuthProvider,
    authenticate,
    store_api_key,
)
from .credentials.extractors import extract_credential
from .credentials.vault_commands import VAULT_COMMANDS, scan_leaked_credentials
from .credentials.vault_config import ConfigPatchError

# -- Doctor + paths ------------------------------------------------------------
from .doctor import agent_doctor_checks
from .paths import mounts_dir

# -- Provider (headless dispatch, instructions, agent config) ------------------
from .provider.agents import AgentConfigSpec, parse_md_agent, prepare_agent_config_dir
from .provider.config import resolve_provider_value
from .provider.headless import (
    CLIOverrides,
    apply_provider_config,
    build_headless_command,
)
from .provider.instructions import bundled_default_instructions, resolve_instructions
from .provider.providers import (
    AGENT_PROVIDERS,
    PROVIDER_NAMES,
    AgentProvider,
    collect_all_auto_approve_env,
    collect_opencode_provider_env,
    get_provider,
)

# -- Roster (agent catalog + config resolution) --------------------------------
from .roster import (
    SidecarSpec,
    VaultRoute,
    ensure_vault_routes,
    get_roster,
    parse_agent_selection,
)

# -- Storage queries (filesystem footprint measurement) -------------------------
from .storage import (
    SharedMountStorageInfo,
    TaskStorageInfo,
    get_shared_mounts_storage,
    get_task_storage,
    get_tasks_storage,
)

# -- Bootstrap YAML roster into module-level dicts ---------------------------
# AGENT_PROVIDERS and AUTH_PROVIDERS are empty dicts populated here to avoid
# circular imports (roster → auth/providers → roster).


def _bootstrap_roster() -> None:
    """Populate module-level provider dicts from the YAML roster."""
    global PROVIDER_NAMES  # noqa: PLW0603 — tuple requires rebind

    import terok_executor.provider.providers as _reg

    from .roster import get_roster

    roster = get_roster()
    AGENT_PROVIDERS.update(roster.providers)
    AUTH_PROVIDERS.update(roster.auth_providers)
    PROVIDER_NAMES = _reg.PROVIDER_NAMES = roster.agent_names


_bootstrap_roster()

__all__ = [
    "__version__",
    # Provider registry
    "AGENT_PROVIDERS",
    "PROVIDER_NAMES",
    "AgentProvider",
    "get_provider",
    "CLIOverrides",
    "apply_provider_config",
    "build_headless_command",
    "collect_opencode_provider_env",
    "collect_all_auto_approve_env",
    # Agent config preparation
    "AgentConfigSpec",
    "prepare_agent_config_dir",
    "parse_md_agent",
    # Auth
    "AUTH_PROVIDERS",
    "AuthProvider",
    "PHANTOM_CREDENTIALS_MARKER",
    "authenticate",
    "store_api_key",
    # Instructions
    "bundled_default_instructions",
    "resolve_instructions",
    # Config stack
    "ConfigScope",
    "ConfigStack",
    "resolve_provider_value",
    # Config schema (executor-owned slice of the shared config.yml)
    "ExecutorConfigView",
    "RawImageSection",
    # Build: image construction + resource staging
    "AGENTS_LABEL",
    "DEFAULT_BASE_IMAGE",
    "INSTALLED_ENV_PATH",
    "BuildError",
    "ImageSet",
    "build_base_images",
    "build_project_image",
    "build_sidecar_image",
    "detect_family",
    "l0_image_tag",
    "l1_image_tag",
    "l1_sidecar_image_tag",
    "render_l1_sidecar",
    "stage_scripts",
    "stage_toad_agents",
    "stage_tmux_config",
    # Vault
    "VaultRoute",
    "ensure_vault_routes",
    "extract_credential",
    # Roster
    "SidecarSpec",
    "get_roster",
    "parse_agent_selection",
    # Command registry
    "AGENT_COMMANDS",
    "VAULT_COMMANDS",
    "CommandDef",
    "mounts_dir",
    "scan_leaked_credentials",
    "ConfigPatchError",
    # Doctor (container health checks)
    "CheckVerdict",
    "DoctorCheck",
    "agent_doctor_checks",
    # Storage queries
    "SharedMountStorageInfo",
    "TaskStorageInfo",
    "get_shared_mounts_storage",
    "get_task_storage",
    "get_tasks_storage",
    # Runner facade
    "AgentRunner",
    # Container environment assembly
    "ContainerEnvSpec",
    "ContainerEnvResult",
    "assemble_container_env",
    # Clone cache
    "seed_workspace_from_clone_cache",
    # Sealed injection helpers
    "inject_agent_config",
    "inject_prompt",
]
