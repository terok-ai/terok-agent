# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-agent: single-agent task runner for hardened Podman containers.

Builds agent images, launches instrumented containers, and manages the
lifecycle of one AI coding agent at a time.  Designed for standalone use
(``terok-agent run claude .``) and as a library for terok orchestration.

Public API::

    # Provider registry
    from terok_agent import HEADLESS_PROVIDERS, HeadlessProvider, get_provider
    from terok_agent import PROVIDER_NAMES, CLIOverrides
    from terok_agent import apply_provider_config, build_headless_command
    from terok_agent import collect_opencode_provider_env, collect_all_auto_approve_env

    # Agent config preparation
    from terok_agent import AgentConfigSpec, prepare_agent_config_dir, parse_md_agent

    # Auth
    from terok_agent import AUTH_PROVIDERS, AuthProvider, authenticate

    # Instructions
    from terok_agent import resolve_instructions, bundled_default_instructions

    # Config stack
    from terok_agent import ConfigStack, ConfigScope, resolve_provider_value

Internal symbols (available via submodule import for white-box tests)::

    from terok_agent.headless_providers import generate_agent_wrapper, generate_all_wrappers
    from terok_agent.headless_providers import OpenCodeProviderConfig, ProviderConfig, WrapperConfig
    from terok_agent.config_stack import deep_merge, load_yaml_scope, load_json_scope
    from terok_agent.instructions import has_custom_instructions
    from terok_agent._util import podman_userns_args
"""

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version

try:
    __version__ = _meta_version("terok-agent")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata

# -- Config resolution ---------------------------------------------------------
from .agent_config import resolve_provider_value

# -- Agent config preparation --------------------------------------------------
from .agents import AgentConfigSpec, parse_md_agent, prepare_agent_config_dir

# -- Auth ----------------------------------------------------------------------
from .auth import AUTH_PROVIDERS, AuthProvider, authenticate, store_api_key

# -- Build: image construction + resource staging ------------------------------
from .build import (
    DEFAULT_BASE_IMAGE,
    BuildError,
    ImageSet,
    build_base_images,
    l0_image_tag,
    l1_image_tag,
    stage_scripts,
    stage_tmux_config,
    stage_toad_agents,
)

# -- Command registry ----------------------------------------------------------
from .commands import COMMANDS as AGENT_COMMANDS, CommandDef

# -- Config stack --------------------------------------------------------------
from .config_stack import ConfigScope, ConfigStack

# -- Credential proxy ----------------------------------------------------------
from .credential_extractors import extract_credential

# -- Provider registry ---------------------------------------------------------
from .headless_providers import (
    HEADLESS_PROVIDERS,
    PROVIDER_NAMES,
    CLIOverrides,
    HeadlessProvider,
    apply_provider_config,
    build_headless_command,
    collect_all_auto_approve_env,
    collect_opencode_provider_env,
    get_provider,
)

# -- Instructions --------------------------------------------------------------
from .instructions import bundled_default_instructions, resolve_instructions
from .registry import CredentialProxyRoute

# -- Runner facade -------------------------------------------------------------
from .runner import AgentRunner

# -- Bootstrap YAML registry into module-level dicts ---------------------------
# HEADLESS_PROVIDERS and AUTH_PROVIDERS are empty dicts populated here to avoid
# circular imports (registry → auth/headless_providers → registry).


def _bootstrap_registry() -> None:
    """Populate module-level provider dicts from the YAML registry."""
    global PROVIDER_NAMES  # noqa: PLW0603 — tuple requires rebind

    import terok_agent.headless_providers as _hp

    from .registry import get_registry

    reg = get_registry()
    HEADLESS_PROVIDERS.update(reg.providers)
    AUTH_PROVIDERS.update(reg.auth_providers)
    PROVIDER_NAMES = _hp.PROVIDER_NAMES = reg.agent_names


_bootstrap_registry()

__all__ = [
    "__version__",
    # Provider registry
    "HEADLESS_PROVIDERS",
    "PROVIDER_NAMES",
    "HeadlessProvider",
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
    "authenticate",
    "store_api_key",
    # Instructions
    "bundled_default_instructions",
    "resolve_instructions",
    # Config stack
    "ConfigScope",
    "ConfigStack",
    "resolve_provider_value",
    # Build: image construction + resource staging
    "DEFAULT_BASE_IMAGE",
    "BuildError",
    "ImageSet",
    "build_base_images",
    "l0_image_tag",
    "l1_image_tag",
    "stage_scripts",
    "stage_toad_agents",
    "stage_tmux_config",
    # Credential proxy
    "CredentialProxyRoute",
    "extract_credential",
    # Command registry
    "AGENT_COMMANDS",
    "CommandDef",
    # Runner facade
    "AgentRunner",
]
