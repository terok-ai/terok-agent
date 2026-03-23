# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-agent: single-agent task runner for hardened Podman containers.

Builds agent images, launches instrumented containers, and manages the
lifecycle of one AI coding agent at a time.  Designed for standalone use
(``terok-agent run claude .``) and as a library for terok orchestration.

Public API re-exports from instrumentation modules::

    # Provider registry & types
    from terok_agent import (
        HEADLESS_PROVIDERS,
        PROVIDER_NAMES,
        HeadlessProvider,
        get_provider,
        CLIOverrides,
        ProviderConfig,
        WrapperConfig,
        OpenCodeProviderConfig,
        apply_provider_config,
        build_headless_command,
        collect_opencode_provider_env,
        collect_all_auto_approve_env,
    )

    # Agent config preparation
    from terok_agent import (
        AgentConfigSpec,
        prepare_agent_config_dir,
        parse_md_agent,
    )

    # Auth
    from terok_agent import AUTH_PROVIDERS, AuthProvider, authenticate

    # Instructions
    from terok_agent import resolve_instructions, bundled_default_instructions

    # Config stack & resolution
    from terok_agent import (
        ConfigStack,
        ConfigScope,
        resolve_provider_value,
        deep_merge,
        load_yaml_scope,
        load_json_scope,
    )

    # Podman utilities
    from terok_agent import podman_userns_args
"""

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version

try:
    __version__ = _meta_version("terok-agent")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata

# -- Podman utilities ----------------------------------------------------------
from ._util import podman_userns_args

# -- Config resolution ---------------------------------------------------------
from .agent_config import resolve_provider_value

# -- Agent config preparation --------------------------------------------------
from .agents import AgentConfigSpec, parse_md_agent, prepare_agent_config_dir

# -- Auth ----------------------------------------------------------------------
from .auth import AUTH_PROVIDERS, AuthProvider, authenticate

# -- Config stack --------------------------------------------------------------
from .config_stack import ConfigScope, ConfigStack, deep_merge, load_json_scope, load_yaml_scope

# -- Provider registry & types ------------------------------------------------
from .headless_providers import (
    HEADLESS_PROVIDERS,
    PROVIDER_NAMES,
    CLIOverrides,
    HeadlessProvider,
    OpenCodeProviderConfig,
    ProviderConfig,
    WrapperConfig,
    apply_provider_config,
    build_headless_command,
    collect_all_auto_approve_env,
    collect_opencode_provider_env,
    get_provider,
)

# -- Instructions --------------------------------------------------------------
from .instructions import bundled_default_instructions, resolve_instructions

__all__ = [
    "__version__",
    # Provider registry & types
    "HEADLESS_PROVIDERS",
    "PROVIDER_NAMES",
    "HeadlessProvider",
    "OpenCodeProviderConfig",
    "get_provider",
    "CLIOverrides",
    "ProviderConfig",
    "WrapperConfig",
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
    # Instructions
    "bundled_default_instructions",
    "resolve_instructions",
    # Config stack & resolution
    "ConfigScope",
    "ConfigStack",
    "resolve_provider_value",
    "deep_merge",
    "load_yaml_scope",
    "load_json_scope",
    # Podman utilities
    "podman_userns_args",
]
