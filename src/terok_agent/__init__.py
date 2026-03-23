# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-agent: single-agent task runner for hardened Podman containers.

Builds agent images, launches instrumented containers, and manages the
lifecycle of one AI coding agent at a time.  Designed for standalone use
(``terok-agent run claude .``) and as a library for terok orchestration.

Public API re-exports from instrumentation modules::

    from terok_agent import HEADLESS_PROVIDERS, get_provider
    from terok_agent import AgentConfigSpec, prepare_agent_config_dir
    from terok_agent import authenticate, AUTH_PROVIDERS
    from terok_agent import resolve_instructions, bundled_default_instructions
    from terok_agent import ConfigStack, ConfigScope, resolve_provider_value
"""

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version

try:
    __version__ = _meta_version("terok-agent")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata

from .agent_config import resolve_provider_value
from .agents import AgentConfigSpec, prepare_agent_config_dir
from .auth import AUTH_PROVIDERS, authenticate
from .config_stack import ConfigScope, ConfigStack
from .headless_providers import HEADLESS_PROVIDERS, HeadlessProvider, get_provider
from .instructions import bundled_default_instructions, resolve_instructions

__all__ = [
    "__version__",
    # Provider registry
    "HEADLESS_PROVIDERS",
    "HeadlessProvider",
    "get_provider",
    # Agent config preparation
    "AgentConfigSpec",
    "prepare_agent_config_dir",
    # Auth
    "AUTH_PROVIDERS",
    "authenticate",
    # Instructions
    "bundled_default_instructions",
    "resolve_instructions",
    # Config stack
    "ConfigScope",
    "ConfigStack",
    "resolve_provider_value",
]
