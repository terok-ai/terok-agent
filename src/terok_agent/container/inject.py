# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Injection helpers for sealed containers.

In sealed isolation mode, the container has no bind mounts — files must be
injected via ``podman cp``.  These helpers complement
:func:`~terok_agent.provider.agents.prepare_agent_config_dir` which prepares
the files on the host side.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def inject_agent_config(container_name: str, config_dir: Path) -> None:
    """Copy a prepared agent-config directory into a sealed container.

    The container must be in the *created* or *stopped* state.  Delegates
    to :meth:`terok_sandbox.Sandbox.copy_to`.
    """
    from terok_sandbox import Sandbox

    Sandbox().copy_to(container_name, config_dir, "/home/dev/.terok")


def inject_prompt(container_name: str, prompt_text: str) -> None:
    """Write a follow-up prompt into a stopped sealed container.

    Writes *prompt_text* to a temp file and copies it into the container
    via ``podman cp``.  Works on stopped containers (unlike ``podman exec``),
    which is the expected state during headless follow-ups.
    """
    from terok_sandbox import Sandbox

    with tempfile.TemporaryDirectory() as td:
        prompt_file = Path(td) / "prompt.txt"
        prompt_file.write_text(prompt_text, encoding="utf-8")
        Sandbox().copy_to(container_name, prompt_file, "/home/dev/.terok/prompt.txt")
