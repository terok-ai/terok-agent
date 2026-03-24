# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Build resource staging for container image construction.

Provides functions to stage bundled resources (scripts, toad agents, tmux
config) from the terok-agent package into a Dockerfile build context
directory.  Used by both ``terok-agent build`` (standalone) and terok's
``docker.py`` (orchestrated).
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path


def _copy_package_tree(package: str, rel_path: str, dest: Path) -> None:
    """Copy a directory tree from package resources to a filesystem path.

    Uses ``importlib.resources`` Traversable API so it works from
    wheels and zip installs.
    """
    root = resources.files(package) / rel_path

    def _recurse(src, dst: Path) -> None:  # type: ignore[no-untyped-def]
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            out = dst / child.name
            if child.is_dir():
                _recurse(child, out)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(child.read_bytes())

    _recurse(root, dest)


def stage_scripts(dest: Path) -> None:
    """Stage container helper scripts into *dest*.

    Copies all files from ``terok_agent/resources/scripts/`` into the given
    directory, replacing any existing contents.  Python bytecode caches and
    ``__init__.py`` markers are excluded.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_agent", "resources/scripts", dest)
    # Clean out packaging artifacts that shouldn't enter the build context
    for unwanted in dest.rglob("__pycache__"):
        shutil.rmtree(unwanted)
    init = dest / "__init__.py"
    if init.exists():
        init.unlink()


def stage_toad_agents(dest: Path) -> None:
    """Stage Toad ACP agent TOML definitions into *dest*.

    These describe OpenCode-based agents (Blablador, KISSKI, etc.) that are
    injected into Toad's bundled agent directory at container build time.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_agent", "resources/toad-agents", dest)
    init = dest / "__init__.py"
    if init.exists():
        init.unlink()


def stage_tmux_config(dest: Path) -> None:
    """Stage the container tmux configuration into *dest*.

    Copies ``container-tmux.conf`` — the green-status-bar config that
    distinguishes container tmux sessions from host tmux.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_agent", "resources/tmux", dest)
    init = dest / "__init__.py"
    if init.exists():
        init.unlink()
