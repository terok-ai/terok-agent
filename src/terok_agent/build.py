# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container image building and build-context staging.

Owns the L0 (base dev) and L1 (agent CLI) Dockerfile templates, resource
staging, image naming, and ``podman build`` invocation.

**Image layer architecture**::

    L0  (base)   — Ubuntu + dev tools + init script + dev user
    L1  (agent)  — All AI agent CLIs, shell environment, ACP wrappers
                   L1 is self-sufficient for standalone use — all user
                   config (repo URL, SSH, branch, gate) is runtime.
    ─── boundary: above owned by terok-agent, below by terok ───
    L2  (project)— Optional: user Dockerfile snippet (custom packages)
                   Only built when project has docker snippet config.

``terok-agent run claude .`` launches directly on the L1 image — no L2
build needed.  terok adds L2 only for project-specific image customisation.

Usage as a library::

    from terok_agent.build import build_base_images

    images = build_base_images("ubuntu:24.04")
    # images.l0 = "terok-l0:ubuntu-24.04"
    # images.l1 = "terok-l1-cli:ubuntu-24.04"

The templates currently use Dockerfile ``ARG``/``${VAR}`` syntax.
Jinja2 is wired in for future use — converting L1 install blocks to
registry-driven ``{% for agent %}`` loops.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_IMAGE = "ubuntu:24.04"
"""Default base OS image when none is specified."""

_DEFAULT_TAG = "ubuntu-24.04"
"""Pre-sanitized tag fragment for the default base image."""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BuildError(RuntimeError):
    """Raised when base-image construction cannot complete.

    The CLI maps this to a user-facing error message; library callers
    can catch it without being terminated by ``SystemExit``.
    """


# ---------------------------------------------------------------------------
# Image naming convention
# ---------------------------------------------------------------------------


def _normalize_base_image(base_image: str | None) -> str:
    """Normalize a base image string, falling back to the default."""
    return (base_image or "").strip() or DEFAULT_BASE_IMAGE


def _base_tag(base_image: str) -> str:
    """Derive a safe OCI tag fragment from an arbitrary base image string.

    Replaces non-alphanumeric characters (except ``_``, ``.``, ``-``) with
    dashes, lowercases, and truncates with a SHA1 suffix if too long.
    """
    raw = _normalize_base_image(base_image)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-.").lower() or _DEFAULT_TAG
    if len(tag) > 120:
        digest = hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
        tag = f"{tag[:111]}-{digest}"
    return tag


def l0_image_tag(base_image: str) -> str:
    """Return the L0 base dev image tag for *base_image*."""
    return f"terok-l0:{_base_tag(base_image)}"


def l1_image_tag(base_image: str) -> str:
    """Return the L1 agent CLI image tag for *base_image*."""
    return f"terok-l1-cli:{_base_tag(base_image)}"


@dataclass(frozen=True)
class ImageSet:
    """L0 + L1 image tags produced by a build."""

    l0: str
    """L0 base dev image tag (e.g. ``terok-l0:ubuntu-24.04``)."""

    l1: str
    """L1 agent CLI image tag (e.g. ``terok-l1-cli:ubuntu-24.04``)."""


# ---------------------------------------------------------------------------
# Build-context resource staging
# ---------------------------------------------------------------------------


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


def _clean_packaging_artifacts(dest: Path) -> None:
    """Remove __pycache__ dirs and __init__.py from a staged directory."""
    for unwanted in dest.rglob("__pycache__"):
        shutil.rmtree(unwanted)
    init = dest / "__init__.py"
    if init.exists():
        init.unlink()


def stage_scripts(dest: Path) -> None:
    """Stage container helper scripts into *dest*.

    Copies all files from ``terok_agent/resources/scripts/`` into the given
    directory, replacing any existing contents.  Python bytecode caches and
    ``__init__.py`` markers are excluded.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_agent", "resources/scripts", dest)
    _clean_packaging_artifacts(dest)


def stage_toad_agents(dest: Path) -> None:
    """Stage Toad ACP agent TOML definitions into *dest*.

    These describe OpenCode-based agents (Blablador, KISSKI, etc.) that are
    injected into Toad's bundled agent directory at container build time.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_agent", "resources/toad-agents", dest)
    _clean_packaging_artifacts(dest)


def stage_tmux_config(dest: Path) -> None:
    """Stage the container tmux configuration into *dest*.

    Copies ``container-tmux.conf`` — the green-status-bar config that
    distinguishes container tmux sessions from host tmux.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_agent", "resources/tmux", dest)
    _clean_packaging_artifacts(dest)


# ---------------------------------------------------------------------------
# Dockerfile template rendering
# ---------------------------------------------------------------------------


def _render_template(template_name: str, variables: dict[str, str]) -> str:
    """Render a Jinja2 Dockerfile template from package resources.

    Templates live in ``resources/templates/``.  Currently they use
    Dockerfile ``ARG``/``${VAR}`` syntax (Jinja2 is a pass-through).
    Future L1 templatisation will use ``{% for agent %}`` loops driven
    by the YAML agent registry.
    """
    from jinja2 import BaseLoader, Environment

    raw = (resources.files("terok_agent") / "resources" / "templates" / template_name).read_text(
        encoding="utf-8"
    )
    env = Environment(  # nosec B701 — Dockerfile output, not HTML
        loader=BaseLoader(), keep_trailing_newline=True, autoescape=False
    )
    tmpl = env.from_string(raw)
    return tmpl.render(**variables)


def render_l0(base_image: str = DEFAULT_BASE_IMAGE) -> str:
    """Render the L0 (base dev) Dockerfile.

    The *base_image* is normalised before rendering so that blank or
    whitespace-only values produce a valid Dockerfile.
    """
    return _render_template(
        "l0.dev.Dockerfile.template",
        {"BASE_IMAGE": _normalize_base_image(base_image)},
    )


def render_l1(l0_image: str, *, cache_bust: str = "0") -> str:
    """Render the L1 (agent CLI) Dockerfile.

    *l0_image* is the tag of the L0 image to build on top of.
    *cache_bust* invalidates the agent-install layers when changed
    (typically set to a Unix timestamp).
    """
    return _render_template(
        "l1.agent-cli.Dockerfile.template",
        {"BASE_IMAGE": l0_image, "AGENT_CACHE_BUST": cache_bust},
    )


# ---------------------------------------------------------------------------
# Build context preparation
# ---------------------------------------------------------------------------


def prepare_build_context(dest: Path) -> None:
    """Stage auxiliary resources into a build context directory.

    After calling this, *dest* contains the resources that Dockerfile
    ``COPY`` directives reference:

    - ``scripts/``     — container helper scripts (init, env, ACP wrappers)
    - ``toad-agents/`` — ACP agent TOML definitions
    - ``tmux/``        — container tmux config

    Dockerfiles themselves are **not** written here — they are rendered
    and placed by :func:`build_base_images` (which calls this function
    internally).
    """
    dest.mkdir(parents=True, exist_ok=True)
    stage_scripts(dest / "scripts")
    stage_toad_agents(dest / "toad-agents")
    stage_tmux_config(dest / "tmux")


# ---------------------------------------------------------------------------
# Image building
# ---------------------------------------------------------------------------


def _check_podman() -> None:
    """Raise :class:`BuildError` if podman is not on PATH."""
    if shutil.which("podman") is None:
        raise BuildError("podman not found; please install podman")


def _image_exists(image: str) -> bool:
    """Check if a container image exists locally."""
    result = subprocess.run(
        ["podman", "image", "exists", image],
        capture_output=True,
    )
    return result.returncode == 0


def build_base_images(
    base_image: str = DEFAULT_BASE_IMAGE,
    *,
    rebuild: bool = False,
    full_rebuild: bool = False,
    build_dir: Path | None = None,
) -> ImageSet:
    """Build L0 + L1 container images and return their tags.

    Skips building if images already exist locally (unless *rebuild* or
    *full_rebuild* is set).  Uses a temporary directory for the build
    context by default; pass *build_dir* to use a specific (empty or
    non-existent) directory instead.

    Args:
        base_image: Base OS image (e.g. ``ubuntu:24.04``, ``nvidia/cuda:...``).
        rebuild: Force rebuild with cache bust (refreshes agent installs).
        full_rebuild: Force rebuild with ``--no-cache --pull=always``.
        build_dir: Build context directory (must be empty or absent).

    Returns:
        :class:`ImageSet` with the L0 and L1 image tags.

    Raises:
        BuildError: If podman is missing or a build step fails.
        ValueError: If *build_dir* exists and is non-empty.
    """
    # Validate arguments before any side effects (podman probe, temp dirs)
    if build_dir is not None:
        if build_dir.is_file():
            raise ValueError(f"build_dir is a file, not a directory: {build_dir}")
        if build_dir.exists() and any(build_dir.iterdir()):
            raise ValueError(f"build_dir must be empty or absent: {build_dir}")

    _check_podman()

    base_image = _normalize_base_image(base_image)
    l0_tag = l0_image_tag(base_image)
    l1_tag = l1_image_tag(base_image)

    # Skip if both images exist and no forced rebuild
    if not rebuild and not full_rebuild:
        if _image_exists(l0_tag) and _image_exists(l1_tag):
            return ImageSet(l0=l0_tag, l1=l1_tag)

    # Prepare build context in a safe directory
    import tempfile

    own_tmp = build_dir is None
    context = build_dir or Path(tempfile.mkdtemp(prefix="terok-agent-build-"))

    try:
        prepare_build_context(context)

        # Single timestamp for both render and build-arg consistency
        cache_bust = str(int(time.time()))

        # Render and write Dockerfiles into the build context
        (context / "L0.Dockerfile").write_text(render_l0(base_image))
        (context / "L1.cli.Dockerfile").write_text(render_l1(l0_tag, cache_bust=cache_bust))

        ctx = str(context)

        # Build L0 — base dev image (Ubuntu + git + SSH + init script)
        cmd_l0 = ["podman", "build", "-f", str(context / "L0.Dockerfile")]
        cmd_l0 += ["--build-arg", f"BASE_IMAGE={base_image}"]
        cmd_l0 += ["-t", l0_tag]
        if full_rebuild:
            cmd_l0 += ["--no-cache", "--pull=always"]
        cmd_l0.append(ctx)

        print("$", " ".join(cmd_l0))
        subprocess.run(cmd_l0, check=True)

        # Build L1 — agent CLI layer (all agent installs, shell env, ACP wrappers)
        cmd_l1 = ["podman", "build", "-f", str(context / "L1.cli.Dockerfile")]
        cmd_l1 += ["--build-arg", f"BASE_IMAGE={l0_tag}"]
        cmd_l1 += ["--build-arg", f"AGENT_CACHE_BUST={cache_bust}"]
        cmd_l1 += ["-t", l1_tag]
        if full_rebuild:
            cmd_l1.append("--no-cache")
        cmd_l1.append(ctx)

        print("$", " ".join(cmd_l1))
        subprocess.run(cmd_l1, check=True)

    except subprocess.CalledProcessError as e:
        raise BuildError(f"Image build failed: {e}") from e
    finally:
        if own_tmp:
            shutil.rmtree(context, ignore_errors=True)

    return ImageSet(l0=l0_tag, l1=l1_tag)
