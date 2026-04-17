# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Builds L0 (base dev) and L1 (agent CLI) container images via podman.

Owns the L0 (base dev) and L1 (agent CLI) Dockerfile templates, resource
staging, image naming, and ``podman build`` invocation.

**Image layer architecture**::

    L0  (base)   — Distro + dev tools + init script + dev user
    L1  (agent)  — All AI agent CLIs, shell environment, ACP wrappers
                   L1 is self-sufficient for standalone use — all user
                   config (repo URL, SSH, branch, gate) is runtime.
    ─── boundary: above owned by terok-executor, below by terok ───
    L2  (project)— Optional: user Dockerfile snippet (custom packages)
                   Only built when project has docker snippet config.

``terok-executor run claude .`` launches directly on the L1 image — no L2
build needed.  terok adds L2 only for project-specific image customisation.

Usage as a library::

    from terok_executor import build_base_images

    images = build_base_images("ubuntu:24.04")
    # images.l0 = "terok-l0:ubuntu-24.04"
    # images.l1 = "terok-l1-cli:ubuntu-24.04"

The L0/L1 templates select between Debian/Ubuntu (``apt``) and Fedora-like
(``dnf``) package managers via a ``family`` Jinja2 variable resolved by
:func:`detect_family` from the base image name (or an explicit override).

L1 is roster-driven: each agent's install steps live in its YAML file
(``install.run_as_root`` / ``install.run_as_dev``), and the L1 template
loops over the resolved selection.  Build emits an OCI label
``ai.terok.agents=<csv>``, an in-container manifest
``/etc/terok/installed.env``, and pre-rendered ``hilfe`` help fragments —
all derived from the same selection.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

# ── Vocabulary ──

DEFAULT_BASE_IMAGE = "ubuntu:24.04"
"""Default base OS image when none is specified."""

AGENTS_LABEL = "ai.terok.agents"
"""OCI label naming the roster entries baked into an L1 image."""

INSTALLED_ENV_PATH = "/etc/terok/installed.env"
"""In-container env file that scripts source to learn what's installed."""

_HELP_SECTION_FILES: dict[str, str] = {"agent": "agents.txt", "dev_tool": "dev-tools.txt"}
"""Maps each :class:`~terok_executor.roster.loader.HelpSection` to its fragment filename."""

_DEFAULT_TAG = "ubuntu-24.04"
"""Pre-sanitized tag fragment for the default base image."""

# Map of known base-image prefixes to their package family.  Each entry
# is either a literal ``"deb"``/``"rpm"`` or a tag-aware resolver — used
# for NVIDIA, where the same repo path ships both Ubuntu (apt) and UBI
# (dnf) variants and only the tag distinguishes them.
# "Officially tested" (per AGENTS.md): ubuntu:24.04, fedora:43,
# quay.io/podman/stable, nvcr.io/nvidia/nvhpc.  Other images in the
# same family path will match but are unsupported.
_NVIDIA_UBI_TAG_RE: re.Pattern[str] = re.compile(r"ubi\d+", re.IGNORECASE)


def _nvidia_family(tag: str) -> str:
    """Pick the family for a matched NVIDIA image from its *tag*.

    NVIDIA tags carry an explicit ``ubuntu`` or ``ubi[N]`` marker; absence
    of either is treated as the historical default of Ubuntu (``deb``).
    """
    return "rpm" if _NVIDIA_UBI_TAG_RE.search(tag) else "deb"


_KNOWN_FAMILIES: tuple[tuple[str, str | Callable[[str], str]], ...] = (
    ("registry.fedoraproject.org/fedora", "rpm"),
    ("quay.io/podman", "rpm"),
    ("nvcr.io/nvidia", _nvidia_family),
    ("nvidia", _nvidia_family),
    ("ubuntu", "deb"),
    ("debian", "deb"),
    ("fedora", "rpm"),
)


class BuildError(RuntimeError):
    """Raised when base-image construction cannot complete.

    The CLI maps this to a user-facing error message; library callers
    can catch it without being terminated by ``SystemExit``.
    """


@dataclass(frozen=True)
class ImageSet:
    """L0 + L1 image tags produced by a build."""

    l0: str
    """L0 base dev image tag (e.g. ``terok-l0:ubuntu-24.04``)."""

    l1: str
    """L1 agent CLI image tag (e.g. ``terok-l1-cli:ubuntu-24.04``)."""

    l1_sidecar: str | None = None
    """L1 sidecar image tag, if built (e.g. ``terok-l1-sidecar:ubuntu-24.04``)."""


# ── Public entry points ──


def detect_family(base_image: str, override: str | None = None) -> str:
    """Resolve the package family (``deb`` or ``rpm``) for *base_image*.

    *override* — when set, must be ``"deb"`` or ``"rpm"`` and wins over
    detection (used to support unknown bases via project config).

    Detection matches a small allowlist of known image prefixes
    (Ubuntu/Debian, Fedora, the official Podman container, NVIDIA CUDA/HPC
    SDK).  NVIDIA images are inspected at the tag level so UBI variants
    (e.g. ``…:13.0.0-devel-ubi9``) resolve to ``rpm`` while Ubuntu
    variants resolve to ``deb``.  Unknown images raise :class:`BuildError`
    with a hint to set ``family:`` explicitly.
    """
    if override is not None:
        if override not in {"deb", "rpm"}:
            raise BuildError(f"family must be 'deb' or 'rpm', got {override!r}")
        return override
    name, tag = _split_image_ref(_normalize_base_image(base_image))
    name_lc = name.lower()
    for prefix, fam in _KNOWN_FAMILIES:
        if name_lc == prefix or name_lc.startswith(prefix + "/"):
            return fam(tag) if callable(fam) else fam
    raise BuildError(
        f"Cannot infer package family for base image {base_image!r}. "
        "Set `family: deb` or `family: rpm` under image: in project.yml."
    )


def build_base_images(
    base_image: str = DEFAULT_BASE_IMAGE,
    *,
    family: str | None = None,
    agents: str | tuple[str, ...] = "all",
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
        family: Override for the package family (``"deb"`` or ``"rpm"``).
            ``None`` means detect from *base_image* via :func:`detect_family`.
        agents: Roster entries to install, as the literal string ``"all"``
            (every entry) or a tuple of names (transitively expanded by
            ``depends_on``).  Same selection drives the OCI label, the L1
            tag suffix, the in-container manifest, and the help fragments.
        rebuild: Force rebuild with cache bust (refreshes agent installs).
        full_rebuild: Force rebuild with ``--no-cache --pull=always``.
        build_dir: Build context directory (must be empty or absent).

    Returns:
        :class:`ImageSet` with the L0 and L1 image tags.

    Raises:
        BuildError: If podman is missing, the family cannot be resolved,
            or a build step fails.
        ValueError: If *build_dir* is a file or a non-empty directory,
            or if *agents* contains unknown roster entries.
    """
    from terok_executor.roster.loader import get_roster

    _validate_build_dir(build_dir)
    _check_podman()

    base_image = _normalize_base_image(base_image)
    selected = get_roster().resolve_selection(agents)

    l0_tag = l0_image_tag(base_image)
    l1_tag = l1_image_tag(base_image, selected)
    l1_alias = l1_image_tag(base_image)

    # Skip if both images exist and no forced rebuild — done before
    # detect_family() so cached images for unknown bases (built earlier
    # with explicit family) can still be reused without supplying it again.
    if not rebuild and not full_rebuild:
        if _image_exists(l0_tag) and _image_exists(l1_tag):
            return ImageSet(l0=l0_tag, l1=l1_tag)

    fam = detect_family(base_image, override=family)

    # Prepare build context in a safe directory
    import tempfile

    own_tmp = build_dir is None
    context = build_dir or Path(tempfile.mkdtemp(prefix="terok-executor-build-"))

    try:
        prepare_build_context(context)
        stage_help_fragments(context / "help.d", selected)

        # Single timestamp for both render and build-arg consistency
        cache_bust = str(int(time.time()))

        # Render and write Dockerfiles into the build context
        (context / "L0.Dockerfile").write_text(render_l0(base_image, family=fam))
        (context / "L1.cli.Dockerfile").write_text(
            render_l1(l0_tag, family=fam, agents=selected, cache_bust=cache_bust)
        )

        ctx = str(context)

        # Build L0 — base dev image (Ubuntu + git + SSH + init script)
        cmd_l0 = ["podman", "build", "-f", str(context / "L0.Dockerfile")]
        cmd_l0 += ["--build-arg", f"BASE_IMAGE={base_image}"]
        cmd_l0 += ["-t", l0_tag]
        if full_rebuild:
            cmd_l0 += ["--no-cache", "--pull=always"]
        cmd_l0.append(ctx)

        print("$", shlex.join(cmd_l0))
        subprocess.run(cmd_l0, check=True)

        # The unsuffixed alias lets `terok run` find the latest L1 without
        # knowing the agent-set suffix; the suffixed tag keeps prior selections
        # individually addressable.
        cmd_l1 = ["podman", "build", "-f", str(context / "L1.cli.Dockerfile")]
        cmd_l1 += ["--build-arg", f"BASE_IMAGE={l0_tag}"]
        cmd_l1 += ["--build-arg", f"AGENT_CACHE_BUST={cache_bust}"]
        cmd_l1 += ["-t", l1_tag, "-t", l1_alias]
        if full_rebuild:
            cmd_l1.append("--no-cache")
        cmd_l1.append(ctx)

        print("$", shlex.join(cmd_l1))
        subprocess.run(cmd_l1, check=True)

    except (OSError, subprocess.CalledProcessError) as e:
        raise BuildError(f"Image build failed: {e}") from e
    finally:
        if own_tmp:
            shutil.rmtree(context, ignore_errors=True)

    return ImageSet(l0=l0_tag, l1=l1_tag)


def build_sidecar_image(
    base_image: str = DEFAULT_BASE_IMAGE,
    *,
    family: str | None = None,
    tool_name: str = "coderabbit",
    rebuild: bool = False,
    full_rebuild: bool = False,
    build_dir: Path | None = None,
) -> str:
    """Build the L1 sidecar image for a specific tool. Returns the image tag.

    Ensures L0 exists first (builds it if missing), then builds the
    sidecar image FROM L0.  The sidecar contains only the named tool —
    no agent CLIs, no LLMs.

    Args:
        base_image: Base OS image (passed through to L0 build).
        family: Override for the package family (``"deb"`` or ``"rpm"``).
            ``None`` means detect from *base_image* via :func:`detect_family`.
        tool_name: Tool to install (selects Jinja2 conditional in template).
        rebuild: Force rebuild with cache bust.
        full_rebuild: Force rebuild with ``--no-cache``.
        build_dir: Build context directory (must be empty or absent).

    Returns:
        The sidecar image tag (e.g. ``terok-l1-sidecar:ubuntu-24.04``).

    Raises:
        BuildError: If podman is missing, the family cannot be resolved,
            or a build step fails.
        ValueError: If *build_dir* is a file or a non-empty directory.
    """
    _validate_build_dir(build_dir)
    _check_podman()

    base_image = _normalize_base_image(base_image)
    l0_tag = l0_image_tag(base_image)
    sidecar_tag = l1_sidecar_image_tag(base_image)

    # Same fast-path as build_base_images: defer detect_family until we
    # know we actually need to render Dockerfiles, so cached sidecars
    # for unknown bases can be reused without re-supplying ``family``.
    if not rebuild and not full_rebuild and _image_exists(sidecar_tag) and _image_exists(l0_tag):
        return sidecar_tag

    fam = detect_family(base_image, override=family)

    # Ensure L0 exists (build if needed)
    if not _image_exists(l0_tag) or full_rebuild:
        build_base_images(base_image, family=fam, rebuild=rebuild, full_rebuild=full_rebuild)

    import tempfile

    own_tmp = build_dir is None
    context = build_dir or Path(tempfile.mkdtemp(prefix="terok-executor-sidecar-"))

    try:
        prepare_build_context(context)
        cache_bust = str(int(time.time()))

        (context / "L1.sidecar.Dockerfile").write_text(
            render_l1_sidecar(l0_tag, family=fam, tool_name=tool_name, cache_bust=cache_bust)
        )

        cmd = ["podman", "build", "-f", str(context / "L1.sidecar.Dockerfile")]
        cmd += ["--build-arg", f"BASE_IMAGE={l0_tag}"]
        cmd += ["--build-arg", f"TOOL_CACHE_BUST={cache_bust}"]
        cmd += ["-t", sidecar_tag]
        if full_rebuild:
            cmd.append("--no-cache")
        cmd.append(str(context))

        print("$", shlex.join(cmd))
        subprocess.run(cmd, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise BuildError(f"Sidecar image build failed: {e}") from e
    finally:
        if own_tmp:
            shutil.rmtree(context, ignore_errors=True)

    return sidecar_tag


# ── Build context ──


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


# ── Dockerfile rendering ──


def render_l0(base_image: str = DEFAULT_BASE_IMAGE, *, family: str | None = None) -> str:
    """Render the L0 (base dev) Dockerfile.

    The *base_image* is normalised before rendering so that blank or
    whitespace-only values produce a valid Dockerfile.  *family*
    (``"deb"`` or ``"rpm"``) selects the package-manager branch of the
    template; ``None`` resolves it via :func:`detect_family`.
    """
    base_image = _normalize_base_image(base_image)
    fam = detect_family(base_image, override=family)
    return _render_template(
        "l0.dev.Dockerfile.template",
        {"BASE_IMAGE": base_image, "family": fam},
    )


def render_l1(
    l0_image: str,
    *,
    family: str,
    agents: tuple[str, ...] | str = "all",
    cache_bust: str = "0",
) -> str:
    """Render the L1 (agent CLI) Dockerfile for the given agent selection.

    *l0_image* is the tag of the L0 image to build on top of.  *family*
    (``"deb"`` or ``"rpm"``) selects the package-manager branch and is
    required — there is no L0 reference to detect from at this point, so
    callers must supply the value resolved at the L0 level (typically via
    :func:`detect_family`).  Each roster install snippet is itself rendered
    as a Jinja template with ``family`` in scope, so snippets can carry
    ``{% if family == "deb" %}…{% else %}…{% endif %}`` branches for
    package-manager-specific commands.  *agents* is a tuple of
    already-resolved roster names (or the literal string ``"all"``); the
    template loops over them and emits each one's install snippets.
    *cache_bust* invalidates the per-agent install layers when changed
    (typically set to a Unix timestamp).
    """
    from terok_executor.roster.loader import get_roster

    roster = get_roster()
    selected = roster.resolve_selection(agents)
    installs = roster.installs

    root_snippets = [
        _render_snippet(installs[n].run_as_root, family) for n in selected if installs[n].run_as_root
    ]
    dev_snippets = [
        _render_snippet(installs[n].run_as_dev, family) for n in selected if installs[n].run_as_dev
    ]

    return _render_template(
        "l1.agent-cli.Dockerfile.template",
        {
            "BASE_IMAGE": l0_image,
            "AGENT_CACHE_BUST": cache_bust,
            "family": family,
            "install_root_snippets": root_snippets,
            "install_dev_snippets": dev_snippets,
            "installed_agents_csv": ",".join(selected),
            "agents_label": AGENTS_LABEL,
            "installed_env_path": INSTALLED_ENV_PATH,
        },
    )


def render_l1_sidecar(
    l0_image: str,
    *,
    family: str,
    tool_name: str = "coderabbit",
    cache_bust: str = "0",
) -> str:
    """Render the L1 sidecar (tool-only) Dockerfile.

    The sidecar image is built FROM L0 (not L1) and installs a single
    tool binary — no agent CLIs, no LLMs.  *family* (required) selects
    the package-manager branch; *tool_name* selects which tool install
    block to activate via Jinja2 conditional.
    """
    return _render_template(
        "l1.sidecar.Dockerfile.template",
        {
            "BASE_IMAGE": l0_image,
            "TOOL_CACHE_BUST": cache_bust,
            "tool_name": tool_name,
            "family": family,
        },
    )


# ── Resource staging ──


def stage_scripts(dest: Path) -> None:
    """Stage container helper scripts into *dest*.

    Copies all files from ``terok_executor/resources/scripts/`` into the given
    directory, replacing any existing contents.  Python bytecode caches and
    ``__init__.py`` markers are excluded.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_executor", "resources/scripts", dest)
    _clean_packaging_artifacts(dest)


def stage_toad_agents(dest: Path) -> None:
    """Stage Toad ACP agent TOML definitions into *dest*.

    These describe OpenCode-based agents (Blablador, KISSKI, etc.) that are
    injected into Toad's bundled agent directory at container build time.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_executor", "resources/toad-agents", dest)
    _clean_packaging_artifacts(dest)


def stage_help_fragments(dest: Path, agents: tuple[str, ...]) -> None:
    """Render per-section ``hilfe`` help fragments into *dest*.

    Writes one file per section (currently ``agent`` and ``dev_tool``)
    containing the labels of the selected agents that have a ``help:``
    section, with backslash escapes (``\\033[...]``) interpreted into
    real ANSI sequences so that ``hilfe`` only needs to ``cat`` them.
    Empty sections are omitted entirely; ``hilfe`` skips missing files.
    """
    from terok_executor.roster.loader import get_roster

    roster = get_roster()
    helps = roster.helps

    by_section: dict[str, list[str]] = {}
    for name in agents:
        spec = helps.get(name)
        if spec is None or not spec.label:
            continue
        by_section.setdefault(spec.section, []).append(spec.label)

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for section, lines in by_section.items():
        decoded = "".join(bytes(line, "utf-8").decode("unicode_escape") + "\n" for line in lines)
        (dest / _HELP_SECTION_FILES[section]).write_text(decoded, encoding="utf-8")


def stage_tmux_config(dest: Path) -> None:
    """Stage the container tmux configuration into *dest*.

    Copies ``container-tmux.conf`` — the green-status-bar config that
    distinguishes container tmux sessions from host tmux.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_executor", "resources/tmux", dest)
    _clean_packaging_artifacts(dest)


# ── Image naming ──


def l0_image_tag(base_image: str) -> str:
    """Return the L0 base dev image tag for *base_image*."""
    return f"terok-l0:{_base_tag(base_image)}"


def l1_image_tag(base_image: str, agents: tuple[str, ...] | None = None) -> str:
    """Return the L1 agent CLI image tag for *base_image* and a selection.

    When *agents* is ``None``, returns the unsuffixed alias
    (e.g. ``terok-l1-cli:ubuntu-24.04``) — the moving handle that always
    points at the most recent build.  When *agents* is a tuple of names,
    appends a sorted ``+a+b+c`` suffix so multiple selections can coexist
    in the local image store and stay individually addressable.
    """
    base = f"terok-l1-cli:{_base_tag(base_image)}"
    if agents is None:
        return base
    suffix = "+".join(sorted(agents)) if agents else "empty"
    return f"{base}+{suffix}"


def l1_sidecar_image_tag(base_image: str) -> str:
    """Return the L1 sidecar (tool-only) image tag for *base_image*."""
    return f"terok-l1-sidecar:{_base_tag(base_image)}"


# ── Private helpers ──


def _validate_build_dir(build_dir: Path | None) -> None:
    """Reject *build_dir* if it is a file or a non-empty directory."""
    if build_dir is None:
        return
    if build_dir.is_file():
        raise ValueError(f"build_dir is a file, not a directory: {build_dir}")
    if build_dir.exists() and any(build_dir.iterdir()):
        raise ValueError(f"build_dir must be empty or absent: {build_dir}")


def _normalize_base_image(base_image: str | None) -> str:
    """Normalize a base image string, falling back to the default."""
    return (base_image or "").strip() or DEFAULT_BASE_IMAGE


def _split_image_ref(ref: str) -> tuple[str, str]:
    """Split an OCI image reference into ``(name_without_tag, tag)``.

    Strips an optional ``@digest`` suffix first, then peels off the
    trailing ``:tag`` only when the last ``:`` lies after the last ``/``
    — so ``localhost:5000/ubuntu:24.04`` keeps the registry port intact
    in *name* and yields ``"24.04"`` as *tag*.  Refs without a tag
    return an empty string for *tag*.
    """
    name = ref.split("@", 1)[0]  # drop digest
    if name.rfind(":") > name.rfind("/"):
        name, _, tag = name.rpartition(":")
        return name, tag
    return name, ""


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


def _render_template(template_name: str, variables: dict[str, str]) -> str:
    """Render a Jinja2 Dockerfile template from package resources.

    Templates live in ``resources/templates/``.  The L0/L1 templates
    branch on a ``family`` variable (``deb``/``rpm``); the L1 template
    also iterates over pre-rendered per-agent install snippets.
    """
    from jinja2 import BaseLoader, Environment

    raw = (resources.files("terok_executor") / "resources" / "templates" / template_name).read_text(
        encoding="utf-8"
    )
    env = Environment(  # nosec B701 — Dockerfile output, not HTML
        loader=BaseLoader(), keep_trailing_newline=True, autoescape=False
    )
    tmpl = env.from_string(raw)
    return tmpl.render(**variables)


def _render_snippet(snippet: str, family: str) -> str:
    """Render a per-agent install snippet as a Jinja string with *family* in scope.

    Roster install snippets are stored as plain Dockerfile text in each
    agent YAML.  They may contain ``{% if family == "deb" %}…{% else %}…
    {% endif %}`` branches so that package-manager-specific commands
    (apt vs dnf, .deb vs .rpm download) live alongside the rest of the
    agent's install steps.  Non-family-aware snippets pass through
    unchanged.
    """
    from jinja2 import BaseLoader, Environment

    env = Environment(  # nosec B701 — Dockerfile output, not HTML
        loader=BaseLoader(), keep_trailing_newline=True, autoescape=False
    )
    return env.from_string(snippet).render(family=family)


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
