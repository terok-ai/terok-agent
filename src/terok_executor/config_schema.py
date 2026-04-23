# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schema for the executor-owned slice of the shared ``config.yml``.

terok-executor owns one top-level section in the shared config:
``image:`` (base image, agent roster, Dockerfile snippets).  This
module defines that section's strict schema and composes it with
sandbox's :class:`~terok_sandbox.config_schema.SandboxConfigView`.

Standalone executor consumers (``terok-executor run``) validate the
file against :class:`ExecutorConfigView`.  Sandbox-owned and
executor-owned sections are strict on their own keys; unknown
top-level sections (terok's ``tui:``, ``logs:`` …) pass through
silently because the view is itself ``extra="allow"``.

Higher layers (terok) inherit from :class:`ExecutorConfigView` and
flip the top level to ``extra="forbid"`` because they know the full
ecosystem set.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from terok_sandbox import SandboxConfigView

# ── Owned sub-section ─────────────────────────────────────────────────


class RawImageSection(BaseModel):
    """The ``image:`` section — base image, agent roster, Dockerfile snippets.

    Strict on its own keys (``extra="forbid"``).  Same shape used in both
    the global ``config.yml`` (defaults across projects) and per-project
    ``project.yml`` (project overrides).
    """

    model_config = ConfigDict(extra="forbid")

    base_image: str = Field(default="ubuntu:24.04", description="Base container image for builds")
    family: Literal["deb", "rpm"] | None = Field(
        default=None,
        description=(
            "Package family for the L0/L1 build (``deb`` or ``rpm``). "
            "Leave unset to auto-detect from *base_image*; set explicitly "
            "when the image is outside the known allowlist."
        ),
    )
    agents: str | None = Field(
        default=None,
        description=(
            'Comma-separated roster entries to install in L1, or "all". '
            "Inherits from the global config when unset."
        ),
    )
    user_snippet_inline: str | None = Field(
        default=None, description="Inline Dockerfile snippet injected into the project image"
    )
    user_snippet_file: str | None = Field(
        default=None, description="Path to a file containing a Dockerfile snippet"
    )


# ── Executor's view of the global config ──────────────────────────────


class ExecutorConfigView(SandboxConfigView):
    """The slice of ``config.yml`` executor owns + sandbox owns (transitively).

    Inherits all eight sandbox-owned sections from
    :class:`~terok_sandbox.config_schema.SandboxConfigView` and adds
    the executor-owned ``image:`` section.  ``extra="allow"`` keeps the
    view tolerant of foreign top-level keys (terok's ``tui:`` /
    ``logs:`` / ``tasks:`` / ``git:`` / ``hooks:``) — standalone
    ``terok-executor run`` flows don't crash on a complete ecosystem
    config, no need to vendor a list of terok's section names here.

    terok's ``RawGlobalConfig`` inherits from this class and flips
    back to ``extra="forbid"``: the topmost layer knows every section,
    so a typo at the top level is caught there.
    """

    model_config = ConfigDict(extra="allow")

    image: RawImageSection = Field(default_factory=RawImageSection)


__all__ = [
    "ExecutorConfigView",
    "RawImageSection",
]
