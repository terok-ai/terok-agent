# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the executor-owned slice of ``config.yml``.

Three properties get pinned:

1. The ``image:`` section is strict on its own keys.
2. ``ExecutorConfigView`` inherits and re-exposes sandbox's section
   strictness (delegated via the parent class).
3. The view tolerates foreign top-level keys (terok's ``tui:``,
   future-package keys).  Standalone ``terok-executor run`` doesn't
   crash on a complete ecosystem config.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from terok_executor.config_schema import ExecutorConfigView, RawImageSection

# ── Owned section strictness ──────────────────────────────────────────


def test_image_section_rejects_unknown_key() -> None:
    """``extra="forbid"`` keeps typos like ``base_iamge`` out of the image section."""
    with pytest.raises(ValidationError, match="base_iamge"):
        RawImageSection.model_validate({"base_iamge": "ubuntu:24.04"})


def test_image_section_rejects_invalid_family_enum() -> None:
    """``family`` is a ``Literal["deb", "rpm"]`` — anything else is rejected."""
    with pytest.raises(ValidationError):
        RawImageSection.model_validate({"family": "alpine"})


def test_image_section_accepts_known_keys() -> None:
    """Documented keys validate cleanly with the right types."""
    section = RawImageSection.model_validate(
        {
            "base_image": "fedora:43",
            "family": "rpm",
            "agents": "claude,codex",
            "user_snippet_inline": "RUN apt update",
        }
    )
    assert section.base_image == "fedora:43"
    assert section.family == "rpm"


# ── Composed view: inherits sandbox strictness, tolerant of foreign keys ──


def test_view_inherits_sandbox_owned_strictness() -> None:
    """Sandbox's ``paths.rooot`` typo still errors when validated through ExecutorConfigView."""
    with pytest.raises(ValidationError, match="rooot"):
        ExecutorConfigView.model_validate({"paths": {"rooot": "/tmp"}})


def test_view_validates_executor_owned_strictness() -> None:
    """Executor's own ``image.foo`` typo errors at the executor layer, not later."""
    with pytest.raises(ValidationError, match="bogus"):
        ExecutorConfigView.model_validate({"image": {"bogus": "value"}})


def test_view_tolerates_terok_owned_top_level_sections() -> None:
    """Foreign top-level keys (``tui:``, ``logs:``, future packages) pass through silently."""
    raw = {
        "image": {"base_image": "ubuntu:24.04"},
        "paths": {"root": "/v"},
        "tui": {"default_tmux": True},  # terok-owned
        "future_package": {"foo": 1},  # not in v0
    }
    view = ExecutorConfigView.model_validate(raw)
    assert view.image.base_image == "ubuntu:24.04"
    assert view.paths.root == "/v"


def test_view_default_construction_is_empty() -> None:
    """An empty ``config.yml`` validates with safe defaults across both layers."""
    view = ExecutorConfigView.model_validate({})
    assert view.image.base_image == "ubuntu:24.04"
    assert view.paths.root is None
    assert view.shield.audit is True
