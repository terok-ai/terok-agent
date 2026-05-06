# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary and schema-walker for ACP's ``category: "model"`` configOption.

Two consumers walk the same nested model-options structure: the probe
reads it (``configOptions[category=model] → choices → ids``) to learn
what an agent can serve; the proxy mutates it on the way out so the
client sees the namespaced ``agent:model`` ids it expects.  Both sit
on top of :func:`iter_model_choice_dicts`, which tolerates the
in-flight ACP schema variants in one place.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

MODEL_OPTION_CATEGORY = "model"
"""ACP semantic category for the model selector configOption."""

MODEL_NAMESPACE_SEP = ":"
"""Separator between agent and model in the namespaced id (e.g.
``claude:opus-4.6``).  Chosen over ``/`` to avoid collisions with
OpenRouter-style ids like ``anthropic/claude-opus-4``."""


def iter_model_choice_dicts(result: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield each dict-shaped choice from ``result.configOptions[category=model]``.

    Tolerant of the in-flight ACP schema variants we observed during
    design — the model selector can nest its choices under ``select``
    or directly, and use ``options`` / ``values`` / ``choices`` as the
    container key.  Skips non-dict entries; consumers that need to
    mutate in place can rely on dict-only semantics.
    """
    options = result.get("configOptions") or []
    for opt in options:
        if not isinstance(opt, dict) or opt.get("category") != MODEL_OPTION_CATEGORY:
            continue
        select = opt.get("select")
        nested = select if isinstance(select, dict) else opt
        if not isinstance(nested, dict):
            continue
        for key in ("options", "values", "choices"):
            choices = nested.get(key)
            if not isinstance(choices, list):
                continue
            for entry in choices:
                if isinstance(entry, dict):
                    yield entry
