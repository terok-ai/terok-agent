# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Minimal YAML loader — vendored from terok.lib.util.yaml.

Only the ``load`` function is needed by terok-agent (for frontmatter parsing
and config stack loading).  Round-trip dump is not required here.
"""

from __future__ import annotations

from typing import Any

from ruamel.yaml import YAML

__all__ = ["load"]

_yaml = YAML(typ="rt")
_yaml.preserve_quotes = True


def load(text: str) -> Any:
    """Round-trip load from a YAML string, preserving comments and order."""
    return _yaml.load(text)
