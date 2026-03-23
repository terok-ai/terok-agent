# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Provider-aware config resolution utility.

The ``resolve_provider_value`` function extracts values from a merged
agent-config dict, supporting flat values and per-provider dicts with
``_default`` fallback.  Pure function — no I/O, no terok dependencies.

The full config stack composition (``build_agent_config_stack``,
``resolve_agent_config``) remains in terok, which owns the global/project/
preset layer semantics.
"""

from __future__ import annotations

from typing import Any


def resolve_provider_value(
    key: str,
    config: dict[str, Any],
    provider_name: str,
) -> Any | None:
    """Extract a provider-aware config value.

    Supports two forms:

    * **Flat value** — ``model: opus`` → same for all providers.
    * **Per-provider dict** — ``model: {claude: opus, codex: o3, _default: fast}``
      → looks up *provider_name*, falls back to ``_default``, then ``None``.

    Returns ``None`` when the key is absent or has no match for the provider.

    **Null override behaviour**: when a per-provider dict maps a provider to
    ``null`` (Python ``None``), that ``None`` is treated as "no value" and the
    resolver falls back to ``_default``.  This is intentional — it allows a
    lower-priority config layer to set a provider-specific value that a
    higher-priority layer can effectively *unset* by mapping it to ``null``,
    letting the ``_default`` (or ``None``) bubble up instead.
    """
    val = config.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        provider_val = val.get(provider_name)
        if provider_val is not None:
            return provider_val
        return val.get("_default")
    return val
