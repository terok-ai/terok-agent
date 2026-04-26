# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-agent model roster cache for the ACP host-proxy.

Probing an agent (initialize + session/new + read configOptions) is
expensive — hundreds of milliseconds — and the result is stable for the
lifetime of an authenticated session.  The cache is keyed
``(image_id, auth_identity, agent_id)``: same image, same auth, same agent
⇒ same model list.

The cache is populated lazily on the first ``session/new`` after a new
auth, and never re-probed mid-session.  ``invalidate_auth`` lets workflows
flush an entire identity's worth of entries when credentials change (today
auth is global so this is rarely useful; the hook exists for future
per-project auth).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheKey:
    """Composite key for one agent's roster within one auth scope.

    ``auth_identity`` is the constant ``"global"`` today (terok auth is
    process-wide); the field exists from day one so per-project auth can
    slot in without a key-schema migration.
    """

    image_id: str
    auth_identity: str
    agent_id: str


class AgentRosterCache:
    """Thread-safe map from :class:`CacheKey` to a tuple of model ids.

    Models are stored as a tuple so cache entries are immutable once
    inserted — callers can return them directly without defensive copying.
    Empty tuples are valid and signal "probe ran but yielded nothing"
    (saved to avoid hammering a misconfigured agent on every session).
    """

    def __init__(self) -> None:
        self._models: dict[CacheKey, tuple[str, ...]] = {}
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> tuple[str, ...] | None:
        """Return cached models for *key*, or ``None`` if not yet probed."""
        with self._lock:
            return self._models.get(key)

    def put(self, key: CacheKey, models: tuple[str, ...]) -> None:
        """Store *models* under *key*, replacing any existing entry."""
        with self._lock:
            self._models[key] = models

    def invalidate_auth(self, auth_identity: str) -> None:
        """Drop every entry tied to *auth_identity*.

        Used when credentials for an identity rotate — the next
        ``session/new`` re-probes affected agents.
        """
        with self._lock:
            self._models = {
                k: v for k, v in self._models.items() if k.auth_identity != auth_identity
            }

    def __len__(self) -> int:
        """Return the number of cached entries (for tests / introspection)."""
        with self._lock:
            return len(self._models)


# Module-level singleton: most callers get this implicitly via
# :class:`ACPRoster`'s default.  Tests inject a fresh
# :class:`AgentRosterCache` via the constructor.
GLOBAL_CACHE = AgentRosterCache()
