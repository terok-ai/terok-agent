# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`AgentRosterCache` — the per-agent model roster cache."""

from __future__ import annotations

import threading

from terok_executor.acp.cache import AgentRosterCache, CacheKey


class TestAgentRosterCache:
    """Get/put/invalidate semantics."""

    def test_miss_returns_none(self) -> None:
        """Reading a key that hasn't been put yields ``None``."""
        cache = AgentRosterCache()
        key = CacheKey(image_id="img", auth_identity="global", agent_id="claude")
        assert cache.get(key) is None

    def test_put_then_get_returns_models(self) -> None:
        """Putting a tuple makes it visible on the next get."""
        cache = AgentRosterCache()
        key = CacheKey(image_id="img", auth_identity="global", agent_id="claude")
        cache.put(key, ("opus-4.6", "haiku-4.5"))
        assert cache.get(key) == ("opus-4.6", "haiku-4.5")

    def test_empty_tuple_is_a_valid_entry(self) -> None:
        """Empty tuples are stored — distinguishes "probed, no models" from "not probed"."""
        cache = AgentRosterCache()
        key = CacheKey(image_id="img", auth_identity="global", agent_id="codex")
        cache.put(key, ())
        assert cache.get(key) == ()

    def test_put_replaces_existing_entry(self) -> None:
        """Subsequent puts overwrite earlier ones."""
        cache = AgentRosterCache()
        key = CacheKey(image_id="img", auth_identity="global", agent_id="claude")
        cache.put(key, ("a",))
        cache.put(key, ("b", "c"))
        assert cache.get(key) == ("b", "c")

    def test_invalidate_auth_drops_only_matching_identity(self) -> None:
        """Invalidating one identity leaves the other intact."""
        cache = AgentRosterCache()
        global_key = CacheKey(image_id="img", auth_identity="global", agent_id="claude")
        project_key = CacheKey(image_id="img", auth_identity="project-x", agent_id="claude")
        cache.put(global_key, ("a",))
        cache.put(project_key, ("b",))
        cache.invalidate_auth("global")
        assert cache.get(global_key) is None
        assert cache.get(project_key) == ("b",)

    def test_keys_differ_by_image(self) -> None:
        """Different image ids yield independent cache entries."""
        cache = AgentRosterCache()
        k1 = CacheKey(image_id="img-a", auth_identity="global", agent_id="claude")
        k2 = CacheKey(image_id="img-b", auth_identity="global", agent_id="claude")
        cache.put(k1, ("models-of-a",))
        cache.put(k2, ("models-of-b",))
        assert cache.get(k1) == ("models-of-a",)
        assert cache.get(k2) == ("models-of-b",)

    def test_concurrent_put_get_is_safe(self) -> None:
        """N threads racing put/get/invalidate don't corrupt the dict."""
        cache = AgentRosterCache()
        keys = [
            CacheKey(image_id="img", auth_identity="global", agent_id=f"a{i}") for i in range(20)
        ]

        def worker(k: CacheKey) -> None:
            for _ in range(100):
                cache.put(k, ("m",))
                cache.get(k)

        threads = [threading.Thread(target=worker, args=(k,)) for k in keys]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Every key got at least one put through
        assert all(cache.get(k) == ("m",) for k in keys)

    def test_len_reflects_size(self) -> None:
        """``__len__`` returns the entry count for tests/introspection."""
        cache = AgentRosterCache()
        assert len(cache) == 0
        cache.put(
            CacheKey(image_id="img", auth_identity="global", agent_id="claude"),
            ("opus-4.6",),
        )
        assert len(cache) == 1
