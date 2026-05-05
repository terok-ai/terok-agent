# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`ACPRoster` — per-task multi-agent aggregation."""

from __future__ import annotations

import asyncio

from terok_sandbox import NullRuntime, Sandbox, SandboxConfig

from terok_executor.acp.cache import AgentRosterCache, CacheKey
from terok_executor.acp.roster import ACPRoster


def _build_sandbox_with_image(agents_csv: str) -> Sandbox:
    """Construct a NullRuntime-backed sandbox advertising *agents_csv*."""
    rt = NullRuntime()
    rt.add_image(
        "img-test",
        repository="terok-l1",
        tag="test",
        labels={"ai.terok.agents": agents_csv},
    )
    return Sandbox(config=SandboxConfig(), runtime=rt)


def _make_roster(
    sandbox: Sandbox,
    *,
    cache: AgentRosterCache | None = None,
    image_id: str = "img-test",
) -> ACPRoster:
    """Build an :class:`ACPRoster` with a fresh cache by default.

    No auth-source injection: the roster doesn't gate probing on
    credentials anymore — successful probes are the source of truth.
    """
    return ACPRoster(
        container_name="c1",
        image_id=image_id,
        sandbox=sandbox,
        cache=cache if cache is not None else AgentRosterCache(),
    )


class TestConfiguredAgents:
    """The image label drives ``configured_agents``."""

    def test_parses_csv_label(self) -> None:
        """Comma-separated values become an ordered tuple."""
        roster = _make_roster(_build_sandbox_with_image("claude,codex,vibe"))
        assert roster.configured_agents == ("claude", "codex", "vibe")

    def test_strips_whitespace_and_drops_empties(self) -> None:
        """Whitespace around commas is tolerated; empty entries are dropped."""
        roster = _make_roster(_build_sandbox_with_image(" claude , , codex "))
        assert roster.configured_agents == ("claude", "codex")

    def test_missing_label_yields_empty(self) -> None:
        """Image without the agents label exposes no configured agents."""
        rt = NullRuntime()
        rt.add_image("img-bare", labels={})
        sandbox = Sandbox(config=SandboxConfig(), runtime=rt)
        roster = _make_roster(sandbox, image_id="img-bare")
        assert roster.configured_agents == ()


class TestListAvailableAgents:
    """Cache-driven walk over every configured agent."""

    def test_returns_namespaced_models_from_cache(self) -> None:
        """Cached agents emit ``agent:model`` ids in image-label order."""
        cache = AgentRosterCache()
        cache.put(
            CacheKey(image_id="img-test", auth_identity="global", agent_id="claude"),
            ("opus-4.6", "haiku-4.5"),
        )
        cache.put(
            CacheKey(image_id="img-test", auth_identity="global", agent_id="codex"),
            ("gpt-5.5",),
        )
        roster = _make_roster(_build_sandbox_with_image("claude,codex"), cache=cache)
        assert asyncio.run(roster.list_available_agents()) == [
            "claude:opus-4.6",
            "claude:haiku-4.5",
            "codex:gpt-5.5",
        ]

    def test_empty_probe_result_is_silently_skipped(self) -> None:
        """An agent whose cache entry is empty contributes no rows.

        The cache stores empty tuples for failed/unauthed probes so
        we don't re-probe; the output skips them so an unauthed agent
        doesn't show up as a no-models row in the picker.
        """
        cache = AgentRosterCache()
        cache.put(
            CacheKey(image_id="img-test", auth_identity="global", agent_id="claude"),
            ("opus-4.6",),
        )
        cache.put(
            CacheKey(image_id="img-test", auth_identity="global", agent_id="codex"),
            (),
        )
        roster = _make_roster(_build_sandbox_with_image("claude,codex"), cache=cache)
        assert asyncio.run(roster.list_available_agents()) == ["claude:opus-4.6"]

    def test_cache_miss_triggers_warm(self) -> None:
        """A cold-cache agent calls warm() once, then its cache entry feeds the output."""
        cache = AgentRosterCache()
        roster = _make_roster(_build_sandbox_with_image("claude"), cache=cache)
        warm_calls: list[str] = []

        async def _fake_warm(agent_id: str) -> tuple[str, ...]:
            warm_calls.append(agent_id)
            cache.put(
                CacheKey(image_id="img-test", auth_identity="global", agent_id=agent_id),
                ("warmed-model",),
            )
            return ("warmed-model",)

        roster.warm = _fake_warm  # type: ignore[method-assign]
        result = asyncio.run(roster.list_available_agents())
        assert result == ["claude:warmed-model"]
        assert warm_calls == ["claude"]

    def test_cold_probes_run_in_parallel(self) -> None:
        """Multiple cold-cache agents probe concurrently, not sequentially."""
        cache = AgentRosterCache()
        roster = _make_roster(_build_sandbox_with_image("claude,codex,vibe"), cache=cache)
        active: set[str] = set()
        max_active = 0

        async def _slow_warm(agent_id: str) -> tuple[str, ...]:
            nonlocal max_active
            active.add(agent_id)
            max_active = max(max_active, len(active))
            await asyncio.sleep(0.05)
            active.discard(agent_id)
            cache.put(
                CacheKey(image_id="img-test", auth_identity="global", agent_id=agent_id),
                (f"{agent_id}-m",),
            )
            return (f"{agent_id}-m",)

        roster.warm = _slow_warm  # type: ignore[method-assign]
        asyncio.run(roster.list_available_agents())
        # Three agents probing in parallel should hit the inner sleep
        # at the same time; sequential would top out at 1.
        assert max_active >= 2

    def test_image_with_no_agents_yields_empty_list(self) -> None:
        """An image without the agents label surfaces no models."""
        rt = NullRuntime()
        rt.add_image("img-bare", labels={})
        sandbox = Sandbox(config=SandboxConfig(), runtime=rt)
        roster = _make_roster(sandbox, image_id="img-bare")
        assert asyncio.run(roster.list_available_agents()) == []
