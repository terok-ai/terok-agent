# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`ACPRoster` — per-task multi-agent aggregation."""

from __future__ import annotations

from unittest import mock

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


class TestConfiguredAgents:
    """The image label drives ``configured_agents``."""

    def test_parses_csv_label(self) -> None:
        """Comma-separated values become an ordered tuple."""
        sandbox = _build_sandbox_with_image("claude,codex,vibe")
        roster = ACPRoster(
            task_id="t1",
            container_name="c1",
            image_id="img-test",
            sandbox=sandbox,
            cache=AgentRosterCache(),
        )
        assert roster.configured_agents == ("claude", "codex", "vibe")

    def test_strips_whitespace_and_drops_empties(self) -> None:
        """Whitespace around commas is tolerated; empty entries are dropped."""
        sandbox = _build_sandbox_with_image(" claude , , codex ")
        roster = ACPRoster(
            task_id="t1",
            container_name="c1",
            image_id="img-test",
            sandbox=sandbox,
            cache=AgentRosterCache(),
        )
        assert roster.configured_agents == ("claude", "codex")

    def test_missing_label_yields_empty(self) -> None:
        """Image without the agents label exposes no configured agents."""
        rt = NullRuntime()
        rt.add_image("img-bare", labels={})
        sandbox = Sandbox(config=SandboxConfig(), runtime=rt)
        roster = ACPRoster(
            task_id="t1",
            container_name="c1",
            image_id="img-bare",
            sandbox=sandbox,
            cache=AgentRosterCache(),
        )
        assert roster.configured_agents == ()


class TestListAvailableAgents:
    """Live walk: configured ∩ authenticated, then namespace from cache."""

    def test_returns_namespaced_models_for_authed_agents(self) -> None:
        """Authenticated agents emit ``agent:model`` ids from the cache."""
        sandbox = _build_sandbox_with_image("claude,codex")
        cache = AgentRosterCache()
        cache.put(
            CacheKey(image_id="img-test", auth_identity="global", agent_id="claude"),
            ("opus-4.6", "haiku-4.5"),
        )
        cache.put(
            CacheKey(image_id="img-test", auth_identity="global", agent_id="codex"),
            ("gpt-5.5",),
        )
        roster = ACPRoster(
            task_id="t1",
            container_name="c1",
            image_id="img-test",
            sandbox=sandbox,
            cache=cache,
        )
        with mock.patch(
            "terok_executor.acp.roster.list_authenticated_agents",
            return_value=["claude", "codex"],
        ):
            assert roster.list_available_agents() == [
                "claude:opus-4.6",
                "claude:haiku-4.5",
                "codex:gpt-5.5",
            ]

    def test_filters_unauthenticated_agents(self) -> None:
        """Configured but un-authed agents are dropped — even if cached."""
        sandbox = _build_sandbox_with_image("claude,codex")
        cache = AgentRosterCache()
        cache.put(
            CacheKey(image_id="img-test", auth_identity="global", agent_id="claude"),
            ("opus-4.6",),
        )
        cache.put(
            CacheKey(image_id="img-test", auth_identity="global", agent_id="codex"),
            ("gpt-5.5",),
        )
        roster = ACPRoster(
            task_id="t1",
            container_name="c1",
            image_id="img-test",
            sandbox=sandbox,
            cache=cache,
        )
        with mock.patch(
            "terok_executor.acp.roster.list_authenticated_agents",
            return_value=["claude"],
        ):
            # Only Claude is authed → only claude:* surfaces.
            assert roster.list_available_agents() == ["claude:opus-4.6"]

    def test_cache_miss_triggers_warm(self) -> None:
        """An authed agent without cache entries calls warm() once."""
        sandbox = _build_sandbox_with_image("claude")
        cache = AgentRosterCache()
        roster = ACPRoster(
            task_id="t1",
            container_name="c1",
            image_id="img-test",
            sandbox=sandbox,
            cache=cache,
        )
        warm_calls: list[str] = []

        def _fake_warm(agent_id: str) -> tuple[str, ...]:
            warm_calls.append(agent_id)
            cache.put(
                CacheKey(image_id="img-test", auth_identity="global", agent_id=agent_id),
                ("warmed-model",),
            )
            return ("warmed-model",)

        with (
            mock.patch(
                "terok_executor.acp.roster.list_authenticated_agents",
                return_value=["claude"],
            ),
            mock.patch.object(roster, "warm", side_effect=_fake_warm),
        ):
            result = roster.list_available_agents()
        assert result == ["claude:warmed-model"]
        assert warm_calls == ["claude"]

    def test_no_authed_agents_yields_empty_list(self) -> None:
        """Task with no authed agents surfaces no models — caller turns
        this into the ``unsupported`` endpoint status."""
        sandbox = _build_sandbox_with_image("claude")
        roster = ACPRoster(
            task_id="t1",
            container_name="c1",
            image_id="img-test",
            sandbox=sandbox,
            cache=AgentRosterCache(),
        )
        with mock.patch(
            "terok_executor.acp.roster.list_authenticated_agents",
            return_value=[],
        ):
            assert roster.list_available_agents() == []


class TestAgentMatrix:
    """The (configured, authenticated) snapshot used by host-side discovery."""

    def test_matrix_intersects_configured_and_authed(self) -> None:
        """Authenticated agents not configured in the image are excluded."""
        sandbox = _build_sandbox_with_image("claude,codex")
        roster = ACPRoster(
            task_id="t1",
            container_name="c1",
            image_id="img-test",
            sandbox=sandbox,
            cache=AgentRosterCache(),
        )
        with mock.patch(
            "terok_executor.acp.roster.list_authenticated_agents",
            return_value=["claude", "vibe"],  # vibe authed but not in image
        ):
            matrix = roster.agent_matrix()
        assert matrix.configured == ("claude", "codex")
        assert matrix.authenticated == frozenset({"claude"})
