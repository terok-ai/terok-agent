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

    Pre-populates the ``acp_capable_agents`` cached_property from the
    configured agents list so tests don't have to mock the in-container
    ``command -v`` check that the production roster runs once at first
    use.  Real deployments need that check (the image label includes
    tools like gh/glab/sonar that don't ship an ACP wrapper); unit
    tests just want to exercise the cache + probe pipeline against
    every configured agent.
    """
    roster = ACPRoster(
        container_name="c1",
        image_id=image_id,
        sandbox=sandbox,
        cache=cache if cache is not None else AgentRosterCache(),
    )
    # ``cached_property`` stores in ``__dict__`` on first access; pre-
    # filling it avoids the in-container exec the production check makes.
    roster.__dict__["acp_capable_agents"] = roster.configured_agents
    return roster


class TestAcpCapableAgents:
    """Wrapper-existence filter on top of ``configured_agents``."""

    def _make_runtime(
        self, *, agents_csv: str, present_wrappers: tuple[str, ...]
    ) -> tuple[NullRuntime, Sandbox]:
        """Build a NullRuntime that reports *present_wrappers* as installed.

        The roster runs ``bash -c '...command -v terok-X-acp ... && echo X...'``
        once to build the capable-agents whitelist; the registered
        ``ExecResult`` returns the agents we want to keep.
        """
        from terok_sandbox.runtime.protocol import ExecResult

        rt = NullRuntime()
        rt.add_image(
            "img-test",
            repository="terok-l1",
            tag="test",
            labels={"ai.terok.agents": agents_csv},
        )
        # Build the same script the roster builds, so the key matches.
        agents = [a.strip() for a in agents_csv.split(",") if a.strip()]
        script = "; ".join(
            f"command -v 'terok-{a}-acp' >/dev/null 2>&1 && echo {a}" for a in agents
        )
        rt.set_exec_result(
            "c1",
            ("bash", "-c", script),
            ExecResult(exit_code=0, stdout="\n".join(present_wrappers), stderr=""),
        )
        return rt, Sandbox(config=SandboxConfig(), runtime=rt)

    def test_filters_to_agents_with_a_wrapper(self) -> None:
        """Agents whose wrapper is missing in the container are dropped."""
        _rt, sandbox = self._make_runtime(
            agents_csv="claude,gh,sonar,opencode",
            present_wrappers=("claude", "opencode"),
        )
        # Don't go through ``_make_roster`` — we want the real
        # ``acp_capable_agents`` to run, not the test pre-fill.
        roster = ACPRoster(
            container_name="c1",
            image_id="img-test",
            sandbox=sandbox,
            cache=AgentRosterCache(),
        )
        assert roster.acp_capable_agents == ("claude", "opencode")

    def test_falls_back_to_full_list_on_check_failure(self) -> None:
        """If the in-container check raises, all configured agents are kept.

        Conservative fallback: better to probe everything (paying the
        timeout cost once per agent) than silently hide the picker.
        """

        class _ExecRaises(NullRuntime):
            def exec(self, container, cmd, *, timeout=None):  # type: ignore[no-untyped-def]
                """Raise so the roster falls back to the unfiltered list."""
                raise RuntimeError("simulated container exec failure")

        rt = _ExecRaises()
        rt.add_image(
            "img-test",
            repository="terok-l1",
            tag="test",
            labels={"ai.terok.agents": "claude,gh"},
        )
        sandbox = Sandbox(config=SandboxConfig(), runtime=rt)
        roster = ACPRoster(
            container_name="c1",
            image_id="img-test",
            sandbox=sandbox,
            cache=AgentRosterCache(),
        )
        assert roster.acp_capable_agents == ("claude", "gh")


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


class TestWarm:
    """Probe success / failure caching contract."""

    def test_failed_probe_is_not_cached(self) -> None:
        """A failing probe leaves the cache cold — the next call re-probes.

        Caching empty tuples on failure used to wedge the daemon
        empty after a single bad first-probe (cold container, slow
        Node start, OAuth refresh racing the 3-second timeout) — only
        a daemon restart could recover.  Now ``warm`` returns ``()``
        but does *not* insert; the next ``list_available_agents`` re-
        runs the probe and a now-warm container can succeed.
        """
        from terok_executor.acp.probe import ProbeError

        cache = AgentRosterCache()
        roster = _make_roster(_build_sandbox_with_image("claude"), cache=cache)
        attempts: list[str] = []

        async def _failing_probe(agent_id: str) -> tuple[str, ...]:
            attempts.append(agent_id)
            raise ProbeError(f"probe timed out for agent {agent_id!r}")

        roster._probe = _failing_probe  # type: ignore[method-assign]
        result = asyncio.run(roster.warm("claude"))
        assert result == ()
        # Cache stays cold — no entry under the failed agent's key.
        assert (
            cache.get(CacheKey(image_id="img-test", auth_identity="global", agent_id="claude"))
            is None
        )
        # And so a re-call goes through `_probe` a second time.
        asyncio.run(roster.warm("claude"))
        assert attempts == ["claude", "claude"]

    def test_successful_probe_caches_result(self) -> None:
        """A successful probe stores the model tuple for the daemon's lifetime."""
        cache = AgentRosterCache()
        roster = _make_roster(_build_sandbox_with_image("claude"), cache=cache)
        attempts: list[str] = []

        async def _succeeding_probe(agent_id: str) -> tuple[str, ...]:
            attempts.append(agent_id)
            return ("opus-4.6",)

        roster._probe = _succeeding_probe  # type: ignore[method-assign]
        first = asyncio.run(roster.warm("claude"))
        # Second list_available_agents call should hit the cache, not re-probe.
        second = asyncio.run(roster.list_available_agents())
        assert first == ("opus-4.6",)
        assert second == ["claude:opus-4.6"]
        assert attempts == ["claude"]
