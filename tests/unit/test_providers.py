# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the provider layer (``resources/providers/*.yaml``).

The keystone invariant: each [`Provider`][terok_executor.roster.types.Provider]'s
``routes.json`` projection reproduces the historical agent-``vault`` route
entry, under the clean provider name.  Locking this *before* the loader is
switched to source routes from providers keeps the byte-identical-except-keys
contract honest — and keeps the sandbox vault untouched.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from terok_executor import providers_config_dir
from terok_executor.credentials.auth import credential_provider
from terok_executor.roster import AgentRoster
from terok_executor.roster.loader import _provider_route_entry, load_roster
from terok_executor.roster.schema import RawProvider
from terok_executor.roster.types import Provider, ProviderAuth
from tests.constants import (
    CANONICAL_PROVIDER_UPSTREAM,
    CONTAINER_HOME,
    EXAMPLE_PROVIDER_UPSTREAM,
    LEGACY_PROVIDER_UPSTREAM,
)

# Agent name → the clean provider name its vault block maps to (identity for
# names that don't rename: coderabbit, openrouter, blablador, kisski).
_AGENT_TO_PROVIDER = {
    "claude": "anthropic",
    "codex": "openai",
    "vibe": "mistral",
    "gh": "github",
    "glab": "gitlab",
    "sonar": "sonarcloud",
}

_EXPECTED_PROVIDERS = {
    "anthropic",
    "openai",
    "mistral",
    "github",
    "gitlab",
    "sonarcloud",
    "coderabbit",
    "openrouter",
    "blablador",
    "kisski",
}


class TestProviderRegistry:
    """The bundled provider set and its relationship to agent vault routes."""

    def test_all_expected_providers_load(self) -> None:
        assert set(load_roster().providers) == _EXPECTED_PROVIDERS

    def test_every_vault_route_has_a_matching_provider(self) -> None:
        roster = load_roster()
        for agent_name in roster.vault_routes:
            provider_name = _AGENT_TO_PROVIDER.get(agent_name, agent_name)
            assert provider_name in roster.providers, agent_name


class TestRouteEntryEquivalence:
    """The byte-identical-except-keys invariant, checked route by route."""

    def test_projection_reproduces_current_route(self) -> None:
        roster = load_roster()
        for agent_name, route in roster.vault_routes.items():
            provider = roster.providers[_AGENT_TO_PROVIDER.get(agent_name, agent_name)]
            expected = {
                "upstream": route.upstream,
                "auth_header": route.auth_header,
                "auth_prefix": route.auth_prefix,
                "path_upstreams": route.path_upstreams or None,
                "oauth_extra_headers": route.oauth_extra_headers or None,
                "oauth_credential_headers": route.oauth_credential_headers or None,
                "oauth_refresh": route.oauth_refresh or None,
            }
            assert _provider_route_entry(provider).model_dump() == expected, agent_name


class TestWireAuth:
    """The OAuth-or-API-key header derivation that replaces ``auth_header: dynamic``."""

    def test_dual_header_modes_are_dynamic(self) -> None:
        # Anthropic: OAuth on Authorization vs API key on x-api-key → sentinel.
        header, prefix, extra = load_roster().providers["anthropic"].wire_auth()
        assert header == "dynamic"
        assert prefix == ""
        assert extra == {"anthropic-beta": "oauth-2025-04-20"}

    def test_single_mode_is_verbatim(self) -> None:
        assert load_roster().providers["mistral"].wire_auth() == ("Authorization", "Bearer ", {})

    def test_same_header_dual_mode_is_not_dynamic(self) -> None:
        provider = Provider(
            name="x",
            upstream=EXAMPLE_PROVIDER_UPSTREAM,
            oauth_auth=ProviderAuth(header="Authorization", prefix="Bearer "),
            api_key_auth=ProviderAuth(header="Authorization", prefix="Bearer "),
        )
        assert provider.wire_auth() == ("Authorization", "Bearer ", {})

    def test_no_auth_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="no auth mode"):
            Provider(name="x", upstream=EXAMPLE_PROVIDER_UPSTREAM).wire_auth()

    def test_same_header_differing_prefix_raises(self) -> None:
        # Same header but disagreeing prefixes can't both be serialised into the
        # single routes.json auth_prefix — must fail loud, not pick one silently.
        provider = Provider(
            name="x",
            upstream=EXAMPLE_PROVIDER_UPSTREAM,
            oauth_auth=ProviderAuth(header="Authorization", prefix="Bearer "),
            api_key_auth=ProviderAuth(header="Authorization", prefix="token "),
        )
        with pytest.raises(ValueError, match="different prefix"):
            provider.wire_auth()


class TestSchemaStrictness:
    """``RawProvider`` rejects typos and credential-less routes."""

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawProvider.model_validate(
                {
                    "upstream": EXAMPLE_PROVIDER_UPSTREAM,
                    "auth": {"api_key": {"header": "Authorization"}},
                    "oops": 1,
                }
            )

    def test_auth_requires_a_mode(self) -> None:
        with pytest.raises(ValidationError):
            RawProvider.model_validate({"upstream": EXAMPLE_PROVIDER_UPSTREAM, "auth": {}})

    def test_missing_upstream_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawProvider.model_validate({"auth": {"api_key": {"header": "Authorization"}}})

    def test_api_key_wire_defaults(self) -> None:
        provider = RawProvider.model_validate(
            {"upstream": EXAMPLE_PROVIDER_UPSTREAM, "auth": {"api_key": {}}}
        ).to_dataclass(name="example")

        assert provider.wire_auth() == ("Authorization", "Bearer ", {})

    @pytest.mark.parametrize("header", ["PRIVATE-TOKEN", "Authorization"])
    def test_explicit_header_preserves_legacy_empty_prefix(self, header: str) -> None:
        provider = RawProvider.model_validate(
            {
                "upstream": EXAMPLE_PROVIDER_UPSTREAM,
                "auth": {"api_key": {"header": header}},
            }
        ).to_dataclass(name="example")

        assert provider.wire_auth() == (header, "", {})

    @pytest.mark.parametrize("field", ["context", "output"])
    def test_model_limits_must_be_positive(self, field: str) -> None:
        with pytest.raises(ValidationError, match=field):
            RawProvider.model_validate(
                {
                    "upstream": EXAMPLE_PROVIDER_UPSTREAM,
                    "auth": {"api_key": {}},
                    "models": {"model-x": {"limit": {field: 0}}},
                }
            )

    def test_partial_model_limits_are_independent(self) -> None:
        provider = RawProvider.model_validate(
            {
                "upstream": EXAMPLE_PROVIDER_UPSTREAM,
                "auth": {"api_key": {}},
                "models": {
                    "context-model": {"limit": {"context": 120_000}},
                    "output-model": {
                        "name": "Output Model",
                        "limit": {"output": 8_192},
                    },
                },
            }
        ).to_dataclass(name="example")

        assert provider.models["context-model"].name == "context-model"
        assert provider.models["context-model"].context_limit == 120_000
        assert provider.models["context-model"].output_limit is None
        assert provider.models["output-model"].name == "Output Model"
        assert provider.models["output-model"].context_limit is None
        assert provider.models["output-model"].output_limit == 8_192


class TestUserProviderFiles:
    """Canonical and compatibility provider YAML layers produce usable endpoints."""

    def test_fresh_roster_reads_changes_without_replacing_shared_snapshot(
        self, tmp_path: Path
    ) -> None:
        """A fresh snapshot reads a new provider and leaves the shared snapshot intact."""
        shared = AgentRoster.shared()
        canonical = tmp_path / "providers"
        canonical.mkdir()
        (canonical / "example.yaml").write_text(
            f"upstream: {EXAMPLE_PROVIDER_UPSTREAM}\nauth:\n  api_key: {{}}\n",
            encoding="utf-8",
        )

        with (
            patch(
                "terok_executor.roster.loader._legacy_user_providers_dir",
                return_value=tmp_path / "legacy",
            ),
            patch("terok_executor.roster.loader.providers_config_dir", return_value=canonical),
        ):
            fresh = AgentRoster.load()

        assert fresh is not shared
        assert "example" in fresh.providers
        assert AgentRoster.shared() is shared
        assert "example" not in shared.providers

    @pytest.mark.parametrize("stem", ["my-provider", "my_provider", "Example"])
    def test_provider_filename_stem_must_form_a_portable_env_handle(
        self, tmp_path: Path, stem: str
    ) -> None:
        """Unsupported provider names fail before generating unusable runtime variables."""
        canonical = tmp_path / "providers"
        canonical.mkdir()
        (canonical / f"{stem}.yaml").write_text(
            f"upstream: {EXAMPLE_PROVIDER_UPSTREAM}\nauth:\n  api_key: {{}}\n",
            encoding="utf-8",
        )

        with (
            patch(
                "terok_executor.roster.loader._legacy_user_providers_dir",
                return_value=tmp_path / "legacy",
            ),
            patch("terok_executor.roster.loader.providers_config_dir", return_value=canonical),
            pytest.raises(ValueError, match=r"YAML file name.*must match '\[a-z0-9\]\+'"),
        ):
            load_roster()

    @pytest.mark.parametrize("stem", ["claude", "gh"])
    def test_provider_filename_cannot_shadow_an_agent_or_tool(
        self, tmp_path: Path, stem: str
    ) -> None:
        """A custom provider cannot replace an existing auth target."""
        canonical = tmp_path / "providers"
        canonical.mkdir()
        (canonical / f"{stem}.yaml").write_text(
            f"upstream: {EXAMPLE_PROVIDER_UPSTREAM}\nauth:\n  api_key: {{}}\n",
            encoding="utf-8",
        )

        with (
            patch(
                "terok_executor.roster.loader._legacy_user_providers_dir",
                return_value=tmp_path / "legacy",
            ),
            patch("terok_executor.roster.loader.providers_config_dir", return_value=canonical),
            pytest.raises(ValueError, match="conflicts with an existing agent or tool"),
        ):
            load_roster()

    def test_user_file_can_override_a_bundled_provider_with_the_same_agent_name(
        self, tmp_path: Path
    ) -> None:
        """A bundled provider remains configurable when its route name matches a tool."""
        canonical = tmp_path / "providers"
        canonical.mkdir()
        (canonical / "coderabbit.yaml").write_text(
            "label: Custom Code Review\n",
            encoding="utf-8",
        )

        with (
            patch(
                "terok_executor.roster.loader._legacy_user_providers_dir",
                return_value=tmp_path / "legacy",
            ),
            patch("terok_executor.roster.loader.providers_config_dir", return_value=canonical),
        ):
            roster = load_roster()

        assert roster.providers["coderabbit"].label == "Custom Code Review"

    def test_canonical_minimal_provider_is_harness_ready(self, tmp_path: Path) -> None:
        canonical = tmp_path / "providers"
        canonical.mkdir()
        (canonical / "example.yaml").write_text(
            "label: Example\n"
            f"upstream: {EXAMPLE_PROVIDER_UPSTREAM}\n"
            "auth:\n  api_key: {}\n"
            "serves:\n  openai-chat: /v1\n"
            "default_model: example-chat\n"
            "models:\n"
            "  example-chat:\n"
            "    name: Example Chat\n"
            "    limit:\n      context: 120000\n",
            encoding="utf-8",
        )

        with (
            patch(
                "terok_executor.roster.loader._legacy_user_providers_dir",
                return_value=tmp_path / "legacy",
            ),
            patch("terok_executor.roster.loader.providers_config_dir", return_value=canonical),
        ):
            roster = load_roster()

        provider = roster.providers["example"]
        assert provider.label == "Example"
        assert provider.default_model == "example-chat"
        assert provider.models["example-chat"].context_limit == 120_000
        assert provider.opencode_config is None
        assert roster.vault_routes["example"].token_env == {"_default": "EXAMPLE_API_KEY"}
        assert roster.auth_providers["example"].container_mount == str(CONTAINER_HOME / ".example")
        assert "example" in roster.all_names
        assert any(m.provider == "example" for m in roster.mounts)
        with pytest.raises(ValueError, match="providers, not installable agents.*--provider"):
            roster.resolve_selection(("example",))

    def test_legacy_opencode_provider_remains_compatible(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "example.yaml").write_text(
            f"upstream: {EXAMPLE_PROVIDER_UPSTREAM}\n"
            "auth:\n"
            "  api_key:\n"
            "    header: Authorization\n"
            '    prefix: "Bearer "\n'
            "serves:\n  openai-chat: /v1\n"
            "opencode:\n"
            "  display_name: Example Legacy\n"
            f"  base_url: {EXAMPLE_PROVIDER_UPSTREAM}/v1\n"
            "  preferred_model: example-chat\n"
            "  fallback_model: example-chat\n"
            "  env_var_prefix: EXAMPLELEGACY\n"
            "  config_dir: .example-legacy\n"
            f"  auth_key_url: {EXAMPLE_PROVIDER_UPSTREAM}/\n",
            encoding="utf-8",
        )

        with (
            patch(
                "terok_executor.roster.loader._legacy_user_providers_dir",
                return_value=legacy,
            ),
            patch(
                "terok_executor.roster.loader.providers_config_dir",
                return_value=tmp_path / "canonical",
            ),
        ):
            roster = load_roster()

        provider = roster.providers["example"]
        assert provider.label == "Example Legacy"
        assert provider.default_model == "example-chat"
        assert provider.opencode_config is not None
        assert roster.vault_routes["example"].token_env == {"_default": "EXAMPLELEGACY_API_KEY"}
        assert roster.auth_providers["example"].container_mount.endswith("/.example-legacy")

    def test_canonical_layer_overrides_legacy_layer(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy"
        canonical = tmp_path / "canonical"
        legacy.mkdir()
        canonical.mkdir()
        (legacy / "local.yaml").write_text(
            "label: Legacy\n"
            f"upstream: {LEGACY_PROVIDER_UPSTREAM}\n"
            "auth:\n  api_key: {}\n"
            "serves:\n  openai-chat: /v1\n",
            encoding="utf-8",
        )
        (canonical / "local.yaml").write_text(
            f"label: Canonical\nupstream: {CANONICAL_PROVIDER_UPSTREAM}\n",
            encoding="utf-8",
        )

        with (
            patch("terok_executor.roster.loader._legacy_user_providers_dir", return_value=legacy),
            patch("terok_executor.roster.loader.providers_config_dir", return_value=canonical),
        ):
            provider = load_roster().providers["local"]

        assert provider.label == "Canonical"
        assert provider.upstream == CANONICAL_PROVIDER_UPSTREAM

    def test_public_accessor_points_below_terok_config_root(self) -> None:
        path = providers_config_dir()

        assert path.name == "providers"
        assert path.parent.name == "terok"


class TestCredentialProviderResolution:
    """``credential_provider`` maps an auth target (agent) to its DB credential key.

    This is what keeps the auth *write* side aligned with the provider-keyed
    routes: ``terok-executor auth claude`` must store under ``anthropic`` so
    ``routed = stored & routes`` intersects at runtime.
    """

    def test_native_resolves_to_default_provider(self) -> None:
        assert credential_provider("claude") == "anthropic"
        assert credential_provider("codex") == "openai"
        assert credential_provider("vibe") == "mistral"

    def test_tool_resolves_to_its_provider(self) -> None:
        assert credential_provider("gh") == "github"
        assert credential_provider("glab") == "gitlab"

    def test_opencode_shim_is_identity(self) -> None:
        # openrouter/blablador/kisski provider names equal their agent names.
        assert credential_provider("blablador") == "blablador"

    def test_unbound_name_passes_through(self) -> None:
        # Harnesses (no binding) and unknown names resolve to themselves.
        assert credential_provider("opencode") == "opencode"
        assert credential_provider("not-an-agent") == "not-an-agent"
