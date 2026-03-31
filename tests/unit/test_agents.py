# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent configuration: parsing, filtering, wrapper generation, and config dir."""

from __future__ import annotations

import json
import tempfile
import unittest.mock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from terok_agent.agents import (
    AgentConfigSpec,
    _generate_claude_wrapper,
    _inject_opencode_instructions,
    _subagents_to_json,
    _write_session_hook,
    parse_md_agent,
    prepare_agent_config_dir,
)
from terok_agent.headless_providers import WrapperConfig
from tests.constants import (
    CONTAINER_CLAUDE_MEMORY_OVERRIDE,
    CONTAINER_CLAUDE_SESSION_PATH,
    CONTAINER_INSTRUCTIONS_PATH,
    NONEXISTENT_AGENT_PATH,
    NONEXISTENT_FILE_PATH,
)


class TestSubagentsToJson:
    """Tests for _subagents_to_json (dict output keyed by agent name)."""

    def test_inline_definition_default_true(self) -> None:
        """Inline sub-agent with default=True is included, output is dict keyed by name."""
        subagents = [
            {
                "name": "reviewer",
                "description": "Code reviewer",
                "tools": ["Read", "Grep"],
                "model": "sonnet",
                "default": True,
                "system_prompt": "You are a code reviewer.",
            }
        ]
        result = json.loads(_subagents_to_json(subagents))
        assert isinstance(result, dict)
        assert "reviewer" in result
        assert result["reviewer"]["prompt"] == "You are a code reviewer."
        assert result["reviewer"]["description"] == "Code reviewer"
        assert "system_prompt" not in result["reviewer"]
        assert "name" not in result["reviewer"]
        assert "default" not in result["reviewer"]

    def test_default_false_excluded_without_selection(self) -> None:
        """Agents with default=False are excluded when not selected."""
        subagents = [
            {"name": "debugger", "default": False, "model": "sonnet", "system_prompt": "Debug."},
        ]
        assert json.loads(_subagents_to_json(subagents)) == {}

    def test_selected_agents_included(self) -> None:
        """Non-default agents are included when passed in selected_agents."""
        subagents = [
            {"name": "debugger", "default": False, "model": "sonnet", "system_prompt": "Debug."},
        ]
        result = json.loads(_subagents_to_json(subagents, selected_agents=["debugger"]))
        assert "debugger" in result

    def test_mixed_default_and_selected(self) -> None:
        """Default agents + selected non-default agents are both included."""
        subagents = [
            {"name": "reviewer", "default": True, "model": "sonnet", "system_prompt": "Review."},
            {"name": "debugger", "default": False, "model": "opus", "system_prompt": "Debug."},
            {"name": "planner", "default": False, "model": "haiku", "system_prompt": "Plan."},
        ]
        result = json.loads(_subagents_to_json(subagents, selected_agents=["debugger"]))
        assert "reviewer" in result
        assert "debugger" in result
        assert "planner" not in result

    def test_file_reference_with_default(self) -> None:
        """File references with default flag are handled correctly."""
        with tempfile.TemporaryDirectory() as td:
            md_file = Path(td) / "reviewer.md"
            md_file.write_text(
                "---\nname: reviewer\ndescription: Code reviewer\n"
                "tools: [Read, Grep]\nmodel: sonnet\n---\n"
                "You are a code reviewer.\n",
                encoding="utf-8",
            )
            subagents = [{"file": str(md_file), "default": True}]
            result = json.loads(_subagents_to_json(subagents))
            assert "reviewer" in result

    def test_missing_file_skipped(self) -> None:
        """Missing file references are skipped."""
        subagents = [{"file": str(NONEXISTENT_AGENT_PATH), "default": True}]
        assert json.loads(_subagents_to_json(subagents)) == {}

    def test_agent_without_name_skipped(self) -> None:
        """Agents without a name are skipped."""
        subagents = [{"default": True, "model": "sonnet", "system_prompt": "No name."}]
        assert json.loads(_subagents_to_json(subagents)) == {}


class TestParseMdAgent:
    """Tests for parse_md_agent."""

    def test_parse_with_frontmatter(self) -> None:
        """Parses YAML frontmatter + body from .md file."""
        with tempfile.TemporaryDirectory() as td:
            md = Path(td) / "test.md"
            md.write_text("---\nname: test\ntools: [Read]\n---\nPrompt body.", encoding="utf-8")
            result = parse_md_agent(str(md))
            assert result["name"] == "test"
            assert result["prompt"] == "Prompt body."

    def test_parse_without_frontmatter(self) -> None:
        """File without frontmatter is treated as raw prompt."""
        with tempfile.TemporaryDirectory() as td:
            md = Path(td) / "test.md"
            md.write_text("Just a prompt.", encoding="utf-8")
            result = parse_md_agent(str(md))
            assert result["prompt"] == "Just a prompt."

    def test_nonexistent_file(self) -> None:
        """Nonexistent file returns empty dict."""
        assert parse_md_agent(str(NONEXISTENT_FILE_PATH)) == {}


class TestGenerateClaudeWrapper:
    """Tests for _generate_claude_wrapper."""

    def test_basic_wrapper(self) -> None:
        """Wrapper includes add-dir / and git env vars."""
        wrapper = _generate_claude_wrapper(WrapperConfig(has_agents=False))
        assert "claude()" in wrapper
        assert '--add-dir "/"' in wrapper
        assert "_terok_apply_git_identity Claude noreply@anthropic.com" in wrapper
        assert "agents.json" not in wrapper

    def test_wrapper_with_agents(self) -> None:
        """Wrapper includes agents.json reference when has_agents=True."""
        assert "agents.json" in _generate_claude_wrapper(WrapperConfig(has_agents=True))

    def test_wrapper_includes_append_system_prompt(self) -> None:
        """Wrapper includes --append-system-prompt when has_instructions=True."""
        wrapper = _generate_claude_wrapper(WrapperConfig(has_agents=False, has_instructions=True))
        assert "--append-system-prompt" in wrapper

    def test_wrapper_timeout_support(self) -> None:
        """Wrapper parses --terok-timeout and wraps claude with timeout."""
        wrapper = _generate_claude_wrapper(WrapperConfig(has_agents=False))
        assert "--terok-timeout" in wrapper
        assert 'timeout "$_timeout" claude' in wrapper
        assert 'command claude "${_args[@]}" "$@"' in wrapper

    def test_wrapper_resume_from_session_file(self) -> None:
        """Wrapper adds --resume from claude-session.txt when it exists."""
        wrapper = _generate_claude_wrapper(WrapperConfig(has_agents=False))
        assert "claude-session.txt" in wrapper
        assert "--resume" in wrapper

    def test_wrapper_sets_memory_override(self) -> None:
        """Wrapper exports CLAUDE_COWORK_MEMORY_PATH_OVERRIDE."""
        wrapper = _generate_claude_wrapper(WrapperConfig(has_agents=False))
        assert f'"{CONTAINER_CLAUDE_MEMORY_OVERRIDE}"' in wrapper


class TestWriteSessionHook:
    """Tests for _write_session_hook."""

    def test_creates_settings_with_hook(self) -> None:
        """Creates settings.json with a SessionStart hook."""
        with tempfile.TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            _write_session_hook(settings_path)
            data = json.loads(settings_path.read_text())
            assert "SessionStart" in data["hooks"]
            command = data["hooks"]["SessionStart"][0]["hooks"][0]["command"]
            assert "session_id" in command

    def test_merges_with_existing_settings(self) -> None:
        """Merges hook into existing settings.json without clobbering."""
        with tempfile.TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            settings_path.write_text('{"permissions": {"allow": ["Read"]}}', encoding="utf-8")
            _write_session_hook(settings_path)
            data = json.loads(settings_path.read_text())
            assert data["permissions"] == {"allow": ["Read"]}
            assert "SessionStart" in data["hooks"]

    def test_idempotent_hook_write(self) -> None:
        """Calling twice doesn't create duplicate hooks."""
        with tempfile.TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            _write_session_hook(settings_path)
            _write_session_hook(settings_path)
            data = json.loads(settings_path.read_text())
            assert len(data["hooks"]["SessionStart"]) == 1

    def test_does_not_rewrite_when_hook_present(self) -> None:
        """If equivalent hook exists, file is left untouched."""
        with tempfile.TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            hook_command = (
                "python3 -c \"import json,sys; print(json.load(sys.stdin)['session_id'])\""
                f" > {CONTAINER_CLAUDE_SESSION_PATH}"
            )
            original = json.dumps(
                {
                    "hooks": {
                        "SessionStart": [{"hooks": [{"type": "command", "command": hook_command}]}]
                    }
                },
                separators=(",", ":"),
            )
            settings_path.write_text(original, encoding="utf-8")
            _write_session_hook(settings_path)
            assert settings_path.read_text(encoding="utf-8") == original

    def test_concurrent_writes_keep_single_hook(self) -> None:
        """Concurrent writes produce a single valid SessionStart entry."""
        with tempfile.TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(_write_session_hook, settings_path) for _ in range(48)]
                for f in futures:
                    f.result()
            data = json.loads(settings_path.read_text())
            assert len(data["hooks"]["SessionStart"]) == 1


class TestPrepareAgentConfigDir:
    """Tests for prepare_agent_config_dir."""

    @staticmethod
    def _make_spec(tasks_root: Path, task_id: str, **kwargs: object) -> AgentConfigSpec:
        """Build a minimal AgentConfigSpec for testing."""
        return AgentConfigSpec(
            tasks_root=tasks_root,
            task_id=task_id,
            subagents=[],
            default_agent=None,
            mounts_base=kwargs.pop("mounts_base", None),
            instructions=kwargs.pop("instructions", None),
            **kwargs,
        )

    @unittest.mock.patch("terok_agent.agents._write_session_hook")
    def test_writes_instructions(self, _mock: object, tmp_path: Path) -> None:
        """Instructions text is written to instructions.md."""
        with tempfile.TemporaryDirectory() as envs:
            spec = self._make_spec(
                tmp_path / "tasks", "t1", instructions="Custom.", mounts_base=Path(envs)
            )
            (tmp_path / "tasks" / "t1").mkdir(parents=True)
            d = prepare_agent_config_dir(spec)
            assert (d / "instructions.md").read_text(encoding="utf-8") == "Custom."

    @unittest.mock.patch("terok_agent.agents._write_session_hook")
    def test_default_instructions_when_none(self, _mock: object, tmp_path: Path) -> None:
        """Default instructions.md written when instructions is None."""
        with tempfile.TemporaryDirectory() as envs:
            spec = self._make_spec(tmp_path / "tasks", "t2", mounts_base=Path(envs))
            (tmp_path / "tasks" / "t2").mkdir(parents=True)
            d = prepare_agent_config_dir(spec)
            assert "conventions" in (d / "instructions.md").read_text(encoding="utf-8")

    @unittest.mock.patch("terok_agent.agents._write_session_hook")
    def test_wrapper_has_append_system_prompt(self, _mock: object, tmp_path: Path) -> None:
        """Claude wrapper includes --append-system-prompt when instructions given."""
        with tempfile.TemporaryDirectory() as envs:
            spec = self._make_spec(
                tmp_path / "tasks", "t3", instructions="Test.", mounts_base=Path(envs)
            )
            (tmp_path / "tasks" / "t3").mkdir(parents=True)
            d = prepare_agent_config_dir(spec)
            wrapper = (d / "terok-agent.sh").read_text(encoding="utf-8")
            assert "--append-system-prompt" in wrapper


class TestInjectOpencodeInstructions:
    """Tests for _inject_opencode_instructions()."""

    def test_creates_file_if_missing(self) -> None:
        """Creates opencode.json with instructions entry and $schema."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "opencode.json"
            _inject_opencode_instructions(config_path)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            assert data["instructions"] == [str(CONTAINER_INSTRUCTIONS_PATH)]
            assert data["$schema"] == "https://opencode.ai/config.json"

    def test_idempotent_when_already_present(self) -> None:
        """Does not duplicate the instructions entry on repeated calls."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "opencode.json"
            _inject_opencode_instructions(config_path)
            _inject_opencode_instructions(config_path)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            assert data["instructions"] == [str(CONTAINER_INSTRUCTIONS_PATH)]

    def test_preserves_existing_instructions(self) -> None:
        """Appends to existing instructions list."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "opencode.json"
            config_path.write_text(
                json.dumps({"instructions": ["/some/other/file.md"]}), encoding="utf-8"
            )
            _inject_opencode_instructions(config_path)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            assert len(data["instructions"]) == 2

    def test_preserves_existing_config_keys(self) -> None:
        """Preserves other keys in the opencode.json file."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "opencode.json"
            config_path.write_text(json.dumps({"model": "test/model"}), encoding="utf-8")
            _inject_opencode_instructions(config_path)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            assert data["model"] == "test/model"

    def test_creates_parent_directories(self) -> None:
        """Creates parent directories if they do not exist."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "nested" / "dir" / "opencode.json"
            _inject_opencode_instructions(config_path)
            assert config_path.is_file()

    def test_handles_invalid_json(self) -> None:
        """Overwrites file with valid config if existing JSON is invalid."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "opencode.json"
            config_path.write_text("not valid json", encoding="utf-8")
            _inject_opencode_instructions(config_path)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            assert data["instructions"] == [str(CONTAINER_INSTRUCTIONS_PATH)]
