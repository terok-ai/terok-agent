# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for sealed-container injection helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from terok_agent.container.inject import inject_agent_config, inject_prompt


class TestInjectAgentConfig:
    """Verify inject_agent_config delegates to Sandbox.copy_to."""

    def test_copies_config_dir_into_container(self, tmp_path: Path) -> None:
        """Config dir is copied to /home/dev/.terok via podman cp."""
        mock_sandbox = MagicMock()
        with patch("terok_sandbox.Sandbox", return_value=mock_sandbox):
            inject_agent_config("my-ctr", tmp_path)

        mock_sandbox.copy_to.assert_called_once_with("my-ctr", tmp_path, "/home/dev/.terok")


class TestInjectPrompt:
    """Verify inject_prompt writes a temp file and copies it in."""

    def test_prompt_text_reaches_container(self) -> None:
        """Prompt text is written to a temp file then podman-cp'd."""
        mock_sandbox = MagicMock()
        with patch("terok_sandbox.Sandbox", return_value=mock_sandbox):
            inject_prompt("sealed-ctr", "Fix the flaky test")

        mock_sandbox.copy_to.assert_called_once()
        _name, src, dest = mock_sandbox.copy_to.call_args[0]
        assert _name == "sealed-ctr"
        assert dest == "/home/dev/.terok/prompt.txt"
        # The temp file is cleaned up, but we can verify the Path was correct
        assert isinstance(src, Path)
        assert src.name == "prompt.txt"

    def test_prompt_text_encoding(self) -> None:
        """Non-ASCII prompt text is written with UTF-8 encoding."""
        written_content = None

        def capture_copy(container, src, dest):
            nonlocal written_content
            # Read the temp file before it's cleaned up
            written_content = src.read_text(encoding="utf-8")

        mock_sandbox = MagicMock()
        mock_sandbox.copy_to.side_effect = capture_copy
        with patch("terok_sandbox.Sandbox", return_value=mock_sandbox):
            inject_prompt("ctr", "Behebe den Fehler — Ünïcödé")

        assert written_content == "Behebe den Fehler — Ünïcödé"

    def test_multiline_prompt(self) -> None:
        """Multi-line prompts are preserved verbatim."""
        written_content = None

        def capture_copy(container, src, dest):
            nonlocal written_content
            written_content = src.read_text(encoding="utf-8")

        mock_sandbox = MagicMock()
        mock_sandbox.copy_to.side_effect = capture_copy
        with patch("terok_sandbox.Sandbox", return_value=mock_sandbox):
            inject_prompt("ctr", "Line one\nLine two\nLine three")

        assert written_content == "Line one\nLine two\nLine three"
