# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Andreas Knüpfer
# SPDX-License-Identifier: Apache-2.0

"""Shell wrapper generation for agent CLI commands.

Produces per-provider bash functions (``claude()``, ``codex()``, ``vibe()``,
etc.) that set git identity, handle session resume, and support both
interactive and headless (``--terok-timeout``) modes.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .providers import AGENT_PROVIDERS, AgentProvider

if TYPE_CHECKING:
    from collections.abc import Callable


INITIAL_PROMPT_PATH = "/home/dev/.terok/initial-prompt.txt"
"""Container path of the per-task initial prompt the TUI/CLI writes at launch."""

INITIAL_PROMPT_CONSUMED_PATH = "/home/dev/.terok/initial-prompt.consumed.txt"
"""Where the prompt file is moved after an agent picks it up (one-shot semantics)."""


@dataclass(frozen=True)
class WrapperConfig:
    """Groups parameters for generating the Claude shell wrapper."""

    has_agents: bool
    has_instructions: bool = False


def generate_agent_wrapper(
    provider: AgentProvider,
    has_agents: bool,
    *,
    claude_wrapper_fn: Callable[[WrapperConfig], str] | None = None,
) -> str:
    """Generate the shell wrapper function content for a single provider.

    For Claude, uses *claude_wrapper_fn* (which should be
    ``agents._generate_claude_wrapper``) to produce the full wrapper with
    ``--add-dir /``, ``--agents``, and session resume support.  The function
    is passed in by the caller to avoid a circular import between this module
    and ``agents``.

    For other providers, produces a simpler wrapper that sets git env vars
    and delegates to the binary.  Instructions are delivered via
    ``opencode.json`` (OpenCode/Blablador), ``model_instructions_file``
    (Codex), or ``--append-system-prompt`` (Claude) -- not via the wrapper.

    Args:
        claude_wrapper_fn: ``(cfg: WrapperConfig) -> str``.
            Required when ``provider.name == "claude"``.

    See also [`generate_all_wrappers`][terok_executor.provider.wrappers.generate_all_wrappers] which produces wrappers for every
    registered provider in one file.
    """
    if provider.name == "claude":
        if claude_wrapper_fn is None:
            raise ValueError("claude_wrapper_fn is required for Claude provider")
        return claude_wrapper_fn(WrapperConfig(has_agents=has_agents))

    return _generate_generic_wrapper(provider)


def generate_all_wrappers(
    has_agents: bool,
    *,
    claude_wrapper_fn: Callable[[WrapperConfig], str] | None = None,
) -> str:
    """Generate shell wrappers for **all** registered providers in one file.

    The output file contains a shell function per provider (``claude()``,
    ``codex()``, ``vibe()``, etc.), each with correct git env vars, timeout
    support, and session resume logic.  This allows interactive CLI users to
    invoke any agent regardless of which provider was configured as default.

    A shared ``_terok_resume_or_fresh`` helper is emitted at the top of the
    file for stale-session fallback (see `_RESUME_FALLBACK_FN`).

    Args:
        claude_wrapper_fn: Required -- produces the Claude wrapper.
    """
    sections: list[str] = [_RESUME_FALLBACK_FN]
    for provider in AGENT_PROVIDERS.values():
        section = generate_agent_wrapper(
            provider,
            has_agents,
            claude_wrapper_fn=claude_wrapper_fn,
        )
        sections.append(section)
    return "\n".join(sections)


# -- Private helpers ----------------------------------------------------------


def _auto_approve_block(provider: AgentProvider) -> list[str]:
    """Emit bash lines for auto-approve flag injection (Codex only today)."""
    if not provider.auto_approve_flags:
        return []
    lines = ["    local _approve_args=()"]
    lines.append('    if [[ "${TEROK_UNRESTRICTED:-}" == "1" ]]; then')
    for flag in provider.auto_approve_flags:
        lines.append(f"        _approve_args+=({shlex.quote(flag)})")
    lines.append("    fi")
    return lines


def _opencode_plugin_block(provider: AgentProvider) -> list[str]:
    """Emit bash lines for OpenCode session plugin symlink setup."""
    if not (provider.session_file and provider.uses_opencode_instructions):
        return []
    if provider.opencode_config is not None:
        plugin_dir = f"$HOME/{provider.opencode_config.config_dir}/opencode/plugins"
    else:
        plugin_dir = "$HOME/.config/opencode/plugins"
    return [
        "    # Ensure OpenCode session plugin is installed",
        "    local _plugin_src=/usr/local/share/terok/opencode-session-plugin.mjs",
        f"    local _plugin_dir={plugin_dir}",
        '    if [ -f "$_plugin_src" ]; then',
        '        mkdir -p "$_plugin_dir"',
        '        ln -sf "$_plugin_src" "$_plugin_dir/terok-session.mjs"',
        "    fi",
    ]


def initial_prompt_block(session_path: str | None) -> list[str]:
    """Emit bash lines that pick up the task's initial prompt as the first message.

    Fires only on a bare interactive launch (no user args, no headless timeout)
    when no session is being resumed.  The file is renamed after consumption so
    a subsequent agent invocation falls through to ``--resume`` (or starts
    fresh) instead of replaying the same prompt.

    The renamed copy is preserved as a paper trail; the user can ``mv`` it
    back to recover from a launch that crashed before the session was saved.
    """
    guards = ['[ -z "$_timeout" ]', "[ $# -eq 0 ]"]
    if session_path:
        guards.append(f"[ ! -s {session_path} ]")
    guards.append(f"[ -s {INITIAL_PROMPT_PATH} ]")
    return [
        "    # Pick up the task's initial prompt as the first message (one-shot).",
        f"    if {' && '.join(guards)}; then",
        f'        set -- "$(cat {INITIAL_PROMPT_PATH})"',
        f"        mv {INITIAL_PROMPT_PATH} {INITIAL_PROMPT_CONSUMED_PATH}",
        "    fi",
    ]


def _session_resume_block(provider: AgentProvider, session_path: str | None) -> list[str]:
    """Emit bash lines for session resume arg injection."""
    if not (session_path and provider.resume_flag):
        return []
    return [
        "    local _resume_args=()",
        f"    if [ -s {session_path} ] && \\",
        '       { [ -n "$_timeout" ] || [ $# -eq 0 ]; }; then',
        f'        _resume_args+=({provider.resume_flag} "$(cat {session_path})")',
        "    fi",
    ]


def _codex_instr_block(provider: AgentProvider) -> list[str]:
    """Emit bash lines for Codex model_instructions_file injection."""
    if provider.name != "codex":
        return []
    return [
        "    local _instr_args=()",
        "    [ -f /home/dev/.terok/instructions.md ] && \\",
        "        _instr_args+=(-c 'model_instructions_file=\"/home/dev/.terok/instructions.md\"')",
    ]


def _vibe_model_sync_block(provider: AgentProvider) -> list[str]:
    """Emit bash lines for lazy Mistral model sync before vibe runs.

    Uses a pure-bash mtime check to avoid Python startup overhead when
    the cache is fresh (<24h).  Only invoked from the vibe() wrapper,
    keeping login shells fast.
    """
    if provider.name != "vibe":
        return []
    return [
        "    # Lazy Mistral model sync (pure-bash mtime check, avoids Python startup)",
        "    if command -v vibe-model-sync >/dev/null 2>&1; then",
        "        local _mc=$HOME/.vibe/mistral-models.txt",
        '        if [ ! -f "$_mc" ] || [ -n "$(find "$_mc" -mmin +1440 2>/dev/null)" ]; then',
        "            vibe-model-sync >/dev/null 2>&1 || true",
        "        fi",
        "    fi",
    ]


def _vibe_capture_fn(provider: AgentProvider, session_path: str | None) -> list[str]:
    """Emit bash lines for Vibe post-run session capture helper."""
    if not (provider.name == "vibe" and session_path):
        return []
    return [
        "    _terok_capture_vibe_session() {",
        '        python3 -c "',
        "import json, os, glob",
        "files = sorted(glob.glob(os.path.expanduser('~/.vibe/logs/session/session_*/meta.json')),",
        "               key=os.path.getmtime, reverse=True)",
        "if files:",
        "    with open(files[0]) as f:",
        "        sid = json.load(f).get('session_id', '')",
        "    if sid:",
        "        print(sid)",
        f'" > {session_path} 2>/dev/null || true',
        "    }",
    ]


def _extra_args_expansion(provider: AgentProvider, session_path: str | None) -> str:
    """Build the extra-args shell expansions between the binary and ``"$@"``."""
    parts: list[str] = []
    if provider.auto_approve_flags:
        parts.append('"${_approve_args[@]}"')
    if session_path and provider.resume_flag:
        parts.append('"${_resume_args[@]}"')
    if provider.name == "codex":
        parts.append('"${_instr_args[@]}"')
    return (" " + " ".join(parts)) if parts else ""


def _wrap_invocation(cmd: str, provider: AgentProvider, session_path: str | None) -> str:
    """Wrap a shell invocation with the stale-session fallback when resume is active."""
    if session_path and provider.resume_flag:
        return f"_terok_resume_or_fresh {session_path} {provider.resume_flag} {cmd}"
    return cmd


_RESUME_FALLBACK_FN = """\
# WORKAROUND: stale-session guard (timing-based heuristic).
#
# When a user starts an agent, exits immediately (no real interaction),
# and re-runs, the captured session ID points to a conversation that was
# never persisted.  The agent then fails with "No conversation found".
#
# This is a best-effort mitigation, not a proper fix: we assume that
# any non-zero exit within 2 seconds of launch is a stale-session error
# and retry without --resume.  This heuristic can misfire (e.g. a fast
# config error would also trigger a retry), but the retry is harmless —
# it just runs without resume, which is the correct fallback anyway.
#
# A proper fix would validate the session ID against the agent's storage
# before injecting --resume, but that requires agent-specific probes
# that don't exist yet.
_terok_resume_or_fresh() {
    local _session_file="$1" _resume_flag="$2"; shift 2
    local _start; _start=$(date +%s)
    "$@"; local _rc=$?
    local _elapsed=$(( $(date +%s) - _start ))
    if [ $_rc -ne 0 ] && [ $_elapsed -lt 2 ] && [ -s "$_session_file" ]; then
        echo "terok: session not found (stale?), retrying without resume" >&2
        rm -f "$_session_file"
        local _retry=() _skip=false
        for _a in "$@"; do
            if $_skip; then _skip=false; continue; fi
            if [ "$_a" = "$_resume_flag" ]; then _skip=true; continue; fi
            _retry+=("$_a")
        done
        "${_retry[@]}"; _rc=$?
    fi
    return $_rc
}
"""


def _generate_generic_wrapper(provider: AgentProvider) -> str:
    """Generate a shell wrapper for non-Claude providers.

    Sets git identity env vars and wraps the binary with optional timeout
    support (``--terok-timeout``), matching the Claude wrapper's interface.

    Session resume logic (for providers with ``session_file``):

    - An OpenCode plugin (or post-run parse for Vibe) captures the session
      ID to ``/home/dev/.terok/<session_file>``.
    - Resume args (``--session <id>`` or ``--resume <id>``) are injected
      only in headless mode (``--terok-timeout`` present) or on bare
      interactive launch (no user args).
    - When the user passes their own arguments, passthrough is transparent
      -- no resume args are injected.
    """
    author_name = shlex.quote(provider.git_author_name)
    author_email = shlex.quote(provider.git_author_email)
    binary = provider.binary
    session_path = f"/home/dev/.terok/{provider.session_file}" if provider.session_file else None
    extra = _extra_args_expansion(provider, session_path)

    lines = [
        "# Generated by terok",
        f"{binary}() {{",
        '    local _timeout=""',
        "    # Extract terok-specific flags (must come before agent flags)",
        "    while [[ $# -gt 0 ]]; do",
        '        case "$1" in',
        '            --terok-timeout) _timeout="$2"; shift 2 ;;',
        "            *) break ;;",
        "        esac",
        "    done",
        "    [ -r /usr/local/share/terok/terok-env-git-identity.sh ] && \\",
        "        . /usr/local/share/terok/terok-env-git-identity.sh",
    ]

    lines.extend(_vibe_model_sync_block(provider))
    lines.extend(_auto_approve_block(provider))
    lines.extend(_opencode_plugin_block(provider))
    lines.extend(_session_resume_block(provider, session_path))
    lines.extend(_codex_instr_block(provider))
    lines.extend(initial_prompt_block(session_path))
    lines.extend(_vibe_capture_fn(provider, session_path))

    headless_cmd = _wrap_invocation(
        f'timeout "$_timeout" {binary}{extra} "$@"', provider, session_path
    )
    interactive_cmd = _wrap_invocation(f'command {binary}{extra} "$@"', provider, session_path)

    # Headless mode (with timeout)
    lines.append('    if [ -n "$_timeout" ]; then')
    lines.append("        (")
    lines.append(f"            _terok_apply_git_identity {author_name} {author_email}")
    if session_path:
        lines.append(f"            export TEROK_SESSION_FILE={session_path}")
    lines.append(f"        {headless_cmd}")
    if provider.name == "vibe" and session_path:
        lines.append("        local _rc=$?; _terok_capture_vibe_session; return $_rc")
    lines.append("        )")

    # Interactive mode (no timeout)
    lines.append("    else")
    lines.append("        (")
    lines.append(f"            _terok_apply_git_identity {author_name} {author_email}")
    if session_path:
        lines.append(f"            export TEROK_SESSION_FILE={session_path}")
    lines.append(f"        {interactive_cmd}")
    if provider.name == "vibe" and session_path:
        lines.append("        local _rc=$?; _terok_capture_vibe_session; return $_rc")
    lines.append("        )")
    lines.append("    fi")
    lines.append("}")

    return "\n".join(lines) + "\n"
