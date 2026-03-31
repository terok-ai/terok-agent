# shellcheck shell=bash
# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

# Core terok container environment — sourced by ALL shell modes:
#   - Non-interactive (bash -c): via BASH_ENV
#   - Interactive:               via /etc/bash.bashrc
#   - Login:                     via /etc/profile.d/
#
# Guard against redundant sourcing.  Login shells hit both profile.d and
# bashrc, and Claude Code's shell-snapshot mechanism may restore env vars
# without restoring functions (declare -f capture can be incomplete).
# We therefore check that key functions actually exist before short-
# circuiting — if a snapshot set the env var but lost the functions,
# re-sourcing repairs the shell.
#
# This guard is safe for per-task wrappers (terok-env-wrappers.sh, sourced
# at the bottom of this file).  Wrappers are bind-mounted at container startup
# before any shell runs, so if _terok_apply_git_identity exists, wrappers
# were already sourced in the same pass.  The [ -r ... ] guard at the
# bottom handles the case where the mount is absent (standalone mode).
_terok_env_ready() {
  declare -F _terok_apply_git_identity >/dev/null 2>&1
}
_terok_env_ready && return 0

# ── PATH ──────────────────────────────────────────────────────────────────────

# Guard against duplicate PATH prepends on re-source.
_terok_prepend_path_once() {
  case ":$PATH:" in
    *":$1:"*) ;;
    *) PATH="$1:$PATH" ;;
  esac
}

_terok_prepend_path_once "$HOME/.opencode/bin"
_terok_prepend_path_once "$HOME/.local/bin"
_terok_prepend_path_once "$HOME/.npm-packages/bin"
export PATH

# ── Git identity ──────────────────────────────────────────────────────────────

# Source the helper that defines _terok_apply_git_identity().
# Wrapper functions (claude, codex, etc.) call this in subshells to set
# GIT_AUTHOR_*/GIT_COMMITTER_* per invocation.
[ -r /usr/local/share/terok/terok-env-git-identity.sh ] && \
    . /usr/local/share/terok/terok-env-git-identity.sh

# gh/glab are human tools — always use the human identity directly.
# No authorship policy dispatch needed (unlike agent wrappers).
gh() {
  (
    export GIT_AUTHOR_NAME="${HUMAN_GIT_NAME:-Nobody}"
    export GIT_AUTHOR_EMAIL="${HUMAN_GIT_EMAIL:-nobody@localhost}"
    export GIT_COMMITTER_NAME="${HUMAN_GIT_NAME:-Nobody}"
    export GIT_COMMITTER_EMAIL="${HUMAN_GIT_EMAIL:-nobody@localhost}"
    command gh "$@"
  )
}
glab() {
  (
    export GIT_AUTHOR_NAME="${HUMAN_GIT_NAME:-Nobody}"
    export GIT_AUTHOR_EMAIL="${HUMAN_GIT_EMAIL:-nobody@localhost}"
    export GIT_COMMITTER_NAME="${HUMAN_GIT_NAME:-Nobody}"
    export GIT_COMMITTER_EMAIL="${HUMAN_GIT_EMAIL:-nobody@localhost}"
    command glab "$@"
  )
}

# ── Proxy bridges ────────────────────────────────────────────────────────────

# Ensure socat bridges are alive (idempotent; self-heals after container restart).
# shellcheck source=ensure-bridges.sh
command -v socat >/dev/null 2>&1 && . ensure-bridges.sh 2>/dev/null

# Export SSH_AUTH_SOCK when the bridge socket exists; unset if it's gone.
if [[ -S /tmp/ssh-agent.sock ]]; then
  export SSH_AUTH_SOCK=/tmp/ssh-agent.sock
elif [[ "${SSH_AUTH_SOCK:-}" == "/tmp/ssh-agent.sock" ]]; then
  unset SSH_AUTH_SOCK
fi

# ── Per-project agent wrappers ────────────────────────────────────────────────

# Source per-task agent wrappers via the L1 symlink to the bind-mounted
# terok-agent.sh.  Defines wrapper functions: claude(), codex(), vibe(), etc.
[ -r /usr/local/share/terok/terok-env-wrappers.sh ] && \
    . /usr/local/share/terok/terok-env-wrappers.sh
