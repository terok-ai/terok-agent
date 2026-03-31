# shellcheck shell=bash
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

# Idempotent socat bridge launcher for container ↔ host-side credential proxy.
#
# Manages two bridges:
#   1. SSH agent  — UNIX socket → ssh-agent-bridge.sh → TCP (phantom-token)
#   2. gh proxy   — UNIX socket → TCP (plain relay to credential proxy)
#
# Uses PID files (not socket existence) to detect dead bridges — stale
# socket files persist after process death and are unreliable sentinels.
#
# Designed to be *sourced* (not executed) so SSH_AUTH_SOCK propagates
# to the caller.  Called from:
#   - init-ssh-and-repo.sh  (first boot)
#   - terok-env.sh          (every shell — self-heal after restart)

_TEROK_PIDDIR=/tmp/.terok
mkdir -p "$_TEROK_PIDDIR" 2>/dev/null

_terok_bridge_alive() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null
}

# ── SSH agent bridge ─────────────────────────────────────────────────────
if [[ -n "${TEROK_SSH_AGENT_PORT:-}" ]] && [[ -n "${TEROK_SSH_AGENT_TOKEN:-}" ]] \
   && command -v socat >/dev/null 2>&1 \
   && ! _terok_bridge_alive "$_TEROK_PIDDIR/ssh-agent.pid"; then
  rm -f /tmp/ssh-agent.sock
  socat "UNIX-LISTEN:/tmp/ssh-agent.sock,fork" "SYSTEM:ssh-agent-bridge.sh" &
  echo $! > "$_TEROK_PIDDIR/ssh-agent.pid"
  export SSH_AUTH_SOCK=/tmp/ssh-agent.sock
fi

# ── gh credential proxy bridge ───────────────────────────────────────────
if [[ -n "${TEROK_PROXY_PORT:-}" ]] && [[ -n "${GH_TOKEN:-}" ]] \
   && command -v socat >/dev/null 2>&1 \
   && ! _terok_bridge_alive "$_TEROK_PIDDIR/gh-proxy.pid"; then
  rm -f /tmp/terok-gh-proxy.sock
  socat UNIX-LISTEN:/tmp/terok-gh-proxy.sock,fork \
    TCP:host.containers.internal:"${TEROK_PROXY_PORT}" &
  echo $! > "$_TEROK_PIDDIR/gh-proxy.pid"
fi
