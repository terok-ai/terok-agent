# Getting started

## Why terok-agent

AI coding agents need network access and credentials to do useful work,
but giving them uncontrolled access to both is a risk — a prompt-injected
or supply-chain-compromised agent can exfiltrate API keys, push to
arbitrary remotes, or reach services it shouldn't.

terok-agent runs each agent in an isolated rootless Podman container with
an egress firewall and a credential proxy that keeps real secrets off the
container filesystem. One command to build, authenticate, and launch.

## Prerequisites

- Python 3.12+
- Podman (rootless) — `podman machine init` on macOS

## Install

```bash
pip install terok-agent
```

## Build container images

```bash
terok-agent build
```

Builds two image layers: a base image (OS, dev tools, init scripts) and
an agent image (all AI agent CLIs, shell wrappers, ACP config). Rebuild
with `--rebuild` to bust caches or `--full-rebuild` for a clean pull.

## Authenticate

```bash
terok-agent auth claude              # OAuth login
terok-agent auth vibe                # interactive API key prompt
terok-agent auth gh --api-key ghp_…  # non-interactive
```

Credentials are stored in a host-side database. Containers never see real
keys — they receive phantom tokens resolved by the credential proxy.
See [Security](security.md) for details.

## First run

```bash
terok-agent run claude . -p "Add type hints to utils.py"
```

This clones the current directory into a hardened container, launches
Claude in headless mode, and streams its output. The egress firewall
and credential proxy are active by default.

See [Launch modes](launch-modes.md) for interactive, web, and tool modes.

## Next steps

- [Agents](agents.md) — supported agents, custom definitions, auth flows
- [Security](security.md) — firewall, credential proxy, restricted mode
- [Launch modes](launch-modes.md) — headless, interactive, web, tool
- [Credential proxy internals](credential-proxy.md) — architecture deep dive
