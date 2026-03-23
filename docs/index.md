# terok-agent

Single-agent task runner for hardened Podman containers.

## Overview

**terok-agent** builds agent images, launches instrumented containers, and
manages the lifecycle of one AI coding agent at a time.  It sits between
[terok-sandbox](https://github.com/terok-ai/terok-sandbox) (container
isolation) and [terok](https://github.com/terok-ai/terok) (project
orchestration) in the dependency chain:

```text
terok → terok-agent → terok-sandbox → terok-shield
```

!!! note "Early development"
    This package currently provides only the project scaffold and a stub CLI.
    The features and commands listed below are planned — not yet implemented.

## Quick start (planned)

```bash
pip install terok-agent
terok-agent run claude .
```

## Planned features

- **Agent registry** — YAML-driven agent definitions (Claude, Codex, Copilot, …)
- **Image building** — parametrized L0 (base) + L1 (agent) Dockerfile layers
- **Auth flows** — per-agent authentication (API keys, OAuth)
- **Shield required** — refuses to run without nftables egress firewall
- **Config stack** — layered config resolution (global → project → preset → CLI)
