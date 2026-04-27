# terok-executor

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![REUSE status](https://api.reuse.software/badge/github.com/terok-ai/terok-executor)](https://api.reuse.software/info/github.com/terok-ai/terok-executor)

One command to run an AI coding agent inside a hardened, rootless
Podman container.

`terok-executor` builds the container, launches the agent against
the directory you point it at, and keeps real credentials on the
host.  Use it on its own as a CLI, or import its `AgentRunner` from
Python when you want library-grade control.

<p align="center">
  <img src="docs/img/architecture.svg" alt="terok ecosystem — terok-executor sits between project orchestration and the hardened runtime">
</p>

## Quick start

```bash
pip install terok-executor                        # Python 3.12+, rootless Podman, nft
terok-executor run claude . -p "Fix the failing test in test_auth.py"
```

The first `run` interactively offers any missing prerequisites — sandbox
services, container images, agent credentials — one `[Y/n]` prompt
at a time.  Mandatory items (services, images) block the launch if
declined; optional ones (SSH key, auth) print the consequence and
proceed.

For non-interactive environments, do the bootstrap explicitly first:

```bash
terok-executor setup                              # install sandbox services + build base images
terok-executor auth claude                        # authenticate (OAuth or API key)
terok-executor run claude . -p "..."              # idempotent — safe to re-run after upgrades
```

## Use as a library

```python
from terok_executor import AgentRunner

runner = AgentRunner()
runner.run_headless(
    agent="claude",
    repo=".",
    prompt="Fix the failing test in test_auth.py",
    max_turns=25,
)
```

`AgentRunner` exposes four launch methods — `run_headless`,
`run_interactive`, `run_web`, `run_tool` — all with the same
hardening guarantees.

## Supported agents

| Agent | Auth | Description |
|-------|------|-------------|
| Claude Code | OAuth, API key | Anthropic Claude Code |
| Codex | OAuth, API key | OpenAI Codex CLI |
| Vibe | API key | Mistral Vibe |
| Copilot | OAuth | GitHub Copilot |
| OpenCode | API key | Generic LLM endpoint driver — bundled defaults for Helmholtz Blablador, KISSKI AcademicCloud, and your own endpoint |
| gh | OAuth, API key | GitHub CLI |
| glab | API key | GitLab CLI |
| CodeRabbit | API key | CodeRabbit (sidecar tool) |
| SonarCloud | API key | SonarCloud scanner (sidecar tool) |

`terok-executor agents` lists the live roster (add `--all` to
include the tool entries).

## Where it sits in the stack

terok-executor is the per-task layer.  Above it,
[terok](https://github.com/terok-ai/terok) composes many concurrent
runs across many projects.  Below it, terok-executor delegates the
host-side security boundary
([terok-sandbox](https://github.com/terok-ai/terok-sandbox)): the
credential vault, the git gate, the egress firewall hooks, the
systemd service lifecycle.

## Commands

| Command | Description |
|---------|-------------|
| `run` | Launch an agent (headless, interactive, or web) |
| `setup` | Bootstrap sandbox services + container images |
| `uninstall` | Remove sandbox services + container images |
| `auth` | Authenticate a provider |
| `agents` | List the agent roster |
| `build` | Build base + agent images explicitly |
| `run-tool` | Run a sidecar tool (CodeRabbit, SonarCloud) |
| `list` | List running containers |
| `stop` | Stop a running container |
| `vault` | Vault management (start, stop, status, install, routes) |

## Documentation

- [Getting started](https://terok-ai.github.io/terok-executor/) — install, build, authenticate, first run
- [Agents](https://terok-ai.github.io/terok-executor/agents/) — catalog, custom definitions, auth flows
- [Launch modes](https://terok-ai.github.io/terok-executor/launch-modes/) — headless, interactive, web, tool
- [Security](https://terok-ai.github.io/terok-executor/security/) — firewall, vault, restricted mode
- [API Reference](https://terok-ai.github.io/terok-executor/reference/) — Python API docs

## Development

See the [Developer Guide](https://terok-ai.github.io/terok-executor/developer/).

## License

[Apache-2.0](LICENSES/Apache-2.0.txt)
