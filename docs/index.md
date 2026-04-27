# Getting started

`terok-executor` runs an AI coding agent inside a hardened, rootless
Podman container — one CLI command to launch, or one Python class to
embed in your own tooling.

![terok ecosystem — terok-executor sits between project orchestration and the hardened runtime](img/architecture.svg)

## Why terok-executor

AI coding agents need network access and credentials to do useful
work, but giving them uncontrolled access to either is a risk: a
prompt-injected or supply-chain-compromised agent can exfiltrate
keys, push to arbitrary remotes, or reach services it shouldn't.

terok-executor runs each agent in an isolated rootless Podman
container with a default-deny egress firewall and a credential
vault that keeps secrets on the host.

## Prerequisites

- Python 3.12+
- Podman (rootless) — `podman machine init` on macOS
- `nft` (nftables CLI)

## Install and run

```bash
pip install terok-executor
terok-executor run claude . -p "Add type hints to utils.py"
```

The first `run` interactively offers any missing prerequisites —
sandbox services, container images, agent credentials — one
`[Y/n]` prompt at a time.  Mandatory items (services, images)
block the launch if declined; optional ones (SSH key, auth) print
the consequence and proceed.

For non-interactive environments (CI, scripts) do the bootstrap
explicitly first:

```bash
terok-executor setup        # install sandbox services + build base images
terok-executor auth claude  # OAuth or API key
terok-executor run claude . -p "..."
```

`setup` is idempotent — safe to re-run after upgrades.  If you want
to do the steps individually, `terok-executor build` only builds
images and `terok-executor vault install` only provisions the
vault.

## Authenticate

```bash
terok-executor auth claude              # OAuth login
terok-executor auth vibe                # interactive API key prompt
terok-executor auth gh --api-key ghp_…  # non-interactive
```

Credentials are stored on the host; containers never see real keys.
See [Security](security.md) for the vault details.

## Use as a library

```python
from terok_executor import AgentRunner

runner = AgentRunner()
runner.run_headless(
    agent="claude",
    repo=".",
    prompt="Add type hints to utils.py",
    max_turns=25,
)
```

`AgentRunner` is the same entry point that `terok-executor run` uses
under the hood — and what
[terok](https://github.com/terok-ai/terok) builds on for multi-task
orchestration.

## Next steps

- [Agents](agents.md) — supported agents, custom definitions, auth flows
- [Security](security.md) — firewall, vault, restricted mode
- [Launch modes](launch-modes.md) — headless, interactive, web, tool
- [Vault internals](vault.md) — architecture deep dive
