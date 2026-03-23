# terok-agent

Single-agent task runner for hardened Podman containers.

**terok-agent** builds agent images, launches instrumented containers, and
manages the lifecycle of one AI coding agent at a time.

> **Status:** Early development — the CLI is a stub. Commands shown below are
> planned features, not yet implemented.

## Ecosystem

```text
terok-shield    →  nftables egress firewall (security boundary)
terok-sandbox   →  hardened container runtime (isolation)
terok-agent     →  single agent task runner (one agent, one mission)
terok           →  project orchestration (TUI, presets, task lifecycle)
```

## Installation

```bash
pip install terok-agent
```

## Usage (planned)

```bash
terok-agent run claude .
terok-agent auth claude
terok-agent agents
```

## Development

```bash
poetry install --with dev,test,docs
make check    # lint + test + tach + security + docstrings + deadcode + reuse
```

## License

[Apache-2.0](LICENSES/Apache-2.0.txt)
