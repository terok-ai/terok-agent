# Launch modes

terok-executor supports four ways to run an agent, plus a separate tool
runner for sidecars.

## Headless

```bash
terok-executor run claude . -p "Fix the failing test"
terok-executor run claude . -p "Refactor auth" --model sonnet --max-turns 10
```

Fire-and-forget. The agent runs autonomously and streams output to the
terminal. Exits when the agent finishes, hits `--max-turns`, or reaches
`--timeout` (default 1800 s).

## Interactive

```bash
terok-executor run claude . --interactive
```

Opens a shell session inside the container. The agent CLI is installed
and ready; credentials and the repository are pre-configured. Use this
to drive the agent manually.

## Web

```bash
terok-executor run claude . --web
terok-executor run claude . --web --port 8080
```

Launches [toad](https://github.com/terok-ai/toad), a multi-agent TUI
served over HTTP. Access it in a browser at the printed URL.

## Tool mode

```bash
terok-executor run-tool coderabbit . -- --pr 42
terok-executor run-tool sonarcloud . --timeout 300
```

Runs a sidecar tool in its own container. Arguments after `--` are
passed to the tool binary. See [Agents](agents.md#sidecar-tools) for
the list of supported tools.

## Managing containers

```bash
terok-executor list            # list running containers
terok-executor stop my-task    # stop a specific container
```

## Common flags

| Flag | Description |
|------|-------------|
| `--gate` / `--no-gate` | Enable or disable the egress firewall (default: on) |
| `--restricted` | No auto-approve, no-new-privileges |
| `--branch <ref>` | Check out a specific git branch |
| `--name <name>` | Container name override |
| `--gpu` | Enable GPU passthrough |
| `--git-identity-from-host` | Use the host's git user.name and user.email |
| `--shared-dir` / `--shared-mount` | Mount a host directory into the container |
| `--timeout <seconds>` | Override the default timeout |
| `--model <name>` | Model override (headless mode) |
| `--max-turns <n>` | Limit agent turns (headless mode) |
