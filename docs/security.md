# Security

terok-agent isolates each agent behind four layers: an egress firewall,
a credential proxy, optional restricted mode, and rootless containers.

## Egress firewall

On by default. The firewall
([terok-shield](https://terok-ai.github.io/terok-shield/)) restricts
outbound traffic to explicitly allowed domains — the agent's API endpoint,
package registries, and git hosts. Everything else is blocked at the
nftables level.

```bash
terok-agent run claude . -p "…"         # firewall on (default)
terok-agent run claude . --no-gate -p "…"  # disable for development
```

## Credential proxy

No real API keys, OAuth tokens, or SSH private keys enter containers.
Instead, each container receives per-task **phantom tokens**. A host-side
proxy ([terok-sandbox](https://terok-ai.github.io/terok-sandbox/))
resolves phantom tokens to real credentials and forwards requests
upstream over TLS.

SSH keys are handled the same way: a host-side SSH agent proxy lets
containers sign git operations without the private key crossing the
container boundary.

This means a compromised agent cannot read, copy, or exfiltrate real
credentials — they exist only on the host and are never written to
container-accessible mounts.

See [Credential proxy internals](credential-proxy.md) for the full
architecture, per-agent routing table, and YAML configuration.

### Managing the proxy

```bash
terok-agent proxy status      # health check
terok-agent proxy start       # start manually
terok-agent proxy stop        # stop
terok-agent proxy install     # install systemd unit
terok-agent proxy routes      # show active routes
terok-agent proxy clean       # remove stale tokens
```

## Restricted mode

```bash
terok-agent run claude . --restricted -p "…"
```

Disables auto-approve flags and sets `--no-new-privileges` on the
container. Use for untrusted prompts or when the agent should confirm
every action with the user.

## Rootless containers

All containers run under rootless Podman — no daemon, no root privileges.
Combined with SELinux labeling, this limits what a compromised agent can
reach on the host filesystem.
