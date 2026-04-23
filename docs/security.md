# Security

terok-executor isolates each agent behind four layers: an egress firewall,
a vault, optional restricted mode, and rootless containers.

## Egress firewall

On by default for every container that starts through terok-executor.
The firewall ([terok-shield](https://terok-ai.github.io/terok-shield/))
restricts outbound traffic to explicitly allowed domains — the agent's
API endpoint, package registries, and git hosts.  Everything else is
blocked at the nftables level.

The firewall is attached via OCI hooks at install time (``terok-executor
setup`` / ``terok-sandbox setup``); there is no per-run opt-out.  To
loosen it for development, edit the shield profile or run
``terok-executor uninstall`` + reinstall without the hooks.

The git gate mirror (``--gate`` / ``--no-gate``) is a separate concern
from the firewall — see [Launch modes](launch-modes.md) for that flag.

## Vault

No real API keys, OAuth tokens, or SSH private keys enter containers.
Instead, each container receives per-task **phantom tokens**. A host-side
token broker ([terok-sandbox](https://terok-ai.github.io/terok-sandbox/))
resolves phantom tokens to real credentials and forwards requests
upstream over TLS.

SSH keys are handled the same way: a host-side SSH signer lets
containers sign git operations without the private key crossing the
container boundary.

This means a compromised agent cannot read, copy, or exfiltrate real
credentials — they exist only on the host and are never written to
container-accessible mounts.

See [Vault internals](vault.md) for the full
architecture, per-agent routing table, and YAML configuration.

### Managing the vault

```bash
terok-executor vault status      # health check
terok-executor vault start       # start manually
terok-executor vault stop        # stop
terok-executor vault install     # install systemd unit
terok-executor vault routes      # show active routes
terok-executor vault clean       # remove stale tokens
```

## Restricted mode

```bash
terok-executor run claude . --restricted -p "…"
```

Disables auto-approve flags and sets `--no-new-privileges` on the
container. Use for untrusted prompts or when the agent should confirm
every action with the user.

## Rootless containers

All containers run under rootless Podman — no daemon, no root privileges.
Combined with SELinux labeling, this limits what a compromised agent can
reach on the host filesystem.
