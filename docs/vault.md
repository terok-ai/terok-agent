# Vault Integration

## Problem

terok bind-mounts vendor config directories into task containers. A
prompt-injected or supply-chain-compromised agent can read and exfiltrate
API keys, OAuth tokens, or SSH private keys from these shared mounts.

## Solution: TCP Token Broker

No real secret enters a task container. Instead:

1. **Credential DB** (host-side) stores API keys and OAuth tokens
2. **Token broker** ([terok-sandbox](https://terok-ai.github.io/terok-sandbox/))
   resolves phantom tokens to real credentials and forwards requests upstream
3. **Per-provider phantom tokens** (per-task, per-provider) are what containers see
4. **SSH signer** ([terok-sandbox](https://terok-ai.github.io/terok-sandbox/))
   lets containers sign with host-side SSH keys without the private keys
   entering the container. A socat bridge inside the container relays
   SSH signer requests over TCP to the host-side proxy.

## Architecture

```text
HOST                                      CONTAINER
-------------------------------           -----------------------------------
Credential DB (sqlite3)                   Per-provider phantom tokens (env vars)
  credentials table                         ANTHROPIC_API_KEY=<claude-phantom>
  proxy_tokens table                        MISTRAL_API_KEY=<vibe-phantom>
    (token, project, task,                  GH_TOKEN=<gh-phantom>
     credential_set, provider)              GITLAB_TOKEN=<glab-phantom>

Token Broker (terok-sandbox)              Agent / tool makes API request
  TCP + Unix socket listeners                with phantom token in auth header
  Resolves phantom → real credential
  Injects auth header, forwards             Routing is by token, not URL path.
  to upstream over TLS                       Token encodes which provider it's for.
```

### Why TCP, not Unix sockets?

SELinux blocks `connect()` on host Unix sockets mounted into rootless
Podman containers (`container_t -> unconfined_t` denied). Containers
reach the token broker via TCP (`host.containers.internal:<port>`) instead.
[terok-shield](https://terok-ai.github.io/terok-shield/) allows the
token broker port through the nftables firewall via `loopback_ports`.

### Per-provider phantom token routing

Each provider gets its own phantom token, created at container launch:

```python
tokens = {
    name: db.create_proxy_token(project, task, credential_set, name)
    for name in routed_providers
}
```

The token broker resolves the route from the token's `provider` field, not from
the URL path. This is essential because some SDKs (Vibe's Mistral SDK,
gh CLI) strip or ignore URL path components.

## Per-Agent Traffic Routing

Different agents reach the token broker in different ways, depending on what
their SDK supports:

| Agent | How it reaches the token broker | Notes |
|-------|-------------------------|-------|
| **Claude** | `ANTHROPIC_BASE_URL=http://host.containers.internal:<port>` | Anthropic SDK respects this env var (default port: 18731) |
| **Codex** | Shared `~/.codex/config.toml` rewrite (`openai_base_url`, `chatgpt_base_url`) | Codex's built-in first-party auth is file/config based, so terok patches the shared Codex config instead of relying on env vars |
| **Vibe** | `config.toml` with `api_base` in shared `~/.vibe` mount | Mistral SDK ignores URL path in api_base, only uses host:port. Written by `shared_config_patch` in YAML |
| **KISSKI** | `TEROK_OC_KISSKI_BASE_URL` env var override | OpenCode reads this; overridden from the real upstream to token broker |
| **Blablador** | `TEROK_OC_BLABLADOR_BASE_URL` env var override | Same pattern as KISSKI |
| **gh** | `http_unix_socket` in `~/.config/gh/config.yml` + socat bridge | gh routes ALL API traffic through a Unix socket. socat bridges it to TCP. See below. |
| **glab** | `GITLAB_API_HOST` + `API_PROTOCOL=http` env vars | glab sends to `http://<api_host>/api/v4/...` |
| **CodeRabbit** | Real API key via sidecar `env_map` | CLI has no base URL override, so token broker routing is not possible. Sidecar receives the real key directly from the credential DB. |
| **SonarCloud** | `SONAR_HOST_URL` + `SONAR_TOKEN` phantom env | Tool agent; scanner uses host URL override |

### gh: socat bridge pattern

gh has no env var for base URL. It supports `http_unix_socket` in its
config file, which routes all API traffic through a Unix socket.

The init script (`init-ssh-and-repo.sh`) starts a socat bridge:

```bash
socat UNIX-LISTEN:/tmp/terok-gh-proxy.sock,fork \
  TCP:host.containers.internal:${TEROK_TOKEN_BROKER_PORT} &
```

The socket is created **inside** the container by the container's own
process. SELinux allows `container_t -> container_t` socket connections.
The TCP hop to the host token broker crosses the container boundary safely.

The `http_unix_socket` path is written to `~/.config/gh/config.yml`
by the `shared_config_patch` mechanism during `terok-executor auth gh`.

### YAML-driven shared_config_patch

Providers that need config file changes (not just env vars) declare a
`shared_config_patch` in their YAML:

```yaml
# Vibe: TOML patch
shared_config_patch:
  file: config.toml
  toml_table: providers
  toml_match: {name: mistral}
  toml_set: {api_base: "{vault_url}/v1"}

# gh: YAML patch
shared_config_patch:
  file: config.yml
  yaml_set: {http_unix_socket: "{vault_socket}"}
```

The patch is applied after auth and reconciled on every task launch.  Only
non-secret values (URLs, socket paths) are written to shared mounts.  The
patcher also writes a `.terok-managed-config.json` sidecar beside the config
file so callers can later disable a provider and remove only values still
owned by terok; user-edited values are preserved.

## Agent YAML Registry

Each agent declares its vault integration in `resources/agents/<name>.yaml`:

```yaml
vault:
  route_prefix: claude           # kept for routes.json generation
  upstream: https://api.anthropic.com
  auth_header: dynamic           # OAuth -> Bearer, API key -> x-api-key
  credential_type: oauth
  credential_file: .credentials.json
  phantom_env:
    ANTHROPIC_API_KEY: true      # env var gets the phantom token
  base_url_env: ANTHROPIC_BASE_URL  # optional: env var for token broker URL
  shared_config_patch: ...       # optional: file patch for token broker URL
```

### Agent-specific settings not in YAML

- **OpenCode base URL override**: For KISSKI and Blablador, the environment
  builder overrides `TEROK_OC_<NAME>_BASE_URL` when the vault is active.
  This is computed at container launch, not declared in YAML.

- **glab env vars**: `GITLAB_API_HOST` and `API_PROTOCOL=http` are injected
  by the environment builder for glab specifically. glab has no YAML field
  for this because it's a routing concern, not a credential concern.

- **socat bridge**: Started by `init-ssh-and-repo.sh` when `TEROK_TOKEN_BROKER_PORT`
  and `GH_TOKEN` are both set. The socket path is hardcoded to
  `/tmp/terok-gh-proxy.sock` (matching the `shared_config_patch`).

## Auth Flow

### Three auth paths

**1. OAuth / interactive login** (Claude, Codex, gh):
Launches a container with the vendor CLI and an empty config directory.
After exit, the extractor captures the OAuth token to the DB.

**2. API key -- interactive prompt** (Vibe, Blablador, KISSKI, glab):
Prompts for an API key on the terminal. No container needed.

**3. API key -- non-interactive** (all providers):
`terok-executor auth <provider> --api-key <key>`

### Post-auth config patching

After storing credentials, `write_vault_config()` applies any
`shared_config_patch` from the YAML registry. This writes token broker URLs
(not secrets) to the provider's shared config mount.  Task launch re-applies
enabled patches and removes disabled provider patches using the managed
sidecar, which keeps global shared config directories from retaining stale
vault routing after a feature mode changes.

## Per-Provider Credential Extractors

| Provider   | File                | Key fields                    |
|------------|---------------------|-------------------------------|
| Claude     | `.credentials.json` | access_token, refresh_token   |
| Codex      | `auth.json`         | access_token, refresh_token   |
| Vibe       | `.env`              | key (MISTRAL_API_KEY)         |
| Blablador  | `config.json`       | key (api_key)                 |
| KISSKI     | `config.json`       | key (api_key)                 |
| gh         | `hosts.yml`         | token (oauth_token)           |
| glab       | `config.yml`        | token (per-host)              |
| CodeRabbit | —                   | API key via `--api-key`       |
| SonarCloud | —                   | API key via `--api-key`       |

## Known Limitations

- **Codex**: ChatGPT/backend-api and realtime websocket traffic are routed
  through the token broker, but any newly added Codex-specific upstream
  surfaces still need explicit vault route coverage.

- **Copilot**: Not proxied yet. No `vault` section in YAML.

## Package Boundaries

- **[terok-sandbox](https://terok-ai.github.io/terok-sandbox/)**: Credential
  DB, token broker (HTTP forwarding, phantom token resolution, OAuth token
  refresh, SSH signer), TCP+Unix listeners, lifecycle management
- **terok-executor** (this package): YAML agent registry, credential extractors,
  auth CLI, container environment wiring (phantom env vars, base URL overrides,
  socat bridges, `shared_config_patch`)
- **[terok](https://terok-ai.github.io/terok/)**: Environment builder, phantom
  token injection for full-stack multi-agent tasks
