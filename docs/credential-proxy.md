# Credential Proxy Integration

## Problem

terok bind-mounts vendor config directories into task containers. A
prompt-injected or supply-chain-compromised agent can read and exfiltrate
API keys, OAuth tokens, or SSH private keys from these shared mounts.

## Solution: TCP Credential Proxy

No real secret enters a task container. Instead:

1. **Credential DB** (sqlite3, host-side) stores API keys and OAuth tokens
2. **Credential proxy** (aiohttp, TCP+Unix socket) injects real auth headers
3. **Per-provider phantom tokens** (per-task, per-provider) are what containers see

## Architecture

```text
HOST                                      CONTAINER
-------------------------------           -----------------------------------
Credential DB (sqlite3)                   Per-provider phantom tokens (env vars)
  credentials table                         ANTHROPIC_API_KEY=<claude-phantom>
  proxy_tokens table                        MISTRAL_API_KEY=<vibe-phantom>
    (token, project, task,                  GH_TOKEN=<gh-phantom>
     credential_set, provider)              GITLAB_TOKEN=<glab-phantom>

Credential Proxy (aiohttp)                Agent / tool makes API request
  TCP: 127.0.0.1:18731                      with phantom token in auth header
  Unix: $XDG_RUNTIME_DIR/terok/sock
  1. Extracts phantom token                 Routing is by token, not URL path.
  2. Looks up provider from token           Token encodes which provider it's for.
  3. Loads real credential from DB
  4. Injects real auth header
  5. Forwards to upstream (genuine TLS)
```

### Why TCP, not Unix sockets?

SELinux blocks `connect()` on Unix sockets mounted into rootless Podman
containers. The `connectto` process label check denies `container_t ->
unconfined_t` regardless of file relabeling. This is intentional per
Red Hat's container security model.

TCP via `host.containers.internal:<port>` is the same pattern the gate
server uses. The phantom token provides authentication. Shield's
`loopback_ports` allows the proxy port through the nftables firewall.

See: [Podman #23972](https://github.com/containers/podman/discussions/23972),
[Dan Walsh: SELinux and Containers](https://danwalsh.livejournal.com/78643.html).

### Per-provider phantom token routing

Each provider gets its own phantom token, created at container launch:

```python
tokens = {
    name: db.create_proxy_token(project, task, credential_set, name)
    for name in routed_providers
}
```

The proxy resolves the route from the token's `provider` field, not from
the URL path. This is essential because some SDKs (Vibe's Mistral SDK,
gh CLI) strip or ignore URL path components.

## Per-Agent Traffic Routing

Different agents reach the proxy in different ways, depending on what
their SDK supports:

| Agent | How it reaches the proxy | Notes |
|-------|-------------------------|-------|
| **Claude** | `ANTHROPIC_BASE_URL=http://host.containers.internal:18731` | Anthropic SDK respects this env var |
| **Codex** | `OPENAI_BASE_URL=http://host.containers.internal:18731` | OpenAI SDK respects this env var (deprecated in Codex v0.117, needs config.toml) |
| **Vibe** | `config.toml` with `api_base` in shared `~/.vibe` mount | Mistral SDK ignores URL path in api_base, only uses host:port. Written by `shared_config_patch` in YAML |
| **KISSKI** | `TEROK_OC_KISSKI_BASE_URL` env var override | OpenCode reads this; overridden from the real upstream to proxy |
| **Blablador** | `TEROK_OC_BLABLADOR_BASE_URL` env var override | Same pattern as KISSKI |
| **gh** | `http_unix_socket` in `~/.config/gh/config.yml` + socat bridge | gh routes ALL API traffic through a Unix socket. socat bridges it to TCP. See below. |
| **glab** | `GITLAB_API_HOST` + `API_PROTOCOL=http` env vars | glab sends to `http://<api_host>/api/v4/...` |

### gh: socat bridge pattern

gh has no env var for base URL. It supports `http_unix_socket` in its
config file, which routes all API traffic through a Unix socket.

The init script (`init-ssh-and-repo.sh`) starts a socat bridge:

```bash
socat UNIX-LISTEN:/tmp/terok-gh-proxy.sock,fork \
  TCP:host.containers.internal:${TEROK_PROXY_PORT} &
```

The socket is created **inside** the container by the container's own
process. SELinux allows `container_t -> container_t` socket connections.
The TCP hop to the host proxy crosses the container boundary safely.

The `http_unix_socket` path is written to `~/.config/gh/config.yml`
by the `shared_config_patch` mechanism during `terok-agent auth gh`.

### YAML-driven shared_config_patch

Providers that need config file changes (not just env vars) declare a
`shared_config_patch` in their YAML:

```yaml
# Vibe: TOML patch
shared_config_patch:
  file: config.toml
  toml_table: providers
  toml_match: {name: mistral}
  toml_set: {api_base: "{proxy_url}/v1"}

# gh: YAML patch
shared_config_patch:
  file: config.yml
  yaml_set: {http_unix_socket: "/tmp/terok-gh-proxy.sock"}
```

The patch is applied after auth (`_write_proxy_config` in `auth.py`).
Only non-secret values (URLs, socket paths) are written to shared mounts.

## Agent YAML Registry

Each agent declares its proxy integration in `resources/agents/<name>.yaml`:

```yaml
credential_proxy:
  route_prefix: claude           # kept for routes.json generation
  upstream: https://api.anthropic.com
  auth_header: dynamic           # OAuth -> Bearer, API key -> x-api-key
  credential_type: oauth
  credential_file: .credentials.json
  phantom_env:
    ANTHROPIC_API_KEY: true      # env var gets the phantom token
  base_url_env: ANTHROPIC_BASE_URL  # optional: env var for proxy URL
  shared_config_patch: ...       # optional: file patch for proxy URL
```

### Agent-specific settings not in YAML

- **OpenCode base URL override**: For KISSKI and Blablador, the environment
  builder overrides `TEROK_OC_<NAME>_BASE_URL` when the proxy is active.
  This is computed at container launch, not declared in YAML.

- **glab env vars**: `GITLAB_API_HOST` and `API_PROTOCOL=http` are injected
  by the environment builder for glab specifically. glab has no YAML field
  for this because it's a routing concern, not a credential concern.

- **socat bridge**: Started by `init-ssh-and-repo.sh` when `TEROK_PROXY_PORT`
  and `GH_TOKEN` are both set. The socket path is hardcoded to
  `/tmp/terok-gh-proxy.sock` (matching the `shared_config_patch`).

- **anthropic-beta header**: The proxy appends `oauth-2025-04-20` to the
  `anthropic-beta` header for OAuth credentials. Claude Code sends its own
  beta features in this header, so the proxy must append (not replace).

## Auth Flow

### Three auth paths

**1. OAuth / interactive login** (Claude, Codex, gh):
Launches a container with the vendor CLI and an empty config directory.
After exit, the extractor captures the OAuth token to the DB.

**2. API key -- interactive prompt** (Vibe, Blablador, KISSKI, glab):
Prompts for an API key on the terminal. No container needed.

**3. API key -- non-interactive** (all providers):
`terok-agent auth <provider> --api-key <key>`

### Post-auth config patching

After storing credentials, `_write_proxy_config()` applies any
`shared_config_patch` from the YAML registry. This writes proxy URLs
(not secrets) to the provider's shared config mount.

## Per-Provider Credential Extractors

| Provider  | File               | Key fields                    |
|-----------|--------------------|-------------------------------|
| Claude    | `.credentials.json`| access_token, refresh_token   |
| Codex     | `auth.json`        | access_token, refresh_token   |
| Vibe      | `.env`             | key (MISTRAL_API_KEY)         |
| Blablador | `config.json`      | key (api_key)                 |
| KISSKI    | `config.json`      | key (api_key)                 |
| gh        | `hosts.yml`        | token (oauth_token)           |
| glab      | `config.yml`       | token (per-host)              |

## Known Limitations

- **Codex**: Needs WebSocket support (proxy only handles HTTP), OAuth token
  refresh (Codex refreshes via `auth.openai.com`), and config.toml base URL
  (env var deprecated in v0.117). Filed as bug issues.

- **OAuth token refresh**: Claude and Codex OAuth tokens expire (~1h). The
  proxy does not refresh them automatically. Re-auth required after expiry.

- **Copilot**: Not proxied yet. No `credential_proxy` section in YAML.

- **SSH keys**: Still bind-mounted as files. SSH agent proxy planned (#551).

## Package Boundaries

- **terok-sandbox**: Credential DB, proxy server, TCP+Unix listener, lifecycle
- **terok-agent**: YAML registry, extractors, auth interceptor, runner proxy env
- **terok**: Environment builder, phantom token injection for full-stack tasks
