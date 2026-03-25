# Credential Proxy Integration

## Problem

terok currently bind-mounts vendor config directories (containing API keys,
OAuth tokens, SSH private keys) into task containers.  A prompt-injected or
supply-chain-compromised agent can read and exfiltrate these secrets to any
allowed domain.

## Solution: Socket-Based Credential Injection

No real secret ever enters a task container.  Instead:

1. **Credential DB** (sqlite3, host-side) stores captured API keys and OAuth tokens
2. **Credential proxy** (aiohttp, Unix socket) injects real auth headers before forwarding to upstream APIs
3. **Phantom tokens** (per-task, worthless outside the proxy) are what containers see

## Architecture

```
HOST                                    CONTAINER
─────────────────────────              ─────────────────────────
Credential DB (sqlite3)                 Phantom API keys (env vars)
  ├── credentials table                 ANTHROPIC_API_KEY=<phantom>
  └── proxy_tokens table                ANTHROPIC_BASE_URL=…/claude

Credential Proxy (aiohttp)              Proxy socket (mounted)
  /run/terok/credential-proxy.sock      /run/terok/credential-proxy.sock
  ├── Validates phantom token           Agent → proxy → real auth → upstream
  ├── Routes by path prefix
  └── Forwards to upstream (genuine TLS)
```

## Auth Flow

### Two kinds of containers

**Auth containers** (ephemeral, one-shot):

1. `terok-agent auth <provider>` creates a temporary container
2. Config dir is mounted to a **temp directory** (not the shared mount)
3. Existing non-secret config (settings, memories) is copied into the temp dir
   so the auth tool has a familiar environment
4. Vendor auth command runs interactively (browser OAuth, API key prompt, etc.)
5. After exit, per-provider **extractors** parse the credential file from the temp dir
6. Extracted credentials are stored in the sqlite **Credential DB**
7. Temp directory is deleted
8. The shared config mount **never sees real credentials**

**Task containers** (long-lived):

1. Shared config dirs mounted as before (settings, memories, plans)
2. Phantom API key env vars injected
3. Base URL env vars point to proxy socket
4. Agent operates in "API key mode" — never touches OAuth, never reads credential files
5. All API calls routed through proxy

### Data flow diagram

```
terok-agent auth claude
         │
         ▼
┌─────────────────────────────┐
│  Temp dir (/tmp/terok-auth-…)│
│  ├── settings.json (copied)  │
│  └── .credentials.json (NEW) │  ← written by Claude's auth flow
└──────────┬──────────────────┘
           │ extract_claude_oauth()
           ▼
┌─────────────────────────────┐
│  Credential DB (sqlite3)     │
│  credentials table:          │
│    default | claude | {…}    │
└─────────────────────────────┘
           │ temp dir deleted
           ▼
  Shared mount (~/.claude/) untouched
  (settings, memories — no secrets)
```

## Per-Provider Credential Extractors

Each extractor is a pure function: `Path → dict`.  They parse vendor-specific
credential files and return a normalized dict with at least one of
`access_token`, `token`, or `key`.

| Provider  | File                   | Extractor                | Key fields           |
|-----------|------------------------|--------------------------|----------------------|
| Claude    | `.credentials.json`    | `extract_claude_oauth`   | access_token, refresh_token |
| Codex     | `auth.json`            | `extract_codex_oauth`    | access_token, refresh_token |
| Vibe      | `.env`                 | `extract_api_key_env`    | key (MISTRAL_API_KEY) |
| Blablador | `config.json`          | `extract_json_api_key`   | key (api_key)        |
| KISSKI    | `config.json`          | `extract_json_api_key`   | key (api_key)        |
| gh        | `hosts.yml`            | `extract_gh_token`       | token (oauth_token)  |
| glab      | `config.yml`           | `extract_glab_token`     | token (per-host)     |

## YAML Registry Extension

Each agent YAML now declares a `credential_proxy:` section:

```yaml
credential_proxy:
  route_prefix: claude             # path prefix in proxy (/claude/v1/…)
  upstream: https://api.anthropic.com
  auth_header: Authorization       # header for the real credential
  auth_prefix: "Bearer "           # prefix before the token value
  credential_type: oauth           # oauth | api_key | oauth_token | pat
  credential_file: .credentials.json
  phantom_env:                     # env vars to inject with phantom tokens
    ANTHROPIC_API_KEY: true
  base_url_env: ANTHROPIC_BASE_URL # env var to override with proxy URL
```

`AgentRegistry.generate_routes_json()` emits the `routes.json` the proxy
server reads — mapping route prefixes to upstream URLs and auth header config.

## Bypass Configuration

The credential proxy is enabled by default.  To disable it (e.g. for debugging
or environments where the proxy can't run), set:

```yaml
credential_proxy:
  bypass_no_secret_protection: true
```

The deliberately long name makes it clear what you're giving up.  When bypassed,
terok falls back to the current shared-mount behavior (real secrets in containers).

## Package Boundaries

- **terok-sandbox**: Credential DB, proxy server, lifecycle management (generic plumbing)
- **terok-agent**: Extractors, YAML registry, auth interceptor (agent-specific knowledge)
- **terok**: Environment integration, phantom token injection, base URL overrides
