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

Both **API key** and **OAuth subscription** users are supported — including
Claude Max, Pro, Team, and Enterprise plans.

## Architecture

```text
HOST                                    CONTAINER
─────────────────────────              ─────────────────────────
Credential DB (sqlite3)                 Phantom API keys (env vars)
  ├── credentials table                 ANTHROPIC_API_KEY=<phantom>
  └── proxy_tokens table                ANTHROPIC_BASE_URL=…/claude

Credential Proxy (aiohttp)              Proxy socket (mounted)
  /run/terok/credential-proxy.sock      /run/terok/credential-proxy.sock
  ├── Validates phantom token           Agent → proxy → real auth → upstream
  ├── Routes by path prefix
  ├── Detects credential type
  │   (OAuth → Bearer, API key → x-api-key)
  └── Forwards to upstream (genuine TLS)
```

## Auth Flow

### Three auth paths

**1. OAuth / interactive login** (Claude, Codex, gh):

Launches a container with the vendor CLI (e.g. `claude`) and an empty config
directory.  The CLI goes through its native auth flow (browser redirect, device
code, etc.).  After exit, the extractor captures the OAuth token to the DB.

**2. API key — interactive prompt** (Vibe, Blablador, KISSKI, glab):

Prompts the user for an API key directly on the terminal.  No container needed.

**3. API key — non-interactive** (all providers):

```bash
terok-agent auth <provider> --api-key <key>
```

Stores the key directly in the credential DB.  No container, no prompt.  Useful
for automated workflows and CI.

### Auth mode selection

For providers supporting both OAuth and API key (Claude, Codex, gh):

```text
$ terok-agent auth claude
Authenticate Claude:

  1. OAuth / interactive login (launches container)
  2. API key (paste key, no container needed)

Choose [1/2]:
```

Auth modes are declared in each agent's YAML (`auth.modes: [oauth, api_key]`).

### Auth container lifecycle

1. `terok-agent auth <provider>` creates an **empty** temp directory
2. The temp dir is mounted as the provider's config dir inside the container
3. The auth tool starts with a clean slate — forced re-authentication
4. After exit, per-provider **extractors** parse the credential files
5. Extracted credentials stored in the sqlite **Credential DB**
6. Temp directory deleted — shared config mount never touched

## Dynamic Auth Header Injection

The proxy detects the credential type and selects the correct auth header:

| Credential type | Header | Format |
|----------------|--------|--------|
| OAuth (`type: oauth`) | `Authorization` | `Bearer <access_token>` |
| API key (`type: api_key`) | `x-api-key` | `<key>` |
| OAuth token (`type: oauth_token`) | Route-configured | Route-configured |
| PAT (`type: pat`) | Route-configured | Route-configured |

This is critical for Claude, where OAuth tokens and API keys use different
headers.  The route config declares `auth_header: dynamic` and the proxy
resolves it based on what's stored in the DB.

### Claude OAuth requirements

Claude's API requires two additional signals for OAuth authentication
(discovered by analyzing Claude Code v2.1.83 source):

1. **`anthropic-beta: oauth-2025-04-20`** header — gates OAuth support
2. **`?beta=true`** query parameter on the URL

Both are sent natively by Claude Code — the proxy forwards them transparently
since it passes all non-auth request headers through.

**Verified end-to-end**: OAuth subscription token → credential DB → phantom
token → proxy → `Authorization: Bearer <real_token>` → Anthropic API → success.

## Per-Provider Credential Extractors

Each extractor is a pure function: `Path → dict`.  They parse vendor-specific
credential files and return a normalized dict.

| Provider  | File                   | Extractor                | Key fields           |
|-----------|------------------------|--------------------------|----------------------|
| Claude    | `.credentials.json`    | `extract_claude_oauth`   | access_token, refresh_token |
| Claude    | `config.json`          | (API key fallback)       | key                  |
| Codex     | `auth.json`            | `extract_codex_oauth`    | access_token, refresh_token |
| Vibe      | `.env`                 | `extract_api_key_env`    | key (MISTRAL_API_KEY) |
| Blablador | `config.json`          | `extract_json_api_key`   | key (api_key)        |
| KISSKI    | `config.json`          | `extract_json_api_key`   | key (api_key)        |
| gh        | `hosts.yml`            | `extract_gh_token`       | token (oauth_token)  |
| glab      | `config.yml`           | `extract_glab_token`     | token (per-host)     |

## YAML Registry Extension

Each agent YAML declares a `credential_proxy:` section:

```yaml
credential_proxy:
  route_prefix: claude             # path prefix in proxy (/claude/v1/…)
  upstream: https://api.anthropic.com
  auth_header: dynamic             # auto-detect: OAuth → Bearer, API key → x-api-key
  auth_prefix: ""
  credential_type: oauth
  credential_file: .credentials.json
  phantom_env:
    ANTHROPIC_API_KEY: true        # env var injected with phantom token
  base_url_env: ANTHROPIC_BASE_URL # env var overridden with proxy URL
```

And an `auth:` section with mode declarations:

```yaml
auth:
  host_dir: _claude-config
  container_mount: /home/dev/.claude
  modes: [oauth, api_key]          # what the user can choose
  command: ["claude"]               # container command for OAuth mode
  api_key_hint: "Get your API key at: https://console.anthropic.com/settings/keys"
```

## Bypass Configuration

The credential proxy is enabled by default.  To disable it:

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
