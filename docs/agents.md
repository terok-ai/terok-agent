# Agents

## Supported agents

| Agent | Auth | Description |
|-------|------|-------------|
| Claude | OAuth, API key | Anthropic Claude Code |
| Codex | OAuth, API key | OpenAI Codex CLI |
| Vibe | API key | Mistral Vibe |
| Copilot | — | GitHub Copilot (not proxied yet) |
| Blablador | API key | Helmholtz Blablador via OpenCode |
| KISSKI | API key | KISSKI AcademicCloud via OpenCode |

### Sidecar tools

Tools run alongside an agent in a separate container:

| Tool | Auth | Description |
|------|------|-------------|
| gh | OAuth, API key | GitHub CLI |
| glab | API key | GitLab CLI |
| CodeRabbit | API key | CodeRabbit code review |
| SonarCloud | API key | SonarCloud scanner |

## Listing agents

```bash
terok-agent agents            # coding agents only
terok-agent agents --all      # include tools (gh, glab, coderabbit, sonarcloud)
```

## Authentication

Three auth paths depending on the provider:

**OAuth / interactive login** (Claude, Codex, gh) — launches a temporary
container with the vendor CLI. After login, the OAuth token is captured
to the host-side credential database.

```bash
terok-agent auth claude
```

**Interactive API key prompt** (Vibe, Blablador, KISSKI, glab) — prompts
for a key on the terminal. No container needed.

```bash
terok-agent auth vibe
```

**Non-interactive** (all providers) — pass the key directly:

```bash
terok-agent auth gh --api-key ghp_…
```

After authentication, containers receive phantom tokens instead of real
credentials. See [Security](security.md) for how this works.

## Running sidecar tools

Tools like CodeRabbit and SonarCloud run via `run-tool`. Arguments after
`--` are passed to the tool binary:

```bash
terok-agent run-tool coderabbit . -- --pr 42
terok-agent run-tool sonarcloud . --timeout 300
```

## Custom agents

Place YAML files in `~/.config/terok/agent/agents/`. The roster merges
user definitions with bundled ones using deep merge for dicts and
`_inherit` splicing for lists.

See the bundled definitions in `resources/agents/` for the schema:
binary, headless flags, credential proxy routing, auth modes, and git
identity.

## Git identity

By default, agents commit under a built-in AI identity. To use the
host machine's git config instead:

```bash
terok-agent run claude . --git-identity-from-host -p "…"
```

This injects `user.name` and `user.email` from the host's git config
into the container.
