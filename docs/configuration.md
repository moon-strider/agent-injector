# Configuration

Configure the process environment or the parent MCP client's `env` section.
The server does not search for `.env` files. `.env.example` is a reference.

## Providers

| Backend name | Key | Base URL | Default model |
| --- | --- | --- | --- |
| `custom` | `LLM_API_KEY` | `LLM_BASE_URL` (required) | `LLM_MODEL` (required) |
| `minimax` | `MINIMAX_API_KEY` | `MINIMAX_BASE_URL`, default `https://api.minimax.io/anthropic` | `MINIMAX_MODEL`, default `MiniMax-M2.5` |
| `zai` | `ZAI_API_KEY` | `ZAI_BASE_URL`, default `https://api.z.ai/api/anthropic` | `ZAI_MODEL`, default `glm-5` |

`custom` can be local or hosted; its name is not a locality guarantee.
MiniMax/Z.AI are enabled only when their key is present. Any `LLM_*` provider field
enables validation of the custom backend. Remote endpoints require a key and
HTTPS. HTTP is accepted only for literal loopback IPs or `localhost`. URLs with
credentials, query strings or fragments are rejected. A keyless loopback backend
receives the placeholder API key `local`.

All endpoints must accept Anthropic Messages. A bare OpenAI-compatible
`/v1/chat/completions` endpoint is insufficient. `LLM_BASE_URL` is the base before
`/v1/messages`; do not append `/v1` unless your gateway explicitly requires that.
Provider presets are configuration conveniences, not a claim about account plans
or successful cloud tests. Confirm access with your provider's current docs.

### Routing

1. A task's `provider` overrides `DEFAULT_PROVIDER`.
2. A task's `model` overrides `DEFAULT_MODEL`, then the selected provider's model.
3. Without a provider, an explicit/default model must uniquely match a configured
   provider's default model.
4. With neither provider nor model, exactly one configured provider is required.

For a model override that is not a configured default, name the provider:

```json
{"prompt":"Explain the input.","provider":"custom","model":"another-model","allowed_tools":[]}
```

Unknown/ambiguous routing is an error. Other providers' keys are not passed to the
child. No provider web search server is installed or launched automatically.

## Server limits

| Variable | Default | Allowed values |
| --- | --- | --- |
| `CLAUDE_BIN` | `claude` | Executable name or absolute path, not a shell command |
| `AGENT_WORKING_ROOT` | Server's current directory | Existing directory, canonicalized |
| `DEFAULT_PROVIDER` | Unset | Configured backend name |
| `DEFAULT_MODEL` | Unset | Model name |
| `AGENT_ALLOWED_TOOLS` | `Read,Glob,Grep,Bash,Edit,Write` | Comma-separated subset; empty disables tools |
| `AGENT_MAX_CONCURRENT` | `4` | 1–32 child processes |
| `AGENT_MAX_QUEUED` | `32` | 0–256 waiting tasks |
| `AGENT_MAX_RETAINED` | `128` | 1–1024 tasks; at least concurrent + queued capacity |
| `AGENT_RETENTION_SECONDS` | `900` | 1–86400 seconds after task completion |
| `AGENT_MAX_TIMEOUT_SECONDS` | `1200` | 1–3600 seconds per task, including queue time |
| `AGENT_MAX_TURNS` | `100` | 1–1000; per-task CLI turn cap |
| `AGENT_MAX_OUTPUT_BYTES` | `1048576` | 4096–16777216 bytes of stdout per task |
| `AGENT_OUTPUT_TOKENS` | `4096` | 16–32000 output tokens per model response |
| `AGENT_SYSTEM_PROMPT` | Unset | Optional replacement system prompt, at most 32000 characters |

Stderr has a separate cap of the smaller of 65536 bytes and
`AGENT_MAX_OUTPUT_BYTES`; only its last 4000 decoded/redacted characters are returned.
Crossing an output cap fails and stops the task. Results are in memory and are
lost on server restart. Older finished tasks may be evicted before their TTL when
capacity is needed. Batches expire when a referenced task expires/is evicted.

If you lower the server timeout/turn limit below the task defaults, pass a matching
smaller value in each task. Task defaults remain 120 seconds and 30 turns.

`AGENT_ALLOWED_TOOLS` is the operator's ceiling. A task's `allowed_tools` selects
from that ceiling. When omitted, tasks get `Read`, `Glob`, `Grep`, intersected with
the ceiling. `[]` means no tools. `required_tools` must be a subset of the task's
available tools and requires corresponding successful tool results.

## Child process isolation

Children use `--safe-mode`, an ephemeral `CLAUDE_CONFIG_DIR`, no user/project/local
settings sources, no discovered MCP servers, no skills, no session persistence,
and an exact `--tools` set. They do not receive a permission bypass flag. Startup
mode is deliberate: tested Claude Code 2.1.266 did not expose `Write` in `--bare`.
Administrator-managed Claude Code policies still apply.

The prompt is sent through stdin instead of command-line arguments. The child
environment retains platform paths, locale, temporary-directory settings, network
proxies and certificate configuration, then adds only the selected provider's
credentials and model settings. This is environment hygiene, not filesystem
isolation. A permitted shell or file tool may still access local account data.
