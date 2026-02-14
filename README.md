# Agent Injector

MCP server that spawns [Claude Code](https://claude.com/claude-code) sub-agents powered by alternative Anthropic-compatible models — **MiniMax**, **GLM**, and any other provider that implements the Anthropic Messages API.

Instead of using Haiku/Sonnet as sub-agents, delegate tasks to models like MiniMax M2.5 or GLM-5 that offer competitive quality at lower cost, while retaining Claude Code's full agentic toolset (Bash, file I/O, MCP servers, skills).

## How It Works

```
Claude Desktop / Claude Code (main session)
    │
    │  MCP tool call
    ▼
Agent Injector (this MCP server)
    │
    │  spawns claude -p ... with swapped env vars
    ▼
Claude Code (headless)  →  MiniMax / GLM / Moonshot
    │                        (Anthropic-compatible endpoint)
    │
    ▼
Full agentic loop: Bash, Read, Edit, Write, Glob, Grep, MCP servers...
```

Both MiniMax and GLM expose **native Anthropic-compatible endpoints** — no gateway or translation layer needed. Agent Injector simply sets `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` to point Claude Code at the alternative provider.

## Tools

| Tool | Description |
|------|-------------|
| `llm_run` | Run a task synchronously (blocks until done). Best for tasks under ~2 min. |
| `llm_start` | Start a background task, get a `task_id` for polling. |
| `llm_poll` | Check status of a running task. |
| `llm_result` | Get the full result of a completed task. |
| `llm_cancel` | Kill a running task. |
| `llm_batch_start` | Start multiple tasks in parallel, get a `batch_id`. |
| `llm_batch_poll` | Check status of all tasks in a batch at once. |
| `llm_list_tasks` | List all active and recent tasks. |

## Installation

Requires Python 3.10+, [uv](https://github.com/astral-sh/uv), and [Claude Code](https://claude.com/claude-code) CLI installed.

### As a Plugin (Claude Desktop / Cowork) — Recommended

The project is packaged as a Claude Desktop plugin that bundles the MCP server and a skill teaching Claude the correct usage patterns.

1. Clone the repo: `git clone https://github.com/moon-strider/agent-injector.git`
2. Open `.mcp.json` in the project root and replace the placeholder values with your actual credentials:
   - `${MINIMAX_API_KEY}` → your MiniMax API key
   - `${MINIMAX_MODEL}` → model name (e.g. `MiniMax-M2.5`)
   - `${MINIMAX_BASE_URL}` → endpoint URL (e.g. `https://api.minimax.io/anthropic`)
3. Zip the project folder into a `.zip` file
4. In Claude Desktop, go to the **Cowork** tab → upload `agent-injector.zip` manually
5. The plugin registers both the MCP server and the usage skill automatically

### Claude Code

```bash
claude mcp add agent-injector \
  -- uvx --from git+https://github.com/moon-strider/agent-injector agent-injector
```

## Supported Providers

Any provider with an Anthropic Messages API compatible endpoint works. Tested:

| Provider | Base URL | Models |
|----------|----------|--------|
| MiniMax | `https://api.minimax.io/anthropic` | `MiniMax-M2.5`, `MiniMax-M2.5-highspeed` |

## Plugin Components

| Component | Description |
|-----------|-------------|
| MCP Server | `agent-injector` — spawns Claude Code sub-agents via `claude -p` |
| Skill | `agent-injector` — teaches Claude the correct usage patterns, tool selection, prompt writing |

## Architecture

Agent Injector spawns Claude Code in headless mode (`claude -p`) with environment variables that redirect all model tiers (opus, sonnet, haiku) to the configured provider:

```
ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic
ANTHROPIC_AUTH_TOKEN=<your-api-key>
ANTHROPIC_DEFAULT_OPUS_MODEL=MiniMax-M2.5
ANTHROPIC_DEFAULT_SONNET_MODEL=MiniMax-M2.5
ANTHROPIC_DEFAULT_HAIKU_MODEL=MiniMax-M2.5
DISABLE_PROMPT_CACHING=1
```

Built-in safeguards:

- **Process timeouts** — configurable per task (default 300s), SIGTERM → SIGKILL escalation
- **Concurrency limits** — max 10 simultaneous tasks
- **Automatic cleanup** — dead tasks garbage-collected every 30s, results expire after 10 min
- **Graceful shutdown** — all child processes terminated on server exit

## License

MIT
