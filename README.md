# Agent Injector

MCP server that spawns [Claude Code](https://claude.com/claude-code) sub-agents powered by alternative Anthropic-compatible models — **MiniMax**, **GLM (Z.AI)**, and any other provider that implements the Anthropic Messages API.

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
Claude Code (headless)  →  MiniMax / GLM / ...
    │                        (Anthropic-compatible endpoint)
    │
    ▼
Full agentic loop: Bash, Read, Edit, Write, Glob, Grep, MCP servers...
```

MiniMax and Z.AI (GLM models) expose **native Anthropic-compatible endpoints** — no gateway or translation layer needed. Agent Injector simply sets `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` to point Claude Code at the alternative provider.

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
| `readme` | Get full usage instructions for sub-agents. |

## Installation

Requires Python 3.11+, [uv](https://github.com/astral-sh/uv), and [Claude Code](https://claude.com/claude-code) CLI installed.

### Claude Desktop

Add to your Claude Desktop config:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "agent-injector": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/moon-strider/agent-injector",
        "agent-injector"
      ],
      "env": {
        "MINIMAX_API_KEY": "your-minimax-api-key",
        "ZAI_API_KEY": "your-zai-api-key",
        "DEFAULT_MODEL": "MiniMax-M2.5"
      }
    }
  }
}
```

Set only the API keys for the providers you use. At least one is required.

## Environment Variables

### Providers

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MINIMAX_API_KEY` | No* | — | MiniMax API key |
| `MINIMAX_BASE_URL` | No | `https://api.minimax.io/anthropic` | MiniMax Anthropic-compatible endpoint |
| `MINIMAX_MODEL` | No | `MiniMax-M2.5` | Default model for MiniMax provider |
| `ZAI_API_KEY` | No* | — | Z.AI API key (from z.ai) |
| `ZAI_BASE_URL` | No | `https://api.z.ai/api/anthropic` | Z.AI Anthropic-compatible endpoint |
| `ZAI_MODEL` | No | `glm-5` | Default model for Z.AI provider |
| `DEFAULT_MODEL` | No | First provider's default | Override default model; also determines which provider is used |

*At least one provider API key must be set.

### Server

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `XDG_STATE_HOME` | No | `~/.local/state` | Base directory for log files |

## Usage Instructions for Claude

"Call the agent-injector:readme tool to get full usage instructions"

## Supported Providers

Any provider with an Anthropic Messages API compatible endpoint works. Tested:

| Provider | Base URL | Default Model | Web Search |
|----------|----------|---------------|------------|
| MiniMax | `https://api.minimax.io/anthropic` | `MiniMax-M2.5` | minimax-coding-plan-mcp (stdio MCP) |
| Z.AI | `https://api.z.ai/api/anthropic` | `glm-5` | web_search_prime (HTTP MCP) |

Each provider's web search MCP is automatically configured for sub-agents when the provider's API key is set.

## Architecture

Agent Injector spawns Claude Code in headless mode (`claude -p`) with environment variables that redirect all model tiers (opus, sonnet, haiku) to the configured provider:

```
ANTHROPIC_BASE_URL=<provider endpoint>
ANTHROPIC_API_KEY=<provider api key>
ANTHROPIC_MODEL=<model name>
ANTHROPIC_DEFAULT_OPUS_MODEL=<model name>
ANTHROPIC_DEFAULT_SONNET_MODEL=<model name>
ANTHROPIC_DEFAULT_HAIKU_MODEL=<model name>
DISABLE_PROMPT_CACHING=1
```

Built-in safeguards:

- **Process timeouts** — configurable per task (default 1200s), SIGTERM → SIGKILL escalation
- **Concurrency limits** — max 10 simultaneous tasks
- **Automatic cleanup** — dead tasks garbage-collected every 30s, results expire after 25 min
- **Graceful shutdown** — all child processes terminated on server exit
- **File logging** — daily logs in `~/.local/state/agent-injector/`, auto-rotated after 7 days

## License

MIT
