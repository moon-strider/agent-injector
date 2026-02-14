# Agent Injector

MCP server that spawns [Claude Code](https://claude.com/claude-code) sub-agents powered by alternative Anthropic-compatible models — **MiniMax**, **GLM**, **Moonshot**, and any other provider that implements the Anthropic Messages API.

Instead of using Haiku/Sonnet as sub-agents, delegate tasks to models like MiniMax M2.5 that offer competitive quality (to Opus) at lower cost, while retaining Claude Code's full agentic toolset (Bash, file I/O, MCP servers, skills).

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

MiniMax, GLM and Moonshot.ai (Kimi models) expose **native Anthropic-compatible endpoints** — no gateway or translation layer needed. Agent Injector simply sets `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` to point Claude Code at the alternative provider.

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
        "MINIMAX_MODEL": "MiniMax-M2.5",
        "MINIMAX_BASE_URL": "https://api.minimax.io/anthropic"
      }
    }
  }
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MINIMAX_API_KEY` | Yes | — | Your MiniMax API key |
| `MINIMAX_BASE_URL` | No | `https://api.minimax.io/anthropic` | Anthropic-compatible endpoint |
| `MINIMAX_MODEL` | No | `MiniMax-M2.5` | Default model name |

## Usage Instructions for Claude (you can put that into rules file for project, or just hope the llm discovers how use the mcp itself)

After installing the MCP server, add the following to your **Project Instructions** (Claude Desktop) or **system prompt** so that Claude knows how to use agent-injector correctly:

```
You have access to the "agent-injector" MCP tools (llm_run, llm_start, llm_poll, llm_result, llm_cancel, llm_batch_start, llm_batch_poll, llm_list_tasks). These tools spawn Claude Code sub-agents powered by MiniMax M2.5.

Rules:
- NEVER pass working_directory. Omit it entirely.
- Prompts must be self-contained — the sub-agent has no access to this conversation. Include all context, file paths, and constraints directly in the prompt.
- Sub-agents do NOT have access to your MCP servers, browser, or Notion.

Tool selection:
- llm_run — quick tasks under ~2 minutes, blocks until done
- llm_start → llm_poll → llm_result — longer tasks (start, poll every 10-15s, fetch result when done)
- llm_batch_start → llm_batch_poll — multiple independent tasks in parallel

Default allowed tools: Read, Glob, Grep, Bash, Edit, Write. Override with allowed_tools parameter. For web research pass: ["web_search", "web_fetch"].
```

## Supported Providers

Any provider with an Anthropic Messages API compatible endpoint works. Tested:

| Provider | Base URL | Models |
|----------|----------|--------|
| MiniMax | `https://api.minimax.io/anthropic` | `MiniMax-M2.5`, `MiniMax-M2.5-highspeed` |

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
