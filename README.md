# Agent Injector

[![ci](https://github.com/moon-strider/agent-injector/actions/workflows/ci.yml/badge.svg)](https://github.com/moon-strider/agent-injector/actions/workflows/ci.yml)

Run Claude Code tasks through MCP using a local model server or a hosted
Anthropic-compatible provider. Start a task, inspect its progress, cancel it, or
submit a batch to a bounded queue.

Agent Injector manages the processes and checks their results. Claude Code
provides the tools; the selected model decides how to use them.

## What you get

- Synchronous and background tasks, a real batch queue, and paginated task listing.
- Explicit provider selection: custom endpoints, MiniMax, and Z.AI presets.
- Read-only tools by default; explicit opt-in to shell commands and file changes.
- Deadlines including queue time, bounded output and retained results, process cleanup.
- Required tool checks that distinguish model text from actual tool execution.
- Configuration diagnostics without starting a model request.

This is a **trusted local stdio server**, not a hosted multi-user service or an OS
sandbox. Tools run with the local account's permissions. Working-directory checks
restrict where a task starts; they do not isolate filesystem or network access.

## Requirements

- Python **3.11 or newer**, [uv](https://docs.astral.sh/uv/).
- [Claude Code](https://code.claude.com/docs/en/setup), available as `claude` or via `CLAUDE_BIN`.
  The integration suite uses **2.1.266**; older CLIs may lack required flags.
- A model endpoint implementing Anthropic Messages, including streaming and tool use
  for agent tasks. An OpenAI-only endpoint needs a compatible gateway.
- Linux or macOS for process-group cleanup. Windows is not currently supported.

A model that answers chat questions may still fail at tools. Start with the
[local inference guide](docs/local-inference.md) and validate the files or results
you care about.

## Quick start with a local server

First run an Anthropic-compatible local server, such as a recent llama.cpp or
Ollama installation. Then configure Agent Injector in your shell:

```bash
export LLM_BASE_URL=http://127.0.0.1:8080
export LLM_MODEL=local-model
export AGENT_WORKING_ROOT=/absolute/path/to/your/workspace

uvx --from git+https://github.com/moon-strider/agent-injector agent-injector --check
```

`--check` prints configured providers, executable availability and limits. It does
not contact a model. The custom backend is called `custom`. Local API keys are
optional for explicit loopback URLs; set `LLM_API_KEY` if your server requires one.

Add the server to your MCP client's configuration (replace the workspace path):

```json
{
  "mcpServers": {
    "agent-injector": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/moon-strider/agent-injector", "agent-injector"],
      "env": {
        "LLM_BASE_URL": "http://127.0.0.1:8080",
        "LLM_MODEL": "local-model",
        "AGENT_WORKING_ROOT": "/absolute/path/to/your/workspace"
      }
    }
  }
}
```

For repeatable installations, append `@<reviewed-commit-sha>` to the Git URL.
This guide installs from Git; it does not assume a PyPI release exists.

From the MCP client, call `llm_run` with:

```json
{
  "prompt": "Read README.md and summarize the installation steps.",
  "provider": "custom",
  "allowed_tools": ["Read"],
  "required_tools": ["Read"],
  "timeout_seconds": 120
}
```

To authorize changes, explicitly request the relevant tools. For example,
`"allowed_tools": ["Read", "Edit", "Write"]` enables file editing. Use
`"allowed_tools": []` for a text-only task. Shell execution requires `"Bash"`.

## Hosted providers

Export the key for the provider you use, then run the same MCP command:

```bash
export MINIMAX_API_KEY=your-key
export DEFAULT_PROVIDER=minimax
# Or: ZAI_API_KEY and DEFAULT_PROVIDER=zai
```

The presets use MiniMax's `/anthropic` endpoint and Z.AI's `/api/anthropic`
endpoint. You can override their base URL and model. Account eligibility, model
availability and billing are determined by the provider; cloud inference was not
used to validate this release.

Set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` for another HTTPS endpoint.
When several providers are configured, set `DEFAULT_PROVIDER` or pass `provider`
in each task. Unknown model names never silently fall through to another backend.

## MCP tools

| Tool | Purpose |
| --- | --- |
| `llm_status` | Configuration, available providers and execution limits |
| `llm_run` | Submit a task and wait for its terminal result |
| `llm_start` | Submit a task and receive its ID immediately |
| `llm_poll` | Status, elapsed time and recent assistant activity |
| `llm_result` | Terminal result, token usage and observed tool calls |
| `llm_cancel` | Cancel a queued/running task and wait for cleanup |
| `llm_batch_start` | Validate and admit a whole batch atomically |
| `llm_batch_poll` | Inspect retained batch tasks and their outcomes |
| `llm_list_tasks` | Filter and paginate active/recent tasks |
| `readme` | In-session usage instructions |

A CLI exit code of zero alone is insufficient. Agent Injector checks the terminal
result, denied permissions and required tools. `completed` still does not establish
that a model's reasoning or generated code is correct. Check the actual output.

## Documentation

- [Configuration](docs/configuration.md): all environment variables, routing and defaults.
- [Tool API](docs/api.md): inputs, results, statuses, queueing and errors.
- [Local inference](docs/local-inference.md): model setup and real smoke checks.
- [Operations](docs/operations.md): installation, containers and troubleshooting.
- [Audit](docs/audit.md): findings, validation evidence and remaining limitations.
- [Contributing](CONTRIBUTING.md) and [security](SECURITY.md).

## Development

```bash
git clone https://github.com/moon-strider/agent-injector
cd agent-injector
uv sync --frozen --group dev
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen mypy src
```

Regular tests use deterministic subprocess and protocol fixtures. The real Claude
Code integration and local model smoke are separate opt-in checks, documented in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE). Claude Code and model weights have their own licenses and terms.
