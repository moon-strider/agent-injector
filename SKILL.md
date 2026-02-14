# Agent Injector — Usage Guide for Claude

You have access to the `agent-injector` MCP server. It spawns Claude Code sub-agents powered by MiniMax M2.5 (or other Anthropic-compatible models). Each sub-agent is a separate `claude -p` process with its own context.

## Critical Rules

1. **NEVER pass `working_directory`** — omit it entirely. The server defaults to the user's home directory. Passing a path that doesn't exist will crash the call.
2. **Keep prompts self-contained** — the sub-agent has no access to your conversation history. Include all necessary context, instructions, and constraints directly in the prompt string.
3. **Don't expect your own tools** — sub-agents have their own limited toolset (Read, Glob, Grep, Bash, Edit, Write by default). They do NOT have access to your MCP servers, browser, or Notion.

## Tool Selection

| Tool | When to use |
|------|-------------|
| `llm_run` | Quick tasks under ~2 minutes. Blocks until done. |
| `llm_start` → `llm_poll` → `llm_result` | Longer tasks. Start, poll periodically, fetch result when done. |
| `llm_batch_start` → `llm_batch_poll` | Multiple independent tasks in parallel. |
| `llm_cancel` | Kill a running task. |
| `llm_list_tasks` | See all active/recent tasks. |

## llm_run — Synchronous Execution

Use for simple, fast tasks. The call blocks until the agent finishes.

```json
{
  "prompt": "Read the file /Users/ilia/project/main.py and list all function names",
  "timeout_seconds": 60,
  "max_turns": 10
}
```

## llm_start / llm_poll / llm_result — Async Execution

Use for tasks that may take longer. Always follow this pattern:

1. Call `llm_start` with the prompt → get `task_id`
2. Wait 10-15 seconds
3. Call `llm_poll` with the `task_id` → check `status`
4. If `status` is `"running"`, wait and poll again
5. When `status` is `"completed"` or `"failed"`, call `llm_result` to get the full output

```json
// Step 1: start
{"prompt": "Analyze the codebase at /Users/ilia/project and write a summary of the architecture"}

// Step 3: poll
{"task_id": "abc12345"}

// Step 5: get result
{"task_id": "abc12345"}
```

## llm_batch_start / llm_batch_poll — Parallel Execution

Use when you have multiple independent research or coding tasks. Each task gets a unique `id`.

```json
{
  "tasks": [
    {"id": "task_a", "prompt": "Research topic A in detail", "max_turns": 15},
    {"id": "task_b", "prompt": "Research topic B in detail", "max_turns": 15},
    {"id": "task_c", "prompt": "Research topic C in detail", "max_turns": 15}
  ]
}
```

Poll with `llm_batch_poll` using the returned `batch_id`. Check `all_completed` field.

## Writing Good Prompts for Sub-Agents

The sub-agent is a fresh Claude Code instance with NO prior context. Your prompt must include everything:

- **What to do** — clear task description
- **Where to look** — full absolute file paths
- **What tools to use** — if you need web search, pass `allowed_tools: ["web_search", "web_fetch"]`
- **Output format** — tell the agent how to structure its response
- **Constraints** — max length, language, focus areas

Bad prompt: "Fix the bug in the auth module"
Good prompt: "Read /Users/ilia/project/src/auth/login.py and fix the bug where the session token is not refreshed after password change. The refresh logic should be in the `refresh_session()` function. Write the fix directly to the file."

## Available `allowed_tools` Values

Default tools (if you don't specify): `Read`, `Glob`, `Grep`, `Bash`, `Edit`, `Write`

Other tools you can pass: `web_search`, `web_fetch`, `TodoRead`, `TodoWrite`

## Limits

- Max 10 concurrent tasks
- Default timeout: 120s for `llm_run`, 300s for `llm_start`
- Both are overridable per-call via `timeout_seconds`
- Completed task results are kept for 10 minutes, then garbage collected
