---
name: agent-injector
description: >
  This skill should be used when the user asks to "run a task with MiniMax",
  "spawn a sub-agent", "run tasks in parallel", "use agent-injector",
  "delegate to another model", or needs to offload work to Claude Code
  sub-agents powered by alternative Anthropic-compatible models.
version: 0.1.0
---

# Agent Injector

Spawn Claude Code sub-agents powered by MiniMax M2.5 (or other Anthropic-compatible models). Each sub-agent runs as an independent `claude -p` process with its own context window.

## Critical Rules

- NEVER pass `working_directory` — omit it entirely, the server defaults to the user's home directory. Passing a nonexistent path crashes the call.
- Keep prompts self-contained — the sub-agent has zero access to the current conversation. Include all context, file paths, instructions, and constraints directly in the prompt.
- Sub-agents do NOT share your MCP servers, browser, or Notion access. They have their own toolset.

## Tool Selection

Use `llm_run` for quick tasks under ~2 minutes (blocks until done).
Use `llm_start` → `llm_poll` → `llm_result` for longer tasks (async pattern).
Use `llm_batch_start` → `llm_batch_poll` for multiple independent tasks in parallel.
Use `llm_cancel` to kill a stuck task, `llm_list_tasks` to see all active/recent tasks.

## Async Pattern

1. Call `llm_start` with the prompt — receive a `task_id`
2. Wait 10-15 seconds
3. Call `llm_poll` with `task_id` — check `status` field
4. If `"running"`, wait and poll again
5. When `"completed"` or `"failed"`, call `llm_result` to get the full output

## Batch Pattern

Call `llm_batch_start` with an array of tasks, each having a unique `id` and `prompt`. Receive a `batch_id`. Poll with `llm_batch_poll` — check the `all_completed` field.

## Writing Effective Prompts

The sub-agent is a fresh Claude Code instance with NO prior context.

Bad: "Fix the bug in the auth module"
Good: "Read /Users/ilia/project/src/auth/login.py and fix the bug where the session token is not refreshed after password change. The refresh logic should be in the refresh_session() function. Write the fix directly to the file."

Always include: what to do, full absolute paths, output format expectations, constraints.

## Allowed Tools

Default (if omitted): Read, Glob, Grep, Bash, Edit, Write

Override by passing `allowed_tools` array. Other options: web_search, web_fetch, TodoRead, TodoWrite.

## Limits

- Max 10 concurrent tasks
- Default timeout: 120s for llm_run, 300s for llm_start (both overridable via timeout_seconds)
- Task results kept in memory for 10 minutes after completion
