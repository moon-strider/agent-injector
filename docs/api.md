# Tool API

Tools return matching JSON in MCP `structuredContent` and a text content block.
Schema errors and operation errors set `isError: true`. Failed, timed-out or
cancelled `llm_run`/`llm_result` results do too. Polling and an explicit successful
cancellation operation return normal MCP results describing the task state.

## Task inputs

`llm_run` and `llm_start` share one strict schema:

| Field | Default | Meaning |
| --- | --- | --- |
| `prompt` | Required | Nonblank text, at most 64000 characters |
| `context` | Unset | Optional nonblank context, at most 64000 characters |
| `provider` | Routing configuration | `custom`, `minimax`, or `zai` if configured |
| `model` | Routing configuration | Model override, at most 256 characters |
| `working_directory` | `AGENT_WORKING_ROOT` | Existing directory inside the root; relative paths resolve from root |
| `allowed_tools` | Read/Glob/Grep within operator policy | Unique supported builtin names; empty disables all |
| `required_tools` | `[]` | Tools that must return a successful tool result |
| `timeout_seconds` | `120` | Integer, 1–3600 and within server limit; queue time included |
| `max_turns` | `30` | Integer, 1–1000 and within server limit |

Unknown fields, booleans used as integers, duplicate tool names, invalid names and
null bytes in task text are rejected. No endpoint, credential or arbitrary CLI
argument can be supplied through a task request.

`context` is prepended inside `<context>` tags. It is model input, not a privileged
instruction boundary. Tasks are independent; conversation resumption is not exposed.

## Status and completion

A task starts `queued`, becomes `running` when a slot is available, and ends in
`completed`, `failed`, `timeout` or `cancelled`. A queued task may expire or be
cancelled without starting a child. Terminal status is stable.

Common fields:

- `task_id`, `provider`, `model`, `status`.
- `elapsed_seconds`: time since submission; frozen at completion.
- `execution_seconds`: time since child execution began, zero for unstarted tasks.
- `exit_code`: child exit code if one was observed.
- `error`: null or an object containing `code` and a diagnostic `message`.

`llm_poll` adds `partial_output` (up to 2000 recent assistant-text characters) and
`activity` with the observed assistant message count and last tool name.

`llm_result` requires a terminal task and adds:

- `result`: the terminal CLI text, when available.
- `stderr`: bounded diagnostic tail, or null.
- `usage`: valid nonnegative integer input/output/cache token counts reported by CLI.
- `tool_calls`: up to 256 observed tool names and call IDs; inputs are not retained here.
- `tool_error_count`: observed tool-result error count.
- `turns`: unique observed assistant messages, not a model-quality measure.

A successful terminal CLI event and exit code are required for `completed`.
Permission denials are errors. For each required tool, at least one call must have
a matching non-error tool result. Textual XML/JSON imitations do not qualify.
A tool result that claims success still cannot establish semantic correctness:
check generated files, tests, requested content or other domain-specific outputs.

Token usage is provider-reported, not a billing ledger. Failed/cancelled requests
may have consumed tokens even without a final usage event. Claude's monetary
estimates are not exposed as authoritative local-model costs.

## Background and batch tasks

Use `llm_start`, then `llm_poll` and finally `llm_result`, passing the returned
`task_id`. `llm_cancel` is idempotent and waits for owned process cleanup.
Cancelling a synchronous MCP request also cancels its task. Disconnecting the
server shuts down its owned workers; background tasks are not durable jobs.

Example batch:

```json
{
  "tasks": [
    {"id":"overview","prompt":"Summarize README.md.","allowed_tools":["Read"]},
    {"id":"tests","prompt":"Inspect the test layout.","allowed_tools":["Glob","Read"]}
  ]
}
```

`llm_batch_start` accepts 1–64 tasks with unique `id` labels and an optional shared
`working_directory`. Each item has the task fields above; an item directory takes
precedence over the shared directory. All items are validated before admission.
If the batch does not fit active+queued capacity, the entire batch is rejected.
Accepted waiting tasks execute as slots become available; there is no fictitious
`queued_limit_reached` status and no silently discarded work.

`llm_batch_poll` accepts `batch_id`. `all_completed` means every task is terminal;
`all_succeeded` means every task completed successfully. For compatibility the
former name is retained even though a terminal task may have failed.

`llm_list_tasks` accepts optional `status`, `offset` (default 0) and `limit`
(default 50, maximum 100). It lists newest tasks first and returns `total`,
`active`, `queued` and `next_offset`. Pagination is a current in-memory view;
concurrent additions/completions can change it.

## Errors

| Code | Meaning |
| --- | --- |
| `invalid_arguments` | Input failed the published schema |
| `configuration_error` | Invalid routing, path, tools or operator limit |
| `cli_not_found` / `spawn_failed` | Claude executable unavailable or could not start |
| `capacity_exceeded` | No admission capacity for the request/batch |
| `server_closed` | Server is shutting down |
| `task_not_found` / `batch_not_found` | Unknown or expired identifier |
| `task_not_finished` | Result requested before a terminal state |
| `cli_exit` / `cli_result_error` | Unsuccessful process or terminal CLI result |
| `invalid_cli_output` | Missing or malformed terminal event |
| `permission_denied` | CLI reported a denied permission |
| `required_tool_missing` | A required tool lacks a successful result |
| `output_limit` | stdout/stderr exceeded the configured bound |
| `timeout` | Whole task deadline exceeded |
| `unknown_tool` | Unrecognized MCP tool |
| `internal_error` | Unexpected implementation error; details omitted |
