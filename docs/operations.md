# Operations

## Installation and diagnostics

Install from a reviewed Git revision or build a wheel from the checkout. The
server does not install Claude Code or model weights at startup. Use `CLAUDE_BIN`
when the parent application's PATH does not include the CLI.

```bash
uv sync --frozen
uv run --frozen agent-injector --check
uv run --frozen agent-injector --version
```

`--check` succeeds even when no providers/CLI are available: it reports diagnostic
state, not model readiness. Invalid operator configuration exits with code 2.
An MCP client can retrieve the same diagnostics through `llm_status`.

Use an explicit `AGENT_WORKING_ROOT` when launching from a desktop application.
The root default is the server process's current directory, which may differ from
the repository you intended. Relative task directories resolve inside that root.

The bundled Claude plugin manifest and `.mcp.json` use the same server entry point.
Set provider variables and the workspace root in the launching environment. The
plugin does not embed credentials or automatically connect child search servers.

## Process lifecycle

The server's stdout is exclusively MCP traffic. stderr carries concise lifecycle
logs without task prompts. Pipe stderr to your own log collector if needed; the
server does not manage a persistent daily log directory.

Each task owns its Claude process and POSIX process group. Deadlines, explicit
cancellation and shutdown terminate the group, escalate when needed and reap the
child. A task cancelled before it starts never launches a process. Deliberately
detached descendants are outside this guarantee; see [SECURITY.md](../SECURITY.md).

Synchronous MCP calls must fit the client's own request timeout. Use `llm_start`
for longer work, then poll. A client timing out or disconnecting is not proof that
external changes have been rolled back. Tasks and batches have no disk-backed
persistence or restart recovery.

## Container

The Docker image includes the pinned Claude CLI, the locked Python package and a
non-root user. It does not include model weights.

```bash
docker build -t agent-injector .
docker run --rm agent-injector --check
```

Use `docker run --rm --init -i` as the MCP command, mount the intended workspace at
`/work`, and pass provider environment variables. The workspace must be writable
by UID 10001 for editing tasks. Mount only the project data the task needs.

For a model on the same Linux host, `--network host` lets the container reach an
explicit loopback URL. Otherwise use a reachable HTTPS gateway. The server rejects
unencrypted HTTP URLs using hostnames such as `host.docker.internal`; do not
assume the container's loopback is the host's loopback. No inbound port is exposed
by this stdio image.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `cli_not_found` | Install Claude Code, verify PATH or set `CLAUDE_BIN` |
| Unknown flag / startup failure | Compare CLI version with the tested 2.1.266 release |
| No providers or ambiguous routing | `llm_status`, provider variables, explicit `provider` |
| HTTP/API errors | Use the Anthropic-compatible base URL and appropriate key |
| `capacity_exceeded` | Wait for active tasks, reduce batch size or adjust operator limits |
| `required_tool_missing` | Inspect actual tool calls/results, model tool support and denied operations |
| CLI says DONE but the task is wrong | Check files/results; `completed` is not semantic verification |
| `timeout` before a child starts | Queue time consumed the whole task deadline |
| `output_limit` | Reduce verbose model output or increase the bounded operator limit |
| Task/batch not found | Results expired, were evicted, or the server restarted |
| A request hangs from the client side | Use background calls; ensure the client's timeout is suitable |

Known configured provider keys are redacted from returned output on a best-effort
basis. Do not paste unreviewed tool results or stderr into public issues.
