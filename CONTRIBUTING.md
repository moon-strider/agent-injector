# Contributing

Use Python 3.11+ and uv. Work on a branch and keep changes focused.

```bash
uv sync --frozen --group dev
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy src
uv run --frozen pytest --cov --cov-report=term-missing
uv build
uv run --frozen pip-audit --progress-spinner off
```

`uv.lock` pins the development/CI dependency set. The package constrains MCP to
its compatible major version. Update the lockfile deliberately, run the gates,
and include the reason for a dependency change.

## Test layers

1. Ordinary pytest tests use strict input contracts and deterministic CLI fixtures.
   The fixture is a real subprocess but is **not Claude Code or an LLM**.
2. The stdio tests start the installed MCP server with legacy and current official
   MCP clients. They cover initialization, tools, batch queues, results, errors,
   cancellation and process cleanup after a client disconnects.
3. The opt-in Claude Code test uses the actual installed CLI with a deterministic
   local Anthropic HTTP/SSE fixture. It verifies actual file reads/writes and
   exact tool availability without model weights, cloud keys or cloud charges:

   ```bash
   AGENT_TEST_CLAUDE=/absolute/path/to/claude \
     uv run --frozen pytest tests/test_claude_integration.py
   ```

4. `scripts/probe_smoke.py` runs the external mcp-probe CLI against protocols
   `2025-11-25` and `2026-07-28`, including active tool cases and strict checks.
   It uses a deterministic Anthropic fixture and verifies actual Claude Read/Write
   results and exact UTF-8 file bytes. Install the probe in a separate environment:

   ```bash
   uv venv /tmp/agent-probe
   uv pip install --python /tmp/agent-probe/bin/python \
     'mcp-probe[full] @ git+https://github.com/moon-strider/mcp-probe.git@21d435e8ab68b216a98a14a5f147730912edef7c'
   uv run --frozen python scripts/probe_smoke.py \
     --claude /absolute/path/to/claude \
     --probe /tmp/agent-probe/bin/mcp-probe \
     --output-dir /tmp/agent-probe-results
   ```

   Reports include skips for unsupported optional features; no fixture result is
   evidence of model intelligence. CI installs the pinned probe and Claude versions.
5. `scripts/live_smoke.py` runs a real CPU model, actual CLI and stdio server, then
   checks the filesystem result. See [local inference](docs/local-inference.md).
   This is opt-in because downloads and inference are much heavier than unit tests.

Do not turn a failed file/content assertion into a mere exit-code assertion.
A successful CLI process may still have failed the actual task.

## Review expectations

- Cover behavioral regressions, cancellation, resource limits and protocol failures.
- Keep stdout reserved for MCP; lifecycle diagnostics go to stderr.
- Do not add prompts, provider keys, weights, runtime caches or generated sessions to Git.
- Document compatibility changes and the precise limits of real-model evidence.
- Commit subjects use lowercase, at most ten words, and no digits.
- Run the commands above before requesting review.
