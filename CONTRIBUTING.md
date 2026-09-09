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
2. The stdio test starts the installed MCP server and uses the official MCP client
   to initialize, list tools, execute tasks and verify error flags.
3. The opt-in Claude Code test uses the actual installed CLI with a deterministic
   local Anthropic HTTP/SSE fixture. It verifies actual file reads/writes and
   exact tool availability without model weights, cloud keys or cloud charges:

   ```bash
   AGENT_TEST_CLAUDE=/absolute/path/to/claude \
     uv run --frozen pytest tests/test_claude_integration.py
   ```

4. `scripts/live_smoke.py` runs a real CPU model, actual CLI and stdio server, then
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
