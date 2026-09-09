# Public-readiness audit

Date: 2026-09-09. Starting commit: `fdf61809c11f4d705d6c06e01e780a06305c9220`.
Scope: MCP contracts, provider routing, subprocess execution, cancellation,
resource bounds, credentials, package/plugin metadata, container setup,
documentation and validation. This is a source/runtime review, not a formal
security certification or a model-quality guarantee.

## Findings and changes

| Area | Original behavior | Resolution |
| --- | --- | --- |
| Installation | Unbounded MCP dependency installed SDK 2.2.0; import failed at `list_tools` | Constrain compatible SDK major; locked development/CI environment |
| Tool policy | `--allowedTools` combined with unconditional permission bypass did not restrict available tools; `[]` restored defaults | Exact `--tools` list, no bypass, read-only default, operator ceiling |
| Root startup | Hardcoded bypass flag caused real CLI to refuse root startup | Normal permission enforcement; no root guard workaround |
| Claude startup | `--bare` experiment did not expose Write in tested CLI | Isolated `--safe-mode` startup, explicit tools, no inherited user/project settings |
| Routing | Unknown model fell through to first configured provider | Explicit provider, deterministic defaults, unknown/ambiguous routing rejected |
| Local models | Only vendor presets; setting a key also installed/connected a search server | Generic Anthropic backend; no automatic search MCP dependencies |
| Environment | All parent environment and provider keys inherited | Allowlisted plumbing plus selected provider key, temporary Claude configuration |
| Prompt privacy | Prompt on command line and debug log | Prompt through stdin; lifecycle-only server logs |
| Success detection | Process exit zero meant completed; result errors ignored | Terminal event, error flags, denied permissions and required successful tool results checked |
| MCP errors | Failed calls returned `isError: false` | Structured error codes with correct MCP error flag |
| Concurrency | Capacity checked before awaited spawn, allowing races | Atomic admission and bounded process slots |
| Batches | Extra tasks labeled queued but never executed; partial admission before later validation errors | Validate/admit whole batch; real bounded queue |
| Cancellation | Cancel/stream completion raced over status | Stable terminal state, owned worker tracking, idempotent cleanup |
| Process cleanup | Only direct child terminated; shutdown did not await all cleanup | POSIX process-group cleanup, escalation and reaping; cancellation-safe spawn ownership |
| Memory | Unlimited process output and never-expired batches | Byte caps, bounded result retention, batch eviction and TTL cleanup |
| Timing | Completed task elapsed time kept increasing | Frozen completion times; deadline includes queue time |
| Input validation | Loose numeric values, unknown fields, unbounded text, weak paths | Shared strict Pydantic/MCP contracts and canonical workspace admission |
| Public presentation | Eight files, no tests/CI/license; README overclaimed tested providers and plan requirements | Cohesive docs, license, tests/CI, versioned package/plugin and explicit evidence boundaries |
| Container | No Claude executable; incomplete package installation | Locked package, pinned CLI and non-root runtime with CI smoke |

Before rewriting the runtime, four direct checks failed against the original code:
empty tool lists, permission bypass removal, rejection of unknown model routing,
and fixed elapsed time for completed jobs. Fresh-install failure and the CLI root
failure were also reproduced in the preceding real smoke investigation.

## Validation layers

- Unit and subprocess regressions cover strict configuration, URL/key handling,
  tool policy, queue admission, batches, cancellation, deadlines, descendant
  termination, UTF-8 streaming, output limits, errors, redaction and retention.
- The official SDK starts the actual server over stdio and verifies initialization,
  advertised tools, task calls, successful results and MCP error flags.
- An opt-in integration runs actual Claude Code against a deterministic local
  Anthropic HTTP/SSE fixture. It observes exactly the requested tools, performs
  real Read/Write calls and verifies file content. This fixture does not perform
  LLM inference.
- CPU model checks run actual llama.cpp, actual CLI and actual MCP server, then
  inspect output bytes. Negative results remain failures even if CLI says DONE.
  [Model details and reproduction](local-inference.md).
- Ruff, strict mypy, build/install checks, dependency auditing and secret scanning
  accompany the runtime tests. CI exercises Python 3.11–3.14, macOS, actual Claude
  Code with the local protocol fixture, and the non-root container.

## Compatibility changes in this release

Python 3.11+ is required. The internal Python modules were reorganized; the
supported interface is the MCP tool contract and console entry point. Existing
nine tool names remain; `llm_status`, required successful tool checks, explicit
provider selection and task pagination are added.

Tasks now default to Read/Glob/Grep, 120 seconds and 30 turns. Request writes and
Bash explicitly. Empty tools disables tools. Results are structured and failures
set MCP `isError`. Unknown fields/model routing are rejected. Search servers and
automatic `.env` loading were removed. Default workspace is the process's current
directory, not the account's home. Old `queued_limit_reached` batches are replaced
with atomic bounded queue admission.

## Limits

- Trusted local clients share task results and local account privileges. This is
  not tenant isolation or an OS sandbox.
- Directory checks are admission checks; they cannot prevent later path races,
  unrestricted shell actions or deliberately detached descendants.
- Tasks, batches, rate/capacity accounting and results are per process, in memory.
  There is no durable queue, retry scheduler, resumable conversation or billing ledger.
- A successful tool response does not validate the model's intended semantics.
  Required tools plus file/content assertions are separate checks.
- Short custom prompts and small CPU models are integration experiments, not
  validation of stock Claude Code on a full production repository.
- Cloud inference, provider subscriptions and a published PyPI release are not
  claimed as verified by this audit.
