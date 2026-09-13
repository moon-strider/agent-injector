# Local CPU inference

Agent Injector needs an Anthropic-compatible streaming endpoint. Recent
[llama.cpp servers](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
and [Ollama](https://docs.ollama.com/integrations/claude-code) provide this API.
A LiteLLM gateway is another option when the underlying endpoint uses a different
protocol. The audit used llama.cpp directly; no Ollama/LiteLLM installation is required.

## Start a model

Install a suitable llama.cpp release and download a model from its publisher.
Check its license, revision and checksum. For example:

```bash
llama-server -m /absolute/path/to/model.gguf \
  --alias local-model --host 127.0.0.1 --port 8080 \
  -c 16384 -t 4 -ngl 0 -np 1 --jinja \
  --api-key local-test-only
```

Then configure the server:

```bash
export LLM_BASE_URL=http://127.0.0.1:8080
export LLM_MODEL=local-model
export LLM_API_KEY=local-test-only
export AGENT_WORKING_ROOT=/absolute/path/to/workspace
export AGENT_MAX_CONCURRENT=1
uv run --frozen agent-injector --check
```

The model alias must match. A model's chat template and tool-call parser must agree;
valid-looking tool markup in text is not a real tool event. Anthropic compatibility
of the transport does not establish tool competence of every model.

Stock Claude Code prompts/tool descriptions can be large. For a constrained local
experiment, `AGENT_SYSTEM_PROMPT` can replace the system prompt with a short one.
This changes the agent's behavior and must be reported with the result. The smoke
driver uses a short prompt, two explicitly authorized tools and a temporary
synthetic workspace. It does not benchmark stock Claude Code on a real codebase.
When replacing the system prompt, include the actual working directory in the
task or use absolute file paths. A process's current directory is not information
the model can infer. Read output also contains display-only line numbers; the
copy task explicitly tells the model to omit them and preserve the final newline.

The September 13 checks use [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
quantized by [Unsloth](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF):

- File: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (2,497,281,120 bytes).
- Quantization repository revision: `a06e946bb6b655725eafa393f4a9745d460374c9`.
- SHA-256: `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`.
- Runtime: llama.cpp `b10867`, commit `f3f1a8f27`; Claude Code `2.1.266`.

Download the [pinned model file](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q4_K_M.gguf)
and verify its hash before running. This is an Apache-2.0 Qwen model with a
third-party quantization. Use the same model alias in llama.cpp and `LLM_MODEL`.

## Run the complete smoke check

The driver starts the local model process itself, then launches the actual MCP
server and Claude CLI in the same local network namespace. Three scenarios check
multiline copying, a single configuration edit, and a prescribed Python range
correction. Each uses a fresh random marker that appears only in the input file.
The model must Read, Write/Edit, then Read again. External checks compare every
expected file byte, including preserved content. The corrected Python fixture
also runs five function cases, only after matching the predetermined safe source.
This code-edit scenario tests following an explicit correction, not autonomous
bug diagnosis. Missing/wrong content exits nonzero even if the CLI claimed success.

```bash
uv run --frozen python scripts/live_smoke.py \
  --claude /absolute/path/to/claude \
  --llama-server /absolute/path/to/llama-server \
  --model-file /absolute/path/to/model.gguf \
  --model-name local-model \
  --scenario all --runs 3 --threads 4 --timeout 300 \
  --log-dir /tmp/agent-live-check
```

Use a new log directory for each run; existing evidence is never overwritten.
`--scenario copy`, `edit` or `code` selects one scenario; `--runs` repeats it with
new markers. The script records model/runtime versions, checksum, exact commands,
custom system prompt, MCP requests/results, timing, and actual/expected files.
It downloads nothing and uses no cloud keys.
It stops its model process and removes its temporary task workspace afterward.
Logs remain in the explicitly requested directory.

The current smoke command uses the Qwen non-thinking sampling defaults: temperature
0.7, top-p 0.8, top-k 20, min-p 0 and a fixed seed. Models other than Qwen may need
different settings. See the publisher's
[sampling guidance](https://huggingface.co/Qwen/Qwen3-1.7B#best-practices).
The server context is 16384 tokens, with four CPU threads and no GPU layers.

## September 13 model checks

The newer Qwen3-4B configuration passed the original single-line copy task in
43.896 seconds; [the result](validation/20260913/initial-copy.json) records actual
Read/Write events and matching content. A stronger nine-case suite then exposed
failures hidden by that simple task: **3/9 passed** (all three configuration edits).
For relative-path copy/code prompts the model sometimes invented an absolute path;
one multiline copy included Read's display-only line numbers. The
[complete first-suite evidence](validation/20260913/relative-paths.json) retains
every failure, expected/actual file, request, result and the original driver.

The current driver explicitly supplies the temporary working directory and
explains how to remove Read's line-number prefixes. Its file, tool and function
assertions are unchanged. These are operational instructions for the short custom
prompt; they do not modify the generated files after the model runs.

With these instructions, **9/9 fresh cases passed** through actual llama.cpp,
Claude Code and stdio MCP:

| Scenario | Passed | Task seconds, range |
| --- | ---: | ---: |
| Exact multiline copy | 3/3 | 30.777–54.027 |
| Change one configuration value | 3/3 | 29.691–51.585 |
| Prescribed code edit plus five function cases | 3/3 | 31.685–33.526 |

[Full evidence](validation/20260913/explicit-context.json) includes every request,
actual/expected file, observed tool sequence, token usage, result and driver source.
[Checksums](validation/20260913/sha256.json) cover the retained evidence files.
Timings exclude model startup; requests share one warmed CPU server. These are
three repetitions of small controlled tasks with new markers, not a reliability
estimate for arbitrary repositories. The short custom prompt, explicit paths and
copy instructions are part of the tested configuration. The previous failures
remain recorded; stock Claude Code prompts and paid providers were not evaluated.

## September 9 evidence and interpretation

The September 9 audit tested the actual Claude Code **2.1.266** binary and llama.cpp **b10867**
(commit `f3f1a8f27`). The deterministic HTTP integration independently proves that
the real CLI can read/write the synthetic files through Agent Injector.

The previously used NER model was `HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF`,
`smollm2-1.7b-instruct-q4_k_m.gguf`, revision
`2d4a76a30b4af41ecd395c35725ac11688d4cfe4`, SHA-256
`decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33`.
It answered a direct text query, but its original tool smoke returned textual
markup rather than an actual `Read` event. This is not evidence of a usable coding agent.

`Qwen/Qwen3-1.7B-GGUF`, file `Qwen3-1.7B-Q8_0.gguf`, revision
`90862c4b9d2787eaed51d12237eafdfe7c5f6077`, SHA-256
`061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`,
made real tool calls but wrote a placeholder instead of the source content in the
copy-file scenario. The external file assertion caught this. Changing sampling
from greedy to the publisher's non-thinking defaults did not fix that observed
failure. Do not treat small-model chat success, tool-call presence, or a CLI success
flag as a model-quality benchmark.

`Qwen/Qwen2.5-3B-Instruct-GGUF`, file `qwen2.5-3b-instruct-q4_k_m.gguf`, revision
`7dabda4d13d513e3e842b20f0d435c732f172cbe`, SHA-256
`626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d`,
also made successful tool calls but wrote `DONE` into the destination. A second
run with a more explicit sequential prompt still failed the unchanged content
assertion. The same Qwen3-oriented sampling settings were used in these two
experiments; they are not a model comparison or a tuned Qwen2.5 benchmark.

| Model / experiment | Task seconds | Actual destination | Exact-copy check |
| --- | ---: | --- | --- |
| Qwen3 1.7B Q8, greedy | 152.354 | `The content of the file.` | failed |
| Qwen3 1.7B Q8, sampled | 112.440 | `The content of the file.` | failed |
| Qwen2.5 3B Q4, sampled | 48.769 | `DONE` | failed |
| Qwen2.5 3B Q4, explicit sequential prompt | 96.517 | `DONE` | failed |

The expected destination was `amber-willow-kite\n` in every case. Timings exclude
model-server startup and are single observations, not performance benchmarks.
[Recorded result excerpts](validation/cpu-smoke.json) preserve the observed
tool IDs, usage, completion flags and output content, including all four failures.
None of these experiments establishes a dependable small-model coding agent.

See [the audit](audit.md) for the validation summary and CI evidence. Cloud
provider presets were not exercised with paid inference during this audit.
