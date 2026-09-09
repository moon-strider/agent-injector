"""Run a real CPU model -> Claude Code -> stdio MCP -> filesystem smoke check.

Weights and executables must already be installed. No cloud API is used.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def exercise(args: argparse.Namespace, root: Path, port: int) -> dict:
    fixture = "amber-willow-kite\n"
    expected_bytes = fixture.encode("utf-8")
    (root / "input.txt").write_bytes(expected_bytes)
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LC_ALL", "TMPDIR"}}
    env.update(
        LLM_BASE_URL=f"http://127.0.0.1:{port}",
        LLM_API_KEY="local-smoke-only",
        LLM_MODEL=args.model_name,
        CLAUDE_BIN=args.claude,
        AGENT_WORKING_ROOT=str(root),
        AGENT_MAX_CONCURRENT="1",
        AGENT_OUTPUT_TOKENS="512",
        AGENT_SYSTEM_PROMPT=(
            "You copy files using tools. Call exactly one tool per response. "
            "First call Read, then wait for its tool result before calling Write. "
            "Write the actual source text, without the Read tool's displayed line numbers. "
            "Never invent file contents. After Write succeeds, reply DONE."
        ),
    )
    params = StdioServerParameters(command=sys.executable, args=["-m", "agent_injector"], env=env)
    with (args.log_dir / "mcp.stderr").open("w") as err:
        async with stdio_client(params, errlog=err) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                status = await session.call_tool("llm_status", {})
                prompt = (
                    f"First use Read to read {root / 'input.txt'}. "
                    f"Then use Write to create {root / 'result.txt'} with the same content. "
                    "Do both tool calls. Finally reply DONE. /no_think"
                )
                started = time.monotonic()
                result = await session.call_tool(
                    "llm_run",
                    {
                        "prompt": prompt,
                        "provider": "custom",
                        "allowed_tools": ["Read", "Write"],
                        "required_tools": ["Read", "Write"],
                        "timeout_seconds": 240,
                        "max_turns": 6,
                    },
                )
                output = root / "result.txt"
                observed = output.read_bytes() if output.exists() else None
                report = {
                    "server": initialized.serverInfo.model_dump(mode="json"),
                    "status": status.structuredContent,
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "mcp_result": result.model_dump(mode="json"),
                    "expected_file_content": fixture,
                    "actual_file_content": observed.decode("utf-8", errors="replace")
                    if observed is not None
                    else None,
                    "passed": not result.isError and observed == expected_bytes,
                }
                print(json.dumps(report, indent=2), flush=True)
                return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", required=True)
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--model-name", default="qwen3-1.7b")
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    args.claude = str(Path(args.claude).resolve())
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        str(Path(args.llama_server).resolve()),
        "-m",
        str(args.model_file.resolve()),
        "--alias",
        args.model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        "16384",
        "-t",
        "4",
        "-ngl",
        "0",
        "-np",
        "1",
        "--jinja",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--temp",
        "0.7",
        "--top-p",
        "0.8",
        "--top-k",
        "20",
        "--min-p",
        "0",
        "--seed",
        "42",
        "--no-webui",
        "--api-key",
        "local-smoke-only",
    ]
    (args.log_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (args.log_dir / "llama-server.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 60
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("Model server exited; inspect llama-server.log")
                try:
                    with http.open(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
            with tempfile.TemporaryDirectory(prefix="agent-live-") as directory:
                report = asyncio.run(exercise(args, Path(directory), port))
            with args.model_file.open("rb") as model:
                report["model_sha256"] = hashlib.file_digest(model, "sha256").hexdigest()
            report["model_file"] = args.model_file.name
            report["claude_version"] = subprocess.check_output(
                [args.claude, "--version"], text=True
            ).strip()
            (args.log_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            if not report["passed"]:
                raise SystemExit(1)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
