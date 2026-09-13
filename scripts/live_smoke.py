"""Verify real CPU inference through Claude Code, stdio MCP, and exact file checks.

Weights and executables must already be installed. No cloud API is used.
Each case has a fresh directory and random marker appearing only in file contents.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SYSTEM_PROMPT = (
    "Use tools to perform the task. Call exactly one tool per response. "
    "First read the actual file and wait for its tool result. Never invent contents "
    "or copy the Read tool's displayed line numbers. After changing a file, "
    "Read it again to verify. Then reply DONE."
)
SCENARIOS = ("copy", "edit", "code")


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    mutation_tool: str
    initial: dict[str, bytes]
    expected: dict[str, bytes]


def make_case(name: str) -> Case:
    marker = secrets.token_hex(12)
    if name == "copy":
        source = f"first line\nmarker={marker}\nlast line\n".encode()
        return Case(
            name,
            "Read input.txt. Use Write to create result.txt with exactly the same complete "
            "contents, preserving input.txt. Read adds line-number prefixes: copy only the "
            "file text after each prefix, never the prefixes or an extra empty EOF line. "
            "Preserve the final newline. Read result.txt after writing, then reply DONE.",
            "Write",
            {"input.txt": source},
            {"input.txt": source, "result.txt": source},
        )
    if name == "edit":
        source = f"enabled=true\nmarker={marker}\nretry_limit=3\nmode=careful\n".encode()
        return Case(
            name,
            "Read config.txt. Use Edit to replace only retry_limit=3 with retry_limit=7. "
            "Preserve every other byte. Read config.txt after editing, then reply DONE.",
            "Edit",
            {"config.txt": source},
            {"config.txt": source.replace(b"retry_limit=3", b"retry_limit=7")},
        )
    if name == "code":
        source = f"# marker={marker}\ndef inclusive_sum(n):\n    return sum(range(1, n))\n".encode()
        return Case(
            name,
            "Read sum.py. Fix inclusive_sum so the upper bound is included: use Edit to "
            "replace range(1, n) with range(1, n + 1), preserving every other byte and "
            "the comment. Read sum.py after editing, then reply DONE.",
            "Edit",
            {"sum.py": source},
            {"sum.py": source.replace(b"range(1, n)", b"range(1, n + 1)")},
        )
    raise ValueError(f"Unknown scenario: {name}")


def verify_case(case: Case, actual: dict[str, bytes | None], wire: dict) -> dict:
    """Check external bytes; execute code only after an exact trusted-code match."""
    payload = wire.get("structuredContent") or {}
    calls = [entry.get("name") for entry in payload.get("tool_calls", [])]
    checks = {
        "mcp_success": wire.get("isError") is False,
        "completed": payload.get("status") == "completed",
        "no_task_error": payload.get("error") is None,
        "no_tool_errors": payload.get("tool_error_count") == 0,
        "at_least_three_tool_calls": len(calls) >= 3,
        "read_before_change": bool(calls) and calls[0] == "Read",
        "mutation_tool_used": case.mutation_tool in calls,
        "final_read": bool(calls) and calls[-1] == "Read",
        "exact_files": all(actual.get(name) == value for name, value in case.expected.items()),
    }
    code_result = None
    if case.name == "code":
        checks["code_test_cases"] = False
        # A different model-written program is never executed, even if it claims success.
        if checks["exact_files"]:
            program = actual["sum.py"]
            assert program is not None
            checked = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import json,sys; namespace={}; "
                    "exec(compile(sys.stdin.read(), '<verified-sum>', 'exec'), namespace); "
                    "print(json.dumps([namespace['inclusive_sum'](n) for n in [-2,0,1,2,5]]))",
                ],
                input=program,
                capture_output=True,
                timeout=5,
                check=False,
            )
            code_result = {
                "inputs": [-2, 0, 1, 2, 5],
                "expected": [0, 0, 1, 3, 15],
                "stdout": checked.stdout.decode(errors="replace"),
                "stderr": checked.stderr.decode(errors="replace"),
                "exit_code": checked.returncode,
            }
            checks["code_test_cases"] = (
                checked.returncode == 0 and json.loads(checked.stdout) == code_result["expected"]
            )
    return {"checks": checks, "code_result": code_result, "passed": all(checks.values())}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def exercise(args: argparse.Namespace, case: Case, root: Path, evidence: Path, port: int):
    await asyncio.to_thread(evidence.mkdir)
    for folder, files in [("before", case.initial), ("expected", case.expected)]:
        (evidence / folder).mkdir()
        for name, content in files.items():
            (evidence / folder / name).write_bytes(content)
    for name, content in case.initial.items():
        (root / name).write_bytes(content)
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LC_ALL", "TMPDIR"}}
    env.update(
        LLM_BASE_URL=f"http://127.0.0.1:{port}",
        LLM_API_KEY="local-smoke-only",
        LLM_MODEL=args.model_name,
        CLAUDE_BIN=args.claude,
        AGENT_WORKING_ROOT=str(root),
        AGENT_MAX_CONCURRENT="1",
        AGENT_MAX_TIMEOUT_SECONDS=str(args.timeout),
        AGENT_OUTPUT_TOKENS="512",
        AGENT_SYSTEM_PROMPT=SYSTEM_PROMPT,
    )
    params = StdioServerParameters(command=sys.executable, args=["-m", "agent_injector"], env=env)
    request = {
        "prompt": (
            f"The task working directory is {root}. Resolve relative file names against "
            f"this directory and pass absolute paths to tools.\n{case.prompt}"
        ),
        "provider": "custom",
        "allowed_tools": ["Read", case.mutation_tool],
        "required_tools": ["Read", case.mutation_tool],
        "timeout_seconds": args.timeout,
        "max_turns": 8,
    }
    save_json(evidence / "request.json", {"tool": "llm_run", "arguments": request})
    save_json(
        evidence / "mcp-command.json", {"command": params.command, "args": params.args, "env": env}
    )
    report = {"scenario": case.name, "passed": False}
    wire: dict = {}
    started = time.monotonic()
    try:
        with (evidence / "mcp.stderr").open("w") as err:
            async with stdio_client(params, errlog=err) as (read, write):
                async with ClientSession(read, write) as session:
                    initialized = await session.initialize()
                    status = await session.call_tool("llm_status", {})
                    report["server"] = initialized.server_info.model_dump(
                        mode="json", by_alias=True
                    )
                    report["status"] = status.structured_content
                    inference_started = time.monotonic()
                    result = await session.call_tool(
                        "llm_run", request, read_timeout_seconds=args.timeout + 30
                    )
                    report["elapsed_s"] = round(time.monotonic() - inference_started, 3)
                    wire = result.model_dump(mode="json", by_alias=True)
    except Exception as exc:
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["elapsed_total_s"] = round(time.monotonic() - started, 3)
        save_json(evidence / "mcp-result.json", wire)
        (evidence / "actual").mkdir()
        actual = {}
        for name in case.expected:
            output = root / name
            observed = output.read_bytes() if output.is_file() else None
            actual[name] = observed
            if observed is not None:
                (evidence / "actual" / name).write_bytes(observed)
        save_json(
            evidence / "files.json",
            {
                name: {
                    "expected_sha256": hashlib.sha256(expected).hexdigest(),
                    "actual_sha256": hashlib.sha256(actual[name]).hexdigest()
                    if actual[name] is not None
                    else None,
                }
                for name, expected in case.expected.items()
            },
        )
        try:
            report.update(verify_case(case, actual, wire))
        except Exception as exc:
            report["verification_exception"] = {"type": type(exc).__name__, "message": str(exc)}
            report["passed"] = False
        if "exception" in report:
            report["passed"] = False
        save_json(evidence / "result.json", report)
    print(
        json.dumps(
            {
                "case": evidence.name,
                "passed": report["passed"],
                "elapsed_s": report.get("elapsed_s"),
            }
        ),
        flush=True,
    )
    return {"case": evidence.name, **report}


async def suite(args: argparse.Namespace, port: int) -> list[dict]:
    reports = []
    names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    for run in range(1, args.runs + 1):
        for name in names:
            evidence = args.log_dir / f"run-{run:03d}-{name}"
            with tempfile.TemporaryDirectory(prefix=f"agent-live-{name}-") as directory:
                reports.append(
                    await exercise(args, make_case(name), Path(directory), evidence, port)
                )
            save_json(args.log_dir / "cases.json", reports)
    return reports


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def bounded_timeout(value: str) -> int:
    number = positive(value)
    if number > 1200:
        raise argparse.ArgumentTypeError("must not exceed 1200 seconds")
    return number


def executable_version(command: str) -> str:
    result = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(f"Version check failed for {command}")
    return (result.stdout + result.stderr).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", required=True)
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--model-name", default="qwen3-4b-instruct-2507")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--runs", type=positive, default=1)
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--timeout", type=bounded_timeout, default=300)
    args = parser.parse_args()
    args.claude = str(Path(args.claude).resolve())
    args.llama_server = str(Path(args.llama_server).resolve())
    args.model_file = args.model_file.resolve()
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {"passed": False, "scenario": args.scenario, "runs": args.runs, "cases": []}
    proc = None
    try:
        with args.model_file.open("rb") as model:
            report["model_sha256"] = hashlib.file_digest(model, "sha256").hexdigest()
        report.update(
            model_file=args.model_file.name,
            model_name=args.model_name,
            claude_version=executable_version(args.claude),
            llama_version=executable_version(args.llama_server),
            python_version=sys.version,
            system_prompt=SYSTEM_PROMPT,
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [
            args.llama_server,
            "-m",
            str(args.model_file),
            "--alias",
            args.model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            "16384",
            "-t",
            str(args.threads),
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
        report["server_command"] = command
        save_json(args.log_dir / "command.json", command)
        save_json(args.log_dir / "provenance.json", report)
        with (args.log_dir / "llama-server.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=log)
            http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 120
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("Model server exited; inspect llama-server.log")
                try:
                    with http.open(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Model server startup timed out") from None
                    time.sleep(0.2)
            report["cases"] = asyncio.run(suite(args, port))
            report["passed"] = all(case["passed"] for case in report["cases"])
    except BaseException as exc:
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        save_json(args.log_dir / "summary.json", report)
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"])}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
