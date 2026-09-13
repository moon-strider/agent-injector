"""Check the real server and Claude CLI with external mcp-probe and a local HTTP fixture.

This checks protocol and CLI integration. It does not measure model intelligence.
Install mcp-probe[full] separately and pass its executable with --probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from anthropic_fixture import FIXTURE_KEY, FIXTURE_MODEL, AnthropicFixture

PROTOCOLS = ("2025-11-25", "2026-07-28")


def run_probe(
    command: list[str], env: dict[str, str], output: Path, timeout: float = 150
) -> tuple[int | None, str | None]:
    """Let the probe close its transports on timeout, then enforce a final deadline."""
    with (output / "stdout.log").open("w") as stdout:
        with (output / "stderr.log").open("w") as stderr:
            try:
                process = subprocess.Popen(
                    command, env=env, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL
                )
            except OSError as exc:
                return None, f"Probe startup failed: {type(exc).__name__}"
            try:
                return process.wait(timeout=timeout), None
            except subprocess.TimeoutExpired:
                # asyncio.run handles SIGINT by cancelling the main task. The probe's
                # transport context can then reap its MCP server and close its pipes.
                if process.poll() is None:
                    try:
                        if os.name == "posix":
                            process.send_signal(signal.SIGINT)
                        else:
                            process.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return process.returncode, "Probe exceeded the outer process deadline"


def exercise(args: argparse.Namespace, protocol: str) -> dict:
    output = args.output_dir / protocol
    output.mkdir(parents=True)
    source = 'violet-river-cedar\nкириллица "quoted" \\ slash\n'
    expected = source.encode()
    with tempfile.TemporaryDirectory(prefix="agent-probe-") as directory:
        root = Path(directory)
        (root / "input.txt").write_bytes(expected)
        cases = {
            "llm_status": {},
            "readme": {},
            "llm_run": {
                "prompt": "Read input.txt, then copy its exact content to result.txt.",
                "allowed_tools": ["Read", "Write"],
                "required_tools": ["Read", "Write"],
                "timeout_seconds": 30,
                "max_turns": 6,
            },
        }
        case_file = output / "cases.json"
        case_file.write_text(json.dumps(cases, indent=2) + "\n")
        probe_path = output / "probe.json"
        with AnthropicFixture(root, source) as fixture:
            env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LC_ALL", "TMPDIR"}}
            env.update(
                LLM_BASE_URL=fixture.base_url,
                LLM_API_KEY=FIXTURE_KEY,
                LLM_MODEL=FIXTURE_MODEL,
                CLAUDE_BIN=args.claude,
                AGENT_WORKING_ROOT=str(root),
                AGENT_MAX_CONCURRENT="1",
                AGENT_SYSTEM_PROMPT="Follow the task and use the provided tools.",
            )
            command = [
                args.probe,
                shlex.join([sys.executable, "-m", "agent_injector"]),
                "--protocol-version",
                protocol,
                "--cases",
                str(case_file),
                "--active",
                "--strict",
                "--timeout",
                "45",
                "--run-timeout",
                "120",
                "--format",
                "json",
                "--output",
                str(probe_path),
            ]
            (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            return_code, process_error = run_probe(command, env, output)
            target = root / "result.txt"
            observed = target.read_bytes() if target.exists() else None
        (output / "anthropic-requests.json").write_text(
            json.dumps(fixture.requests, indent=2, ensure_ascii=False) + "\n"
        )
        report_error = None
        try:
            probe = json.loads(probe_path.read_text())
            if not isinstance(probe, dict):
                raise ValueError("Expected a JSON object")
        except (OSError, ValueError) as exc:
            probe = {}
            report_error = f"Cannot read the probe report: {type(exc).__name__}"
        checks = {
            check["id"]: check["status"]
            for suite in probe.get("suites", [])
            for check in suite["checks"]
        }
        report = {
            "protocol": protocol,
            "fixture_only": True,
            "probe_exit_code": return_code,
            "process_error": process_error,
            "report_error": report_error,
            "probe_summary": probe.get("summary"),
            "fixture_errors": fixture.errors,
            "anthropic_requests": len(fixture.requests),
            "successful_tool_results": fixture.observed_tools,
            "file_matches": observed == expected,
            "expected_sha256": hashlib.sha256(expected).hexdigest(),
            "observed_sha256": hashlib.sha256(observed).hexdigest()
            if observed is not None
            else None,
            "passed": (
                return_code == 0
                and process_error is None
                and report_error is None
                and probe.get("exit_code") == 0
                and probe.get("spec_version") == protocol
                and probe.get("incomplete") is False
                and all(checks.get(key) == "PASS" for key in ("TOOL-003", "TOOL-004", "TOOL-005"))
                and not fixture.errors
                and len(fixture.requests) == 3
                and fixture.observed_tools == ["Read", "Write"]
                and observed == expected
            ),
        }
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.claude = str(args.claude.resolve())
    args.probe = str(args.probe.resolve())
    args.output_dir = args.output_dir.resolve()
    try:
        args.output_dir.mkdir(parents=True)
    except FileExistsError:
        parser.error("--output-dir already exists; use a fresh directory for each run")
    reports = [exercise(args, protocol) for protocol in PROTOCOLS]
    summary = {
        "claude_version": executable_version(args.claude),
        "probe_version": executable_version(args.probe),
        "protocols": reports,
        "passed": all(report["passed"] for report in reports),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


def executable_version(executable: str) -> str:
    try:
        return subprocess.check_output([executable, "--version"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Unavailable: {type(exc).__name__}"


if __name__ == "__main__":
    main()
