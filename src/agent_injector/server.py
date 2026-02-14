import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from mcp.server.models import InitializationOptions
from mcp.server import Server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("agent-injector")

MINIMAX_BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.5")

DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TURNS = 50
MAX_CONCURRENT = 10
CLEANUP_INTERVAL = 30
TASK_TTL = 600

tasks: dict[str, dict[str, Any]] = {}
batches: dict[str, list[dict[str, Any]]] = {}

server = Server("agent-injector")


def _load_dotenv():
    env_path = Path(__file__).parent.parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        eq = line.find("=")
        if eq == -1:
            continue
        key = line[:eq].strip()
        value = line[eq + 1:].strip()
        if key not in os.environ:
            os.environ[key] = value


def _build_env(model: str) -> dict[str, str]:
    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = MINIMAX_BASE_URL
    env["ANTHROPIC_API_KEY"] = MINIMAX_API_KEY
    env["ANTHROPIC_AUTH_TOKEN"] = MINIMAX_API_KEY
    env["ANTHROPIC_MODEL"] = model
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    env["DISABLE_PROMPT_CACHING"] = "1"
    return env


def _active_count() -> int:
    return sum(1 for t in tasks.values() if t["status"] == "running")


def _now() -> float:
    return asyncio.get_running_loop().time()


async def _graceful_kill(proc: asyncio.subprocess.Process, timeout_term: int = 5, timeout_kill: int = 3):
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=timeout_term)
    except asyncio.TimeoutError:
        if proc.returncode is None:
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout_kill)
            except asyncio.TimeoutError:
                pass
    except ProcessLookupError:
        pass


async def _spawn_task(
    prompt: str,
    context: str | None = None,
    model: str | None = None,
    working_directory: str | None = None,
    allowed_tools: list[str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> str:
    task_id = uuid.uuid4().hex[:8]
    model = model or MINIMAX_MODEL
    tools = allowed_tools if allowed_tools else ["Read", "Glob", "Grep", "Bash", "Edit", "Write"]

    cwd = working_directory or Path.home().as_posix()
    if not Path(cwd).is_dir():
        raise FileNotFoundError(f"Working directory does not exist: {cwd}")

    full_prompt = f"<context>\n{context}\n</context>\n\n{prompt}" if context else prompt

    args = [
        "claude",
        "-p", full_prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--model", "sonnet",
        "--max-turns", str(max_turns),
        "--dangerously-skip-permissions",
    ]

    if tools:
        args.extend(["--allowedTools", ",".join(tools)])

    env = _build_env(model)

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    task = {
        "id": task_id,
        "status": "running",
        "model": model,
        "prompt": prompt,
        "output": "",
        "stderr_output": "",
        "partial_output": "",
        "exit_code": None,
        "started_at": _now(),
        "completed_at": None,
        "timeout": timeout_seconds,
        "process": proc,
    }
    tasks[task_id] = task

    asyncio.create_task(_read_stream(task, proc, timeout_seconds))

    return task_id


async def _read_stream(task: dict, proc: asyncio.subprocess.Process, timeout: int):
    try:
        async def read_stdout():
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                task["output"] += text
                task["partial_output"] = task["output"][-2000:]

        async def read_stderr():
            assert proc.stderr is not None
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    break
                task["stderr_output"] += chunk.decode("utf-8", errors="replace")

        await asyncio.wait_for(
            asyncio.gather(read_stdout(), read_stderr(), proc.wait()),
            timeout=timeout,
        )

        task["exit_code"] = proc.returncode
        task["status"] = "completed" if proc.returncode == 0 else "failed"

    except asyncio.TimeoutError:
        logger.warning(f"Task {task['id']} timed out after {timeout}s")
        await _graceful_kill(proc)
        task["status"] = "timeout"

    except asyncio.CancelledError:
        await _graceful_kill(proc)
        task["status"] = "cancelled"
        raise

    except Exception as e:
        logger.error(f"Task {task['id']} stream error: {e}")
        await _graceful_kill(proc)
        task["status"] = "failed"
        task["stderr_output"] += f"\nInternal error: {e}"

    finally:
        try:
            task["completed_at"] = _now()
        except RuntimeError:
            task["completed_at"] = task["started_at"]
        task["process"] = None


def _extract_result(raw: str) -> str:
    lines = raw.strip().split("\n")
    for line in reversed(lines):
        try:
            obj = json.loads(line)
            if "result" in obj:
                return obj["result"]
            if obj.get("type") == "result":
                return obj.get("result") or obj.get("text") or json.dumps(obj)
        except (json.JSONDecodeError, TypeError):
            continue
    return raw[-4000:] if raw else ""


def _task_summary(task: dict) -> dict:
    try:
        now = _now()
    except RuntimeError:
        now = task["started_at"]
    return {
        "task_id": task["id"],
        "status": task["status"],
        "model": task["model"],
        "elapsed_seconds": round(now - task["started_at"]),
        "exit_code": task["exit_code"],
    }


async def _cleanup_loop():
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            now = _now()
            to_delete = []
            for tid, task in list(tasks.items()):
                if task["status"] != "running" and task["completed_at"]:
                    if (now - task["completed_at"]) > TASK_TTL:
                        to_delete.append(tid)
            for tid in to_delete:
                del tasks[tid]
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Cleanup loop error: {e}")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="llm_run",
            description=(
                "Run a task using Claude Code with MiniMax as the backend model. "
                "Blocks until completion or timeout. Use for tasks under ~2 minutes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Task description for the agent"},
                    "context": {"type": "string", "description": "Task context generated by the calling agent — background info, relevant data, constraints. Passed to the sub-agent inside <context> tags."},
                    "model": {"type": "string", "description": f"Model name (default: {MINIMAX_MODEL})"},
                    "working_directory": {"type": "string", "description": "Working directory"},
                    "allowed_tools": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Tools the agent can use",
                    },
                    "timeout_seconds": {"type": "number", "description": "Max execution time", "default": 120},
                    "max_turns": {"type": "number", "description": "Max agentic iterations", "default": 30},
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="llm_start",
            description=(
                "Start a background task using Claude Code with MiniMax. "
                "Returns a task_id for polling. Use for tasks that may take longer than 2 minutes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Task description for the agent"},
                    "context": {"type": "string", "description": "Task context generated by the calling agent — background info, relevant data, constraints. Passed to the sub-agent inside <context> tags."},
                    "model": {"type": "string", "description": f"Model name (default: {MINIMAX_MODEL})"},
                    "working_directory": {"type": "string", "description": "Working directory"},
                    "allowed_tools": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Tools the agent can use",
                    },
                    "timeout_seconds": {"type": "number", "description": "Max execution time", "default": 300},
                    "max_turns": {"type": "number", "description": "Max agentic iterations", "default": 50},
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="llm_poll",
            description="Check the status of a running task. Returns status, elapsed time, and partial output.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID from llm_start"},
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="llm_result",
            description="Get the full result of a completed task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID from llm_start"},
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="llm_cancel",
            description="Cancel a running task by killing its process.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to cancel"},
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="llm_batch_start",
            description="Start multiple tasks in parallel. Returns a batch_id for polling all at once.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Unique subtask identifier"},
                                "prompt": {"type": "string"},
                                "context": {"type": "string", "description": "Task context for the sub-agent"},
                                "model": {"type": "string"},
                                "allowed_tools": {"type": "array", "items": {"type": "string"}},
                                "timeout_seconds": {"type": "number"},
                                "max_turns": {"type": "number"},
                            },
                            "required": ["id", "prompt"],
                        },
                    },
                    "working_directory": {"type": "string", "description": "Shared working directory"},
                },
                "required": ["tasks"],
            },
        ),
        Tool(
            name="llm_batch_poll",
            description="Check status of all tasks in a batch.",
            inputSchema={
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "Batch ID from llm_batch_start"},
                },
                "required": ["batch_id"],
            },
        ),
        Tool(
            name="llm_list_tasks",
            description="List all active and recent tasks with their status.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        return await _handle_tool(name, arguments)
    except Exception as e:
        logger.error(f"Tool {name} error: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "llm_run":
        if _active_count() >= MAX_CONCURRENT:
            return [TextContent(type="text", text=json.dumps({"error": "Too many concurrent tasks", "active": _active_count(), "max": MAX_CONCURRENT}))]

        task_id = await _spawn_task(
            prompt=arguments["prompt"],
            context=arguments.get("context"),
            model=arguments.get("model"),
            working_directory=arguments.get("working_directory"),
            allowed_tools=arguments.get("allowed_tools"),
            timeout_seconds=arguments.get("timeout_seconds", 120),
            max_turns=arguments.get("max_turns", 30),
        )

        task = tasks[task_id]
        while task["status"] == "running":
            await asyncio.sleep(0.5)

        result = _extract_result(task["output"])
        return [TextContent(type="text", text=json.dumps({
            **_task_summary(task),
            "result": result,
            "stderr": task["stderr_output"][-1000:] if task["stderr_output"] else None,
        }))]

    elif name == "llm_start":
        if _active_count() >= MAX_CONCURRENT:
            return [TextContent(type="text", text=json.dumps({"error": "Too many concurrent tasks", "active": _active_count(), "max": MAX_CONCURRENT}))]

        task_id = await _spawn_task(
            prompt=arguments["prompt"],
            context=arguments.get("context"),
            model=arguments.get("model"),
            working_directory=arguments.get("working_directory"),
            allowed_tools=arguments.get("allowed_tools"),
            timeout_seconds=arguments.get("timeout_seconds", DEFAULT_TIMEOUT),
            max_turns=arguments.get("max_turns", DEFAULT_MAX_TURNS),
        )

        return [TextContent(type="text", text=json.dumps({"task_id": task_id, "status": "running", "model": tasks[task_id]["model"]}))]

    elif name == "llm_poll":
        task = tasks.get(arguments["task_id"])
        if not task:
            return [TextContent(type="text", text=json.dumps({"error": "Task not found", "task_id": arguments["task_id"]}))]
        return [TextContent(type="text", text=json.dumps({
            **_task_summary(task),
            "partial_output": task["partial_output"][-500:],
        }))]

    elif name == "llm_result":
        task = tasks.get(arguments["task_id"])
        if not task:
            return [TextContent(type="text", text=json.dumps({"error": "Task not found", "task_id": arguments["task_id"]}))]
        if task["status"] == "running":
            return [TextContent(type="text", text=json.dumps({"error": "Task still running. Use llm_poll to check status.", "task_id": arguments["task_id"]}))]
        result = _extract_result(task["output"])
        return [TextContent(type="text", text=json.dumps({
            **_task_summary(task),
            "result": result,
            "stderr": task["stderr_output"][-1000:] if task["stderr_output"] else None,
        }))]

    elif name == "llm_cancel":
        task = tasks.get(arguments["task_id"])
        if not task:
            return [TextContent(type="text", text=json.dumps({"error": "Task not found"}))]
        proc = task["process"]
        if task["status"] == "running" and proc is not None:
            await _graceful_kill(proc)
            task["status"] = "cancelled"
            try:
                task["completed_at"] = _now()
            except RuntimeError:
                task["completed_at"] = task["started_at"]
        return [TextContent(type="text", text=json.dumps({"task_id": task["id"], "status": task["status"]}))]

    elif name == "llm_batch_start":
        batch_id = uuid.uuid4().hex[:8]
        entries = []
        for t in arguments["tasks"]:
            if _active_count() >= MAX_CONCURRENT:
                entries.append({"id": t["id"], "task_id": None, "status": "queued_limit_reached"})
                continue
            tid = await _spawn_task(
                prompt=t["prompt"],
                context=t.get("context"),
                model=t.get("model"),
                working_directory=arguments.get("working_directory"),
                allowed_tools=t.get("allowed_tools"),
                timeout_seconds=t.get("timeout_seconds", DEFAULT_TIMEOUT),
                max_turns=t.get("max_turns", DEFAULT_MAX_TURNS),
            )
            entries.append({"id": t["id"], "task_id": tid, "status": "running"})
        batches[batch_id] = entries
        return [TextContent(type="text", text=json.dumps({"batch_id": batch_id, "tasks": entries}))]

    elif name == "llm_batch_poll":
        batch = batches.get(arguments["batch_id"])
        if not batch:
            return [TextContent(type="text", text=json.dumps({"error": "Batch not found"}))]
        results = []
        for entry in batch:
            if not entry.get("task_id"):
                results.append({"id": entry["id"], "status": entry["status"]})
                continue
            task = tasks.get(entry["task_id"])
            if not task:
                results.append({"id": entry["id"], "status": "not_found"})
                continue
            results.append({"id": entry["id"], **_task_summary(task)})
        all_done = all(r["status"] != "running" for r in results)
        return [TextContent(type="text", text=json.dumps({"batch_id": arguments["batch_id"], "all_completed": all_done, "tasks": results}))]

    elif name == "llm_list_tasks":
        task_list = [_task_summary(t) for t in tasks.values()]
        return [TextContent(type="text", text=json.dumps({"tasks": task_list, "active": _active_count()}))]

    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main_async():
    _load_dotenv()

    global MINIMAX_BASE_URL, MINIMAX_API_KEY, MINIMAX_MODEL
    MINIMAX_BASE_URL = os.environ.get("MINIMAX_BASE_URL", MINIMAX_BASE_URL)
    MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", MINIMAX_API_KEY)
    MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL", MINIMAX_MODEL)

    if not MINIMAX_API_KEY:
        logger.warning("MINIMAX_API_KEY is not set. Tasks will fail without an API key.")

    logger.info(f"Agent Injector starting — model={MINIMAX_MODEL} base_url={MINIMAX_BASE_URL}")

    cleanup_task = asyncio.create_task(_cleanup_loop())

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="agent-injector",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        cleanup_task.cancel()
        for task in tasks.values():
            proc = task.get("process")
            if task["status"] == "running" and proc is not None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
