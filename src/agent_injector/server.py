import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from mcp.server.models import InitializationOptions
from mcp.server import Server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

LOG_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "agent-injector"
LOG_RETENTION_DAYS = 7

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("agent-injector")


def _setup_file_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logger.info(f"Log file: {log_file}")
    _rotate_logs()


def _rotate_logs():
    cutoff = time.time() - LOG_RETENTION_DAYS * 86400
    for f in LOG_DIR.glob("*.log"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                logger.debug(f"Rotated old log: {f.name}")
        except OSError:
            pass


@dataclass
class Provider:
    name: str
    api_key: str
    base_url: str
    default_model: str
    mcp_servers: dict = field(default_factory=dict)
    known_models: list[str] = field(default_factory=list)


PROVIDERS: dict[str, Provider] = {}
DEFAULT_MODEL: str | None = None

DEFAULT_TIMEOUT = 1200
DEFAULT_MAX_TURNS = 150
MAX_CONCURRENT = 10
CLEANUP_INTERVAL = 30
TASK_TTL = 1500

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


def _build_minimax_mcp(api_key: str, base_url: str) -> dict:
    api_host = base_url.replace("/anthropic", "") if base_url.endswith("/anthropic") else base_url
    return {
        "minimax-search": {
            "command": "uvx",
            "args": ["minimax-coding-plan-mcp", "-y"],
            "env": {
                "MINIMAX_API_KEY": api_key,
                "MINIMAX_API_HOST": api_host,
            },
        }
    }


def _build_zai_mcp(api_key: str, base_url: str) -> dict:
    return {
        "zai-search": {
            "type": "http",
            "url": "https://api.z.ai/api/mcp/web_search_prime/mcp",
            "headers": {
                "Authorization": f"Bearer {api_key}",
            },
        }
    }


PROVIDER_DEFAULTS = {
    "minimax": {
        "key_var": "MINIMAX_API_KEY",
        "url_var": "MINIMAX_BASE_URL",
        "model_var": "MINIMAX_MODEL",
        "default_url": "https://api.minimax.io/anthropic",
        "default_model": "MiniMax-M2.5",
        "known_models": ["MiniMax-M2.5", "MiniMax-M2.5-highspeed"],
        "mcp_builder": _build_minimax_mcp,
    },
    "zai": {
        "key_var": "ZAI_API_KEY",
        "url_var": "ZAI_BASE_URL",
        "model_var": "ZAI_MODEL",
        "default_url": "https://api.z.ai/api/anthropic",
        "default_model": "glm-5",
        "known_models": ["glm-5", "glm-4.7", "glm-4.6", "glm-4.5-air"],
        "mcp_builder": _build_zai_mcp,
    },
}


def _init_providers():
    global DEFAULT_MODEL
    for name, cfg in PROVIDER_DEFAULTS.items():
        api_key = os.environ.get(cfg["key_var"], "")
        if not api_key:
            continue
        base_url = os.environ.get(cfg["url_var"], cfg["default_url"])
        model = os.environ.get(cfg["model_var"], cfg["default_model"])
        mcp_servers = cfg["mcp_builder"](api_key, base_url)
        PROVIDERS[name] = Provider(
            name=name,
            api_key=api_key,
            base_url=base_url,
            default_model=model,
            mcp_servers=mcp_servers,
            known_models=cfg["known_models"],
        )
        logger.info(f"Provider '{name}' registered: model={model} base_url={base_url}")

    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL")
    if DEFAULT_MODEL:
        logger.info(f"DEFAULT_MODEL={DEFAULT_MODEL}")

    if not PROVIDERS:
        logger.warning("No provider API keys set. Tasks will fail.")


def _resolve_provider(model: str | None = None) -> tuple[Provider, str]:
    if not PROVIDERS:
        raise RuntimeError("No providers configured. Set at least one API key (MINIMAX_API_KEY or ZAI_API_KEY).")

    target_model = model or DEFAULT_MODEL

    if target_model:
        for provider in PROVIDERS.values():
            if target_model == provider.default_model or target_model in provider.known_models:
                return provider, target_model
        first = next(iter(PROVIDERS.values()))
        logger.debug(f"Model '{target_model}' not recognized, routing to provider '{first.name}'")
        return first, target_model

    first = next(iter(PROVIDERS.values()))
    return first, first.default_model


def _build_env(provider: Provider, model: str) -> dict[str, str]:
    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = provider.base_url
    env["ANTHROPIC_API_KEY"] = provider.api_key
    env["ANTHROPIC_AUTH_TOKEN"] = provider.api_key
    env["ANTHROPIC_MODEL"] = model
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    env["DISABLE_PROMPT_CACHING"] = "1"
    return env


def _build_mcp_config(provider: Provider) -> str | None:
    if not provider.mcp_servers:
        return None
    return json.dumps({"mcpServers": provider.mcp_servers})


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
    provider, resolved_model = _resolve_provider(model)
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
        "--disallowedTools", "",
    ]

    mcp_config = _build_mcp_config(provider)
    if mcp_config:
        args.extend(["--mcp-config", mcp_config])

    if tools:
        args.extend(["--allowedTools", ",".join(tools)])

    env = _build_env(provider, resolved_model)

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    logger.info(f"Task {task_id} spawned: provider={provider.name} model={resolved_model} cwd={cwd} max_turns={max_turns} timeout={timeout_seconds}s")
    logger.debug(f"Task {task_id} prompt: {prompt[:200]}")

    task = {
        "id": task_id,
        "status": "running",
        "model": resolved_model,
        "provider": provider.name,
        "prompt": prompt,
        "output": "",
        "stderr_output": "",
        "partial_output": "",
        "exit_code": None,
        "started_at": _now(),
        "completed_at": None,
        "timeout": timeout_seconds,
        "process": proc,
        "activity": {"summary": "Initializing...", "tool": None, "turns": 0, "last_message": ""},
        "_line_buf": "",
    }
    tasks[task_id] = task

    asyncio.create_task(_read_stream(task, proc, timeout_seconds))

    return task_id


def _parse_activity(task: dict, line: str):
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return

    msg_type = obj.get("type")
    activity = task["activity"]

    if msg_type == "assistant":
        activity["turns"] += 1
        message = obj.get("message", {})
        content_blocks = message.get("content", [])
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    activity["last_message"] = text[:200]
                    activity["summary"] = f"Turn {activity['turns']}: thinking..."
            elif isinstance(block, dict) and block.get("type") == "tool_use":
                tool_name = block.get("name", "unknown")
                activity["tool"] = tool_name
                tool_input = block.get("input", {})
                detail = ""
                if tool_name in ("Read", "Glob", "Grep") and "path" in tool_input:
                    detail = f" {tool_input['path']}"
                elif tool_name == "Edit" and "file_path" in tool_input:
                    detail = f" {tool_input['file_path']}"
                elif tool_name == "Write" and "file_path" in tool_input:
                    detail = f" {tool_input['file_path']}"
                elif tool_name == "Bash" and "command" in tool_input:
                    cmd = tool_input["command"]
                    detail = f" `{cmd[:80]}`"
                elif tool_name == "WebSearch" and "query" in tool_input:
                    detail = f" \"{tool_input['query'][:80]}\""
                elif tool_name == "WebFetch" and "url" in tool_input:
                    detail = f" {tool_input['url'][:80]}"
                activity["summary"] = f"Turn {activity['turns']}: {tool_name}{detail}"

    elif msg_type == "tool_use":
        tool_name = obj.get("name") or obj.get("tool", "unknown")
        activity["tool"] = tool_name
        activity["summary"] = f"Turn {activity['turns']}: using {tool_name}"

    elif msg_type == "tool_result":
        activity["summary"] = f"Turn {activity['turns']}: processing {activity.get('tool', 'tool')} result"

    elif msg_type == "result":
        activity["summary"] = "Completed"


async def _read_stream(task: dict, proc: asyncio.subprocess.Process, timeout: int):
    try:
        async def read_stdout():
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    if task["_line_buf"]:
                        _parse_activity(task, task["_line_buf"])
                        task["_line_buf"] = ""
                    break
                text = chunk.decode("utf-8", errors="replace")
                task["output"] += text
                task["partial_output"] = task["output"][-2000:]
                task["_line_buf"] += text
                while "\n" in task["_line_buf"]:
                    line, task["_line_buf"] = task["_line_buf"].split("\n", 1)
                    line = line.strip()
                    if line:
                        _parse_activity(task, line)

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
        logger.info(f"Task {task['id']} {task['status']} provider={task['provider']} exit_code={proc.returncode} turns={task['activity']['turns']}")
        if task["status"] == "failed" and task["stderr_output"]:
            logger.error(f"Task {task['id']} stderr: {task['stderr_output'][-500:]}")

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


def _get_default_model_description() -> str:
    if DEFAULT_MODEL:
        return DEFAULT_MODEL
    if PROVIDERS:
        first = next(iter(PROVIDERS.values()))
        return first.default_model
    return "auto"


def _build_usage_instructions() -> str:
    providers_info = []
    for p in PROVIDERS.values():
        search_info = "web search MCP available" if p.mcp_servers else "no web search MCP"
        providers_info.append(f"  - {p.name}: model={p.default_model}, {search_info}")
    providers_block = "\n".join(providers_info) if providers_info else "  (none configured)"

    default_model = _get_default_model_description()

    return f"""You have access to the "agent-injector" MCP tools. These tools spawn Claude Code sub-agents powered by alternative model backends.

Active providers:
{providers_block}

Default model: {default_model}

Available tools:
- llm_run — Run a task synchronously (blocks until done). Best for quick tasks under ~2 min.
- llm_start — Start a background task, returns a task_id for polling.
- llm_poll — Check status and current activity of a running task.
- llm_result — Get the full result of a completed task.
- llm_cancel — Kill a running task.
- llm_batch_start — Start multiple tasks in parallel, get a batch_id.
- llm_batch_poll — Check status of all tasks in a batch at once.
- llm_list_tasks — List all active and recent tasks.

Rules:
- NEVER pass working_directory. Omit it entirely.
- Sub-agents do NOT have access to your MCP servers, browser, or Notion.
- ALWAYS pass the "context" parameter with a summary of relevant information the sub-agent needs. The sub-agent has NO access to this conversation — you must generate context yourself.

Context parameter:
- Every call to llm_run, llm_start, or llm_batch_start accepts an optional "context" string.
- Use it to pass background info, file contents, constraints, prior decisions — anything the sub-agent needs to do its job.
- The context is injected into the sub-agent's prompt inside <context> tags before the task itself.
- Keep context concise but sufficient. Summarize — don't dump entire conversations.
- Example: if the user asks to refactor a file, read the file yourself first, then pass its contents (or a summary) as context along with the refactoring instructions as the prompt.

Model parameter:
- Pass "model" to override the default model. The provider is auto-detected from the model name.
- Available models depend on which provider API keys are configured.

Tool selection:
- llm_run — quick tasks under ~2 minutes, blocks until done
- llm_start → llm_poll → llm_result — longer tasks (start, poll every 10-15s, fetch result when done)
- llm_batch_start → llm_batch_poll — multiple independent tasks in parallel

Polling:
- llm_poll returns an "activity" object with: summary (what the agent is doing right now), tool (current tool), turns (how many turns completed), last_message (agent's last thought).
- Use this to monitor progress without waiting for completion.

Default allowed tools for sub-agents: Read, Glob, Grep, Bash, Edit, Write. Override with allowed_tools parameter.
Sub-agents also have access to provider-specific web search MCP tools by default (when available for the selected provider)."""


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
        "provider": task.get("provider", "unknown"),
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
    default_model = _get_default_model_description()
    return [
        Tool(
            name="llm_run",
            description=(
                "Run a task using Claude Code with an alternative model backend. "
                "Blocks until completion or timeout. Use for tasks under ~2 minutes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Task description for the agent"},
                    "context": {"type": "string", "description": "Task context generated by the calling agent — background info, relevant data, constraints. Passed to the sub-agent inside <context> tags."},
                    "model": {"type": "string", "description": f"Model name (default: {default_model})"},
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
                "Start a background task using Claude Code with an alternative model backend. "
                "Returns a task_id for polling. Use for tasks that may take longer than 2 minutes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Task description for the agent"},
                    "context": {"type": "string", "description": "Task context generated by the calling agent — background info, relevant data, constraints. Passed to the sub-agent inside <context> tags."},
                    "model": {"type": "string", "description": f"Model name (default: {default_model})"},
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
        Tool(
            name="readme",
            description="Get full usage instructions for all agent-injector tools. Call this first to learn how to use sub-agents properly.",
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
    logger.debug(f"Tool call: {name} args={json.dumps(arguments)[:300]}")

    if name == "readme":
        return [TextContent(type="text", text=_build_usage_instructions())]

    elif name == "llm_run":
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

        task = tasks[task_id]
        return [TextContent(type="text", text=json.dumps({"task_id": task_id, "status": "running", "model": task["model"], "provider": task["provider"]}))]

    elif name == "llm_poll":
        task = tasks.get(arguments["task_id"])
        if not task:
            return [TextContent(type="text", text=json.dumps({"error": "Task not found", "task_id": arguments["task_id"]}))]
        return [TextContent(type="text", text=json.dumps({
            **_task_summary(task),
            "activity": task.get("activity", {}),
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
            logger.info(f"Task {task['id']} cancelled by user after {task['activity']['turns']} turns")
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
            summary = {"id": entry["id"], **_task_summary(task)}
            if task["status"] == "running":
                summary["activity"] = task.get("activity", {})
            results.append(summary)
        all_done = all(r["status"] != "running" for r in results)
        return [TextContent(type="text", text=json.dumps({"batch_id": arguments["batch_id"], "all_completed": all_done, "tasks": results}))]

    elif name == "llm_list_tasks":
        task_list = [_task_summary(t) for t in tasks.values()]
        return [TextContent(type="text", text=json.dumps({"tasks": task_list, "active": _active_count()}))]

    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main_async():
    _load_dotenv()
    _setup_file_logging()
    _init_providers()

    if PROVIDERS:
        names = ", ".join(f"{p.name}({p.default_model})" for p in PROVIDERS.values())
        logger.info(f"Agent Injector starting — providers: {names}")
    else:
        logger.info("Agent Injector starting — no providers configured")
    logger.info(f"Logs: {LOG_DIR}, retention: {LOG_RETENTION_DAYS} days")

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
