"""MCP transport and public tool dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import ValidationError

from . import __version__
from .config import ConfigurationError, Settings, load_settings
from .models import BatchId, BatchRequest, Input, ListRequest, TaskId, TaskRequest
from .runner import ACTIVE, TaskError, TaskManager

USAGE = """Agent Injector runs bounded Claude Code tasks through configured model providers.

1. Call llm_status to see providers, executable availability and execution limits.
2. Use llm_run for short tasks or llm_start then llm_poll/llm_result for background work.
3. provider explicitly selects a configured backend. Unknown model names require provider.
4. allowed_tools defaults to Read, Glob, Grep, intersected with server policy. [] disables
   tools. Bash, Edit and Write must be explicitly requested and permitted by the server.
5. required_tools can require observed tool calls before a task can complete successfully.
   Text resembling a tool call does not count. This does not verify the semantic result.
6. working_directory must be inside AGENT_WORKING_ROOT. This is an admission check,
   not an operating-system sandbox. Authorized tools retain local account privileges.
7. llm_batch_start validates and admits the entire batch or rejects it. Queued tasks run
   when a process slot becomes available; timeout_seconds includes queue time.
8. Cancel queued/running tasks with llm_cancel. Results expire and may be evicted when
   retained capacity is reached. llm_list_tasks supports status/offset/limit.
9. completed means the CLI returned a successful result and required tools were observed.
   Inspect result, tool_calls and tool_error_count; validate your task's actual output.

No provider search MCP servers, other local MCP configurations, skills or permission
bypass are enabled automatically. For remote providers, task contents go to that provider.
Use this stdio server with trusted local clients only. See README and docs/ for setup.
"""

CONTRACTS: dict[str, tuple[type[Input], str]] = {
    "llm_run": (TaskRequest, "Run a bounded task and wait for its terminal result."),
    "llm_start": (TaskRequest, "Queue a task and return its task_id immediately."),
    "llm_poll": (TaskId, "Get task status, observed tools and recent assistant text."),
    "llm_result": (TaskId, "Get a finished task's result, usage and tool calls."),
    "llm_cancel": (TaskId, "Cancel a queued or running task and wait for process cleanup."),
    "llm_batch_start": (BatchRequest, "Atomically admit a batch into the bounded task queue."),
    "llm_batch_poll": (BatchId, "Get every task status in a retained batch."),
    "llm_list_tasks": (ListRequest, "List retained tasks with optional status and pagination."),
    "llm_status": (Input, "Inspect provider names, CLI availability and limits without inference."),
    "readme": (Input, "Read the task execution contract and usage instructions."),
}


def response(payload: dict[str, Any], *, error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structuredContent=payload,
        isError=error,
    )


class Application:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.manager = TaskManager(settings)
        self.server: Server[Any, Any] = Server("agent-injector", version=__version__)
        self.server.list_tools()(self.list_tools)  # type: ignore[no-untyped-call]
        # Use the published Pydantic contract without reflecting raw input in SDK errors.
        self.server.call_tool(validate_input=False)(self.call_tool)

    async def list_tools(self) -> list[Tool]:
        return [
            Tool(name=name, description=description, inputSchema=model.model_json_schema())
            for name, (model, description) in CONTRACTS.items()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        try:
            contract = CONTRACTS.get(name)
            if contract is None:
                raise TaskError("unknown_tool", "Unknown tool")
            parsed = contract[0].model_validate(arguments)
            payload = await self.dispatch(name, parsed)
            failed = name in {"llm_run", "llm_result"} and payload.get("status") in {
                "failed",
                "timeout",
                "cancelled",
            }
            return response(payload, error=failed)
        except ValidationError as exc:
            details = [
                {"field": ".".join(map(str, e["loc"])), "message": e["msg"]}
                for e in exc.errors(include_input=False, include_url=False)
            ]
            return response(
                {
                    "error": {
                        "code": "invalid_arguments",
                        "message": "Invalid tool arguments",
                        "details": details,
                    }
                },
                error=True,
            )
        except ConfigurationError as exc:
            return response(
                {"error": {"code": "configuration_error", "message": str(exc)}}, error=True
            )
        except TaskError as exc:
            return response({"error": {"code": exc.code, "message": str(exc)}}, error=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger(__name__).error("Unexpected tool dispatch failure")
            return response(
                {"error": {"code": "internal_error", "message": "Unexpected server error"}},
                error=True,
            )

    async def dispatch(self, name: str, value: Input) -> dict[str, Any]:
        manager = self.manager
        if name in {"llm_run", "llm_start"}:
            assert isinstance(value, TaskRequest)
            job = await manager.start(value)
            if name == "llm_start":
                return job.summary()
            try:
                await job.done.wait()
            except asyncio.CancelledError:
                await manager.cancel(job)
                raise
            return manager.result(job)
        if isinstance(value, TaskId):
            job = manager.get(value.task_id)
            if name == "llm_cancel":
                await manager.cancel(job)
                return job.summary()
            if name == "llm_poll":
                return {
                    **job.summary(),
                    "partial_output": job.partial_output,
                    "activity": {
                        "turns": job.turns,
                        "tool": job.tool_calls[-1]["name"] if job.tool_calls else None,
                    },
                }
            if job.status in ACTIVE:
                raise TaskError("task_not_finished", "Task is queued or running; use llm_poll")
            return manager.result(job)
        if isinstance(value, BatchRequest):
            return await manager.start_batch(value)
        if isinstance(value, BatchId):
            manager.cleanup()
            entries = manager.batches.get(value.batch_id)
            if entries is None:
                raise TaskError("batch_not_found", "Batch not found or expired")
            tasks = [{"id": label, **manager.jobs[task_id].summary()} for label, task_id in entries]
            return {
                "batch_id": value.batch_id,
                "tasks": tasks,
                "all_completed": all(task["status"] not in ACTIVE for task in tasks),
                "all_succeeded": all(task["status"] == "completed" for task in tasks),
            }
        if isinstance(value, ListRequest):
            manager.cleanup()
            jobs = list(reversed(manager.jobs.values()))
            selected = [job for job in jobs if value.status is None or job.status == value.status]
            end = value.offset + value.limit
            return {
                "tasks": [job.summary() for job in selected[value.offset : end]],
                "total": len(selected),
                "active": sum(job.status == "running" for job in jobs),
                "queued": sum(job.status == "queued" for job in jobs),
                "next_offset": end if end < len(selected) else None,
            }
        if name == "llm_status":
            settings = self.settings
            return {
                "version": __version__,
                "claude_available": shutil.which(settings.claude_command) is not None,
                "providers": [
                    {"name": p.name, "model": p.default_model} for p in settings.providers.values()
                ],
                "default_provider": settings.default_provider,
                "default_model": settings.default_model,
                "working_root": str(settings.working_root),
                "allowed_tools": sorted(settings.allowed_tools),
                "limits": {
                    "concurrent": settings.max_concurrent,
                    "queued": settings.max_queued,
                    "retained": settings.max_retained,
                    "timeout_seconds": settings.max_timeout_seconds,
                    "turns": settings.max_turns,
                    "output_bytes": settings.max_output_bytes,
                    "retention_seconds": settings.retention_seconds,
                },
            }
        return {"instructions": USAGE}

    async def run(self) -> None:
        async def cleanup() -> None:
            while True:
                await asyncio.sleep(min(30, self.settings.retention_seconds))
                self.manager.cleanup()

        cleaner = asyncio.create_task(cleanup())
        try:
            async with stdio_server() as (read, write):
                await self.server.run(read, write, self.server.create_initialization_options())
        finally:
            cleaner.cancel()
            await asyncio.gather(cleaner, return_exceptions=True)
            await self.manager.close()


async def main_async() -> None:
    await Application(load_settings()).run()
