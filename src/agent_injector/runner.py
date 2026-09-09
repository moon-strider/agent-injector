"""Bounded task queue and ownership of every child process."""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import os
import shutil
import signal
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import READ_TOOLS, ConfigurationError, Provider, Settings, child_environment
from .models import BatchRequest, TaskRequest

logger = logging.getLogger(__name__)
ACTIVE = {"queued", "running"}


class TaskError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass
class Job:
    request: TaskRequest = field(repr=False)
    provider: Provider = field(repr=False)
    model: str
    directory: Path
    tools: list[str]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    completed_at: float | None = None
    status: str = "queued"
    exit_code: int | None = None
    result: str = ""
    stderr: str = ""
    partial_output: str = ""
    error: dict[str, str] | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    tool_error_count: int = 0
    successful_tool_ids: set[str] = field(default_factory=set, repr=False)
    turns: int = 0
    assistant_ids: set[str] = field(default_factory=set, repr=False)
    terminal: dict[str, Any] | None = field(default=None, repr=False)
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    worker: asyncio.Task[None] | None = field(default=None, repr=False)
    done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def summary(self) -> dict[str, Any]:
        end = self.completed_at if self.completed_at is not None else time.monotonic()
        return {
            "task_id": self.id,
            "status": self.status,
            "provider": self.provider.name,
            "model": self.model,
            "elapsed_seconds": round(end - self.created_at, 3),
            "execution_seconds": round(end - self.started_at, 3)
            if self.started_at is not None
            else 0,
            "exit_code": self.exit_code,
            "error": self.error,
        }


def command(settings: Settings, job: Job) -> list[str]:
    args = [
        settings.claude_command,
        "--safe-mode",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        job.model,
        "--max-turns",
        str(job.request.max_turns),
        "--no-session-persistence",
        "--disable-slash-commands",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--tools",
        ",".join(job.tools),
    ]
    if job.tools:
        args.extend(["--allowedTools", ",".join(job.tools)])
    if settings.system_prompt:
        args.extend(["--system-prompt", settings.system_prompt])
    return args


async def terminate_process(proc: asyncio.subprocess.Process) -> None:
    """Terminate the owned process group on POSIX, then reap the direct child."""

    def send(sig: signal.Signals) -> None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, sig)
            elif proc.returncode is None:  # pragma: no cover - Windows CI exercises CLI
                proc.terminate() if sig == signal.SIGTERM else proc.kill()
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), 0.5)
    except TimeoutError:
        pass
    finally:
        # Descendants may retain pipes even after the leader exited.
        send(signal.SIGKILL)
    await asyncio.wait_for(proc.wait(), 3)


class TaskManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self.batches: dict[str, list[tuple[str, str]]] = {}
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(settings.max_concurrent)
        self._closed = False

    def prepare(self, request: TaskRequest) -> Job:
        provider, model = self.settings.resolve(request.provider, request.model)
        directory = self.settings.directory(request.working_directory)
        tools = (
            request.allowed_tools
            if request.allowed_tools is not None
            else [t for t in READ_TOOLS if t in self.settings.allowed_tools]
        )
        if not set(tools) <= self.settings.allowed_tools:
            raise ConfigurationError("Requested tools exceed AGENT_ALLOWED_TOOLS")
        if not set(request.required_tools) <= set(tools):
            raise ConfigurationError("required_tools must be included in allowed_tools")
        if (
            request.timeout_seconds > self.settings.max_timeout_seconds
            or request.max_turns > self.settings.max_turns
        ):
            raise ConfigurationError("Task timeout or turn count exceeds the server limit")
        if not shutil.which(self.settings.claude_command):
            raise TaskError(
                "cli_not_found", "Claude Code executable not found; install it or set CLAUDE_BIN"
            )
        return Job(
            request=request, provider=provider, model=model, directory=directory, tools=list(tools)
        )

    async def _admit(
        self, jobs: list[Job], batch: list[tuple[str, str]] | None = None
    ) -> str | None:
        async with self._lock:
            if self._closed:
                raise TaskError("server_closed", "Server is shutting down")
            self.cleanup()
            active = sum(j.status in ACTIVE for j in self.jobs.values())
            if active + len(jobs) > self.settings.max_concurrent + self.settings.max_queued:
                raise TaskError("capacity_exceeded", "Task queue is full; retry after tasks finish")
            while len(self.jobs) + len(jobs) > self.settings.max_retained:
                oldest = next(
                    (key for key, job in self.jobs.items() if job.status not in ACTIVE), None
                )
                if oldest is None:
                    raise TaskError("capacity_exceeded", "Task capacity exceeded")
                del self.jobs[oldest]
            self._clean_batches()
            for job in jobs:
                self.jobs[job.id] = job
                job.worker = asyncio.create_task(self._execute(job), name=f"agent-{job.id}")
            if batch is not None:
                batch_id = uuid.uuid4().hex
                self.batches[batch_id] = batch
                return batch_id
            return None

    async def start(self, request: TaskRequest) -> Job:
        job = self.prepare(request)
        await self._admit([job])
        return job

    async def start_batch(self, request: BatchRequest) -> dict[str, Any]:
        jobs: list[Job] = []
        entries: list[tuple[str, str]] = []
        for item in request.tasks:
            payload = item.model_dump(exclude={"id"})
            payload["working_directory"] = item.working_directory or request.working_directory
            job = self.prepare(TaskRequest.model_validate(payload))
            jobs.append(job)
            entries.append((item.id, job.id))
        batch_id = await self._admit(jobs, entries)
        return {
            "batch_id": batch_id,
            "tasks": [
                {"id": label, **job.summary()}
                for (label, _), job in zip(entries, jobs, strict=True)
            ],
        }

    def get(self, task_id: str) -> Job:
        self.cleanup()
        try:
            return self.jobs[task_id]
        except KeyError as exc:
            raise TaskError("task_not_found", "Task not found or expired") from exc

    def result(self, job: Job) -> dict[str, Any]:
        return {
            **job.summary(),
            "result": job.result,
            "stderr": job.stderr or None,
            "usage": job.usage,
            "tool_calls": job.tool_calls,
            "tool_error_count": job.tool_error_count,
            "turns": job.turns,
        }

    async def cancel(self, job: Job) -> None:
        if job.status not in ACTIVE:
            if job.status == "cancelled" and not job.done.is_set():
                await job.done.wait()
            return
        job.status = "cancelled"
        if job.worker:
            job.worker.cancel()
            await asyncio.gather(job.worker, return_exceptions=True)
        if (
            not job.done.is_set()
        ):  # A task cancelled before its coroutine starts has no finally block.
            job.completed_at = time.monotonic()
            job.done.set()

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
        await asyncio.gather(
            *(self.cancel(job) for job in list(self.jobs.values()) if job.status in ACTIVE)
        )
        await asyncio.gather(
            *(job.worker for job in self.jobs.values() if job.worker is not None),
            return_exceptions=True,
        )

    def _clean_batches(self) -> None:
        for key, entries in list(self.batches.items()):
            if any(task_id not in self.jobs for _, task_id in entries):
                del self.batches[key]

    def cleanup(self) -> None:
        now = time.monotonic()
        for key, job in list(self.jobs.items()):
            if (
                job.completed_at is not None
                and now - job.completed_at >= self.settings.retention_seconds
            ):
                del self.jobs[key]
        self._clean_batches()

    def redact(self, value: str) -> str:
        for provider in self.settings.providers.values():
            if len(provider.api_key) >= 4:
                value = value.replace(provider.api_key, "[redacted]")
        return value

    def _event(self, job: Job, line: str) -> None:
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            return
        if not isinstance(event, dict):
            return
        if event.get("type") == "result":
            job.terminal = event
            return
        message = event.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if not isinstance(content, list):
            return
        if event.get("type") == "assistant":
            message_id = message.get("id")
            if isinstance(message_id, str) and message_id not in job.assistant_ids:
                job.assistant_ids.add(message_id)
                job.turns += 1
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and isinstance(block.get("name"), str):
                    record = {"name": block["name"], "id": str(block.get("id", ""))}
                    if record not in job.tool_calls and len(job.tool_calls) < 256:
                        job.tool_calls.append(record)
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    job.partial_output = self.redact(block["text"][-2000:])
        elif event.get("type") == "user":
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("is_error") is not True
                    and isinstance(block.get("tool_use_id"), str)
                ):
                    job.successful_tool_ids.add(block["tool_use_id"])
            job.tool_error_count += sum(
                isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error") is True
                for b in content
            )

    async def _read_stdout(self, job: Job, stream: asyncio.StreamReader) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""
        total = 0
        while chunk := await stream.read(8192):
            total += len(chunk)
            if total > self.settings.max_output_bytes:
                raise TaskError(
                    "output_limit", "Claude Code output exceeded the configured byte limit"
                )
            pending += decoder.decode(chunk)
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                self._event(job, line)
        pending += decoder.decode(b"", final=True)
        if pending:
            self._event(job, pending)

    async def _read_stderr(self, job: Job, stream: asyncio.StreamReader) -> None:
        data = bytearray()
        while chunk := await stream.read(8192):
            data.extend(chunk)
            job.stderr = self.redact(data.decode("utf-8", errors="replace"))[-4000:]
            if len(data) > min(self.settings.max_output_bytes, 65536):
                raise TaskError(
                    "output_limit", "Claude Code stderr exceeded the configured byte limit"
                )

    def _finish(self, job: Job) -> None:
        event = job.terminal
        if job.exit_code != 0:
            raise TaskError("cli_exit", "Claude Code exited unsuccessfully; inspect stderr")
        if event is None or not isinstance(event.get("result", ""), str):
            raise TaskError(
                "invalid_cli_output", "Claude Code did not emit a valid terminal result"
            )
        job.result = self.redact(event.get("result", ""))
        if event.get("is_error") is not False or event.get("subtype") != "success":
            raise TaskError(
                "cli_result_error", "Claude Code reported an unsuccessful terminal result"
            )
        if event.get("permission_denials"):
            raise TaskError("permission_denied", "Claude Code reported denied tool permissions")
        observed = {
            entry["name"] for entry in job.tool_calls if entry["id"] in job.successful_tool_ids
        }
        if not set(job.request.required_tools) <= observed:
            raise TaskError(
                "required_tool_missing", "Required tools did not return successful results"
            )
        usage = event.get("usage", {})
        if isinstance(usage, dict):
            job.usage = {
                key: value
                for key, value in usage.items()
                if key
                in {
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                }
                and type(value) is int
                and value >= 0
            }
        job.status = "completed"

    async def _execute(self, job: Job) -> None:
        try:
            # The timeout includes time spent waiting for admission to a process slot.
            remaining = max(0, job.created_at + job.request.timeout_seconds - time.monotonic())
            async with asyncio.timeout(remaining):
                async with self._slots:
                    job.status = "running"
                    job.started_at = time.monotonic()
                    with tempfile.TemporaryDirectory(prefix="agent-injector-") as config_dir:
                        spawning = asyncio.create_task(
                            asyncio.create_subprocess_exec(
                                *command(self.settings, job),
                                cwd=job.directory,
                                env=child_environment(
                                    self.settings, job.provider, job.model, config_dir
                                ),
                                stdin=asyncio.subprocess.PIPE,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                start_new_session=os.name == "posix",
                            )
                        )
                        try:
                            proc = await asyncio.shield(spawning)
                        except asyncio.CancelledError:
                            # Acquire ownership before propagating cancellation.
                            proc = await spawning
                            job.process = proc
                            await terminate_process(proc)
                            raise
                        job.process = proc
                        assert proc.stdin and proc.stdout and proc.stderr
                        prompt = job.request.prompt
                        if job.request.context:
                            prompt = f"<context>\n{job.request.context}\n</context>\n\n{prompt}"

                        async def write_prompt() -> None:
                            assert proc.stdin
                            try:
                                proc.stdin.write(prompt.encode("utf-8"))
                                await proc.stdin.drain()
                            except (BrokenPipeError, ConnectionResetError):
                                pass
                            finally:
                                proc.stdin.close()

                        readers = [
                            asyncio.create_task(self._read_stdout(job, proc.stdout)),
                            asyncio.create_task(self._read_stderr(job, proc.stderr)),
                            asyncio.create_task(write_prompt()),
                        ]
                        try:
                            await asyncio.gather(*readers)
                            await proc.wait()
                            job.exit_code = proc.returncode
                            self._finish(job)
                        finally:
                            for reader in readers:
                                reader.cancel()
                            await asyncio.gather(*readers, return_exceptions=True)
                            await terminate_process(proc)
        except TimeoutError:
            job.status = "timeout"
            job.error = {
                "code": "timeout",
                "message": "Task deadline exceeded, including queue time",
            }
        except asyncio.CancelledError:
            job.status = "cancelled"
        except TaskError as exc:
            job.status = "failed"
            job.error = {"code": exc.code, "message": str(exc)}
        except OSError:
            job.status = "failed"
            job.error = {"code": "spawn_failed", "message": "Could not start Claude Code"}
        except Exception:
            job.status = "failed"
            job.error = {"code": "internal_error", "message": "Unexpected task execution failure"}
            logger.error("Task %s encountered an internal error", job.id)
        finally:
            if job.process is not None:
                if job.process.returncode is None:
                    await terminate_process(job.process)
                job.exit_code = job.process.returncode
                job.process = None
            job.completed_at = time.monotonic()
            job.done.set()
            logger.info("Task %s %s provider=%s", job.id, job.status, job.provider.name)
