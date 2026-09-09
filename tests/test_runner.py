import asyncio
import dataclasses
import json
import os
import time

import pytest

from agent_injector.config import ConfigurationError
from agent_injector.models import BatchRequest, TaskRequest
from agent_injector.runner import TaskError, TaskManager, command


async def finish(manager, prompt, **kwargs):
    job = await manager.start(TaskRequest(prompt=prompt, **kwargs))
    await asyncio.wait_for(job.done.wait(), 5)
    return job


async def started(job):
    async with asyncio.timeout(5):
        while job.process is None or not (job.directory / "running").exists():
            await asyncio.sleep(0.01)


async def test_text_and_fixed_elapsed(settings):
    manager = TaskManager(settings)
    job = await finish(manager, "hello")
    assert job.status == "completed" and job.result == "hello" and job.exit_code == 0
    elapsed = job.summary()["elapsed_seconds"]
    await asyncio.sleep(0.02)
    assert job.summary()["elapsed_seconds"] == elapsed
    assert job.usage == {"input_tokens": 4, "output_tokens": 2}
    assert job.process is None


async def test_tools_restrict_availability_and_empty_means_none(settings):
    manager = TaskManager(settings)
    for tools in [[], ["Read"], ["Read", "Write"]]:
        job = await finish(manager, "argv", allowed_tools=tools)
        args = json.loads(job.result)
        assert args[args.index("--tools") + 1] == ",".join(tools)
        assert "--dangerously-skip-permissions" not in args
        assert "--strict-mcp-config" in args and "--safe-mode" in args
        assert "argv" not in args  # prompts travel through stdin
    default = manager.prepare(TaskRequest(prompt="test"))
    assert default.tools == ["Read", "Glob", "Grep"]
    custom = dataclasses.replace(settings, system_prompt="custom system")
    args = command(custom, default)
    assert args[args.index("--system-prompt") + 1] == "custom system"


@pytest.mark.parametrize(
    "prompt,code",
    [
        ("exit-error", "cli_exit"),
        ("no-result", "invalid_cli_output"),
        ("cli-error", "cli_result_error"),
        ("permission", "permission_denied"),
        ("bad-result", "invalid_cli_output"),
    ],
)
async def test_failures_are_not_success(settings, prompt, code):
    job = await finish(TaskManager(settings), prompt)
    assert job.status == "failed" and job.error["code"] == code
    assert "fixture-secret-key" not in job.stderr


async def test_observed_tools_required_and_duplicates(settings):
    manager = TaskManager(settings)
    fake = await finish(manager, "fake-tool", allowed_tools=["Read"], required_tools=["Read"])
    assert fake.status == "failed" and fake.error["code"] == "required_tool_missing"
    denied = await finish(manager, "failed-tool", allowed_tools=["Read"], required_tools=["Read"])
    assert denied.status == "failed" and denied.error["code"] == "required_tool_missing"
    real = await finish(manager, "tool", allowed_tools=["Read"], required_tools=["Read"])
    assert real.status == "completed" and real.turns == 1
    assert real.tool_calls == [{"name": "Read", "id": "tool-one"}]
    assert real.tool_error_count == 1
    assert real.partial_output == "reading"


async def test_cancellation_during_spawn_acquires_child_ownership(settings, monkeypatch):
    manager = TaskManager(settings)
    original = asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    release = asyncio.Event()
    processes = []

    async def delayed_spawn(*args, **kwargs):
        proc = await original(*args, **kwargs)
        processes.append(proc)
        spawned.set()
        await release.wait()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    job = await manager.start(TaskRequest(prompt="slow"))
    await asyncio.wait_for(spawned.wait(), 3)
    cancelling = asyncio.create_task(manager.cancel(job))
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(cancelling, 3)
    assert job.status == "cancelled" and job.done.is_set()
    assert processes[0].returncode is not None


async def test_incremental_unicode_malformed_events_and_redaction(settings):
    manager = TaskManager(settings)
    job = await finish(manager, "unicode")
    assert job.partial_output == "Привет 東京 😀"
    assert (await finish(manager, "malformed")).status == "completed"
    assert (await finish(manager, "redact")).result == "[redacted]"


@pytest.mark.parametrize("prompt", ["output-limit", "stderr-limit"])
async def test_output_is_bounded(settings, prompt):
    manager = TaskManager(dataclasses.replace(settings, max_output_bytes=4096))
    job = await finish(manager, prompt)
    assert job.status == "failed" and job.error["code"] == "output_limit"
    assert len(job.stderr) <= 4000


@pytest.mark.parametrize("prompt", ["slow", "ignore-term", "descendant"])
async def test_cancel_reaps_process_group(settings, prompt):
    manager = TaskManager(settings)
    job = await manager.start(TaskRequest(prompt=prompt))
    await started(job)
    proc = job.process
    await manager.cancel(job)
    assert job.status == "cancelled" and proc.returncode is not None and job.process is None
    await asyncio.sleep(0.05)
    assert job.status == "cancelled"
    if prompt == "descendant" and os.path.isdir("/proc"):
        from pathlib import Path

        pid = (settings.working_root / "descendant.pid").read_text()
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or stat.read_text().split()[2] == "Z"


async def test_cancel_before_execution_and_queue_timeout(settings):
    manager = TaskManager(settings)
    first = await manager.start(TaskRequest(prompt="slow"))
    cancelled = await manager.start(TaskRequest(prompt="never-started"))
    await manager.cancel(cancelled)
    assert cancelled.status == "cancelled" and cancelled.done.is_set()
    queued = await manager.start(TaskRequest(prompt="queued", timeout_seconds=1))
    await asyncio.wait_for(queued.done.wait(), 3)
    assert queued.status == "timeout" and queued.started_at is None
    await manager.close()
    assert first.status == "cancelled"


async def test_running_timeout(settings):
    job = await finish(TaskManager(settings), "slow", timeout_seconds=1)
    assert job.status == "timeout" and job.exit_code is not None


async def test_atomic_concurrency_admission(settings):
    manager = TaskManager(dataclasses.replace(settings, max_queued=0))
    results = await asyncio.gather(
        *(manager.start(TaskRequest(prompt="slow")) for _ in range(10)), return_exceptions=True
    )
    assert sum(not isinstance(r, Exception) for r in results) == 1
    assert all(
        isinstance(r, TaskError) and r.code == "capacity_exceeded"
        for r in results
        if isinstance(r, Exception)
    )
    await manager.close()


async def test_batch_queue_really_executes_and_invalid_batch_has_no_effect(settings):
    manager = TaskManager(settings)
    batch = await manager.start_batch(
        BatchRequest(tasks=[{"id": str(i), "prompt": str(i)} for i in range(3)])
    )
    assert len(manager.jobs) == 3
    await asyncio.gather(*(job.done.wait() for job in manager.jobs.values()))
    assert [job.result for job in manager.jobs.values()] == ["0", "1", "2"]
    assert all(job.status == "completed" for job in manager.jobs.values())
    before = set(manager.jobs)
    with pytest.raises(ConfigurationError):
        await manager.start_batch(
            BatchRequest(
                tasks=[
                    {"id": "good", "prompt": "ok"},
                    {"id": "bad", "prompt": "oops", "provider": "unknown"},
                ]
            )
        )
    assert set(manager.jobs) == before and len(manager.batches) == 1
    assert batch["batch_id"] in manager.batches


async def test_batch_over_capacity_is_atomic(settings):
    manager = TaskManager(settings)
    with pytest.raises(TaskError, match="queue"):
        await manager.start_batch(
            BatchRequest(tasks=[{"id": str(i), "prompt": "slow"} for i in range(4)])
        )
    assert not manager.jobs and not manager.batches


async def test_retention_bounds_tasks_and_batches(settings):
    manager = TaskManager(settings)
    first = await finish(manager, "first")
    for i in range(8):
        batch = await manager.start_batch(BatchRequest(tasks=[{"id": "one", "prompt": str(i)}]))
        await manager.jobs[batch["tasks"][0]["task_id"]].done.wait()
    assert len(manager.jobs) == 4 and len(manager.batches) == 4
    with pytest.raises(TaskError):
        manager.get(first.id)
    for job in manager.jobs.values():
        job.completed_at = time.monotonic() - 1000
    manager.cleanup()
    assert not manager.jobs and not manager.batches


async def test_server_policy_and_closed_manager(settings):
    manager = TaskManager(dataclasses.replace(settings, allowed_tools=frozenset({"Read"})))
    for request in [
        TaskRequest(prompt="x", allowed_tools=["Write"]),
        TaskRequest(prompt="x", required_tools=["Write"]),
        TaskRequest(prompt="x", timeout_seconds=3600),
        TaskRequest(prompt="x", max_turns=1000),
    ]:
        with pytest.raises(ConfigurationError):
            manager.prepare(request)
    missing = TaskManager(dataclasses.replace(settings, claude_command="/missing/claude"))
    with pytest.raises(TaskError, match="executable"):
        missing.prepare(TaskRequest(prompt="x"))
    await manager.close()
    with pytest.raises(TaskError, match="shutting down"):
        await manager.start(TaskRequest(prompt="x"))


async def test_environment_and_temporary_config_cleanup(settings, monkeypatch):
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-forward")
    job = await finish(TaskManager(settings), "environment")
    env = json.loads(job.result)
    assert env["UNRELATED_SECRET"] is None and env["ANTHROPIC_AUTH_TOKEN"] is None
    assert env["ANTHROPIC_MODEL"] == "fixture"
    assert not os.path.exists(env["CLAUDE_CONFIG_DIR"])
