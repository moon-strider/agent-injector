import asyncio
import dataclasses
import json
import os
import subprocess
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_injector.server import Application


def payload(result):
    assert result.structuredContent == json.loads(result.content[0].text)
    return result.structuredContent


async def call(app, name, args=None):
    return await app.call_tool(name, args or {})


async def test_contracts_errors_and_diagnostics(settings):
    app = Application(settings)
    schemas = await app.list_tools()
    assert len(schemas) == 10
    assert all(t.inputSchema["additionalProperties"] is False for t in schemas)
    status = payload(await call(app, "llm_status"))
    assert status["claude_available"] and status["providers"] == [
        {"name": "custom", "model": "fixture"}
    ]
    assert "fixture-secret-key" not in json.dumps(status)
    assert "required_tools" in payload(await call(app, "readme"))["instructions"]
    for name, args, code in [
        ("missing", {}, "unknown_tool"),
        ("llm_run", {}, "invalid_arguments"),
        ("llm_poll", {"task_id": "missing"}, "task_not_found"),
        ("llm_batch_poll", {"batch_id": "missing"}, "batch_not_found"),
        ("llm_run", {"prompt": "x", "provider": "missing"}, "configuration_error"),
    ]:
        result = await call(app, name, args)
        assert result.isError and payload(result)["error"]["code"] == code
    invalid = await call(app, "llm_run", {"prompt": ["private-sensitive-input"]})
    assert "private-sensitive-input" not in json.dumps(payload(invalid))


async def test_sync_async_poll_result_and_cancel(settings):
    app = Application(settings)
    success = await call(app, "llm_run", {"prompt": "hello"})
    assert not success.isError and payload(success)["result"] == "hello"
    failed = await call(app, "llm_run", {"prompt": "cli-error"})
    assert failed.isError and payload(failed)["status"] == "failed"
    started = payload(await call(app, "llm_start", {"prompt": "slow"}))
    identity = {"task_id": started["task_id"]}
    poll = payload(await call(app, "llm_poll", identity))
    assert poll["status"] in {"queued", "running"} and "partial_output" in poll
    assert (await call(app, "llm_result", identity)).isError
    cancelled = await call(app, "llm_cancel", identity)
    assert not cancelled.isError and payload(cancelled)["status"] == "cancelled"
    result = await call(app, "llm_result", identity)
    assert result.isError and payload(result)["status"] == "cancelled"
    assert payload(await call(app, "llm_cancel", identity))["status"] == "cancelled"


async def test_cancelling_sync_request_cleans_child(settings):
    app = Application(settings)
    pending = asyncio.create_task(call(app, "llm_run", {"prompt": "slow"}))
    while not app.manager.jobs:
        await asyncio.sleep(0.01)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert all(job.status == "cancelled" and job.done.is_set() for job in app.manager.jobs.values())


async def test_batch_status_and_pagination(settings):
    app = Application(settings)
    response = await call(
        app,
        "llm_batch_start",
        {"tasks": [{"id": "a", "prompt": "a"}, {"id": "b", "prompt": "cli-error"}]},
    )
    batch_id = payload(response)["batch_id"]
    initial = payload(await call(app, "llm_batch_poll", {"batch_id": batch_id}))
    assert initial["all_completed"] is False
    await asyncio.gather(*(job.done.wait() for job in app.manager.jobs.values()))
    final = payload(await call(app, "llm_batch_poll", {"batch_id": batch_id}))
    assert final["all_completed"] and not final["all_succeeded"]
    listed = payload(await call(app, "llm_list_tasks", {"limit": 1}))
    assert listed["total"] == 2 and listed["next_offset"] == 1 and len(listed["tasks"]) == 1
    filtered = payload(await call(app, "llm_list_tasks", {"status": "failed"}))
    assert filtered["total"] == 1 and filtered["tasks"][0]["status"] == "failed"
    assert payload(await call(app, "llm_list_tasks", {"offset": 10}))["tasks"] == []


async def test_dispatch_internal_error_sanitized(settings, monkeypatch):
    app = Application(settings)

    async def boom(*args):
        raise RuntimeError("private-secret")

    monkeypatch.setattr(app, "dispatch", boom)
    result = await call(app, "llm_status")
    assert result.isError and "private-secret" not in str(result)


async def test_real_stdio_mcp_transport(settings, tmp_path):
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG"}}
    env.update(
        LLM_BASE_URL="http://127.0.0.1:8080",
        LLM_API_KEY="fixture-secret-key",
        LLM_MODEL="fixture",
        AGENT_WORKING_ROOT=str(tmp_path),
        CLAUDE_BIN=settings.claude_command,
    )
    params = StdioServerParameters(command=sys.executable, args=["-m", "agent_injector"], env=env)
    with (tmp_path / "server.stderr").open("w") as err:
        async with stdio_client(params, errlog=err) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.version == "0.2.0"
                assert len((await session.list_tools()).tools) == 10
                result = await session.call_tool(
                    "llm_run",
                    {"prompt": "tool", "allowed_tools": ["Read"], "required_tools": ["Read"]},
                )
                assert not result.isError and payload(result)["tool_calls"][0]["name"] == "Read"
                result = await session.call_tool("llm_run", {"prompt": "cli-error"})
                assert result.isError and payload(result)["status"] == "failed"
                invalid = await session.call_tool(
                    "llm_run", {"prompt": "x", "timeout_seconds": False}
                )
                assert invalid.isError


def test_cli_check_version_invalid_configuration(settings, tmp_path):
    base = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG"}}
    base["CLAUDE_BIN"] = settings.claude_command
    run = subprocess.run(
        [sys.executable, "-m", "agent_injector", "--check"],
        env=base,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(run.stdout)["claude_available"]
    run = subprocess.run(
        [sys.executable, "-m", "agent_injector", "--version"],
        env=base,
        capture_output=True,
        text=True,
        check=True,
    )
    assert run.stdout.strip() == "0.2.0"
    run = subprocess.run(
        [sys.executable, "-m", "agent_injector", "--check"],
        env={**base, "AGENT_MAX_CONCURRENT": "0"},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2 and "Configuration error" in run.stderr


async def test_spawn_os_error_is_terminal(settings, monkeypatch):
    app = Application(settings)

    async def fail(*args, **kwargs):
        raise OSError("private-path")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)
    result = await call(app, "llm_run", {"prompt": "x"})
    assert result.isError and payload(result)["error"]["code"] == "spawn_failed"
    assert "private-path" not in str(result)


async def test_default_tools_respect_operator_policy(settings):
    app = Application(dataclasses.replace(settings, allowed_tools=frozenset()))
    result = await call(app, "llm_run", {"prompt": "argv"})
    args = json.loads(payload(result)["result"])
    assert args[args.index("--tools") + 1] == ""
