"""Exercise both MCP protocol eras through a real server subprocess."""

import asyncio
import json
import os
import signal
import sys

import pytest
from mcp import Client, StdioServerParameters


@pytest.fixture(params=["legacy", "2026-07-28"])
def client(settings, tmp_path, request):
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG"}}
    env.update(
        LLM_BASE_URL="http://127.0.0.1:8080",
        LLM_API_KEY="fixture-secret-key",
        LLM_MODEL="fixture",
        AGENT_WORKING_ROOT=str(tmp_path),
        CLAUDE_BIN=settings.claude_command,
        AGENT_MAX_CONCURRENT="1",
        AGENT_MAX_QUEUED="2",
    )
    params = StdioServerParameters(command=sys.executable, args=["-m", "agent_injector"], env=env)
    return Client(params, mode=request.param, read_timeout_seconds=10)


def payload(result):
    assert result.structured_content == json.loads(result.content[0].text)
    return result.structured_content


async def wait_for_running(tmp_path):
    async with asyncio.timeout(5):
        while not (tmp_path / "running").exists():
            await asyncio.sleep(0.01)
    return int((tmp_path / "running").read_text())


def process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def wait_for_exit(pid):
    async with asyncio.timeout(5):
        while process_exists(pid):
            await asyncio.sleep(0.01)


async def test_queue_batch_poll_and_errors_over_stdio(client, tmp_path):
    async with client:
        if client.mode == "legacy":
            assert client.protocol_version < "2026-07-28"
        else:
            assert client.protocol_version == "2026-07-28"
        tools = (await client.list_tools()).tools
        assert len(tools) == 10
        assert all(tool.input_schema["additionalProperties"] is False for tool in tools)
        status = payload(await client.call_tool("llm_status"))
        assert status["limits"]["concurrent"] == 1
        assert "fixture-secret-key" not in json.dumps(status)

        batch = payload(
            await client.call_tool(
                "llm_batch_start",
                {"tasks": [{"id": "a", "prompt": "slow"}, {"id": "b", "prompt": "tool"}]},
            )
        )
        pid = await wait_for_running(tmp_path)
        first = {"task_id": batch["tasks"][0]["task_id"]}
        second = {"task_id": batch["tasks"][1]["task_id"]}
        poll = payload(await client.call_tool("llm_poll", first))
        assert poll["status"] == "running" and "partial_output" in poll
        assert payload(await client.call_tool("llm_poll", second))["status"] == "queued"
        assert (await client.call_tool("llm_result", first)).is_error
        await client.call_tool("llm_cancel", first)
        await wait_for_exit(pid)

        async with asyncio.timeout(5):
            while True:
                complete = payload(
                    await client.call_tool("llm_batch_poll", {"batch_id": batch["batch_id"]})
                )
                if complete["all_completed"]:
                    break
                await asyncio.sleep(0.01)
        assert not complete["all_succeeded"]
        assert (await client.call_tool("llm_result", first)).is_error
        result = await client.call_tool("llm_result", second)
        assert not result.is_error and payload(result)["tool_calls"][0]["name"] == "Read"
        listed = payload(await client.call_tool("llm_list_tasks", {"status": "cancelled"}))
        assert listed["total"] == 1
        for name, arguments, code in [
            ("missing", {}, "unknown_tool"),
            ("llm_run", {"prompt": ["private-input"]}, "invalid_arguments"),
            ("llm_run", {"prompt": "private-input", "timeout_seconds": False}, "invalid_arguments"),
            ("llm_run", {"prompt": "private-input", "provider": "missing"}, "configuration_error"),
        ]:
            failure = await client.call_tool(name, arguments)
            assert failure.is_error and payload(failure)["error"]["code"] == code
            assert "private-input" not in json.dumps(payload(failure))
        assert "required_tools" in payload(await client.call_tool("readme"))["instructions"]


async def test_request_cancellation_reaps_child_over_stdio(client, tmp_path):
    async with client:
        pending = asyncio.create_task(client.call_tool("llm_run", {"prompt": "ignore-term"}))
        pid = await wait_for_running(tmp_path)
        try:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            await wait_for_exit(pid)
            listed = payload(await client.call_tool("llm_list_tasks"))
            assert [task["status"] for task in listed["tasks"]] == ["cancelled"]
            assert listed["active"] == listed["queued"] == 0
            next_result = await client.call_tool("llm_run", {"prompt": "slot-released"})
            assert not next_result.is_error and payload(next_result)["result"] == "slot-released"
        finally:
            if process_exists(pid):
                os.kill(pid, signal.SIGKILL)


async def test_stdio_disconnect_reaps_background_child(client, tmp_path):
    pid = None
    try:
        async with client:
            await client.call_tool("llm_start", {"prompt": "ignore-term"})
            pid = await wait_for_running(tmp_path)
            assert process_exists(pid)
        await wait_for_exit(pid)
    finally:
        if pid is not None and process_exists(pid):
            os.kill(pid, signal.SIGKILL)
