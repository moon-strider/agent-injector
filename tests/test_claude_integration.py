"""Real Claude Code, deterministic Anthropic HTTP fixture; no LLM inference.

Opt in with AGENT_TEST_CLAUDE=/absolute/path/to/claude.
"""

import asyncio
import dataclasses
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_injector.config import Provider
from agent_injector.server import Application

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("AGENT_TEST_CLAUDE"), reason="real Claude Code integration is opt in"
    ),
]


async def test_real_claude_reads_and_writes_file(settings, tmp_path, monkeypatch):
    marker = "violet-river-cedar\n"
    (tmp_path / "input.txt").write_text(marker)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def json_response(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.json_response({"data": []})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if "count_tokens" in self.path:
                self.json_response({"input_tokens": 128})
                return
            requests.append(request)
            assert self.headers.get("x-api-key") == "fixture-local-key"
            results = [
                block
                for message in request.get("messages", [])
                if isinstance(message.get("content"), list)
                for block in message["content"]
                if block.get("type") == "tool_result"
            ]
            if any(block.get("tool_use_id") == "write-one" for block in results):
                block = {"type": "text", "text": "DONE"}
                stop = "end_turn"
            elif any(block.get("tool_use_id") == "read-one" for block in results):
                block = {
                    "type": "tool_use",
                    "id": "write-one",
                    "name": "Write",
                    "input": {"file_path": str(tmp_path / "result.txt"), "content": marker},
                }
                stop = "tool_use"
            else:
                block = {
                    "type": "tool_use",
                    "id": "read-one",
                    "name": "Read",
                    "input": {"file_path": str(tmp_path / "input.txt")},
                }
                stop = "tool_use"
            events = [
                {
                    "type": "message_start",
                    "message": {
                        "id": f"message-{len(requests)}",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "fixture-model",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 128, "output_tokens": 0},
                    },
                }
            ]
            if block["type"] == "tool_use":
                events.extend(
                    [
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {**block, "input": {}},
                        },
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(block["input"]),
                            },
                        },
                    ]
                )
            else:
                events.extend(
                    [
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": block["text"]},
                        },
                    ]
                )
            events.extend(
                [
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop, "stop_sequence": None},
                        "usage": {"output_tokens": 16},
                    },
                    {"type": "message_stop"},
                ]
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in events:
                self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        monkeypatch.delenv(key, raising=False)
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    provider = Provider(
        "custom", "fixture-local-key", f"http://127.0.0.1:{http.server_port}", "fixture-model"
    )
    app = Application(
        dataclasses.replace(
            settings,
            providers={"custom": provider},
            claude_command=os.environ["AGENT_TEST_CLAUDE"],
            system_prompt="Follow the task and use the provided tools.",
        )
    )
    try:
        result = await asyncio.wait_for(
            app.call_tool(
                "llm_run",
                {
                    "prompt": "Read input.txt, then write its content to result.txt.",
                    "allowed_tools": ["Read", "Write"],
                    "required_tools": ["Read", "Write"],
                    "max_turns": 6,
                    "timeout_seconds": 30,
                },
            ),
            40,
        )
        assert not result.isError, result.structuredContent
        assert (tmp_path / "result.txt").exists(), {
            "result": result.structuredContent,
            "tools": [t["name"] for t in requests[0].get("tools", [])],
            "messages": requests[-1].get("messages"),
        }
        assert (tmp_path / "result.txt").read_text() == marker
        assert [entry["name"] for entry in result.structuredContent["tool_calls"]] == [
            "Read",
            "Write",
        ]
        assert len(requests) == 3
        assert all(r["model"] == "fixture-model" for r in requests)
        assert {t["name"] for t in requests[0]["tools"]} == {"Read", "Write"}
        assert result.structuredContent["result"] == "DONE"
    finally:
        await app.manager.close()
        await asyncio.to_thread(http.shutdown)
        http.server_close()
        thread.join(timeout=2)
