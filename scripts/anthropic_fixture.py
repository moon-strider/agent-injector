"""Deterministic Anthropic tool responses for integration checks, not model inference."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

FIXTURE_KEY = "fixture-local-key"
FIXTURE_MODEL = "fixture-model"


class AnthropicFixture:
    """Direct a real Claude CLI to Read and Write inside a disposable directory."""

    def __init__(self, root: Path, content: str):
        self.root = root
        self.content = content
        self.requests: list[dict] = []
        self.errors: list[str] = []
        self.observed_tools: list[str] = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def json_response(self, value: dict, status: int = 200):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.json_response({"data": []})

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 2_097_152:
                        raise ValueError("Fixture request exceeds its body limit")
                    self.connection.settimeout(10)
                    request = json.loads(self.rfile.read(length))
                    path = urlsplit(self.path).path
                    if path == "/v1/messages/count_tokens":
                        self.json_response({"input_tokens": 128})
                        return
                    if path != "/v1/messages":
                        raise ValueError("Unexpected fixture endpoint")
                    if self.headers.get("x-api-key") != FIXTURE_KEY:
                        raise ValueError("Missing fixture authentication")
                    block = fixture.reply(request)
                except (ValueError, KeyError, TypeError) as exc:
                    fixture.errors.append(str(exc))
                    self.json_response({"error": {"message": str(exc)}}, status=400)
                    return
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for event in fixture.events(block):
                        self.wfile.write(
                            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                        )
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    fixture.errors.append("CLI disconnected before reading the fixture response")

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.daemon_threads = True
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.http.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=2)

    def reply(self, request: dict) -> dict:
        self.requests.append(request)
        if request["model"] != FIXTURE_MODEL:
            raise ValueError("CLI did not use the configured model")
        if {tool["name"] for tool in request["tools"]} != {"Read", "Write"}:
            raise ValueError("CLI tool exposure differs from the explicit allowlist")
        results = {
            block.get("tool_use_id"): block
            for message in request["messages"]
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if block.get("type") == "tool_result"
        }
        if any(result.get("is_error") for result in results.values()):
            raise ValueError("CLI reported a failed tool result")
        if "write-one" in results:
            self.observed_tools.append("Write")
            return {"type": "text", "text": "DONE"}
        if "read-one" in results:
            observed = results["read-one"].get("content", "")
            if isinstance(observed, list):
                observed = "\n".join(block.get("text", "") for block in observed)
            if not all(line in observed for line in self.content.splitlines()):
                raise ValueError("Read tool result did not contain the source content")
            self.observed_tools.append("Read")
            return {
                "type": "tool_use",
                "id": "write-one",
                "name": "Write",
                "input": {"file_path": str(self.root / "result.txt"), "content": self.content},
            }
        return {
            "type": "tool_use",
            "id": "read-one",
            "name": "Read",
            "input": {"file_path": str(self.root / "input.txt")},
        }

    def events(self, block: dict) -> list[dict]:
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": f"message-{len(self.requests)}",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": FIXTURE_MODEL,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 128, "output_tokens": 0},
                },
            }
        ]
        tool = block["type"] == "tool_use"
        initial = {**block, "input": {}} if tool else {"type": "text", "text": ""}
        delta = (
            {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            if tool
            else {"type": "text_delta", "text": block["text"]}
        )
        events.extend(
            [
                {"type": "content_block_start", "index": 0, "content_block": initial},
                {"type": "content_block_delta", "index": 0, "delta": delta},
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "tool_use" if tool else "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": 16},
                },
                {"type": "message_stop"},
            ]
        )
        return events
