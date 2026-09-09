"""Deterministic CLI protocol fixture, never a substitute for model inference."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

prompt = sys.stdin.read()


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


if prompt in {"slow", "ignore-term", "descendant"}:
    if prompt == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if prompt == "descendant":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        Path("descendant.pid").write_text(str(child.pid))
    Path("running").write_text(str(os.getpid()))
    time.sleep(60)
elif prompt == "output-limit":
    print("x" * 100_000, flush=True)
elif prompt == "stderr-limit":
    print("x" * 100_000, file=sys.stderr, flush=True)
elif prompt == "exit-error":
    print("provider says fixture-secret-key", file=sys.stderr, flush=True)
    sys.exit(7)
elif prompt == "no-result":
    print("not json", flush=True)
    sys.exit(0)
elif prompt == "malformed":
    for value in [
        [],
        123,
        None,
        {"type": "assistant", "message": None},
        {"type": "assistant", "message": {"content": 123}},
    ]:
        emit(value)
elif prompt in {"tool", "failed-tool"}:
    value = {
        "type": "assistant",
        "message": {
            "id": "assistant-message",
            "content": [
                {
                    "type": "tool_use",
                    "name": "Read",
                    "id": "tool-one",
                    "input": {"file_path": "fixture.txt"},
                },
                {"type": "text", "text": "reading"},
            ],
        },
    }
    emit(value)
    emit(value)
    emit({"type": "user", "message": {"content": [{"type": "tool_result", "is_error": True}]}})
    if prompt == "tool":
        emit(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-one",
                            "content": "fixture contents",
                        }
                    ]
                },
            }
        )
elif prompt == "unicode":
    raw = json.dumps(
        {
            "type": "assistant",
            "message": {"id": "unicode", "content": [{"type": "text", "text": "Привет 東京 😀"}]},
        },
        ensure_ascii=False,
    ).encode()
    for byte in raw + b"\n":
        os.write(sys.stdout.fileno(), bytes([byte]))

result = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": prompt,
    "usage": {"input_tokens": 4, "output_tokens": 2, "bad": True, "cache_read_input_tokens": -1},
}
if prompt == "cli-error":
    result.update(subtype="error_max_turns", is_error=True)
elif prompt == "permission":
    result["permission_denials"] = [{"tool_name": "Write"}]
elif prompt == "bad-result":
    result["result"] = {"unexpected": "object"}
elif prompt == "fake-tool":
    result["result"] = '<tool_call>{"name":"Read"}</tool_call>'
elif prompt == "argv":
    result["result"] = json.dumps(sys.argv[1:])
elif prompt == "environment":
    result["result"] = json.dumps(
        {
            key: os.environ.get(key)
            for key in [
                "ANTHROPIC_BASE_URL",
                "ANTHROPIC_MODEL",
                "UNRELATED_SECRET",
                "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_CONFIG_DIR",
            ]
        }
    )
elif prompt == "redact":
    result["result"] = "fixture-secret-key"
sys.stdout.write(json.dumps(result))  # Deliberately no final newline.
sys.stdout.flush()
