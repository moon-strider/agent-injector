from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent_injector.config import Provider, Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    executable = tmp_path / "claude-fixture"
    fixture = Path(__file__).parent / "fixtures" / "cli.py"
    executable.write_text(f"#!{sys.executable}\n" + fixture.read_text())
    executable.chmod(0o755)
    return Settings(
        providers={
            "custom": Provider("custom", "fixture-secret-key", "http://localhost:8080", "fixture")
        },
        working_root=tmp_path,
        claude_command=str(executable),
        max_concurrent=1,
        max_queued=2,
        max_retained=4,
    )
