"""The inference harness must reject success claims without verified effects."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_smoke.py"
SPEC = importlib.util.spec_from_file_location("injector_live_smoke", SCRIPT)
live = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = live
SPEC.loader.exec_module(live)


def success_wire(mutation):
    return {
        "isError": False,
        "structuredContent": {
            "status": "completed",
            "error": None,
            "tool_error_count": 0,
            "result": "DONE",
            "tool_calls": [{"name": name} for name in ["Read", mutation, "Read"]],
        },
    }


@pytest.mark.parametrize("name", live.SCENARIOS)
def test_exact_files_and_observed_tools_pass(name):
    case = live.make_case(name)
    checked = live.verify_case(case, case.expected, success_wire(case.mutation_tool))
    assert checked["passed"]
    if name == "code":
        assert json.loads(checked["code_result"]["stdout"]) == [0, 0, 1, 3, 15]


def test_copy_success_claim_cannot_hide_wrong_bytes_or_changed_source():
    case = live.make_case("copy")
    wire = success_wire("Write")
    wrong_output = {**case.expected, "result.txt": b"DONE\n"}
    assert not live.verify_case(case, wrong_output, wire)["passed"]
    changed_source = {**case.expected, "input.txt": b"overwritten\n"}
    assert not live.verify_case(case, changed_source, wire)["passed"]


def test_correct_files_need_completed_task_and_final_read():
    case = live.make_case("edit")
    wire = success_wire("Edit")
    wire["structuredContent"]["tool_calls"] = []
    assert not live.verify_case(case, case.expected, wire)["passed"]
    wire = success_wire("Edit")
    wire["structuredContent"]["tool_calls"][-1] = {"name": "Edit"}
    assert not live.verify_case(case, case.expected, wire)["passed"]
    wire = success_wire("Edit")
    wire["structuredContent"]["status"] = "timeout"
    assert not live.verify_case(case, case.expected, wire)["passed"]


def test_unapproved_code_is_never_executed(monkeypatch):
    case = live.make_case("code")
    actual = {"sum.py": case.expected["sum.py"] + b"raise RuntimeError('unexpected code')\n"}

    def forbidden(*args, **kwargs):
        pytest.fail("The harness must not execute different generated code")

    monkeypatch.setattr(live.subprocess, "run", forbidden)
    checked = live.verify_case(case, actual, success_wire("Edit"))
    assert not checked["passed"] and checked["code_result"] is None


def test_markers_are_fresh_and_absent_from_prompts():
    markers = set()
    for name in live.SCENARIOS * 2:
        case = live.make_case(name)
        source = next(iter(case.initial.values())).decode()
        marker = re.search(r"marker=([0-9a-f]{24})", source).group(1)
        assert marker not in markers
        assert marker not in case.prompt and marker not in live.SYSTEM_PROMPT
        markers.add(marker)


def test_existing_evidence_directory_is_not_overwritten(tmp_path):
    sentinel = tmp_path / "summary.json"
    sentinel.write_text("previous experiment")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--claude",
            "unused",
            "--llama-server",
            "unused",
            "--model-file",
            "unused",
            "--log-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0 and "FileExistsError" in result.stderr
    assert sentinel.read_text() == "previous experiment"
