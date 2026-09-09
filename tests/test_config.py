import dataclasses
import os

import pytest
from pydantic import ValidationError

from agent_injector.config import (
    ConfigurationError,
    child_environment,
    load_settings,
    validate_endpoint,
)
from agent_injector.models import BatchRequest, TaskRequest


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8080", "http://[::1]:11434", "http://localhost:8080"]
)
def test_loopback_urls(url):
    assert validate_endpoint(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "ftp://localhost",
        "https://key:secret@example.com",
        "https://example.com?api_key=secret",
        "https://example.com/#x",
        "https://example.com:invalid",
        "",
        "https://bad host",
    ],
)
def test_bad_urls_never_reflect_secrets(url):
    with pytest.raises(ConfigurationError) as error:
        validate_endpoint(url)
    assert "secret" not in str(error.value)


def test_keyless_local_and_remote_requirements(tmp_path):
    base = {
        "AGENT_WORKING_ROOT": str(tmp_path),
        "LLM_BASE_URL": "http://localhost:8080/",
        "LLM_MODEL": "smollm",
    }
    settings = load_settings(base)
    provider, model = settings.resolve(None, None)
    assert provider.name == "custom" and provider.api_key == "local" and model == "smollm"
    assert provider.base_url == "http://localhost:8080"
    with pytest.raises(ConfigurationError, match="API_KEY"):
        load_settings({**base, "LLM_BASE_URL": "https://example.com"})
    hosted = load_settings(
        {**base, "LLM_BASE_URL": "https://example.com", "LLM_API_KEY": "secret-key"}
    )
    assert hosted.providers["custom"].api_key == "secret-key"
    assert "secret-key" not in repr(hosted)


def test_explicit_routing_and_no_silent_fallback(tmp_path):
    settings = load_settings(
        {
            "AGENT_WORKING_ROOT": str(tmp_path),
            "MINIMAX_API_KEY": "mini-secret",
            "ZAI_API_KEY": "zai-secret",
        }
    )
    with pytest.raises(ConfigurationError, match="Multiple"):
        settings.resolve(None, None)
    assert settings.resolve(None, "glm-5")[0].name == "zai"
    assert settings.resolve("minimax", "custom-model")[1] == "custom-model"
    for provider, model in [(None, "typo"), ("typo", None)]:
        with pytest.raises(ConfigurationError):
            settings.resolve(provider, model)
    configured = dataclasses.replace(settings, default_provider="minimax")
    assert configured.resolve(None, None)[0].name == "minimax"
    with pytest.raises(ConfigurationError):
        load_settings({"DEFAULT_PROVIDER": "missing"})
    with pytest.raises(ConfigurationError):
        load_settings({}).resolve(None, None)


def test_directory_canonicalization(settings, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    assert settings.directory("sub") == sub
    assert settings.directory(None) == tmp_path
    for path in ["../", "missing"]:
        with pytest.raises(ConfigurationError):
            settings.directory(path)
    link = tmp_path / "escape"
    link.symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ConfigurationError):
        settings.directory(str(link))


@pytest.mark.parametrize(
    "name,value",
    [
        ("AGENT_MAX_CONCURRENT", "0"),
        ("AGENT_MAX_QUEUED", "-1"),
        ("AGENT_MAX_TURNS", "nan"),
        ("AGENT_MAX_TIMEOUT_SECONDS", "3601"),
        ("AGENT_MAX_RETAINED", "2"),
        ("AGENT_ALLOWED_TOOLS", "Task"),
        ("AGENT_SYSTEM_PROMPT", "x" * 32001),
        ("AGENT_SYSTEM_PROMPT", "x\x00"),
        ("DEFAULT_MODEL", "\n"),
        ("CLAUDE_BIN", ""),
        ("AGENT_WORKING_ROOT", "/does-not-exist-agent-test"),
    ],
)
def test_operator_limits(name, value):
    with pytest.raises(ConfigurationError):
        load_settings({name: value})


def test_child_environment_isolated(settings, monkeypatch):
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-forward")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "old-credential")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("HTTPS_PROXY", "http://network-proxy")
    env = child_environment(settings, settings.providers["custom"], "fixture", "/tmp/task-config")
    assert env["HTTPS_PROXY"] == "http://network-proxy"
    assert env["ANTHROPIC_API_KEY"] == "fixture-secret-key"
    assert all(
        key not in env
        for key in ["UNRELATED_SECRET", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK"]
    )
    assert env["CLAUDE_CONFIG_DIR"] == "/tmp/task-config"
    assert os.environ["UNRELATED_SECRET"] == "do-not-forward"


@pytest.mark.parametrize(
    "extra",
    [
        {"prompt": " "},
        {"prompt": "x\x00y"},
        {"timeout_seconds": False},
        {"timeout_seconds": 1.5},
        {"timeout_seconds": "10"},
        {"timeout_seconds": 0},
        {"max_turns": 0},
        {"unknown": "x"},
        {"allowed_tools": ["Read", "Read"]},
        {"allowed_tools": ["Read,Bash"]},
        {"required_tools": ["Task"]},
        {"prompt": "x" * 64001},
        {"context": "x" * 64001},
    ],
)
def test_strict_task_contract(extra):
    with pytest.raises(ValidationError):
        TaskRequest.model_validate({"prompt": "test", **extra})


def test_batch_validation_is_complete():
    for tasks in [
        [],
        [{"id": "x", "prompt": "ok"}, {"id": "x", "prompt": "duplicate"}],
        [{"id": "x", "prompt": "ok"}, {"id": "y", "prompt": "ok", "max_turns": False}],
    ]:
        with pytest.raises(ValidationError):
            BatchRequest.model_validate({"tasks": tasks})
