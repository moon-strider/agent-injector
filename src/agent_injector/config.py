"""Explicit provider routing and server-owned execution limits."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

BUILTIN_TOOLS = frozenset({"Read", "Glob", "Grep", "Bash", "Edit", "Write"})
READ_TOOLS = ("Read", "Glob", "Grep")


class ConfigurationError(ValueError):
    """Invalid operator configuration; messages never include credentials."""


def validate_endpoint(url: str) -> bool:
    """Validate URL and return whether it targets an explicit loopback host."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        _ = parsed.port
        loopback = host == "localhost"
        if host and not loopback:
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                pass
        if (
            not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or any(c.isspace() for c in url)
            or parsed.scheme not in ("https", "http")
            or (parsed.scheme == "http" and not loopback)
        ):
            raise ValueError
    except ValueError as exc:
        raise ConfigurationError(
            "Provider URL must be HTTPS or loopback HTTP, without credentials/query/fragment"
        ) from exc
    return loopback


def _text(value: str, name: str, maximum: int = 256) -> str:
    if not value.strip() or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ConfigurationError(f"{name} must be nonempty text without control characters")
    return value


@dataclass(frozen=True)
class Provider:
    name: str
    api_key: str = field(repr=False)
    base_url: str
    default_model: str


@dataclass(frozen=True)
class Settings:
    providers: dict[str, Provider]
    working_root: Path
    default_provider: str | None = None
    default_model: str | None = None
    claude_command: str = "claude"
    max_concurrent: int = 4
    max_queued: int = 32
    max_retained: int = 128
    retention_seconds: int = 900
    max_output_bytes: int = 1_048_576
    max_timeout_seconds: int = 1200
    max_turns: int = 100
    allowed_tools: frozenset[str] = BUILTIN_TOOLS
    system_prompt: str | None = None
    output_tokens: int = 4096

    def resolve(self, provider: str | None, model: str | None) -> tuple[Provider, str]:
        if not self.providers:
            raise ConfigurationError(
                "No providers configured; set LLM_BASE_URL and LLM_MODEL or a provider key"
            )
        selected = provider or self.default_provider
        target = model or self.default_model
        if selected:
            if selected not in self.providers:
                raise ConfigurationError(
                    "Unknown provider; use llm_status to list configured providers"
                )
            backend = self.providers[selected]
        elif target:
            matches = [p for p in self.providers.values() if p.default_model == target]
            if len(matches) != 1:
                raise ConfigurationError(
                    "Model routing is ambiguous or unknown; specify provider explicitly"
                )
            backend = matches[0]
        elif len(self.providers) == 1:
            backend = next(iter(self.providers.values()))
        else:
            raise ConfigurationError(
                "Multiple providers configured; set DEFAULT_PROVIDER or pass provider"
            )
        return backend, target or backend.default_model

    def directory(self, value: str | None) -> Path:
        candidate = Path(value).expanduser() if value else self.working_root
        if not candidate.is_absolute():
            candidate = self.working_root / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(self.working_root) or not candidate.is_dir():
            raise ConfigurationError(
                "working_directory must be an existing directory inside AGENT_WORKING_ROOT"
            )
        return candidate


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    providers: dict[str, Provider] = {}
    definitions = (
        ("custom", "LLM", "", ""),
        ("minimax", "MINIMAX", "https://api.minimax.io/anthropic", "MiniMax-M2.5"),
        ("zai", "ZAI", "https://api.z.ai/api/anthropic", "glm-5"),
    )
    for name, prefix, url_default, model_default in definitions:
        key = env.get(f"{prefix}_API_KEY", "").strip()
        url = env.get(f"{prefix}_BASE_URL", url_default).strip().rstrip("/")
        model = env.get(f"{prefix}_MODEL", model_default).strip()
        enabled = (
            bool(key)
            if prefix != "LLM"
            else any(env.get(f"LLM_{s}") for s in ("BASE_URL", "MODEL", "API_KEY"))
        )
        if not enabled:
            continue
        loopback = validate_endpoint(url)
        if not key and not loopback:
            raise ConfigurationError(f"{prefix}_API_KEY is required for remote endpoints")
        _text(model, f"{prefix}_MODEL")
        _text(key or "local", f"{prefix}_API_KEY", 8192)
        providers[name] = Provider(name, key or "local", url, model)

    def integer(name: str, default: int, low: int, high: int) -> int:
        try:
            value = int(env.get(name, str(default)))
        except ValueError as exc:
            raise ConfigurationError(f"{name} must be an integer") from exc
        if not low <= value <= high:
            raise ConfigurationError(f"{name} must be between {low} and {high}")
        return value

    root = Path(env.get("AGENT_WORKING_ROOT", str(Path.cwd()))).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError("AGENT_WORKING_ROOT must be an existing directory")
    tools = frozenset(
        t.strip()
        for t in env.get("AGENT_ALLOWED_TOOLS", ",".join(sorted(BUILTIN_TOOLS))).split(",")
        if t.strip()
    )
    if not tools <= BUILTIN_TOOLS:
        raise ConfigurationError("AGENT_ALLOWED_TOOLS contains unsupported tools")
    selected = env.get("DEFAULT_PROVIDER") or None
    default_model = env.get("DEFAULT_MODEL") or None
    if selected and selected not in providers:
        raise ConfigurationError("DEFAULT_PROVIDER is not configured")
    if default_model:
        _text(default_model, "DEFAULT_MODEL")
    prompt = env.get("AGENT_SYSTEM_PROMPT") or None
    if prompt and (len(prompt) > 32_000 or "\x00" in prompt):
        raise ConfigurationError("AGENT_SYSTEM_PROMPT is too long or contains a null byte")
    settings = Settings(
        providers=providers,
        working_root=root,
        default_provider=selected,
        default_model=default_model,
        claude_command=_text(env.get("CLAUDE_BIN", "claude"), "CLAUDE_BIN", 4096),
        max_concurrent=integer("AGENT_MAX_CONCURRENT", 4, 1, 32),
        max_queued=integer("AGENT_MAX_QUEUED", 32, 0, 256),
        max_retained=integer("AGENT_MAX_RETAINED", 128, 1, 1024),
        retention_seconds=integer("AGENT_RETENTION_SECONDS", 900, 1, 86400),
        max_output_bytes=integer("AGENT_MAX_OUTPUT_BYTES", 1_048_576, 4096, 16_777_216),
        max_timeout_seconds=integer("AGENT_MAX_TIMEOUT_SECONDS", 1200, 1, 3600),
        max_turns=integer("AGENT_MAX_TURNS", 100, 1, 1000),
        output_tokens=integer("AGENT_OUTPUT_TOKENS", 4096, 16, 32000),
        allowed_tools=tools,
        system_prompt=prompt,
    )
    if settings.max_retained < settings.max_concurrent + settings.max_queued:
        raise ConfigurationError("AGENT_MAX_RETAINED must cover concurrent and queued capacity")
    return settings


def child_environment(
    settings: Settings, provider: Provider, model: str, config_dir: str
) -> dict[str, str]:
    # Preserve platform/network plumbing, not arbitrary application secrets or Claude overrides.
    keep = {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SYSTEMROOT",
        "SystemRoot",
        "COMSPEC",
        "PATHEXT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
    }
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update(
        {
            "ANTHROPIC_BASE_URL": provider.base_url,
            "ANTHROPIC_API_KEY": provider.api_key,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "CLAUDE_CONFIG_DIR": config_dir,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_PROMPT_CACHING": "1",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(settings.output_tokens),
            "MAX_THINKING_TOKENS": "0",
        }
    )
    return env
