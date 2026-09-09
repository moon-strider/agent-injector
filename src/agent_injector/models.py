"""Input contracts shared by MCP schemas and direct dispatch."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import BUILTIN_TOOLS

ShortText = Annotated[str, Field(min_length=1, max_length=256)]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TaskRequest(Input):
    prompt: Annotated[str, Field(min_length=1, max_length=64_000)]
    context: Annotated[str, Field(max_length=64_000)] | None = None
    provider: ShortText | None = None
    model: ShortText | None = None
    working_directory: Annotated[str, Field(min_length=1, max_length=4096)] | None = None
    allowed_tools: Annotated[list[str], Field(max_length=6)] | None = None
    required_tools: Annotated[list[str], Field(max_length=6)] = Field(default_factory=list)
    timeout_seconds: Annotated[int, Field(ge=1, le=3600)] = 120
    max_turns: Annotated[int, Field(ge=1, le=1000)] = 30

    @field_validator("prompt", "context", "provider", "model", "working_directory")
    @classmethod
    def text_without_null(cls, value: str | None) -> str | None:
        if value is not None and ("\x00" in value or not value.strip()):
            raise ValueError("Text must not be blank or contain null bytes")
        return value

    @field_validator("allowed_tools", "required_tools")
    @classmethod
    def tool_names(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (len(set(value)) != len(value) or not set(value) <= BUILTIN_TOOLS):
            raise ValueError("Tools must be unique supported builtin names")
        return value


class BatchItem(TaskRequest):
    id: ShortText


class BatchRequest(Input):
    tasks: Annotated[list[BatchItem], Field(min_length=1, max_length=64)]
    working_directory: Annotated[str, Field(min_length=1, max_length=4096)] | None = None

    @model_validator(mode="after")
    def unique_ids(self) -> BatchRequest:
        if len({task.id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Batch item ids must be unique")
        return self


class TaskId(Input):
    task_id: ShortText


class BatchId(Input):
    batch_id: ShortText


class ListRequest(Input):
    status: (
        Annotated[str, Field(pattern="^(queued|running|completed|failed|timeout|cancelled)$")]
        | None
    ) = None
    offset: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1, le=100)] = 50
