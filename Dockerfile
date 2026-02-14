FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml /app/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /app/pyproject.toml --no-dev --no-editable

ADD src /app/src

ENV MINIMAX_API_KEY=""
ENV MINIMAX_MODEL="MiniMax-M2.5"
ENV MINIMAX_BASE_URL="https://api.minimax.io/anthropic"

ENTRYPOINT ["uv", "run", "agent-injector"]
