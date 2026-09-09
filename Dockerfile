FROM node:22-bookworm-slim AS claude
ARG CLAUDE_CODE_VERSION=2.1.266
RUN npm install --prefix /opt/claude --no-audit --no-fund \
    @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

FROM ghcr.io/astral-sh/uv:0.12.8 AS uv
FROM python:3.12-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=claude /opt/claude /opt/claude
ENV PATH="/opt/claude/node_modules/.bin:/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable \
    && useradd --create-home --uid 10001 agent \
    && mkdir /work && chown agent:agent /work
ENV AGENT_WORKING_ROOT=/work
USER agent
WORKDIR /work
ENTRYPOINT ["agent-injector"]
