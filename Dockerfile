FROM python:3.11.15-slim

# Copy UV to docker
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups

COPY src/ ./src/
COPY config/ ./config/

# from_root anchors on .git or .project-root. .git is dockerignored, so give
# it an explicit anchor at /app.
RUN touch .project-root

# Use the venv built above directly. Do NOT use `uv run` at CMD — it re-syncs
# with default groups and drags training deps (sagemaker/xgboost/torch/cuda)
# back into the running container.
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
