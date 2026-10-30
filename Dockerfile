# Image that runs the feature pipeline. GitHub Actions runs the scheduled jobs
# inside this image, so what is tested and graded is what runs in production.
FROM python:3.13-slim

# uv installs the exact versions from uv.lock (reproducible builds).
COPY --from=ghcr.io/astral-sh/uv:0.5.9 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first: this layer is cached as long as the lockfile is unchanged.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY config ./config
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Usage: docker run --rm --env-file .env -v "$GOOGLE_APPLICATION_CREDENTIALS:/secrets/key.json:ro" \
#          -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/key.json soil-pipeline ingest
ENTRYPOINT ["python", "-m", "features"]
CMD ["--help"]
