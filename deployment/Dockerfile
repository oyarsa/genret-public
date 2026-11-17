# Graft: torch-free image. The encoder runs on onnxruntime, so only the
# base dependencies are installed -- no CPU torch extra, no training
# extra. Corpus data and the exported ONNX encoder are far too large to
# bake in; they are bind-mounted at /data (see docker-compose.yml).

# --- Frontend build -------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Runtime --------------------------------------------------------
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
# Every dependency resolves to a wheel, so no build toolchain is needed.
# --no-editable installs the project into the venv, so nothing outside it
# is required at runtime.
RUN uv sync --frozen --no-dev --no-editable --no-cache

COPY --from=frontend /build/dist ./frontend/dist

EXPOSE 8000

# Loopback is the safe default for the CLI, but inside a container the
# port is only reachable on the compose network, and Caddy proxies to it.
# --proxy-headers makes the rate limiter count real client addresses
# rather than the proxy's.
CMD ["/app/.venv/bin/graft", "serve", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--edge-dir", "/data/edges", \
     "--dense-dir", "/data/dense", \
     "--facets-path", "/data/dense/facets.jsonl.zst", \
     "--metadata-dir", "/data/metadata", \
     "--s2orc-dir", "/data/s2orc", \
     "--onnx-dir", "/data/minilm_onnx", \
     "--cache-dir", "/cache", \
     "--frontend-dir", "/app/frontend/dist"]
