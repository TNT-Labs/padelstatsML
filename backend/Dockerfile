# Single image for both services (api and worker) on Raspberry Pi 5 / ARM64.
#
# Build context is the repository root:
#   docker compose -f docker-compose.pi.yml build
#
# Stage 1 builds the web UI; stage 2 is the runtime. The API serves the built
# UI itself, so there is no nginx container.

# ── Stage 1: web UI ──────────────────────────────────────────────────────────
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

# opencv-python-headless still needs glib; everything else in the dependency
# set ships prebuilt aarch64 wheels, so nothing is compiled here.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Raspberry Pi AI HAT+ (Hailo). HailoRT is not installed here: the worker
# uses the host's own (hailo-all), mounted by docker-compose.hailo.yml, so
# the runtime is always the version of the host's driver — a different one
# refuses to talk to the board. Here only what its Python binding needs
# around it: Debian's Python 3.11 packages it imports, found after pip's
# (the .pth goes last, so pip's numpy and the rest still win). Without the
# board none of this is used.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libstdc++6 \
      python3-netaddr python3-netifaces python3-future python3-argcomplete python3-contextlib2 \
 && echo /usr/lib/python3/dist-packages \
      > "$(python -c 'import site; print(site.getsitepackages()[0])')/zz-debian.pth" \
 && rm -rf /var/lib/apt/lists/*

COPY backend/app    ./app
COPY backend/scripts ./scripts
COPY --from=web /web/dist ./web

# Exported models (detector, re-ID) are mounted at runtime, not baked in.
VOLUME ["/data", "/srv/weights"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
