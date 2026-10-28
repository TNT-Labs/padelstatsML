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
 && apt-get install -y --no-install-recommends libglib2.0-0 curl gpg \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Runtime for the Raspberry Pi AI HAT+ (Hailo), from the Raspberry Pi
# repository: HailoRT, its Python binding and the compiled networks
# (/usr/share/hailo-models). Used only when the board is fitted and the
# worker is given /dev/hailo0 (docker-compose.hailo.yml); without it the
# analysis runs on the CPU. Best effort: an image built where the repository
# cannot be reached still works, on the CPU. The binding is for Debian's
# Python 3.11, the same as this image's; the .pth puts Debian's packages
# after pip's, so pip's numpy and the rest still win.
ARG HAILO=auto
RUN if [ "$HAILO" != "no" ] && [ "$(dpkg --print-architecture)" = "arm64" ]; then \
      ( curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
          | gpg --dearmor -o /usr/share/keyrings/raspberrypi.gpg \
        && echo "deb [signed-by=/usr/share/keyrings/raspberrypi.gpg] https://archive.raspberrypi.com/debian/ bookworm main" \
          > /etc/apt/sources.list.d/raspi.list \
        && apt-get update \
        && apt-get install -y --no-install-recommends hailort python3-hailort hailo-models \
        && echo /usr/lib/python3/dist-packages > "$(python -c 'import site; print(site.getsitepackages()[0])')/zz-debian.pth" \
        && python -c "import hailo_platform" \
        && echo "HailoRT installato" ) \
      || echo "ATTENZIONE: HailoRT non installato, l'acceleratore non sarà usato"; \
      rm -rf /var/lib/apt/lists/*; \
    fi

COPY backend/app    ./app
COPY backend/scripts ./scripts
COPY --from=web /web/dist ./web

# Exported models (detector, re-ID) are mounted at runtime, not baked in.
VOLUME ["/data", "/srv/weights"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
