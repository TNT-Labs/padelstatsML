FROM python:3.12-slim

WORKDIR /app

# OpenCV headless runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight deps first — layer stays cached on requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install large ML wheels separately with a generous read timeout
COPY requirements.ml.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.ml.txt

# App source
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
