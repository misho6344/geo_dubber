FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GEODUBBER_HOME=/var/data/geodubber

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgomp1 libglib2.0-0 libgl1 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-online.txt ./requirements-online.txt
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements-online.txt

COPY . .
RUN mkdir -p /var/data/geodubber/projects /var/data/geodubber/cache \
    && python -m compileall -q config core online_api.py

EXPOSE 10000
CMD ["sh", "-c", "uvicorn online_api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
