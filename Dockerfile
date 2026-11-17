FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GEODUBBER_HOME=/var/data/geodubber \
    HF_HOME=/var/data/geodubber/hf-cache \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-online.txt ./requirements-online.txt
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install -r requirements-online.txt

COPY . .
RUN useradd --system --uid 10001 --create-home geodubber \
    && mkdir -p /var/data/geodubber/projects /var/data/geodubber/cache /var/data/geodubber/models \
    && chown -R geodubber:geodubber /var/data/geodubber \
    && python -m compileall -q config core online_api.py

USER 10001:10001
EXPOSE 10000
CMD ["uvicorn", "online_api:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
