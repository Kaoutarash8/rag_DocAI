FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data HF_HOME=/data/cache/hf PADDLE_PDX_CACHE_HOME=/data/cache/paddlex

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 app && mkdir -p /data && chown app:app /data

WORKDIR /code
COPY requirements.txt .
# PyTorch version CPU (évite ~2 Go de bibliothèques CUDA inutiles ici)
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# PaddleX télécharge des polices dans son propre dossier au premier OCR :
# l'utilisateur non-root doit pouvoir y écrire.
RUN chown -R app:app /usr/local/lib/python3.11/site-packages/paddlex

COPY alembic.ini .
COPY alembic ./alembic
COPY app ./app
COPY scripts ./scripts
RUN chown -R app:app /code
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# 1 worker : ChromaDB local et la file de traitement sont mono-process.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers"]