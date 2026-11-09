FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser models/champion.keras models/champion.json ./models/
RUN mkdir -p instance && chown appuser:appuser /app /app/instance

USER appuser

ENV MODEL_PATH=models/champion.keras \
    MODEL_META_PATH=models/champion.json \
    THRESHOLD=0.5 \
    MAX_UPLOAD_MB=5 \
    PORT=5050

EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5050/health', timeout=4).status == 200 else 1)"

CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "1", "--threads", "4", "--timeout", "120", "app.main:app"]
