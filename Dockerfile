FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LITSCOUT_DB_PATH=/data/litscout.db
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY litscout ./litscout
RUN pip install --no-cache-dir . && useradd -m app && mkdir /data && chown app /data
USER app
VOLUME ["/data"]
# Cloud Run and most PaaS hosts inject $PORT; default to 8000 locally
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/healthz')"
CMD ["sh", "-c", "exec uvicorn litscout.main:app --host 0.0.0.0 --port ${PORT}"]
