FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LITSCOUT_DB_PATH=/data/litscout.db
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY litscout ./litscout
RUN pip install --no-cache-dir . && useradd -m app && mkdir /data && chown app /data
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "litscout.main:app", "--host", "0.0.0.0", "--port", "8000"]
