FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt
COPY . .
RUN groupadd --system fraud && useradd --system --gid fraud --home /app fraud \
    && chown -R fraud:fraud /app
USER fraud
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/live', timeout=3)" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
