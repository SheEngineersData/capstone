FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 TF_USE_LEGACY_KERAS=1 DATA_DIR=/data ARTIFACT_DIR=/app/artifacts
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY potatofl/ potatofl/
COPY service/ service/
COPY artifacts/ artifacts/
COPY samples/ samples/
# ImageNet backbone weights are not needed: the full model, backbone included, is in artifacts/model.weights.h5
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
