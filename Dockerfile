FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./pyproject.toml
COPY params.yaml ./params.yaml
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/outputs \
    && chown -R appuser:appuser /app
USER appuser

CMD ["sh", "-c", "python -m iris_pipeline.data.prepare && python -m iris_pipeline.training.train"]
