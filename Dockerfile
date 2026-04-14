FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home appuser

# Install dependencies in a cached layer that only invalidates when
# pyproject.toml changes. The dep list is extracted from pyproject.toml at
# build time so there's no duplication between the two files.
COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir --upgrade pip && \
    python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy the full source tree and register the project itself. --no-deps is
# safe because the deps layer above has already resolved everything, and it
# keeps this layer fast: editing a single source file re-runs a few-second
# package registration instead of a multi-minute dependency solve.
COPY . /app
RUN pip install --no-cache-dir --no-deps .

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "chargeback_ops.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
