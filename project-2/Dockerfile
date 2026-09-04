FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN addgroup --system sam && adduser --system --ingroup sam sam
COPY pyproject.toml README.md LICENSE ./
COPY sam_analytics ./sam_analytics
COPY migrations ./migrations
COPY routes ./routes
COPY templates ./templates
COPY static ./static
COPY app.py wsgi.py worker.py ./
RUN pip install --upgrade pip && pip install ".[modeling]"
USER sam
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "30", "wsgi:app"]
