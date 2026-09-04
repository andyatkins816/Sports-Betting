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
# Render supplies PORT at runtime (10000 by default).  Keep 8000 as the local
# fallback, use Render's recommended worker count when available, and send
# request/error logs to stdout for the deployment log explorer.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers ${WEB_CONCURRENCY:-2} --threads 4 --timeout 30 --access-logfile - --error-logfile - wsgi:app"]
