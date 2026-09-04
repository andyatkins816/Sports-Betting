"""Flask application factory for SAM Analytics.

The old application imported model classes and provider clients at module load,
which made every web request depend on unvalidated demo data. The factory now
starts only the reviewable API surface; ingestion and model jobs run in workers.
"""

from __future__ import annotations

import logging
import uuid

from flask import Flask, g, request

from sam_analytics.settings import Settings


def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings.from_environment()
    app = Flask(__name__)
    app.config.update(
        ENVIRONMENT=settings.environment,
        SECRET_KEY=settings.secret_key or "development-only-not-for-production",
        SAM_SETTINGS=settings,
        MAX_CONTENT_LENGTH=64 * 1024,
    )
    app.json.sort_keys = False
    app.logger.setLevel(logging.INFO if settings.is_production else logging.DEBUG)

    @app.before_request
    def attach_request_id() -> None:
        # Do not trust a client-provided identifier for logs or audit correlation.
        g.request_id = str(uuid.uuid4())

    @app.after_request
    def security_headers(response):
        response.headers["X-Request-ID"] = g.get("request_id", "")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; base-uri 'self'; frame-ancestors 'none'"
        if request.is_secure or settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        origin = request.headers.get("Origin")
        if origin and origin in settings.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Vary"] = "Origin"
        return response

    from routes.api import bp as api_bp
    from routes.views import bp as views_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(views_bp)
    return app


# Flask and Gunicorn entry point. Validation remains fail-closed in production.
app = create_app()
