"""Celery entry point for isolated ingestion and settlement jobs.

Production workers receive provider credentials from the platform secret store.
They are not loaded in the Flask web process and never run model training inline.
"""

from __future__ import annotations

import os

from celery import Celery

celery_app = Celery(
    "sam_analytics",
    broker=os.getenv("REDIS_URL", "redis://redis:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://redis:6379/0"),
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_schedule={
        "ingest-pregame-odds": {
            "task": "sam_analytics.ingest_quotes",
            "schedule": 60.0,
        },
        "settle-completed-events": {
            "task": "sam_analytics.settle_events",
            "schedule": 300.0,
        },
    },
)


@celery_app.task(name="sam_analytics.ingest_quotes")
def ingest_quotes() -> dict:
    """Queue hook for a licensed provider adapter.

    Implement the adapter only after a provider contract supplies the permitted
    sports, refresh rate, storage/redistribution terms, and stable IDs.
    """
    if not os.getenv("ODDS_PROVIDER"):
        return {"status": "skipped", "reason": "ODDS_PROVIDER is not configured"}
    if os.getenv("ODDS_PROVIDER") == "the_odds_api" and not os.getenv("ODDS_PROVIDER_API_KEY"):
        return {"status": "skipped", "reason": "ODDS_PROVIDER_API_KEY is not configured"}
    return {
        "status": "not_configured",
        "reason": "provider adapter is available; audited persistence must be configured before polling",
    }


@celery_app.task(name="sam_analytics.settle_events")
def settle_events() -> dict:
    """Queue hook for a licensed results provider; never synthesizes results."""
    if not os.getenv("RESULTS_PROVIDER"):
        return {"status": "skipped", "reason": "RESULTS_PROVIDER is not configured"}
    return {"status": "not_configured", "reason": "results adapter required"}
