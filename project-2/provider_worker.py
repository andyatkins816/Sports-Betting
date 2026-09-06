"""Fail-closed Celery entry point for one manual provider-shadow request.

This is intentionally separate from both the public web service and the
synthetic storage-probe worker.  It can make one narrowly scoped The Odds API
request only when every staging, license, database, broker, and evidence-store
setting passes admission.  It has no schedule, retry path, results backend, or
public-output path.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import replace
from uuid import UUID

from celery import Celery
from kombu import Queue

from sam_analytics.ingestion import RawOddsQuote
from sam_analytics.ingestion_run_repository import PostgresIngestionRunRepository
from sam_analytics.ingestion_runs import IngestionFailureCode
from sam_analytics.odds_ledger import OddsLedger
from sam_analytics.provider_contracts import (
    ApprovedProviderContract,
    ProviderContractRegistry,
)
from sam_analytics.provider_shadow import (
    ManualProviderShadowOrchestrator,
    ProviderShadowFetchFailure,
)
from sam_analytics.provider_shadow_settings import (
    PROVIDER_SHADOW_ADMISSION_RUN_ID,
    ProviderShadowConfigurationError,
    ProviderShadowSettings,
)
from sam_analytics.providers.the_odds_api import (
    OddsApiFetch,
    OddsApiRequestScope,
    TheOddsApiClient,
    TheOddsApiError,
)
from sam_analytics.s3_payload_store import S3CompatibleRawPayloadStore

WorkerConfigurationError = ProviderShadowConfigurationError

_PROVIDER_SHADOW_QUEUE = "sam_provider_shadow"
_PROVIDER_SHADOW_TASK = "sam_analytics.ingest_the_odds_api_shadow"
_HTTP_STATUS_RE = re.compile(r"^The Odds API returned HTTP ([1-5][0-9]{2})$")
_RAW_ONLY_PROVIDER_ADDED_MARKETS = frozenset({"h2h_lay"})


def create_celery_app(environ: Mapping[str, str] | None = None) -> Celery:
    """Build the worker only inside the exact provider-shadow boundary."""

    source = os.environ if environ is None else environ
    ProviderShadowSettings.from_environment(source)
    app = Celery(
        "sam_provider_shadow",
        broker=source["REDIS_URL"],
        backend="disabled://",
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_ignore_result=True,
        task_store_errors_even_if_ignored=False,
        task_track_started=False,
        # In the same durable audit database, the fixed, code-reviewed run ID
        # prevents redelivery or repeated manual publication from reaching the
        # provider.
        task_acks_late=False,
        task_reject_on_worker_lost=False,
        worker_prefetch_multiplier=1,
        worker_concurrency=1,
        task_soft_time_limit=75,
        task_time_limit=90,
        task_default_queue=_PROVIDER_SHADOW_QUEUE,
        task_queues=(Queue(_PROVIDER_SHADOW_QUEUE),),
        task_create_missing_queues=False,
        task_routes={
            _PROVIDER_SHADOW_TASK: {"queue": _PROVIDER_SHADOW_QUEUE},
        },
        task_send_sent_event=False,
        worker_send_task_events=False,
        worker_enable_remote_control=False,
        beat_schedule={},
    )
    return app


def _celery_job_identity(task_id: object) -> str:
    """Convert only a canonical Celery UUID into a safe audit identity."""

    if not isinstance(task_id, str):
        raise WorkerConfigurationError("the manual provider-shadow task identity is invalid")
    parsed: UUID | None = None
    parse_failed = False
    try:
        parsed = UUID(task_id)
    except (AttributeError, TypeError, ValueError):
        parse_failed = True
    if parse_failed or parsed is None or str(parsed) != task_id.lower():
        raise WorkerConfigurationError("the manual provider-shadow task identity is invalid")
    return f"celery:{parsed}"


def _classify_provider_error(error: TheOddsApiError) -> IngestionFailureCode:
    """Collapse a credential-safe adapter error into one finite audit code."""

    message = str(error)
    matched = _HTTP_STATUS_RE.fullmatch(message)
    if matched is not None:
        status = int(matched.group(1))
        if status == 429:
            return IngestionFailureCode.PROVIDER_RATE_LIMITED
        if status >= 500:
            return IngestionFailureCode.PROVIDER_TEMPORARY_UNAVAILABLE
        return IngestionFailureCode.PROVIDER_RESPONSE_INVALID
    if message == "The Odds API request failed":
        return IngestionFailureCode.PROVIDER_TEMPORARY_UNAVAILABLE
    return IngestionFailureCode.PROVIDER_RESPONSE_INVALID


def _fetch_once(
    client: TheOddsApiClient,
    settings: ProviderShadowSettings,
) -> OddsApiFetch:
    """Fetch the admitted scope and require a bounded provider quota receipt."""

    failure_code: IngestionFailureCode | None = None
    fetched: OddsApiFetch | None = None
    try:
        fetched = client.fetch_pregame_odds(
            settings.sport_key,
            regions=",".join(settings.regions),
            markets=settings.markets,
        )
    except TheOddsApiError as error:
        failure_code = _classify_provider_error(error)
    if failure_code is not None:
        # Raise after leaving the handler so the provider exception (which may
        # retain implementation details programmatically) is not chained.
        raise ProviderShadowFetchFailure(failure_code)
    if not isinstance(fetched, OddsApiFetch):
        raise ProviderShadowFetchFailure(IngestionFailureCode.PROVIDER_RESPONSE_INVALID)

    quota_values = (
        fetched.requests_remaining,
        fetched.requests_used,
        fetched.request_cost,
    )
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in quota_values
    ) or fetched.request_cost not in (0, 1):
        raise ProviderShadowFetchFailure(IngestionFailureCode.PROVIDER_RESPONSE_INVALID)
    expected_scope = OddsApiRequestScope(
        sport_key=settings.sport_key,
        regions=settings.regions,
        markets=settings.markets,
    )
    if fetched.request_scope != expected_scope:
        raise ProviderShadowFetchFailure(IngestionFailureCode.PROVIDER_RESPONSE_INVALID)
    if not isinstance(fetched.quotes, list):
        raise ProviderShadowFetchFailure(IngestionFailureCode.PROVIDER_RESPONSE_INVALID)

    # The provider documents that an exchange can add h2h_lay when h2h is
    # requested.  Preserve those bytes in raw evidence, but do not normalize
    # the added market into the exact h2h-only analytics boundary.  Reject any
    # other returned provider, sport, or market mismatch.
    admitted_quotes: list[RawOddsQuote] = []
    for quote in fetched.quotes:
        if (
            not isinstance(quote, RawOddsQuote)
            or quote.provider != settings.provider
            or quote.sport != settings.sport_key
        ):
            raise ProviderShadowFetchFailure(IngestionFailureCode.PROVIDER_RESPONSE_INVALID)
        if quote.market in settings.markets:
            admitted_quotes.append(quote)
        elif quote.market not in _RAW_ONLY_PROVIDER_ADDED_MARKETS:
            raise ProviderShadowFetchFailure(IngestionFailureCode.PROVIDER_RESPONSE_INVALID)

    return (
        fetched
        if len(admitted_quotes) == len(fetched.quotes)
        else replace(fetched, quotes=admitted_quotes)
    )


def _execute_provider_shadow(
    *,
    task_id: object,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Revalidate admission, construct private dependencies, and run once."""

    source = os.environ if environ is None else environ
    settings = ProviderShadowSettings.from_environment(source)
    job_identity = _celery_job_identity(task_id)

    client = TheOddsApiClient(
        source["ODDS_PROVIDER_API_KEY"],
        max_response_bytes=settings.raw_evidence_max_bytes,
    )
    raw_payload_store = S3CompatibleRawPayloadStore.from_environment(source)
    contracts = ProviderContractRegistry(
        [
            ApprovedProviderContract(
                provider=settings.provider,
                license_scope=settings.license_scope,
                license_version=settings.license_version,
                permitted_source_types=frozenset({"odds"}),
            )
        ]
    )
    ledger = OddsLedger(
        source["DATABASE_URL"],
        raw_payload_store=raw_payload_store,
        provider_contracts=contracts,
    )
    run_repository = PostgresIngestionRunRepository(source["DATABASE_URL"])
    orchestrator = ManualProviderShadowOrchestrator(
        provider_fetch=lambda: _fetch_once(client, settings),
        odds_ledger=ledger,
        ingestion_run_repository=run_repository,
    )
    # Celery result storage is disabled.  Do not return a digest, receipt,
    # object location, quota count, normalized value, or provider content.
    orchestrator.run(
        job_identity=job_identity,
        license_scope=settings.license_scope,
        license_version=settings.license_version,
        run_id=PROVIDER_SHADOW_ADMISSION_RUN_ID,
    )


celery_app = create_celery_app()


@celery_app.task(
    bind=True,
    name=_PROVIDER_SHADOW_TASK,
    queue=_PROVIDER_SHADOW_QUEUE,
    ignore_result=True,
    acks_late=False,
    reject_on_worker_lost=False,
    max_retries=0,
)
def ingest_the_odds_api_shadow(task) -> None:
    """Run one operator-dispatched private provider-shadow ingestion."""

    _execute_provider_shadow(task_id=task.request.id)
