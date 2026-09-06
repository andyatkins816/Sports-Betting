"""Fail-closed Celery entry point for bounded provider ingestion.

This is intentionally separate from both the public web service and the
synthetic storage-probe worker. It makes narrowly scoped The Odds API requests
only when every staging, license, database, broker, and evidence-store setting
passes admission. Pregame odds run every five minutes and recent final scores
run hourly, with no retry path, Celery result backend, or public-output path.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import joblib
import psycopg
from celery import Celery
from kombu import Queue

from sam_analytics.ingestion import RawOddsQuote
from sam_analytics.ingestion_run_repository import PostgresIngestionRunRepository
from sam_analytics.ingestion_runs import (
    IngestionFailureCode,
    IngestionRun,
    IngestionRunStateTransition,
    mark_failed,
    mark_succeeded,
    new_manual_shadow_run,
    start_next_attempt,
)
from sam_analytics.modeling import (
    PREGAME_H2H_FEATURE_SCHEMA,
    CandidateEvaluation,
    FittedProbabilityModel,
    ProbabilityMetrics,
    ProbabilityModelEvaluator,
    PromotionDecision,
    PromotionPolicy,
    RollingTimeSplitter,
    default_model_candidates,
    evaluate_candidate_promotion,
    feature_schema_fingerprint,
    fit_approved_model,
    load_h2h_market_training_rows,
    select_best_candidate,
)
from sam_analytics.odds_ledger import (
    OddsLedger,
    OddsLedgerValidationError,
    ResultsLedgerWriteResult,
    prepare_the_odds_api_results_payload,
)
from sam_analytics.provider_contracts import (
    ApprovedProviderContract,
    ProviderContractRegistry,
)
from sam_analytics.provider_shadow import (
    ManualProviderShadowOrchestrator,
    ProviderShadowFetchFailure,
    ProviderShadowRunFailed,
    ProviderShadowUnavailable,
)
from sam_analytics.provider_shadow_settings import (
    ProviderShadowConfigurationError,
    ProviderShadowSettings,
)
from sam_analytics.providers.the_odds_api import (
    OddsApiFetch,
    OddsApiRequestScope,
    ScoresApiFetch,
    ScoresApiRequestScope,
    TheOddsApiClient,
    TheOddsApiError,
)
from sam_analytics.s3_payload_store import S3CompatibleRawPayloadStore

WorkerConfigurationError = ProviderShadowConfigurationError

_PROVIDER_SHADOW_QUEUE = "sam_provider_shadow"
_PROVIDER_SHADOW_TASK = "sam_analytics.ingest_the_odds_api_shadow"
_PROVIDER_SHADOW_INTERVAL_SECONDS = 5 * 60
_PROVIDER_SCORES_TASK = "sam_analytics.settle_the_odds_api_scores"
_PROVIDER_SCORES_INTERVAL_SECONDS = 60 * 60
_PROVIDER_SCORES_EXPIRY_SECONDS = 55 * 60
_MODEL_TRAINING_TASK = "sam_analytics.train_model_candidate"
_MODEL_MINIMUM_ROWS = 750
_MODEL_ARTIFACT_FORMAT = "joblib-sklearn-v1"
_PROVIDER_RESULTS_SOURCE_TYPE = "result"
_ACCEPTED_RESULTS_LEDGER_STATUSES = frozenset({"accepted", "accepted_empty"})
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
        # A redelivered task keeps its Celery UUID, which is also the durable
        # ingestion-run ID. The database therefore rejects that redelivery
        # before another provider request can start.
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
            _PROVIDER_SCORES_TASK: {"queue": _PROVIDER_SHADOW_QUEUE},
            _MODEL_TRAINING_TASK: {"queue": _PROVIDER_SHADOW_QUEUE},
        },
        task_send_sent_event=False,
        worker_send_task_events=False,
        worker_enable_remote_control=False,
        beat_schedule_filename=os.path.join(
            tempfile.gettempdir(), "sam-provider-shadow-celerybeat-schedule"
        ),
        beat_schedule={
            "ingest-the-odds-api-shadow-every-five-minutes": {
                "task": _PROVIDER_SHADOW_TASK,
                "schedule": _PROVIDER_SHADOW_INTERVAL_SECONDS,
                # Never drain stale scheduled work in a burst after downtime.
                "options": {"queue": _PROVIDER_SHADOW_QUEUE, "expires": 4 * 60},
            },
            "settle-the-odds-api-scores-hourly": {
                "task": _PROVIDER_SCORES_TASK,
                "schedule": _PROVIDER_SCORES_INTERVAL_SECONDS,
                "options": {
                    "queue": _PROVIDER_SHADOW_QUEUE,
                    "expires": _PROVIDER_SCORES_EXPIRY_SECONDS,
                },
            },
        },
    )
    return app


def _canonical_task_uuid(task_id: object) -> UUID:
    """Return only a canonical Celery UUID for durable run identity."""

    if not isinstance(task_id, str):
        raise WorkerConfigurationError("the provider-shadow task identity is invalid")
    parsed: UUID | None = None
    parse_failed = False
    try:
        parsed = UUID(task_id)
    except (AttributeError, TypeError, ValueError):
        parse_failed = True
    if parse_failed or parsed is None or str(parsed) != task_id.lower():
        raise WorkerConfigurationError("the provider-shadow task identity is invalid")
    return parsed


def _celery_job_identity(task_id: object) -> str:
    """Convert only a canonical Celery UUID into a safe audit identity."""

    return f"celery:{_canonical_task_uuid(task_id)}"


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


def _fetch_scores_once(
    client: TheOddsApiClient,
    settings: ProviderShadowSettings,
) -> ScoresApiFetch:
    """Fetch one exact recent-score scope and sanitize provider failures."""

    fetched: ScoresApiFetch | None = None
    failure_code: IngestionFailureCode | None = None
    try:
        fetched = client.fetch_scores(settings.sport_key, days_from=3)
    except TheOddsApiError as error:
        failure_code = _classify_provider_error(error)
    if failure_code is not None:
        raise ProviderShadowFetchFailure(failure_code)
    if not isinstance(fetched, ScoresApiFetch):
        raise ProviderShadowFetchFailure(IngestionFailureCode.PROVIDER_RESPONSE_INVALID)
    quota_values = (
        fetched.requests_remaining,
        fetched.requests_used,
        fetched.request_cost,
    )
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in quota_values
    ) or fetched.request_cost != 2:
        raise ProviderShadowFetchFailure(IngestionFailureCode.PROVIDER_RESPONSE_INVALID)
    if fetched.request_scope != ScoresApiRequestScope(
        sport_key=settings.sport_key,
        days_from=3,
    ) or not isinstance(fetched.scores, tuple):
        raise ProviderShadowFetchFailure(IngestionFailureCode.PROVIDER_RESPONSE_INVALID)
    return fetched


def _execute_provider_shadow(
    *,
    task_id: object,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Revalidate admission, construct private dependencies, and run once."""

    source = os.environ if environ is None else environ
    settings = ProviderShadowSettings.from_environment(source)
    run_id = _canonical_task_uuid(task_id)
    job_identity = f"celery:{run_id}"

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
                permitted_source_types=frozenset({"odds", "result"}),
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
        run_id=run_id,
    )


def _append_score_transition(
    repository: PostgresIngestionRunRepository,
    run: IngestionRun,
    previous: IngestionRunStateTransition,
    transition: IngestionRunStateTransition,
) -> None:
    persisted: object | None = None
    repository_failed = False
    try:
        persisted = repository.append_transition(run, previous, transition)
    except Exception:
        repository_failed = True
    if repository_failed or persisted != transition:
        raise ProviderShadowUnavailable("provider score audit repository is unavailable")


def _fail_score_run(
    repository: PostgresIngestionRunRepository,
    run: IngestionRun,
    running: IngestionRunStateTransition,
    failure_code: IngestionFailureCode,
) -> None:
    failed = mark_failed(
        run,
        running,
        failure_code=failure_code,
        occurred_at=datetime.now(UTC),
    )
    _append_score_transition(repository, run, running, failed)
    raise ProviderShadowRunFailed(
        ingestion_run_id=run.id,
        failure_code=failure_code,
    )


def _execute_score_settlement(
    *,
    task_id: object,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fetch and persist one audited recent-final-score response."""

    source = os.environ if environ is None else environ
    settings = ProviderShadowSettings.from_environment(source)
    run_id = _canonical_task_uuid(task_id)
    job_identity = f"celery:{run_id}"

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
                permitted_source_types=frozenset({"odds", "result"}),
            )
        ]
    )
    ledger = OddsLedger(
        source["DATABASE_URL"],
        raw_payload_store=raw_payload_store,
        provider_contracts=contracts,
    )
    repository = PostgresIngestionRunRepository(source["DATABASE_URL"])

    created_at = datetime.now(UTC)
    run, queued = new_manual_shadow_run(
        provider=settings.provider,
        job_identity=job_identity,
        source_type=_PROVIDER_RESULTS_SOURCE_TYPE,
        max_attempts=1,
        created_at=created_at,
        run_id=run_id,
    )
    persisted: object | None = None
    repository_failed = False
    try:
        persisted = repository.create_run(run, queued)
    except Exception:
        repository_failed = True
    if repository_failed or persisted != queued:
        raise ProviderShadowUnavailable("provider score audit repository is unavailable")

    running = start_next_attempt(run, queued, occurred_at=datetime.now(UTC))
    _append_score_transition(repository, run, queued, running)

    fetched: ScoresApiFetch | None = None
    failure_code: IngestionFailureCode | None = None
    try:
        fetched = _fetch_scores_once(client, settings)
    except ProviderShadowFetchFailure as error:
        failure_code = error.failure_code
    except Exception:
        failure_code = IngestionFailureCode.INTERNAL_TRANSIENT
    if failure_code is not None:
        _fail_score_run(repository, run, running, failure_code)

    payload = None
    preparation_failed = False
    try:
        payload = prepare_the_odds_api_results_payload(
            fetched,
            license_scope=settings.license_scope,
            license_version=settings.license_version,
        )
    except Exception:
        preparation_failed = True
    if preparation_failed:
        _fail_score_run(
            repository,
            run,
            running,
            IngestionFailureCode.PROVIDER_RESPONSE_INVALID,
        )

    ledger_result: ResultsLedgerWriteResult | None = None
    failure_code = None
    try:
        ledger_result = ledger.persist_results(payload, now=datetime.now(UTC))
    except OddsLedgerValidationError:
        failure_code = IngestionFailureCode.EVIDENCE_VALIDATION_FAILED
    except Exception:
        failure_code = IngestionFailureCode.INTERNAL_TRANSIENT
    if failure_code is not None:
        _fail_score_run(repository, run, running, failure_code)
    if (
        not isinstance(ledger_result, ResultsLedgerWriteResult)
        or ledger_result.status not in _ACCEPTED_RESULTS_LEDGER_STATUSES
    ):
        _fail_score_run(
            repository,
            run,
            running,
            IngestionFailureCode.EVIDENCE_VALIDATION_FAILED,
        )

    succeeded = mark_succeeded(run, running, occurred_at=datetime.now(UTC))
    _append_score_transition(repository, run, running, succeeded)


def _metric_payload(metrics: ProbabilityMetrics) -> dict[str, int | float]:
    return {
        "sample_size": metrics.sample_size,
        "brier": metrics.brier,
        "logloss": metrics.logloss,
        "expected_calibration_error": metrics.expected_calibration_error,
    }


def _evaluation_payload(
    evaluation: CandidateEvaluation,
    decision: PromotionDecision,
) -> dict[str, Any]:
    return {
        "candidate": {
            "name": evaluation.candidate.name,
            "family": evaluation.candidate.family,
            "random_state": evaluation.candidate.random_state,
            "hyperparameters": dict(evaluation.candidate.hyperparameters),
        },
        "data_fingerprint_sha256": evaluation.data_fingerprint,
        "evaluated_rows": evaluation.score.evaluated_rows,
        "total_rows": evaluation.score.total_rows,
        "fold_count": evaluation.score.fold_count,
        "coverage": evaluation.score.coverage,
        "raw_metrics": _metric_payload(evaluation.raw_metrics),
        "calibrated_metrics": _metric_payload(evaluation.score.metrics),
        "promotion_gates_passed": decision.approved,
        "promotion_gate_reasons": list(decision.reasons),
    }


def _serialize_model_candidate(model: FittedProbabilityModel) -> tuple[bytes, str]:
    """Serialize only the fitted candidate; governance supplies release time later."""

    artifact_payload = {
        "format_version": 1,
        "candidate": {
            "name": model.candidate.name,
            "family": model.candidate.family,
            "random_state": model.candidate.random_state,
            "hyperparameters": dict(model.candidate.hyperparameters),
        },
        "schema": {
            "schema_id": model.schema.schema_id,
            "features": [
                {
                    "name": feature.name,
                    "minimum": feature.minimum,
                    "maximum": feature.maximum,
                }
                for feature in model.schema.features
            ],
            "reject_unknown_features": model.schema.reject_unknown_features,
        },
        "estimator": model.estimator,
        "calibrator": model.calibrator,
        "training_rows": model.training_rows,
    }
    buffer = io.BytesIO()
    joblib.dump(artifact_payload, buffer, compress=3, protocol=5)
    artifact = buffer.getvalue()
    if not artifact:
        raise RuntimeError("model candidate serialization produced an empty artifact")
    return artifact, hashlib.sha256(artifact).hexdigest()


def _persist_model_candidate(
    *,
    database_url: str,
    version: str,
    sport: str,
    schema_sha256: str,
    artifact: bytes,
    artifact_sha256: str,
    training_cutoff: datetime,
    validation_report: Mapping[str, Any],
) -> bool:
    """Insert one immutable, unapproved model candidate and a safe status fact."""

    connection = None
    inserted = False
    failed = False
    try:
        connection = psycopg.connect(
            database_url,
            application_name="sam-model-training",
            connect_timeout=5,
            options="-c statement_timeout=10000",
        )
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO model_registry (
                    version, sport, target_definition, feature_contract_sha256,
                    artifact_uri, artifact_sha256, artifact_format, artifact_bytes,
                    training_data_cutoff, validation_report, approval_status
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'candidate'
                )
                ON CONFLICT (version) DO NOTHING
                RETURNING id
                """,
                (
                    version,
                    sport,
                    "home_team_wins",
                    schema_sha256,
                    f"postgresql:model_registry/{version}",
                    artifact_sha256,
                    _MODEL_ARTIFACT_FORMAT,
                    artifact,
                    training_cutoff,
                    json.dumps(
                        validation_report,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                ),
            )
            inserted = cursor.fetchone() is not None
            if inserted:
                cursor.execute(
                    """
                    INSERT INTO operational_signal (
                        signal_type, observed_at, source, provenance_sha256, payload
                    ) VALUES ('model', %s, %s, %s, %s::jsonb)
                    """,
                    (
                        training_cutoff,
                        "model-training-worker",
                        artifact_sha256,
                        json.dumps(
                            {
                                "status": "candidate_registered",
                                "model_version": version,
                                "approval_status": "candidate",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
        connection.commit()
    except Exception:
        failed = True
    finally:
        if connection is not None:
            if failed:
                try:
                    connection.rollback()
                except Exception:
                    failed = True
            try:
                connection.close()
            except Exception:
                failed = True
    if failed:
        raise WorkerConfigurationError("model candidate registry is unavailable") from None
    return inserted


def _execute_model_training(environ: Mapping[str, str] | None = None) -> None:
    """Evaluate real settled rows and register a gated candidate, never an approval."""

    source = os.environ if environ is None else environ
    settings = ProviderShadowSettings.from_environment(source)
    training_cutoff = datetime.now(UTC)
    try:
        rows = load_h2h_market_training_rows(
            source["DATABASE_URL"],
            sport=settings.sport_key,
            training_cutoff=training_cutoff,
        )
    except Exception:
        raise WorkerConfigurationError("model training data is unavailable") from None
    if len(rows) < _MODEL_MINIMUM_ROWS:
        print(
            "model training waiting for settled history: "
            f"{len(rows)} eligible rows; {_MODEL_MINIMUM_ROWS} required"
        )
        return

    splitter = RollingTimeSplitter()
    policy = PromotionPolicy()
    evaluator = ProbabilityModelEvaluator(PREGAME_H2H_FEATURE_SCHEMA, splitter)
    evaluations = evaluator.evaluate_many(default_model_candidates(), rows)
    decisions = tuple(
        evaluate_candidate_promotion(evaluation, policy)
        for evaluation in evaluations
    )
    eligible = tuple(
        evaluation
        for evaluation, decision in zip(evaluations, decisions)
        if decision.approved
    )
    if not eligible:
        raise WorkerConfigurationError("no model candidate passed the promotion gates")

    selected = select_best_candidate(eligible)
    trained_at = datetime.now(UTC)
    fitted = fit_approved_model(
        selected,
        rows,
        PREGAME_H2H_FEATURE_SCHEMA,
        trained_at=trained_at,
        promotion_policy=policy,
    )
    try:
        artifact, artifact_sha256 = _serialize_model_candidate(fitted)
    except Exception:
        raise WorkerConfigurationError("model candidate serialization failed") from None
    schema_sha256 = feature_schema_fingerprint(PREGAME_H2H_FEATURE_SCHEMA)
    version = (
        f"sam-{settings.sport_key}-{schema_sha256[:12]}-"
        f"{selected.data_fingerprint[:12]}-{selected.candidate.name}-"
        f"{artifact_sha256[:12]}"
    )
    validation_report = {
        "schema_id": PREGAME_H2H_FEATURE_SCHEMA.schema_id,
        "schema_sha256": schema_sha256,
        "data_fingerprint_sha256": selected.data_fingerprint,
        "training_cutoff": training_cutoff.isoformat(),
        "trained_at": trained_at.isoformat(),
        "training_rows": len(rows),
        "selected_candidate": selected.candidate.name,
        "splitter": {
            "min_train_rows": splitter.min_train_rows,
            "validation_rows": splitter.validation_rows,
            "step_rows": splitter.effective_step_rows,
            "embargo_rows": splitter.embargo_rows,
        },
        "promotion_policy": {
            "minimum_evaluated_rows": policy.minimum_evaluated_rows,
            "minimum_coverage": policy.minimum_coverage,
            "maximum_brier": policy.maximum_brier,
            "maximum_logloss": policy.maximum_logloss,
            "maximum_expected_calibration_error": (
                policy.maximum_expected_calibration_error
            ),
        },
        "candidate_evaluations": [
            _evaluation_payload(evaluation, decision)
            for evaluation, decision in zip(evaluations, decisions)
        ],
        "governance_status": "candidate_requires_independent_approval",
    }
    inserted = _persist_model_candidate(
        database_url=source["DATABASE_URL"],
        version=version,
        sport=settings.sport_key,
        schema_sha256=schema_sha256,
        artifact=artifact,
        artifact_sha256=artifact_sha256,
        training_cutoff=training_cutoff,
        validation_report=validation_report,
    )
    print("model candidate registered" if inserted else "model candidate already registered")


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
    """Run one scheduled or operator-dispatched provider-shadow ingestion."""

    _execute_provider_shadow(task_id=task.request.id)


@celery_app.task(
    bind=True,
    name=_PROVIDER_SCORES_TASK,
    queue=_PROVIDER_SHADOW_QUEUE,
    ignore_result=True,
    acks_late=False,
    reject_on_worker_lost=False,
    max_retries=0,
)
def settle_the_odds_api_scores(task) -> None:
    """Persist one scheduled or operator-dispatched recent-score response."""

    _execute_score_settlement(task_id=task.request.id)


@celery_app.task(
    name=_MODEL_TRAINING_TASK,
    queue=_PROVIDER_SHADOW_QUEUE,
    ignore_result=True,
    acks_late=False,
    reject_on_worker_lost=False,
    max_retries=0,
    soft_time_limit=10 * 60,
    time_limit=12 * 60,
)
def train_model_candidate() -> None:
    """Run an operator-triggered private training pass and return no model material."""

    _execute_model_training()
