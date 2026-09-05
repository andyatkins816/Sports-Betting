-- Append-only audit facts for future, bounded manual shadow-ingestion runs.
--
-- These tables intentionally persist safe identifiers, attempt counts, state,
-- and finite failure codes only.  Credentials, request URLs, response content,
-- exception text, and object references do not belong in this operational
-- ledger.  Immutable evidence remains in the separate receipt/store boundary.

CREATE TABLE ingestion_run (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider TEXT NOT NULL CHECK (provider ~ '^[a-z][a-z0-9_-]{0,63}$'),
    job_identity TEXT NOT NULL CHECK (
        job_identity ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
        AND job_identity !~* '(api[-_]?key|token|secret|password|authorization|credential|cookie|bearer)'
    ),
    source_type TEXT NOT NULL CHECK (source_type ~ '^[a-z][a-z0-9_]{0,63}$'),
    run_mode TEXT NOT NULL DEFAULT 'manual_shadow'
        CHECK (run_mode = 'manual_shadow'),
    max_attempts SMALLINT NOT NULL DEFAULT 1 CHECK (max_attempts BETWEEN 1 AND 5),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (provider, job_identity)
);
CREATE INDEX ingestion_run_provider_created_idx
    ON ingestion_run (provider, created_at DESC);

CREATE TABLE ingestion_run_state_transition (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ingestion_run_id UUID NOT NULL REFERENCES ingestion_run(id),
    state_sequence INTEGER NOT NULL CHECK (state_sequence >= 1),
    state TEXT NOT NULL CHECK (state IN (
        'queued', 'running', 'succeeded', 'failed', 'blocked', 'cancelled'
    )),
    attempt_count SMALLINT NOT NULL CHECK (attempt_count >= 0),
    failure_class TEXT CHECK (failure_class IN ('retryable', 'non_retryable')),
    failure_code TEXT CHECK (failure_code IN (
        'provider_rate_limited',
        'provider_temporary_unavailable',
        'network_timeout',
        'storage_unavailable',
        'database_unavailable',
        'queue_unavailable',
        'internal_transient',
        'provider_contract_unapproved',
        'license_not_permitted',
        'configuration_invalid',
        'provider_response_invalid',
        'evidence_validation_failed',
        'idempotency_conflict'
    )),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ingestion_run_id, state_sequence),
    CHECK (
        (failure_class IS NULL AND failure_code IS NULL)
        OR (failure_class = 'retryable' AND failure_code IS NOT NULL AND failure_code IN (
            'provider_rate_limited',
            'provider_temporary_unavailable',
            'network_timeout',
            'storage_unavailable',
            'database_unavailable',
            'queue_unavailable',
            'internal_transient'
        ))
        OR (failure_class = 'non_retryable' AND failure_code IS NOT NULL AND failure_code IN (
            'provider_contract_unapproved',
            'license_not_permitted',
            'configuration_invalid',
            'provider_response_invalid',
            'evidence_validation_failed',
            'idempotency_conflict'
        ))
    ),
    CHECK (
        (state IN ('failed', 'blocked') AND failure_class IS NOT NULL)
        OR (state NOT IN ('failed', 'blocked') AND failure_class IS NULL)
    ),
    CHECK (state <> 'blocked' OR failure_class = 'non_retryable'),
    CHECK (state <> 'queued' OR attempt_count = 0),
    CHECK (
        state NOT IN ('running', 'succeeded', 'failed')
        OR attempt_count >= 1
    )
);
CREATE INDEX ingestion_run_state_transition_latest_idx
    ON ingestion_run_state_transition (ingestion_run_id, state_sequence DESC);
CREATE INDEX ingestion_run_state_transition_state_time_idx
    ON ingestion_run_state_transition (state, occurred_at DESC);

-- Serialize state appends for a run by locking its immutable identity row.
-- The transition rule makes an explicit human retry possible only after a
-- retryable failure; no background retry process is created by this migration.
CREATE OR REPLACE FUNCTION enforce_ingestion_run_state_transition() RETURNS trigger AS $$
DECLARE
    previous_state TEXT;
    previous_sequence INTEGER;
    previous_attempt_count SMALLINT;
    previous_failure_class TEXT;
    run_max_attempts SMALLINT;
BEGIN
    SELECT max_attempts
      INTO run_max_attempts
      FROM ingestion_run
     WHERE id = NEW.ingestion_run_id
       FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ingestion run state references an unavailable run';
    END IF;

    SELECT state, state_sequence, attempt_count, failure_class
      INTO previous_state, previous_sequence, previous_attempt_count, previous_failure_class
      FROM ingestion_run_state_transition
     WHERE ingestion_run_id = NEW.ingestion_run_id
     ORDER BY state_sequence DESC
     LIMIT 1;

    IF previous_state IS NULL THEN
        IF NEW.state_sequence <> 1
           OR NEW.state <> 'queued'
           OR NEW.attempt_count <> 0 THEN
            RAISE EXCEPTION 'an ingestion run must begin with a queued zero-attempt state';
        END IF;
        RETURN NEW;
    END IF;

    IF NEW.state_sequence <> previous_sequence + 1 THEN
        RAISE EXCEPTION 'ingestion run state sequence must be contiguous';
    END IF;

    IF previous_state = 'queued' THEN
        IF (NEW.state = 'running' AND NEW.attempt_count = 1)
           OR (NEW.state IN ('blocked', 'cancelled') AND NEW.attempt_count = 0) THEN
            RETURN NEW;
        END IF;
    ELSIF previous_state = 'running' THEN
        IF NEW.state IN ('succeeded', 'failed', 'blocked', 'cancelled')
           AND NEW.attempt_count = previous_attempt_count THEN
            RETURN NEW;
        END IF;
    ELSIF previous_state = 'failed' AND previous_failure_class = 'retryable' THEN
        IF NEW.state = 'running'
           AND NEW.attempt_count = previous_attempt_count + 1
           AND NEW.attempt_count <= run_max_attempts THEN
            RETURN NEW;
        ELSIF NEW.state = 'cancelled'
           AND NEW.attempt_count = previous_attempt_count THEN
            RETURN NEW;
        END IF;
    END IF;

    RAISE EXCEPTION 'invalid ingestion run state transition';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ingestion_run_state_transition_integrity
    BEFORE INSERT ON ingestion_run_state_transition
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_run_state_transition();
CREATE TRIGGER ingestion_run_append_only
    BEFORE UPDATE OR DELETE ON ingestion_run
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_run_state_transition_append_only
    BEFORE UPDATE OR DELETE ON ingestion_run_state_transition
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
