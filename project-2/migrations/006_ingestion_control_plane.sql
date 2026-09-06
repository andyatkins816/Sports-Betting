-- Append-only persistence boundary for a future bounded ingestion dispatcher.
--
-- This migration adds records only.  It does not enable a policy, publish a
-- message, schedule work, contact a provider, or connect these tables to a
-- process.  A future repository must insert a dispatch, its quota reservation,
-- its first outbox message, and its initial transition in one transaction.

CREATE TABLE ingestion_dispatch (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider TEXT NOT NULL CHECK (provider ~ '^[a-z][a-z0-9_-]{0,63}$'),
    source_type TEXT NOT NULL CHECK (source_type ~ '^[a-z][a-z0-9_]{0,63}$'),
    request_fingerprint_sha256 CHAR(64) NOT NULL
        CHECK (request_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    estimated_cost INTEGER NOT NULL CHECK (estimated_cost BETWEEN 1 AND 10000),
    policy_version TEXT NOT NULL
        CHECK (policy_version ~ '^[a-z0-9][a-z0-9._-]{0,63}$'),
    max_attempts SMALLINT NOT NULL CHECK (max_attempts BETWEEN 1 AND 5),
    admitted_at TIMESTAMPTZ NOT NULL,
    -- The dispatcher supplies the SHA-256 of its reviewed canonical identity.
    -- Global key uniqueness and identity-tuple uniqueness together prevent
    -- both key reuse and assigning two keys to the same logical request.
    idempotency_key CHAR(64) NOT NULL
        CONSTRAINT ingestion_dispatch_idempotency_key_unique UNIQUE
        CHECK (idempotency_key ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ingestion_dispatch_identity_unique UNIQUE (
        provider,
        request_fingerprint_sha256,
        source_type,
        window_start,
        window_end
    ),
    CHECK (window_start < window_end),
    CHECK (window_end <= window_start + interval '7 days'),
    CHECK (admitted_at <= created_at + interval '5 minutes')
);
CREATE INDEX ingestion_dispatch_provider_admitted_idx
    ON ingestion_dispatch (provider, admitted_at DESC, id);

-- Health adapters query the latest durable quote receipt per provider.  This
-- index is read-only infrastructure; it neither schedules nor creates work.
CREATE INDEX IF NOT EXISTS odds_snapshot_provider_received_idx
    ON odds_snapshot (provider, received_at DESC);

-- Reservations never expire implicitly.  An inconclusive publication must be
-- reconciled with a new audit fact rather than silently returning quota.
CREATE TABLE ingestion_quota_reservation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ingestion_dispatch_id UUID NOT NULL REFERENCES ingestion_dispatch(id),
    attempt_number SMALLINT NOT NULL CHECK (attempt_number BETWEEN 1 AND 5),
    reserved_credits INTEGER NOT NULL CHECK (reserved_credits BETWEEN 1 AND 10000),
    reserved_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (reserved_at <= created_at + interval '5 minutes'),
    UNIQUE (ingestion_dispatch_id, attempt_number)
);
CREATE INDEX ingestion_quota_reservation_time_idx
    ON ingestion_quota_reservation (
        reserved_at DESC,
        ingestion_dispatch_id,
        attempt_number
    );

-- An outbox message contains only the immutable dispatch reference.  Consumers
-- rehydrate its safe fields from ingestion_dispatch, so no arbitrary payload
-- or delivery mutation is stored here.  Later attempts receive new records.
CREATE TABLE ingestion_dispatch_outbox (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ingestion_dispatch_id UUID NOT NULL REFERENCES ingestion_dispatch(id),
    attempt_number SMALLINT NOT NULL CHECK (attempt_number BETWEEN 1 AND 5),
    message_type TEXT NOT NULL DEFAULT 'ingestion_dispatch_requested'
        CHECK (message_type = 'ingestion_dispatch_requested'),
    message_schema_version SMALLINT NOT NULL DEFAULT 1
        CHECK (message_schema_version = 1),
    available_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ingestion_dispatch_id, attempt_number)
);
CREATE INDEX ingestion_dispatch_outbox_backlog_idx
    ON ingestion_dispatch_outbox (
        available_at,
        ingestion_dispatch_id,
        attempt_number
    );

CREATE TABLE ingestion_dispatch_transition (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ingestion_dispatch_id UUID NOT NULL REFERENCES ingestion_dispatch(id),
    state_sequence INTEGER NOT NULL CHECK (state_sequence >= 1),
    state TEXT NOT NULL CHECK (state IN (
        'pending',
        'queued',
        'running',
        'retry_wait',
        'succeeded',
        'dead_lettered',
        'cancelled'
    )),
    -- This is zero before the first attempt, and counts the current or most
    -- recently completed attempt thereafter.
    attempt_count SMALLINT NOT NULL CHECK (attempt_count BETWEEN 0 AND 5),
    worker_identity TEXT CHECK (
        worker_identity IS NULL
        OR (
            worker_identity ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
            AND worker_identity !~* '(api[-_]?key|token|secret|password|authorization|credential|cookie|bearer)'
        )
    ),
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
    dead_letter_reason TEXT CHECK (dead_letter_reason IN (
        'policy_disabled',
        'non_retryable',
        'attempts_exhausted',
        'retry_after_exceeds_limit'
    )),
    retry_not_before_at TIMESTAMPTZ,
    occurred_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ingestion_dispatch_id, state_sequence),
    CHECK (occurred_at <= created_at + interval '5 minutes'),
    CHECK (
        (state = 'pending'
            AND attempt_count = 0
            AND worker_identity IS NULL
            AND failure_code IS NULL
            AND dead_letter_reason IS NULL
            AND retry_not_before_at IS NULL)
        OR (state = 'queued'
            AND worker_identity IS NULL
            AND failure_code IS NULL
            AND dead_letter_reason IS NULL
            AND retry_not_before_at IS NULL)
        OR (state = 'running'
            AND attempt_count >= 1
            AND worker_identity IS NOT NULL
            AND failure_code IS NULL
            AND dead_letter_reason IS NULL
            AND retry_not_before_at IS NULL)
        OR (state = 'retry_wait'
            AND attempt_count >= 1
            AND worker_identity IS NOT NULL
            AND failure_code IS NOT NULL
            AND dead_letter_reason IS NULL
            AND retry_not_before_at > occurred_at
            AND retry_not_before_at <= occurred_at + interval '7 days')
        OR (state = 'succeeded'
            AND attempt_count >= 1
            AND worker_identity IS NOT NULL
            AND failure_code IS NULL
            AND dead_letter_reason IS NULL
            AND retry_not_before_at IS NULL)
        OR (state = 'dead_lettered'
            AND attempt_count >= 1
            AND worker_identity IS NOT NULL
            AND failure_code IS NOT NULL
            AND dead_letter_reason IS NOT NULL
            AND retry_not_before_at IS NULL)
        OR (state = 'cancelled'
            AND worker_identity IS NULL
            AND failure_code IS NULL
            AND dead_letter_reason IS NULL
            AND retry_not_before_at IS NULL)
    ),
    CHECK (
        state <> 'retry_wait'
        OR failure_code IN (
            'provider_rate_limited',
            'provider_temporary_unavailable',
            'network_timeout',
            'storage_unavailable',
            'database_unavailable',
            'queue_unavailable',
            'internal_transient'
        )
    ),
    CHECK (
        state <> 'dead_lettered'
        OR (dead_letter_reason = 'non_retryable' AND failure_code IN (
            'provider_contract_unapproved',
            'license_not_permitted',
            'configuration_invalid',
            'provider_response_invalid',
            'evidence_validation_failed',
            'idempotency_conflict'
        ))
        OR (dead_letter_reason = 'policy_disabled' AND failure_code IS NOT NULL)
        OR (dead_letter_reason IN (
            'attempts_exhausted',
            'retry_after_exceeds_limit'
        ) AND failure_code IN (
            'provider_rate_limited',
            'provider_temporary_unavailable',
            'network_timeout',
            'storage_unavailable',
            'database_unavailable',
            'queue_unavailable',
            'internal_transient'
        ))
    )
);
CREATE INDEX ingestion_dispatch_transition_latest_idx
    ON ingestion_dispatch_transition (
        ingestion_dispatch_id,
        state_sequence DESC
    ) INCLUDE (
        state,
        attempt_count,
        occurred_at,
        retry_not_before_at,
        worker_identity
    );
CREATE INDEX ingestion_dispatch_transition_backlog_idx
    ON ingestion_dispatch_transition (state, occurred_at, ingestion_dispatch_id)
    WHERE state IN ('pending', 'queued', 'retry_wait');
CREATE INDEX ingestion_dispatch_transition_worker_activity_idx
    ON ingestion_dispatch_transition (worker_identity, occurred_at DESC)
    WHERE worker_identity IS NOT NULL;

-- Database time owns record creation so supplied future timestamps cannot make
-- an early transition or outbox record appear eligible.
CREATE OR REPLACE FUNCTION set_ingestion_control_created_at() RETURNS trigger AS $$
BEGIN
    NEW.created_at := clock_timestamp();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Lock the immutable dispatch row so concurrent transition appends cannot both
-- claim the same next sequence.  The state machine has no automatic branch:
-- every queued, retry, terminal, or cancellation fact requires a new insert.
CREATE OR REPLACE FUNCTION enforce_ingestion_dispatch_transition() RETURNS trigger AS $$
DECLARE
    dispatch_admitted_at TIMESTAMPTZ;
    dispatch_max_attempts SMALLINT;
    previous_state TEXT;
    previous_sequence INTEGER;
    previous_attempt_count SMALLINT;
    previous_worker_identity TEXT;
    previous_retry_not_before_at TIMESTAMPTZ;
    previous_occurred_at TIMESTAMPTZ;
BEGIN
    SELECT admitted_at, max_attempts
      INTO dispatch_admitted_at, dispatch_max_attempts
      FROM ingestion_dispatch
     WHERE id = NEW.ingestion_dispatch_id
       FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ingestion dispatch transition references an unavailable dispatch';
    END IF;
    IF NEW.attempt_count > dispatch_max_attempts THEN
        RAISE EXCEPTION 'ingestion dispatch attempt exceeds its reviewed limit';
    END IF;
    IF NEW.occurred_at < dispatch_admitted_at THEN
        RAISE EXCEPTION 'ingestion dispatch transition cannot predate admission';
    END IF;

    SELECT state, state_sequence, attempt_count, worker_identity,
           retry_not_before_at, occurred_at
      INTO previous_state, previous_sequence, previous_attempt_count,
           previous_worker_identity, previous_retry_not_before_at,
           previous_occurred_at
      FROM ingestion_dispatch_transition
     WHERE ingestion_dispatch_id = NEW.ingestion_dispatch_id
     ORDER BY state_sequence DESC
     LIMIT 1;

    IF previous_state IS NULL THEN
        IF NEW.state_sequence <> 1
           OR NEW.state <> 'pending'
           OR NEW.attempt_count <> 0 THEN
            RAISE EXCEPTION 'an ingestion dispatch must begin pending with zero attempts';
        END IF;
        RETURN NEW;
    END IF;

    IF NEW.state_sequence <> previous_sequence + 1 THEN
        RAISE EXCEPTION 'ingestion dispatch transition sequence must be contiguous';
    END IF;
    IF NEW.occurred_at < previous_occurred_at THEN
        RAISE EXCEPTION 'ingestion dispatch transition time cannot move backward';
    END IF;

    IF previous_state = 'pending' THEN
        IF (NEW.state = 'queued' AND NEW.attempt_count = 0)
           OR (NEW.state = 'cancelled' AND NEW.attempt_count = 0) THEN
            RETURN NEW;
        END IF;
    ELSIF previous_state = 'queued' THEN
        IF NEW.state = 'running'
           AND NEW.attempt_count = previous_attempt_count + 1 THEN
            RETURN NEW;
        ELSIF NEW.state = 'cancelled'
           AND NEW.attempt_count = previous_attempt_count THEN
            RETURN NEW;
        END IF;
    ELSIF previous_state = 'running' THEN
        IF NEW.state IN ('succeeded', 'dead_lettered')
           AND NEW.attempt_count = previous_attempt_count
           AND NEW.worker_identity = previous_worker_identity THEN
            IF NEW.state = 'dead_lettered'
               AND NEW.dead_letter_reason = 'attempts_exhausted'
               AND NEW.attempt_count <> dispatch_max_attempts THEN
                RAISE EXCEPTION 'attempts-exhausted dead letter requires the final attempt';
            END IF;
            RETURN NEW;
        ELSIF NEW.state = 'retry_wait'
           AND NEW.attempt_count = previous_attempt_count
           AND NEW.worker_identity = previous_worker_identity
           AND NEW.attempt_count < dispatch_max_attempts THEN
            RETURN NEW;
        END IF;
    ELSIF previous_state = 'retry_wait' THEN
        IF NEW.state = 'queued'
           AND NEW.attempt_count = previous_attempt_count
           AND NEW.occurred_at >= previous_retry_not_before_at
           AND clock_timestamp() >= previous_retry_not_before_at THEN
            RETURN NEW;
        ELSIF NEW.state = 'cancelled'
           AND NEW.attempt_count = previous_attempt_count THEN
            RETURN NEW;
        END IF;
    END IF;

    RAISE EXCEPTION 'invalid ingestion dispatch transition';
END;
$$ LANGUAGE plpgsql;

-- A provider attempt receives exactly one equal-cost reservation.  Retry
-- attempts must reserve again rather than reusing the first request's credit.
CREATE OR REPLACE FUNCTION enforce_ingestion_quota_reservation() RETURNS trigger AS $$
DECLARE
    dispatch_admitted_at TIMESTAMPTZ;
    dispatch_estimated_cost INTEGER;
    dispatch_max_attempts SMALLINT;
BEGIN
    SELECT admitted_at, estimated_cost, max_attempts
      INTO dispatch_admitted_at, dispatch_estimated_cost, dispatch_max_attempts
      FROM ingestion_dispatch
     WHERE id = NEW.ingestion_dispatch_id
       FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ingestion quota reservation references an unavailable dispatch';
    END IF;
    IF NEW.attempt_number > dispatch_max_attempts THEN
        RAISE EXCEPTION 'ingestion quota reservation attempt exceeds its reviewed limit';
    END IF;
    IF NEW.reserved_credits <> dispatch_estimated_cost THEN
        RAISE EXCEPTION 'ingestion quota reservation must equal the estimated attempt cost';
    END IF;
    IF NEW.reserved_at < dispatch_admitted_at THEN
        RAISE EXCEPTION 'ingestion quota reservation cannot predate admission';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Each outbox attempt must be within the immutable dispatch attempt bound and
-- cannot become eligible before the dispatch was admitted.
CREATE OR REPLACE FUNCTION enforce_ingestion_dispatch_outbox() RETURNS trigger AS $$
DECLARE
    dispatch_admitted_at TIMESTAMPTZ;
    dispatch_max_attempts SMALLINT;
BEGIN
    SELECT admitted_at, max_attempts
      INTO dispatch_admitted_at, dispatch_max_attempts
      FROM ingestion_dispatch
     WHERE id = NEW.ingestion_dispatch_id
       FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ingestion outbox references an unavailable dispatch';
    END IF;
    IF NEW.attempt_number > dispatch_max_attempts THEN
        RAISE EXCEPTION 'ingestion outbox attempt exceeds its reviewed limit';
    END IF;
    IF NEW.available_at < dispatch_admitted_at THEN
        RAISE EXCEPTION 'ingestion outbox availability cannot predate admission';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Deferred checks let a repository insert related facts in any order inside a
-- transaction while making an incomplete commit impossible.
CREATE OR REPLACE FUNCTION enforce_ingestion_dispatch_initial_bundle() RETURNS trigger AS $$
DECLARE
    reservation_count INTEGER;
    reservation_credits INTEGER;
    reservation_time TIMESTAMPTZ;
    initial_outbox_count INTEGER;
    initial_outbox_time TIMESTAMPTZ;
    initial_transition_count INTEGER;
BEGIN
    SELECT count(*), min(reserved_credits), min(reserved_at)
      INTO reservation_count, reservation_credits, reservation_time
      FROM ingestion_quota_reservation
     WHERE ingestion_dispatch_id = NEW.id
       AND attempt_number = 1;

    SELECT count(*), min(available_at)
      INTO initial_outbox_count, initial_outbox_time
      FROM ingestion_dispatch_outbox
     WHERE ingestion_dispatch_id = NEW.id
       AND attempt_number = 1;

    SELECT count(*)
      INTO initial_transition_count
      FROM ingestion_dispatch_transition
     WHERE ingestion_dispatch_id = NEW.id
       AND state_sequence = 1
       AND state = 'pending'
       AND attempt_count = 0;

    IF reservation_count <> 1
       OR reservation_credits <> NEW.estimated_cost
       OR reservation_time < NEW.admitted_at
       OR initial_outbox_count <> 1
       OR initial_outbox_time <> NEW.admitted_at
       OR reservation_time > initial_outbox_time
       OR initial_transition_count <> 1 THEN
        RAISE EXCEPTION 'ingestion dispatch requires one atomic reservation, outbox, and initial transition';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- A queued fact is the immutable publication fact for its numbered outbox
-- record.  A retry-wait fact must atomically create the next attempt record.
CREATE OR REPLACE FUNCTION enforce_ingestion_dispatch_outbox_transition_pair() RETURNS trigger AS $$
DECLARE
    paired_outbox_count INTEGER;
    paired_available_at TIMESTAMPTZ;
BEGIN
    IF NEW.state = 'queued' THEN
        SELECT count(*), min(available_at)
          INTO paired_outbox_count, paired_available_at
          FROM ingestion_dispatch_outbox
         WHERE ingestion_dispatch_id = NEW.ingestion_dispatch_id
           AND attempt_number = NEW.attempt_count + 1;
        IF paired_outbox_count <> 1
           OR paired_available_at > NEW.occurred_at
           OR paired_available_at > clock_timestamp() THEN
            RAISE EXCEPTION 'queued transition requires an eligible outbox record';
        END IF;
    ELSIF NEW.state = 'retry_wait' THEN
        SELECT count(*), min(available_at)
          INTO paired_outbox_count, paired_available_at
          FROM ingestion_dispatch_outbox
         WHERE ingestion_dispatch_id = NEW.ingestion_dispatch_id
           AND attempt_number = NEW.attempt_count + 1;
        IF paired_outbox_count <> 1
           OR paired_available_at <> NEW.retry_not_before_at THEN
            RAISE EXCEPTION 'retry-wait transition requires the next transactional outbox record';
        END IF;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Conversely, no later-attempt outbox record may exist without its matching
-- retry-wait fact.  Both sides are checked at commit.
CREATE OR REPLACE FUNCTION enforce_ingestion_dispatch_retry_outbox_pair() RETURNS trigger AS $$
DECLARE
    paired_transition_count INTEGER;
    paired_retry_time TIMESTAMPTZ;
    paired_retry_occurred_at TIMESTAMPTZ;
    paired_reservation_count INTEGER;
    paired_reservation_time TIMESTAMPTZ;
BEGIN
    SELECT count(*), min(reserved_at)
      INTO paired_reservation_count, paired_reservation_time
      FROM ingestion_quota_reservation
     WHERE ingestion_dispatch_id = NEW.ingestion_dispatch_id
       AND attempt_number = NEW.attempt_number;

    IF paired_reservation_count <> 1
       OR paired_reservation_time > NEW.available_at THEN
        RAISE EXCEPTION 'ingestion outbox requires its matching quota reservation';
    END IF;

    IF NEW.attempt_number = 1 THEN
        RETURN NULL;
    END IF;

    SELECT count(*), min(retry_not_before_at), min(occurred_at)
      INTO paired_transition_count, paired_retry_time, paired_retry_occurred_at
      FROM ingestion_dispatch_transition
     WHERE ingestion_dispatch_id = NEW.ingestion_dispatch_id
       AND state = 'retry_wait'
       AND attempt_count = NEW.attempt_number - 1;

    IF paired_transition_count <> 1
       OR paired_retry_time <> NEW.available_at
       OR paired_reservation_time < paired_retry_occurred_at THEN
        RAISE EXCEPTION 'retry outbox requires its matching retry-wait transition';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- The reverse deferred check prevents an unused future reservation from being
-- committed without the outbox record that owns that attempt.
CREATE OR REPLACE FUNCTION enforce_ingestion_quota_reservation_outbox_pair() RETURNS trigger AS $$
DECLARE
    paired_outbox_count INTEGER;
BEGIN
    SELECT count(*)
      INTO paired_outbox_count
      FROM ingestion_dispatch_outbox
     WHERE ingestion_dispatch_id = NEW.ingestion_dispatch_id
       AND attempt_number = NEW.attempt_number;

    IF paired_outbox_count <> 1 THEN
        RAISE EXCEPTION 'ingestion quota reservation requires its matching outbox record';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ingestion_dispatch_created_at
    BEFORE INSERT ON ingestion_dispatch
    FOR EACH ROW EXECUTE FUNCTION set_ingestion_control_created_at();
CREATE TRIGGER ingestion_quota_reservation_created_at
    BEFORE INSERT ON ingestion_quota_reservation
    FOR EACH ROW EXECUTE FUNCTION set_ingestion_control_created_at();
CREATE TRIGGER ingestion_dispatch_outbox_created_at
    BEFORE INSERT ON ingestion_dispatch_outbox
    FOR EACH ROW EXECUTE FUNCTION set_ingestion_control_created_at();
CREATE TRIGGER ingestion_dispatch_transition_created_at
    BEFORE INSERT ON ingestion_dispatch_transition
    FOR EACH ROW EXECUTE FUNCTION set_ingestion_control_created_at();

CREATE TRIGGER ingestion_dispatch_transition_integrity
    BEFORE INSERT ON ingestion_dispatch_transition
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_dispatch_transition();
CREATE TRIGGER ingestion_dispatch_outbox_integrity
    BEFORE INSERT ON ingestion_dispatch_outbox
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_dispatch_outbox();
CREATE TRIGGER ingestion_quota_reservation_integrity
    BEFORE INSERT ON ingestion_quota_reservation
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_quota_reservation();
CREATE CONSTRAINT TRIGGER ingestion_dispatch_initial_bundle_integrity
    AFTER INSERT ON ingestion_dispatch
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_dispatch_initial_bundle();
CREATE CONSTRAINT TRIGGER ingestion_dispatch_transition_outbox_integrity
    AFTER INSERT ON ingestion_dispatch_transition
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_dispatch_outbox_transition_pair();
CREATE CONSTRAINT TRIGGER ingestion_dispatch_retry_outbox_integrity
    AFTER INSERT ON ingestion_dispatch_outbox
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_dispatch_retry_outbox_pair();
CREATE CONSTRAINT TRIGGER ingestion_quota_reservation_outbox_integrity
    AFTER INSERT ON ingestion_quota_reservation
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_quota_reservation_outbox_pair();

-- Read models are derived from immutable transitions.  The activity view is
-- historical evidence of worker-owned state changes, not a liveness claim.
CREATE VIEW ingestion_dispatch_latest_state AS
SELECT dispatch.id AS ingestion_dispatch_id,
       dispatch.provider,
       dispatch.source_type,
       dispatch.idempotency_key,
       transition.state_sequence,
       transition.state,
       transition.attempt_count,
       transition.occurred_at,
       transition.retry_not_before_at
FROM ingestion_dispatch AS dispatch
JOIN LATERAL (
    SELECT state_sequence, state, attempt_count, occurred_at, retry_not_before_at
    FROM ingestion_dispatch_transition
    WHERE ingestion_dispatch_id = dispatch.id
    ORDER BY state_sequence DESC
    LIMIT 1
) AS transition ON TRUE;

CREATE VIEW ingestion_worker_activity AS
SELECT dispatch.provider,
       transition.worker_identity,
       max(transition.occurred_at) AS latest_worker_activity_at,
       max(transition.occurred_at)
           FILTER (WHERE transition.state = 'running') AS latest_attempt_started_at,
       max(transition.occurred_at)
           FILTER (WHERE transition.state IN (
               'succeeded', 'retry_wait', 'dead_lettered'
           )) AS latest_attempt_finished_at,
       count(*) FILTER (WHERE transition.state = 'running') AS attempts_started
FROM ingestion_dispatch_transition AS transition
JOIN ingestion_dispatch AS dispatch
  ON dispatch.id = transition.ingestion_dispatch_id
WHERE transition.worker_identity IS NOT NULL
GROUP BY dispatch.provider, transition.worker_identity;

CREATE TRIGGER ingestion_dispatch_append_only
    BEFORE UPDATE OR DELETE ON ingestion_dispatch
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_quota_reservation_append_only
    BEFORE UPDATE OR DELETE ON ingestion_quota_reservation
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_dispatch_outbox_append_only
    BEFORE UPDATE OR DELETE ON ingestion_dispatch_outbox
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_dispatch_transition_append_only
    BEFORE UPDATE OR DELETE ON ingestion_dispatch_transition
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_dispatch_append_only_truncate
    BEFORE TRUNCATE ON ingestion_dispatch
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_quota_reservation_append_only_truncate
    BEFORE TRUNCATE ON ingestion_quota_reservation
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_dispatch_outbox_append_only_truncate
    BEFORE TRUNCATE ON ingestion_dispatch_outbox
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_dispatch_transition_append_only_truncate
    BEFORE TRUNCATE ON ingestion_dispatch_transition
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_mutation();
