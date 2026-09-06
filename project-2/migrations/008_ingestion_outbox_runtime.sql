-- Append-only persistence boundary for the reviewed outbox publisher and
-- provider-call worker runtime.
--
-- A publisher claim is only a short, recoverable right to attempt a broker
-- send.  It never marks an outbox row delivered before the send.  A worker
-- claim is deliberately different: once its running transition commits, that
-- exact provider attempt is never callable again, even after its lease expires.
-- This inactive milestone deliberately has no expiry-reconciliation operation:
-- without an execution fence held across the external call, expiry cannot prove
-- that a paused worker will not resume.  An inconclusive claim therefore blocks
-- the provider lane until a later, separately reviewed recovery design exists.

-- The migration runner owns a transaction for this file.  Pin unqualified DDL
-- creation to the trusted application schema and put temporary relations last.
-- pg_catalog is deliberately omitted: PostgreSQL then searches it implicitly
-- before this explicit path, while `public` remains the creation target.
SET LOCAL search_path = public, pg_temp;

-- Runtime UUIDs are capability identities, not nullable sentinels.  Reject
-- zero-valued identities for all new authorization and receipt facts while
-- leaving any historic rows available for an explicit audit/remediation.
ALTER TABLE provider_use_authorization
    ADD CONSTRAINT provider_use_authorization_nonzero_id
        CHECK (id <> '00000000-0000-0000-0000-000000000000'::UUID) NOT VALID;
ALTER TABLE provider_payload_receipt
    ADD CONSTRAINT provider_payload_receipt_nonzero_id
        CHECK (id <> '00000000-0000-0000-0000-000000000000'::UUID) NOT VALID;

-- Runtime policy values are copied onto each dispatch so delayed execution can
-- re-enforce the exact reviewed rate and quota boundary without trusting a
-- mutable in-process registry.  Historic 006/007 rows remain readable, while
-- NOT VALID checks require every new or changed dispatch to carry the snapshot.
ALTER TABLE ingestion_dispatch
    ADD COLUMN min_request_interval INTERVAL CHECK (
        min_request_interval >= interval '0 seconds'
        AND min_request_interval <= interval '7 days'
    ),
    ADD COLUMN quota_floor INTEGER CHECK (
        quota_floor BETWEEN 0 AND 2147483647
    ),
    ADD COLUMN quota_max_age INTERVAL CHECK (
        quota_max_age > interval '0 seconds'
        AND quota_max_age <= interval '7 days'
    ),
    ADD COLUMN retry_schedule_sha256 CHAR(64) CHECK (
        retry_schedule_sha256 ~ '^[0-9a-f]{64}$'
    );
ALTER TABLE ingestion_dispatch
    ADD CONSTRAINT ingestion_dispatch_runtime_policy_required_for_new_records
        CHECK (
            min_request_interval IS NOT NULL
            AND quota_floor IS NOT NULL
            AND quota_max_age IS NOT NULL
            AND retry_schedule_sha256 IS NOT NULL
        ) NOT VALID;

-- Receipt creation time participates in execution-time quota freshness.  New
-- writers that omit it receive a statement-time database value; deterministic
-- historical fixtures may still provide an earlier value.  Reject inconsistent
-- or future-dated fields against one database-owned observation.  Backdating
-- only makes a receipt older (and therefore less useful) under the freshness
-- checks below.  A later least-privilege insert routine can fully own creation
-- time without granting callers direct table writes.
ALTER TABLE provider_payload_receipt
    ALTER COLUMN created_at SET DEFAULT clock_timestamp();

CREATE OR REPLACE FUNCTION enforce_provider_payload_receipt_runtime_times()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    database_now TIMESTAMPTZ;
BEGIN
    database_now := clock_timestamp();
    IF NOT isfinite(NEW.created_at)
       OR NOT isfinite(NEW.received_at)
       OR NOT isfinite(NEW.captured_at)
       OR NEW.received_at > NEW.created_at
       OR NEW.captured_at > NEW.created_at
       OR NEW.created_at > database_now
       OR NEW.received_at > database_now
       OR NEW.captured_at > database_now THEN
        RAISE EXCEPTION 'provider receipt timestamps are inconsistent or in the future';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER provider_payload_receipt_runtime_times
    BEFORE INSERT ON provider_payload_receipt
    FOR EACH ROW EXECUTE FUNCTION enforce_provider_payload_receipt_runtime_times();

CREATE TABLE ingestion_outbox_publication_claim (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ingestion_dispatch_outbox_id UUID NOT NULL
        REFERENCES ingestion_dispatch_outbox(id),
    claim_sequence INTEGER NOT NULL CHECK (claim_sequence >= 1),
    publisher_identity TEXT NOT NULL CHECK (
        publisher_identity ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
        AND publisher_identity !~* '(api[-_]?key|token|secret|password|authorization|credential|cookie|bearer)'
    ),
    lease_token UUID NOT NULL UNIQUE,
    claimed_at TIMESTAMPTZ NOT NULL,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ingestion_dispatch_outbox_id, claim_sequence),
    CHECK (claimed_at < lease_expires_at)
);
CREATE INDEX ingestion_outbox_publication_claim_lease_idx
    ON ingestion_outbox_publication_claim (
        ingestion_dispatch_outbox_id,
        lease_expires_at DESC,
        claim_sequence DESC
    );

-- This fact is appended only after the broker-send operation returns.  A crash
-- before this insert leaves the outbox eligible after the short claim lease;
-- a crash after the send may duplicate delivery, which the consumer claim
-- below makes harmless.
CREATE TABLE ingestion_outbox_publication_delivery (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    publication_claim_id UUID NOT NULL UNIQUE
        REFERENCES ingestion_outbox_publication_claim(id),
    ingestion_dispatch_outbox_id UUID NOT NULL UNIQUE
        REFERENCES ingestion_dispatch_outbox(id),
    publication_id UUID NOT NULL UNIQUE,
    delivered_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One immutable claim is the sole permission for one exact provider call.
-- There is intentionally no reclaim path for this table.  Expiry means the
-- outcome is inconclusive, not that another worker may call the provider.
CREATE TABLE ingestion_dispatch_attempt_claim (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    publication_delivery_id UUID NOT NULL UNIQUE
        REFERENCES ingestion_outbox_publication_delivery(id),
    ingestion_dispatch_id UUID NOT NULL REFERENCES ingestion_dispatch(id),
    attempt_number SMALLINT NOT NULL CHECK (attempt_number BETWEEN 1 AND 5),
    provider_use_authorization_id UUID NOT NULL
        REFERENCES provider_use_authorization(id),
    quota_reservation_id UUID NOT NULL UNIQUE
        REFERENCES ingestion_quota_reservation(id),
    quota_receipt_id UUID NOT NULL REFERENCES provider_payload_receipt(id),
    running_transition_id UUID NOT NULL UNIQUE
        REFERENCES ingestion_dispatch_transition(id)
        DEFERRABLE INITIALLY DEFERRED,
    provider TEXT NOT NULL CHECK (provider ~ '^[a-z][a-z0-9_-]{0,63}$'),
    source_type TEXT NOT NULL CHECK (source_type ~ '^[a-z][a-z0-9_]{0,63}$'),
    request_fingerprint_sha256 CHAR(64) NOT NULL
        CHECK (request_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    license_scope TEXT NOT NULL
        CHECK (license_scope ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    license_version TEXT NOT NULL
        CHECK (license_version ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    exposure TEXT NOT NULL CHECK (exposure = 'private_raw'),
    worker_identity TEXT NOT NULL CHECK (
        worker_identity ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
        AND worker_identity !~* '(api[-_]?key|token|secret|password|authorization|credential|cookie|bearer)'
    ),
    lease_token UUID NOT NULL UNIQUE,
    claimed_at TIMESTAMPTZ NOT NULL,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ingestion_dispatch_id, attempt_number),
    CHECK (window_start < window_end),
    CHECK (claimed_at < lease_expires_at)
);
CREATE INDEX ingestion_dispatch_attempt_claim_provider_idx
    ON ingestion_dispatch_attempt_claim (provider, claimed_at DESC, id);

-- A response receipt is copied into the attempt lineage so later review does
-- not need to infer which authorization, request window, or provider call
-- produced a retained payload.
CREATE TABLE ingestion_dispatch_attempt_receipt (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    attempt_claim_id UUID NOT NULL UNIQUE
        REFERENCES ingestion_dispatch_attempt_claim(id),
    ingestion_dispatch_id UUID NOT NULL REFERENCES ingestion_dispatch(id),
    attempt_number SMALLINT NOT NULL CHECK (attempt_number BETWEEN 1 AND 5),
    provider_use_authorization_id UUID NOT NULL
        REFERENCES provider_use_authorization(id),
    provider_payload_receipt_id UUID NOT NULL UNIQUE
        REFERENCES provider_payload_receipt(id),
    provider TEXT NOT NULL CHECK (provider ~ '^[a-z][a-z0-9_-]{0,63}$'),
    source_type TEXT NOT NULL CHECK (source_type ~ '^[a-z][a-z0-9_]{0,63}$'),
    request_fingerprint_sha256 CHAR(64) NOT NULL
        CHECK (request_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    license_scope TEXT NOT NULL,
    license_version TEXT NOT NULL,
    linked_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ingestion_dispatch_id, attempt_number)
);

CREATE TABLE ingestion_dispatch_attempt_completion (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    attempt_claim_id UUID NOT NULL UNIQUE
        REFERENCES ingestion_dispatch_attempt_claim(id),
    ingestion_dispatch_id UUID NOT NULL REFERENCES ingestion_dispatch(id),
    attempt_number SMALLINT NOT NULL CHECK (attempt_number BETWEEN 1 AND 5),
    completion_transition_id UUID NOT NULL UNIQUE
        REFERENCES ingestion_dispatch_transition(id)
        DEFERRABLE INITIALLY DEFERRED,
    attempt_receipt_id UUID UNIQUE
        REFERENCES ingestion_dispatch_attempt_receipt(id),
    outcome TEXT NOT NULL CHECK (outcome IN (
        'succeeded', 'retry_wait', 'dead_lettered'
    )),
    worker_identity TEXT NOT NULL CHECK (
        worker_identity ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
        AND worker_identity !~* '(api[-_]?key|token|secret|password|authorization|credential|cookie|bearer)'
    ),
    resolution_kind TEXT NOT NULL CHECK (resolution_kind = 'worker'),
    resolver_identity TEXT NOT NULL CHECK (
        resolver_identity ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
        AND resolver_identity !~* '(api[-_]?key|token|secret|password|authorization|credential|cookie|bearer)'
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
    retry_safety TEXT CHECK (retry_safety = 'request_not_sent'),
    dead_letter_reason TEXT CHECK (dead_letter_reason IN (
        'policy_disabled', 'non_retryable', 'attempts_exhausted',
        'retry_after_exceeds_limit'
    )),
    retry_not_before_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ingestion_dispatch_id, attempt_number),
    CHECK (
        (outcome = 'succeeded'
            AND attempt_receipt_id IS NOT NULL
            AND failure_code IS NULL
            AND retry_safety IS NULL
            AND dead_letter_reason IS NULL
            AND retry_not_before_at IS NULL)
        OR (outcome = 'retry_wait'
            AND failure_code IS NOT NULL
            AND failure_code IN (
                'provider_rate_limited',
                'provider_temporary_unavailable',
                'network_timeout',
                'storage_unavailable',
                'database_unavailable',
                'queue_unavailable',
                'internal_transient'
            )
            AND retry_safety = 'request_not_sent'
            AND attempt_receipt_id IS NULL
            AND dead_letter_reason IS NULL
            AND retry_not_before_at > completed_at
            AND retry_not_before_at <= completed_at + interval '7 days')
        OR (outcome = 'dead_lettered'
            AND failure_code IS NOT NULL
            AND retry_safety IS NULL
            AND dead_letter_reason IS NOT NULL
            AND retry_not_before_at IS NULL)
    )
);
CREATE INDEX ingestion_dispatch_attempt_completion_time_idx
    ON ingestion_dispatch_attempt_completion (completed_at DESC, id);

-- Database-owned clocks prevent callers extending either lease or forging an
-- earlier publication/completion fact.
CREATE OR REPLACE FUNCTION enforce_ingestion_publication_claim_insert()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    dispatch_provider TEXT;
    prior_claim_count INTEGER;
    active_claim_count INTEGER;
    delivered_count INTEGER;
    latest_state TEXT;
    latest_attempt_count SMALLINT;
BEGIN
    SELECT dispatch.provider
      INTO dispatch_provider
      FROM ingestion_dispatch_outbox AS outbox
      JOIN ingestion_dispatch AS dispatch
        ON dispatch.id = outbox.ingestion_dispatch_id
     WHERE outbox.id = NEW.ingestion_dispatch_outbox_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'publication claim references an unavailable outbox record';
    END IF;
    PERFORM lock_ingestion_provider(dispatch_provider);
    PERFORM 1
      FROM ingestion_dispatch_outbox
     WHERE id = NEW.ingestion_dispatch_outbox_id
       FOR UPDATE;

    SELECT count(*)
      INTO delivered_count
      FROM ingestion_outbox_publication_delivery
     WHERE ingestion_dispatch_outbox_id = NEW.ingestion_dispatch_outbox_id;
    IF delivered_count <> 0 THEN
        RAISE EXCEPTION 'delivered ingestion outbox records cannot be reclaimed';
    END IF;

    SELECT count(*)
      INTO active_claim_count
      FROM ingestion_outbox_publication_claim
     WHERE ingestion_dispatch_outbox_id = NEW.ingestion_dispatch_outbox_id
       AND lease_expires_at > clock_timestamp();
    IF active_claim_count <> 0 THEN
        RAISE EXCEPTION 'ingestion outbox already has an active publication claim';
    END IF;

    SELECT count(*)
      INTO prior_claim_count
      FROM ingestion_outbox_publication_claim
     WHERE ingestion_dispatch_outbox_id = NEW.ingestion_dispatch_outbox_id;

    SELECT transition.state, transition.attempt_count
      INTO latest_state, latest_attempt_count
      FROM ingestion_dispatch_outbox AS outbox
      JOIN LATERAL (
          SELECT state, attempt_count
          FROM ingestion_dispatch_transition
          WHERE ingestion_dispatch_id = outbox.ingestion_dispatch_id
          ORDER BY state_sequence DESC
          LIMIT 1
      ) AS transition ON TRUE
     WHERE outbox.id = NEW.ingestion_dispatch_outbox_id
       AND outbox.available_at <= clock_timestamp();

    IF NOT FOUND
       OR latest_attempt_count IS DISTINCT FROM (
           SELECT attempt_number - 1
           FROM ingestion_dispatch_outbox
           WHERE id = NEW.ingestion_dispatch_outbox_id
       )
       OR latest_state IS DISTINCT FROM CASE
           WHEN latest_attempt_count = 0 THEN 'pending'
           ELSE 'retry_wait'
       END THEN
        RAISE EXCEPTION 'ingestion outbox is not publishable in its latest state';
    END IF;

    NEW.claim_sequence := prior_claim_count + 1;
    NEW.claimed_at := clock_timestamp();
    NEW.lease_expires_at := NEW.claimed_at + interval '2 minutes';
    NEW.created_at := NEW.claimed_at;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_ingestion_publication_delivery_insert()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    claim_outbox_id UUID;
    claim_lease_token UUID;
    claim_lease_expires_at TIMESTAMPTZ;
    dispatch_provider TEXT;
BEGIN
    SELECT claim.ingestion_dispatch_outbox_id, claim.lease_token,
           claim.lease_expires_at, dispatch.provider
      INTO claim_outbox_id, claim_lease_token, claim_lease_expires_at,
           dispatch_provider
      FROM ingestion_outbox_publication_claim AS claim
      JOIN ingestion_dispatch_outbox AS outbox
        ON outbox.id = claim.ingestion_dispatch_outbox_id
      JOIN ingestion_dispatch AS dispatch
        ON dispatch.id = outbox.ingestion_dispatch_id
     WHERE claim.id = NEW.publication_claim_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'publication delivery references an unavailable claim';
    END IF;
    PERFORM lock_ingestion_provider(dispatch_provider);
    IF claim_outbox_id IS DISTINCT FROM NEW.ingestion_dispatch_outbox_id
       OR claim_lease_token IS DISTINCT FROM NEW.publication_id THEN
        RAISE EXCEPTION 'publication delivery must match its exact claim';
    END IF;

    NEW.delivered_at := clock_timestamp();
    NEW.created_at := NEW.delivered_at;
    IF NEW.delivered_at >= claim_lease_expires_at THEN
        RAISE EXCEPTION 'an expired publication claim cannot record delivery';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_ingestion_attempt_claim_insert()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    dispatch_provider TEXT;
    dispatch_source_type TEXT;
    dispatch_request_fingerprint CHAR(64);
    dispatch_window_start TIMESTAMPTZ;
    dispatch_window_end TIMESTAMPTZ;
    dispatch_max_attempts SMALLINT;
    dispatch_min_request_interval INTERVAL;
    dispatch_quota_floor INTEGER;
    dispatch_quota_max_age INTERVAL;
    dispatch_authorization_id UUID;
    authorization_provider TEXT;
    authorization_source_type TEXT;
    authorization_license_scope TEXT;
    authorization_license_version TEXT;
    authorization_exposure TEXT;
    authorization_reviewed_at TIMESTAMPTZ;
    authorization_effective_from TIMESTAMPTZ;
    authorization_effective_until TIMESTAMPTZ;
    expected_reservation_id UUID;
    expected_quota_receipt_id UUID;
    expected_quota_remaining INTEGER;
    expected_quota_received_at TIMESTAMPTZ;
    expected_quota_created_at TIMESTAMPTZ;
    reserved_credits BIGINT;
    prior_completed_at TIMESTAMPTZ;
    prior_min_request_interval INTERVAL;
    expected_delivery_id UUID;
    latest_state TEXT;
    latest_attempt_count SMALLINT;
    provider_open_claim_count INTEGER;
BEGIN
    SELECT provider
      INTO dispatch_provider
      FROM ingestion_dispatch
     WHERE id = NEW.ingestion_dispatch_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'attempt claim references an unavailable dispatch';
    END IF;
    PERFORM lock_ingestion_provider(dispatch_provider);

    SELECT count(*)
      INTO provider_open_claim_count
      FROM ingestion_dispatch_attempt_claim AS attempt_claim
      LEFT JOIN ingestion_dispatch_attempt_completion AS completion
        ON completion.attempt_claim_id = attempt_claim.id
     WHERE attempt_claim.provider = dispatch_provider
       AND attempt_claim.id IS DISTINCT FROM NEW.id
       AND completion.id IS NULL;
    IF provider_open_claim_count <> 0 THEN
        RAISE EXCEPTION 'provider already has an unresolved ingestion attempt';
    END IF;

    SELECT dispatch.provider, dispatch.source_type,
           dispatch.request_fingerprint_sha256, dispatch.window_start,
           dispatch.window_end, dispatch.max_attempts,
           dispatch.min_request_interval, dispatch.quota_floor,
           dispatch.quota_max_age,
           dispatch.provider_use_authorization_id,
           authorization.provider, authorization.source_type,
           authorization.license_scope, authorization.license_version,
           authorization.exposure, authorization.reviewed_at,
           authorization.effective_from, authorization.effective_until
      INTO dispatch_provider, dispatch_source_type,
           dispatch_request_fingerprint, dispatch_window_start,
           dispatch_window_end, dispatch_max_attempts,
           dispatch_min_request_interval, dispatch_quota_floor,
           dispatch_quota_max_age,
           dispatch_authorization_id, authorization_provider,
           authorization_source_type, authorization_license_scope,
           authorization_license_version, authorization_exposure,
           authorization_reviewed_at, authorization_effective_from,
           authorization_effective_until
      FROM ingestion_dispatch AS dispatch
      JOIN provider_use_authorization AS authorization
        ON authorization.id = dispatch.provider_use_authorization_id
     WHERE dispatch.id = NEW.ingestion_dispatch_id
       FOR UPDATE OF dispatch;

    IF NOT FOUND
       OR NEW.attempt_number > dispatch_max_attempts
       OR dispatch_min_request_interval IS NULL
       OR dispatch_quota_floor IS NULL
       OR dispatch_quota_max_age IS NULL
       OR authorization_provider IS DISTINCT FROM dispatch_provider
       OR authorization_source_type IS DISTINCT FROM dispatch_source_type
       OR authorization_exposure IS DISTINCT FROM 'private_raw' THEN
        RAISE EXCEPTION 'attempt claim does not match its exact dispatch authorization';
    END IF;

    NEW.claimed_at := clock_timestamp();
    NEW.lease_expires_at := NEW.claimed_at + interval '5 minutes';
    NEW.created_at := NEW.claimed_at;
    IF NEW.claimed_at < greatest(
            authorization_reviewed_at,
            authorization_effective_from
       )
       OR NEW.lease_expires_at > authorization_effective_until THEN
        RAISE EXCEPTION 'attempt claim falls outside its provider authorization window';
    END IF;

    SELECT completion.completed_at, prior_dispatch.min_request_interval
      INTO prior_completed_at, prior_min_request_interval
      FROM ingestion_dispatch_attempt_completion AS completion
      JOIN ingestion_dispatch_attempt_claim AS prior_claim
        ON prior_claim.id = completion.attempt_claim_id
      JOIN ingestion_dispatch AS prior_dispatch
        ON prior_dispatch.id = prior_claim.ingestion_dispatch_id
     WHERE prior_claim.provider = dispatch_provider
     ORDER BY completion.completed_at DESC, completion.id DESC
     LIMIT 1;
    IF prior_completed_at IS NOT NULL
       AND (
           prior_min_request_interval IS NULL
           OR NEW.claimed_at < prior_completed_at + greatest(
               dispatch_min_request_interval,
               prior_min_request_interval
           )
       ) THEN
        RAISE EXCEPTION 'attempt claim violates the reviewed provider interval';
    END IF;

    SELECT reservation.id
      INTO expected_reservation_id
      FROM ingestion_quota_reservation AS reservation
     WHERE reservation.ingestion_dispatch_id = NEW.ingestion_dispatch_id
       AND reservation.attempt_number = NEW.attempt_number;

    SELECT receipt.id, receipt.provider_quota_remaining,
           receipt.received_at, receipt.created_at
      INTO expected_quota_receipt_id, expected_quota_remaining,
           expected_quota_received_at, expected_quota_created_at
      FROM provider_payload_receipt AS receipt
     WHERE receipt.provider = dispatch_provider
       AND receipt.license_scope IS NOT DISTINCT FROM
            authorization_license_scope
       AND receipt.license_version IS NOT DISTINCT FROM
            authorization_license_version
       AND receipt.provider_quota_remaining IS NOT NULL
       AND receipt.received_at <= NEW.claimed_at
       AND receipt.created_at <= NEW.claimed_at
     ORDER BY receipt.received_at DESC, receipt.created_at DESC,
              receipt.provider_quota_remaining ASC, receipt.id DESC
     LIMIT 1
       FOR KEY SHARE;
    SELECT COALESCE(sum(reservation.reserved_credits), 0)
      INTO reserved_credits
      FROM ingestion_quota_reservation AS reservation
      JOIN ingestion_dispatch AS reserved_dispatch
        ON reserved_dispatch.id = reservation.ingestion_dispatch_id
     WHERE reserved_dispatch.provider = dispatch_provider;
    IF expected_quota_receipt_id IS NULL
       OR expected_quota_received_at > expected_quota_created_at
       OR NEW.claimed_at - expected_quota_received_at > dispatch_quota_max_age
       OR expected_quota_remaining::BIGINT - reserved_credits <
            dispatch_quota_floor THEN
        RAISE EXCEPTION 'attempt claim lacks current reviewed provider quota';
    END IF;

    SELECT delivery.id
      INTO expected_delivery_id
      FROM ingestion_dispatch_outbox AS outbox
      JOIN ingestion_outbox_publication_delivery AS delivery
        ON delivery.ingestion_dispatch_outbox_id = outbox.id
     WHERE outbox.ingestion_dispatch_id = NEW.ingestion_dispatch_id
       AND outbox.attempt_number = NEW.attempt_number;

    SELECT state, attempt_count
      INTO latest_state, latest_attempt_count
      FROM ingestion_dispatch_transition
     WHERE ingestion_dispatch_id = NEW.ingestion_dispatch_id
     ORDER BY state_sequence DESC
     LIMIT 1;

    IF expected_reservation_id IS NULL
       OR expected_quota_receipt_id IS NULL
       OR expected_delivery_id IS NULL
       OR latest_state IS DISTINCT FROM 'queued'
       OR latest_attempt_count IS DISTINCT FROM NEW.attempt_number - 1 THEN
        RAISE EXCEPTION 'attempt claim requires its delivered queued reservation bundle';
    END IF;
    IF NEW.publication_delivery_id IS DISTINCT FROM expected_delivery_id
       OR NEW.quota_reservation_id IS DISTINCT FROM expected_reservation_id
       OR NEW.quota_receipt_id IS DISTINCT FROM expected_quota_receipt_id THEN
        RAISE EXCEPTION 'attempt claim identifiers do not match their exact dispatch bundle';
    END IF;

    NEW.provider_use_authorization_id := dispatch_authorization_id;
    NEW.provider := dispatch_provider;
    NEW.source_type := dispatch_source_type;
    NEW.request_fingerprint_sha256 := dispatch_request_fingerprint;
    NEW.window_start := dispatch_window_start;
    NEW.window_end := dispatch_window_end;
    NEW.license_scope := authorization_license_scope;
    NEW.license_version := authorization_license_version;
    NEW.exposure := authorization_exposure;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_ingestion_attempt_receipt_insert()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    claim_record ingestion_dispatch_attempt_claim%ROWTYPE;
    receipt_provider TEXT;
    receipt_source_type TEXT;
    receipt_request_fingerprint CHAR(64);
    receipt_license_scope TEXT;
    receipt_license_version TEXT;
    receipt_received_at TIMESTAMPTZ;
    receipt_created_at TIMESTAMPTZ;
    authorization_effective_from TIMESTAMPTZ;
    authorization_effective_until TIMESTAMPTZ;
BEGIN
    NEW.linked_at := clock_timestamp();
    NEW.created_at := NEW.linked_at;
    SELECT *
      INTO claim_record
      FROM ingestion_dispatch_attempt_claim
     WHERE id = NEW.attempt_claim_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'attempt receipt references an unavailable claim';
    END IF;
    PERFORM lock_ingestion_provider(claim_record.provider);

    IF clock_timestamp() >= claim_record.lease_expires_at THEN
        RAISE EXCEPTION 'an expired provider attempt cannot acquire a response receipt';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM ingestion_dispatch_attempt_completion
        WHERE attempt_claim_id = claim_record.id
    ) THEN
        RAISE EXCEPTION 'a completed provider attempt cannot acquire another receipt';
    END IF;

    SELECT receipt.provider, receipt.source_type,
           receipt.request_fingerprint_sha256, receipt.license_scope,
           receipt.license_version, receipt.received_at, receipt.created_at
      INTO receipt_provider, receipt_source_type,
           receipt_request_fingerprint, receipt_license_scope,
           receipt_license_version, receipt_received_at, receipt_created_at
      FROM provider_payload_receipt AS receipt
     WHERE receipt.id = NEW.provider_payload_receipt_id
       FOR KEY SHARE;
    IF NOT FOUND
       OR receipt_provider IS DISTINCT FROM claim_record.provider
       OR receipt_source_type IS DISTINCT FROM claim_record.source_type
       OR receipt_request_fingerprint IS DISTINCT FROM
            claim_record.request_fingerprint_sha256
       OR receipt_license_scope IS DISTINCT FROM claim_record.license_scope
       OR receipt_license_version IS DISTINCT FROM claim_record.license_version
       OR receipt_received_at < claim_record.claimed_at
       OR receipt_received_at > claim_record.lease_expires_at
       OR receipt_received_at > receipt_created_at
       OR receipt_received_at > NEW.linked_at
       OR receipt_created_at > NEW.linked_at THEN
        RAISE EXCEPTION 'provider response receipt does not match its exact attempt lineage';
    END IF;

    SELECT effective_from, effective_until
      INTO authorization_effective_from, authorization_effective_until
      FROM provider_use_authorization
     WHERE id = claim_record.provider_use_authorization_id
       FOR KEY SHARE;
    IF receipt_received_at < authorization_effective_from
       OR receipt_received_at >= authorization_effective_until THEN
        RAISE EXCEPTION 'provider response receipt falls outside its authorization window';
    END IF;

    NEW.ingestion_dispatch_id := claim_record.ingestion_dispatch_id;
    NEW.attempt_number := claim_record.attempt_number;
    NEW.provider_use_authorization_id :=
        claim_record.provider_use_authorization_id;
    NEW.provider := claim_record.provider;
    NEW.source_type := claim_record.source_type;
    NEW.request_fingerprint_sha256 :=
        claim_record.request_fingerprint_sha256;
    NEW.window_start := claim_record.window_start;
    NEW.window_end := claim_record.window_end;
    NEW.license_scope := claim_record.license_scope;
    NEW.license_version := claim_record.license_version;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_ingestion_attempt_completion_insert()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    claim_record ingestion_dispatch_attempt_claim%ROWTYPE;
    receipt_claim_id UUID;
    latest_state TEXT;
    latest_attempt_count SMALLINT;
BEGIN
    SELECT *
      INTO claim_record
      FROM ingestion_dispatch_attempt_claim
     WHERE id = NEW.attempt_claim_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'attempt completion references an unavailable claim';
    END IF;
    PERFORM lock_ingestion_provider(claim_record.provider);

    SELECT state, attempt_count
      INTO latest_state, latest_attempt_count
      FROM ingestion_dispatch_transition
     WHERE ingestion_dispatch_id = claim_record.ingestion_dispatch_id
     ORDER BY state_sequence DESC
     LIMIT 1;
    IF latest_state IS DISTINCT FROM 'running'
       OR latest_attempt_count IS DISTINCT FROM claim_record.attempt_number THEN
        RAISE EXCEPTION 'attempt completion requires its latest running transition';
    END IF;

    NEW.completed_at := clock_timestamp();
    NEW.created_at := NEW.completed_at;
    NEW.ingestion_dispatch_id := claim_record.ingestion_dispatch_id;
    NEW.attempt_number := claim_record.attempt_number;
    NEW.worker_identity := claim_record.worker_identity;

    IF NEW.resolution_kind IS DISTINCT FROM 'worker'
       OR NEW.resolver_identity IS DISTINCT FROM claim_record.worker_identity
       OR NEW.completed_at >= claim_record.lease_expires_at THEN
        RAISE EXCEPTION 'worker completion requires its exact active attempt lease';
    END IF;

    IF NEW.attempt_receipt_id IS NOT NULL THEN
        SELECT attempt_claim_id
          INTO receipt_claim_id
          FROM ingestion_dispatch_attempt_receipt
         WHERE id = NEW.attempt_receipt_id;
        IF receipt_claim_id IS DISTINCT FROM claim_record.id THEN
            RAISE EXCEPTION 'attempt completion receipt does not match its claim';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER ingestion_outbox_publication_claim_integrity
    BEFORE INSERT ON ingestion_outbox_publication_claim
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_publication_claim_insert();
CREATE TRIGGER ingestion_outbox_publication_delivery_integrity
    BEFORE INSERT ON ingestion_outbox_publication_delivery
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_publication_delivery_insert();
CREATE TRIGGER ingestion_dispatch_attempt_claim_integrity
    BEFORE INSERT ON ingestion_dispatch_attempt_claim
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_attempt_claim_insert();
CREATE TRIGGER ingestion_dispatch_attempt_receipt_integrity
    BEFORE INSERT ON ingestion_dispatch_attempt_receipt
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_attempt_receipt_insert();
CREATE TRIGGER ingestion_dispatch_attempt_completion_integrity
    BEFORE INSERT ON ingestion_dispatch_attempt_completion
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_attempt_completion_insert();

-- Deferred two-way bundle checks make every runtime fact prove the transition
-- it caused.  The reverse transition checks are conditional on an 008 runtime
-- fact so historic and deliberately unwired 006 records remain readable.
CREATE OR REPLACE FUNCTION enforce_ingestion_publication_delivery_transition_pair()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    paired_transition_count INTEGER;
BEGIN
    SELECT count(*)
      INTO paired_transition_count
      FROM ingestion_dispatch_outbox AS outbox
      JOIN ingestion_dispatch_transition AS transition
        ON transition.ingestion_dispatch_id = outbox.ingestion_dispatch_id
       AND transition.state = 'queued'
       AND transition.attempt_count = outbox.attempt_number - 1
       AND transition.occurred_at = NEW.delivered_at
     WHERE outbox.id = NEW.ingestion_dispatch_outbox_id;
    IF paired_transition_count <> 1 THEN
        RAISE EXCEPTION 'publication delivery requires its exact queued transition';
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_ingestion_attempt_claim_transition_pair()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    paired_transition_count INTEGER;
BEGIN
    SELECT count(*)
      INTO paired_transition_count
      FROM ingestion_dispatch_transition
     WHERE id = NEW.running_transition_id
       AND ingestion_dispatch_id = NEW.ingestion_dispatch_id
       AND state = 'running'
       AND attempt_count = NEW.attempt_number
       AND worker_identity = NEW.worker_identity
       AND occurred_at = NEW.claimed_at;
    IF paired_transition_count <> 1 THEN
        RAISE EXCEPTION 'attempt claim requires its exact committed running transition';
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_ingestion_attempt_receipt_completion_pair()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    paired_completion_count INTEGER;
BEGIN
    SELECT count(*)
      INTO paired_completion_count
      FROM ingestion_dispatch_attempt_completion AS completion
      JOIN provider_payload_receipt AS receipt
        ON receipt.id = NEW.provider_payload_receipt_id
     WHERE completion.attempt_receipt_id = NEW.id
       AND completion.attempt_claim_id = NEW.attempt_claim_id
       AND completion.ingestion_dispatch_id = NEW.ingestion_dispatch_id
       AND completion.attempt_number = NEW.attempt_number
       AND (
           completion.outcome <> 'succeeded'
           OR receipt.provider_response_status BETWEEN 200 AND 299
       );
    IF paired_completion_count <> 1 THEN
        RAISE EXCEPTION 'attempt receipt requires an exact status-compatible completion bundle';
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_ingestion_attempt_completion_transition_pair()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    paired_transition_count INTEGER;
BEGIN
    SELECT count(*)
      INTO paired_transition_count
      FROM ingestion_dispatch_transition
     WHERE id = NEW.completion_transition_id
       AND ingestion_dispatch_id = NEW.ingestion_dispatch_id
       AND state = NEW.outcome
       AND attempt_count = NEW.attempt_number
       AND worker_identity = NEW.worker_identity
       AND failure_code IS NOT DISTINCT FROM NEW.failure_code
       AND dead_letter_reason IS NOT DISTINCT FROM NEW.dead_letter_reason
       AND retry_not_before_at IS NOT DISTINCT FROM NEW.retry_not_before_at
       AND occurred_at = NEW.completed_at;
    IF paired_transition_count <> 1 THEN
        RAISE EXCEPTION 'attempt completion requires its exact terminal transition';
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_ingestion_runtime_transition_reverse_pair()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    paired_outbox_id UUID;
    runtime_fact_count INTEGER;
BEGIN
    IF NEW.state = 'queued' THEN
        SELECT id
          INTO paired_outbox_id
          FROM ingestion_dispatch_outbox
         WHERE ingestion_dispatch_id = NEW.ingestion_dispatch_id
           AND attempt_number = NEW.attempt_count + 1;
        IF EXISTS (
            SELECT 1
            FROM ingestion_outbox_publication_claim
            WHERE ingestion_dispatch_outbox_id = paired_outbox_id
        ) THEN
            SELECT count(*)
              INTO runtime_fact_count
              FROM ingestion_outbox_publication_delivery
             WHERE ingestion_dispatch_outbox_id = paired_outbox_id
               AND delivered_at = NEW.occurred_at;
            IF runtime_fact_count <> 1 THEN
                RAISE EXCEPTION 'runtime queued transition requires its publication delivery';
            END IF;
        END IF;
    ELSIF NEW.state = 'running' THEN
        IF EXISTS (
            SELECT 1
            FROM ingestion_dispatch_outbox AS outbox
            JOIN ingestion_outbox_publication_delivery AS delivery
              ON delivery.ingestion_dispatch_outbox_id = outbox.id
            WHERE outbox.ingestion_dispatch_id = NEW.ingestion_dispatch_id
              AND outbox.attempt_number = NEW.attempt_count
        ) THEN
            SELECT count(*)
              INTO runtime_fact_count
              FROM ingestion_dispatch_attempt_claim
             WHERE running_transition_id = NEW.id
               AND ingestion_dispatch_id = NEW.ingestion_dispatch_id
               AND attempt_number = NEW.attempt_count
               AND worker_identity = NEW.worker_identity
               AND claimed_at = NEW.occurred_at;
            IF runtime_fact_count <> 1 THEN
                RAISE EXCEPTION 'runtime running transition requires its provider-attempt claim';
            END IF;
        END IF;
    ELSIF NEW.state IN ('succeeded', 'retry_wait', 'dead_lettered') THEN
        IF EXISTS (
            SELECT 1
            FROM ingestion_dispatch_attempt_claim
            WHERE ingestion_dispatch_id = NEW.ingestion_dispatch_id
              AND attempt_number = NEW.attempt_count
        ) THEN
            SELECT count(*)
              INTO runtime_fact_count
              FROM ingestion_dispatch_attempt_completion
             WHERE completion_transition_id = NEW.id
               AND ingestion_dispatch_id = NEW.ingestion_dispatch_id
               AND attempt_number = NEW.attempt_count
               AND outcome = NEW.state
               AND completed_at = NEW.occurred_at;
            IF runtime_fact_count <> 1 THEN
                RAISE EXCEPTION 'runtime terminal transition requires its attempt completion';
            END IF;
        END IF;
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER ingestion_publication_delivery_transition_integrity
    AFTER INSERT ON ingestion_outbox_publication_delivery
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_publication_delivery_transition_pair();
CREATE CONSTRAINT TRIGGER ingestion_attempt_claim_transition_integrity
    AFTER INSERT ON ingestion_dispatch_attempt_claim
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_attempt_claim_transition_pair();
CREATE CONSTRAINT TRIGGER ingestion_attempt_receipt_completion_integrity
    AFTER INSERT ON ingestion_dispatch_attempt_receipt
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_attempt_receipt_completion_pair();
CREATE CONSTRAINT TRIGGER ingestion_attempt_completion_transition_integrity
    AFTER INSERT ON ingestion_dispatch_attempt_completion
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_attempt_completion_transition_pair();
CREATE CONSTRAINT TRIGGER ingestion_runtime_transition_reverse_integrity
    AFTER INSERT ON ingestion_dispatch_transition
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_runtime_transition_reverse_pair();

-- Claim one eligible outbox row.  Zero rows means there is currently no work.
-- Repeating an active token is safe and returns the same claim; an expired
-- token returns `expired` and never renews itself.  Publisher leases are the
-- only leases that can be followed by a new claim because no provider call has
-- occurred at this stage.
CREATE OR REPLACE FUNCTION claim_ingestion_outbox_publication(
    p_publisher_identity TEXT,
    p_lease_token UUID
) RETURNS TABLE (
    disposition TEXT,
    publication_claim_id UUID,
    outbox_id UUID,
    dispatch_id UUID,
    attempt_number SMALLINT,
    claimed_at TIMESTAMPTZ,
    lease_expires_at TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    existing_claim_id UUID;
    existing_outbox_id UUID;
    existing_dispatch_id UUID;
    existing_attempt_number SMALLINT;
    existing_publisher_identity TEXT;
    existing_claimed_at TIMESTAMPTZ;
    existing_lease_expires_at TIMESTAMPTZ;
    existing_delivery_count INTEGER;
    candidate_outbox_id UUID;
    candidate_dispatch_id UUID;
    candidate_attempt_number SMALLINT;
    candidate_provider TEXT;
    new_claim_id UUID;
    new_claimed_at TIMESTAMPTZ;
    new_lease_expires_at TIMESTAMPTZ;
BEGIN
    IF p_publisher_identity IS NULL
       OR p_publisher_identity !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
       OR p_publisher_identity ~* '(api[-_]?key|token|secret|password|authorization|credential|cookie|bearer)'
       OR p_lease_token IS NULL THEN
        RAISE EXCEPTION 'publication claim identity or lease token is invalid';
    END IF;

    SELECT claim.id, claim.ingestion_dispatch_outbox_id,
           outbox.ingestion_dispatch_id, outbox.attempt_number,
           claim.publisher_identity, claim.claimed_at,
           claim.lease_expires_at,
           count(delivery.id)::INTEGER
      INTO existing_claim_id, existing_outbox_id, existing_dispatch_id,
           existing_attempt_number, existing_publisher_identity,
           existing_claimed_at, existing_lease_expires_at,
           existing_delivery_count
      FROM ingestion_outbox_publication_claim AS claim
      JOIN ingestion_dispatch_outbox AS outbox
        ON outbox.id = claim.ingestion_dispatch_outbox_id
      LEFT JOIN ingestion_outbox_publication_delivery AS delivery
        ON delivery.publication_claim_id = claim.id
     WHERE claim.lease_token = p_lease_token
     GROUP BY claim.id, outbox.ingestion_dispatch_id, outbox.attempt_number;

    IF existing_claim_id IS NOT NULL THEN
        IF existing_publisher_identity IS DISTINCT FROM p_publisher_identity THEN
            RAISE EXCEPTION 'publication lease token belongs to another publisher identity';
        END IF;
        RETURN QUERY SELECT
            CASE
                WHEN existing_delivery_count = 1 THEN 'delivered'
                WHEN existing_lease_expires_at <= clock_timestamp() THEN 'expired'
                ELSE 'publishable'
            END,
            existing_claim_id,
            existing_outbox_id,
            existing_dispatch_id,
            existing_attempt_number,
            existing_claimed_at,
            existing_lease_expires_at;
        RETURN;
    END IF;

    LOOP
        SELECT outbox.id, outbox.ingestion_dispatch_id,
               outbox.attempt_number, dispatch.provider
          INTO candidate_outbox_id, candidate_dispatch_id,
               candidate_attempt_number, candidate_provider
          FROM ingestion_dispatch_outbox AS outbox
          JOIN ingestion_dispatch AS dispatch
            ON dispatch.id = outbox.ingestion_dispatch_id
          JOIN LATERAL (
              SELECT state, attempt_count
              FROM ingestion_dispatch_transition
              WHERE ingestion_dispatch_id = outbox.ingestion_dispatch_id
              ORDER BY state_sequence DESC
              LIMIT 1
          ) AS transition ON TRUE
         WHERE outbox.available_at <= clock_timestamp()
           AND transition.attempt_count = outbox.attempt_number - 1
           AND transition.state = CASE
               WHEN outbox.attempt_number = 1 THEN 'pending'
               ELSE 'retry_wait'
           END
           AND NOT EXISTS (
               SELECT 1
               FROM ingestion_outbox_publication_delivery AS delivery
               WHERE delivery.ingestion_dispatch_outbox_id = outbox.id
           )
           AND NOT EXISTS (
               SELECT 1
               FROM ingestion_outbox_publication_claim AS claim
               WHERE claim.ingestion_dispatch_outbox_id = outbox.id
                 AND claim.lease_expires_at > clock_timestamp()
           )
         ORDER BY outbox.available_at, outbox.created_at, outbox.id
         LIMIT 1;

        IF candidate_outbox_id IS NULL THEN
            RETURN;
        END IF;

        PERFORM lock_ingestion_provider(candidate_provider);
        PERFORM 1
          FROM ingestion_dispatch_outbox
         WHERE id = candidate_outbox_id
           FOR UPDATE;

        IF EXISTS (
               SELECT 1
               FROM ingestion_outbox_publication_delivery AS delivery
               WHERE delivery.ingestion_dispatch_outbox_id = candidate_outbox_id
           )
           OR EXISTS (
               SELECT 1
               FROM ingestion_outbox_publication_claim AS claim
               WHERE claim.ingestion_dispatch_outbox_id = candidate_outbox_id
                 AND claim.lease_expires_at > clock_timestamp()
           )
           OR NOT EXISTS (
               SELECT 1
               FROM ingestion_dispatch_outbox AS outbox
               JOIN LATERAL (
                   SELECT state, attempt_count
                   FROM ingestion_dispatch_transition
                   WHERE ingestion_dispatch_id = outbox.ingestion_dispatch_id
                   ORDER BY state_sequence DESC
                   LIMIT 1
               ) AS transition ON TRUE
               WHERE outbox.id = candidate_outbox_id
                 AND outbox.available_at <= clock_timestamp()
                 AND transition.attempt_count = outbox.attempt_number - 1
                 AND transition.state = CASE
                     WHEN outbox.attempt_number = 1 THEN 'pending'
                     ELSE 'retry_wait'
                 END
           ) THEN
            candidate_outbox_id := NULL;
            CONTINUE;
        END IF;

        new_claimed_at := clock_timestamp();
        INSERT INTO ingestion_outbox_publication_claim (
            ingestion_dispatch_outbox_id, claim_sequence,
            publisher_identity, lease_token, claimed_at,
            lease_expires_at
        ) VALUES (
            candidate_outbox_id, 1, p_publisher_identity, p_lease_token,
            new_claimed_at, new_claimed_at + interval '2 minutes'
        )
        RETURNING id, ingestion_outbox_publication_claim.claimed_at,
                  ingestion_outbox_publication_claim.lease_expires_at
             INTO new_claim_id, new_claimed_at, new_lease_expires_at;

        RETURN QUERY SELECT
            'publishable'::TEXT,
            new_claim_id,
            candidate_outbox_id,
            candidate_dispatch_id,
            candidate_attempt_number,
            new_claimed_at,
            new_lease_expires_at;
        RETURN;
    END LOOP;
END;
$$;

-- Claiming a dispatch appends the provider-attempt claim and its running
-- transition in one transaction.  The repository must commit that transaction
-- before exposing a `started` result to provider-calling code.  Every later
-- delivery of the same broker envelope returns `inconclusive` (or `terminal`)
-- and can never regain provider-call permission.
CREATE OR REPLACE FUNCTION claim_ingestion_dispatch_attempt(
    p_dispatch_id UUID,
    p_attempt_number SMALLINT,
    p_worker_identity TEXT,
    p_lease_token UUID
) RETURNS TABLE (
    disposition TEXT,
    provider_call_permitted BOOLEAN,
    claim_id UUID,
    running_transition_id UUID,
    provider_use_authorization_id UUID,
    quota_receipt_id UUID,
    provider TEXT,
    source_type TEXT,
    request_fingerprint_sha256 CHAR(64),
    window_start TIMESTAMPTZ,
    window_end TIMESTAMPTZ,
    estimated_cost INTEGER,
    policy_version TEXT,
    max_attempts SMALLINT,
    license_scope TEXT,
    license_version TEXT,
    exposure TEXT,
    started_at TIMESTAMPTZ,
    lease_expires_at TIMESTAMPTZ,
    min_request_interval INTERVAL,
    quota_floor INTEGER,
    quota_max_age INTERVAL,
    retry_schedule_sha256 CHAR(64),
    authorization_effective_until TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_dispatch_provider TEXT;
    v_dispatch_source_type TEXT;
    v_request_fingerprint CHAR(64);
    v_window_start TIMESTAMPTZ;
    v_window_end TIMESTAMPTZ;
    v_estimated_cost INTEGER;
    v_policy_version TEXT;
    v_max_attempts SMALLINT;
    v_min_request_interval INTERVAL;
    v_quota_floor INTEGER;
    v_quota_max_age INTERVAL;
    v_retry_schedule_sha256 CHAR(64);
    v_authorization_id UUID;
    v_license_scope TEXT;
    v_license_version TEXT;
    v_exposure TEXT;
    v_reviewed_at TIMESTAMPTZ;
    v_effective_from TIMESTAMPTZ;
    v_effective_until TIMESTAMPTZ;
    v_reservation_id UUID;
    v_quota_receipt_id UUID;
    v_quota_remaining INTEGER;
    v_quota_received_at TIMESTAMPTZ;
    v_quota_created_at TIMESTAMPTZ;
    v_reserved_credits BIGINT;
    v_prior_completed_at TIMESTAMPTZ;
    v_prior_min_request_interval INTERVAL;
    v_delivery_id UUID;
    v_existing_claim_id UUID;
    v_existing_running_transition_id UUID;
    v_existing_completion_id UUID;
    v_latest_sequence INTEGER;
    v_latest_state TEXT;
    v_latest_attempt_count SMALLINT;
    v_claim_id UUID;
    v_running_transition_id UUID;
    v_started_at TIMESTAMPTZ;
    v_lease_expires_at TIMESTAMPTZ;
BEGIN
    IF p_dispatch_id IS NULL
       OR p_attempt_number IS NULL
       OR p_attempt_number < 1
       OR p_attempt_number > 5
       OR p_worker_identity IS NULL
       OR p_worker_identity !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
       OR p_worker_identity ~* '(api[-_]?key|token|secret|password|authorization|credential|cookie|bearer)'
       OR p_lease_token IS NULL THEN
        disposition := 'rejected';
        provider_call_permitted := FALSE;
        RETURN NEXT;
        RETURN;
    END IF;

    SELECT dispatch.provider
      INTO v_dispatch_provider
      FROM ingestion_dispatch AS dispatch
     WHERE dispatch.id = p_dispatch_id;
    IF NOT FOUND THEN
        disposition := 'rejected';
        provider_call_permitted := FALSE;
        RETURN NEXT;
        RETURN;
    END IF;
    PERFORM lock_ingestion_provider(v_dispatch_provider);

    SELECT attempt_claim.id, attempt_claim.running_transition_id,
           completion.id
      INTO v_existing_claim_id, v_existing_running_transition_id,
           v_existing_completion_id
      FROM ingestion_dispatch_attempt_claim AS attempt_claim
      LEFT JOIN ingestion_dispatch_attempt_completion AS completion
        ON completion.attempt_claim_id = attempt_claim.id
     WHERE attempt_claim.ingestion_dispatch_id = p_dispatch_id
       AND attempt_claim.attempt_number = p_attempt_number;
    IF v_existing_claim_id IS NOT NULL THEN
        disposition := CASE
            WHEN v_existing_completion_id IS NULL THEN 'inconclusive'
            ELSE 'terminal'
        END;
        provider_call_permitted := FALSE;
        claim_id := v_existing_claim_id;
        running_transition_id := v_existing_running_transition_id;
        RETURN NEXT;
        RETURN;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM ingestion_dispatch_attempt_claim AS attempt_claim
        LEFT JOIN ingestion_dispatch_attempt_completion AS completion
          ON completion.attempt_claim_id = attempt_claim.id
        WHERE attempt_claim.provider = v_dispatch_provider
          AND completion.id IS NULL
    ) THEN
        disposition := 'not_ready';
        provider_call_permitted := FALSE;
        RETURN NEXT;
        RETURN;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM ingestion_dispatch_attempt_claim AS attempt_claim
        WHERE attempt_claim.lease_token = p_lease_token
    ) THEN
        disposition := 'rejected';
        provider_call_permitted := FALSE;
        RETURN NEXT;
        RETURN;
    END IF;

    SELECT dispatch.provider, dispatch.source_type,
           dispatch.request_fingerprint_sha256, dispatch.window_start,
           dispatch.window_end, dispatch.estimated_cost,
           dispatch.policy_version, dispatch.max_attempts,
           dispatch.min_request_interval, dispatch.quota_floor,
           dispatch.quota_max_age, dispatch.retry_schedule_sha256,
           dispatch.provider_use_authorization_id,
           authorization.license_scope, authorization.license_version,
           authorization.exposure, authorization.reviewed_at,
           authorization.effective_from, authorization.effective_until
      INTO v_dispatch_provider, v_dispatch_source_type,
           v_request_fingerprint, v_window_start, v_window_end,
           v_estimated_cost, v_policy_version, v_max_attempts,
           v_min_request_interval, v_quota_floor, v_quota_max_age,
           v_retry_schedule_sha256,
           v_authorization_id, v_license_scope, v_license_version,
           v_exposure, v_reviewed_at, v_effective_from, v_effective_until
      FROM ingestion_dispatch AS dispatch
      JOIN provider_use_authorization AS authorization
        ON authorization.id = dispatch.provider_use_authorization_id
     WHERE dispatch.id = p_dispatch_id
       FOR UPDATE OF dispatch;

    SELECT transition.state_sequence, transition.state,
           transition.attempt_count
      INTO v_latest_sequence, v_latest_state, v_latest_attempt_count
      FROM ingestion_dispatch_transition AS transition
     WHERE transition.ingestion_dispatch_id = p_dispatch_id
     ORDER BY transition.state_sequence DESC
     LIMIT 1;
    IF v_latest_state IN ('succeeded', 'dead_lettered', 'cancelled') THEN
        disposition := 'terminal';
        provider_call_permitted := FALSE;
        RETURN NEXT;
        RETURN;
    END IF;
    IF v_latest_state IS DISTINCT FROM 'queued'
       OR v_latest_attempt_count IS DISTINCT FROM p_attempt_number - 1 THEN
        disposition := 'not_ready';
        provider_call_permitted := FALSE;
        RETURN NEXT;
        RETURN;
    END IF;

    v_started_at := clock_timestamp();
    IF p_attempt_number > v_max_attempts
       OR v_dispatch_provider IS NULL
       OR v_min_request_interval IS NULL
       OR v_quota_floor IS NULL
       OR v_quota_max_age IS NULL
       OR v_retry_schedule_sha256 IS NULL
       OR v_exposure IS DISTINCT FROM 'private_raw'
       OR v_started_at < greatest(v_reviewed_at, v_effective_from)
       OR v_started_at + interval '5 minutes' > v_effective_until THEN
        INSERT INTO ingestion_dispatch_transition (
            ingestion_dispatch_id, state_sequence, state,
            attempt_count, occurred_at
        ) VALUES (
            p_dispatch_id, v_latest_sequence + 1, 'cancelled',
            p_attempt_number - 1, v_started_at
        );
        disposition := 'terminal';
        provider_call_permitted := FALSE;
        RETURN NEXT;
        RETURN;
    END IF;

    SELECT completion.completed_at, prior_dispatch.min_request_interval
      INTO v_prior_completed_at, v_prior_min_request_interval
      FROM ingestion_dispatch_attempt_completion AS completion
      JOIN ingestion_dispatch_attempt_claim AS prior_claim
        ON prior_claim.id = completion.attempt_claim_id
      JOIN ingestion_dispatch AS prior_dispatch
        ON prior_dispatch.id = prior_claim.ingestion_dispatch_id
     WHERE prior_claim.provider = v_dispatch_provider
     ORDER BY completion.completed_at DESC, completion.id DESC
     LIMIT 1;
    IF v_prior_completed_at IS NOT NULL
       AND (
           v_prior_min_request_interval IS NULL
           OR v_started_at < v_prior_completed_at + greatest(
               v_min_request_interval,
               v_prior_min_request_interval
           )
       ) THEN
        disposition := 'not_ready';
        provider_call_permitted := FALSE;
        RETURN NEXT;
        RETURN;
    END IF;

    SELECT reservation.id
      INTO v_reservation_id
      FROM ingestion_quota_reservation AS reservation
     WHERE reservation.ingestion_dispatch_id = p_dispatch_id
       AND reservation.attempt_number = p_attempt_number;
    SELECT delivery.id
      INTO v_delivery_id
      FROM ingestion_dispatch_outbox AS outbox
      JOIN ingestion_outbox_publication_delivery AS delivery
        ON delivery.ingestion_dispatch_outbox_id = outbox.id
     WHERE outbox.ingestion_dispatch_id = p_dispatch_id
       AND outbox.attempt_number = p_attempt_number;
    IF v_reservation_id IS NULL
       OR v_delivery_id IS NULL THEN
        disposition := 'not_ready';
        provider_call_permitted := FALSE;
        RETURN NEXT;
        RETURN;
    END IF;

    SELECT receipt.id, receipt.provider_quota_remaining,
           receipt.received_at, receipt.created_at
      INTO v_quota_receipt_id, v_quota_remaining,
           v_quota_received_at, v_quota_created_at
      FROM provider_payload_receipt AS receipt
     WHERE receipt.provider = v_dispatch_provider
       AND receipt.license_scope IS NOT DISTINCT FROM v_license_scope
       AND receipt.license_version IS NOT DISTINCT FROM v_license_version
       AND receipt.provider_quota_remaining IS NOT NULL
       AND receipt.received_at <= v_started_at
       AND receipt.created_at <= v_started_at
     ORDER BY receipt.received_at DESC, receipt.created_at DESC,
              receipt.provider_quota_remaining ASC, receipt.id DESC
     LIMIT 1
       FOR KEY SHARE;
    SELECT COALESCE(sum(reservation.reserved_credits), 0)
      INTO v_reserved_credits
      FROM ingestion_quota_reservation AS reservation
      JOIN ingestion_dispatch AS reserved_dispatch
        ON reserved_dispatch.id = reservation.ingestion_dispatch_id
     WHERE reserved_dispatch.provider = v_dispatch_provider;
    IF v_quota_receipt_id IS NULL
       OR v_quota_received_at > v_quota_created_at
       OR v_started_at - v_quota_received_at > v_quota_max_age
       OR v_quota_remaining::BIGINT - v_reserved_credits < v_quota_floor THEN
        disposition := 'not_ready';
        provider_call_permitted := FALSE;
        RETURN NEXT;
        RETURN;
    END IF;

    v_claim_id := gen_random_uuid();
    v_running_transition_id := gen_random_uuid();
    INSERT INTO ingestion_dispatch_attempt_claim (
        id, publication_delivery_id, ingestion_dispatch_id, attempt_number,
        provider_use_authorization_id, quota_reservation_id,
        quota_receipt_id, running_transition_id, provider, source_type,
        request_fingerprint_sha256, window_start, window_end,
        license_scope, license_version, exposure, worker_identity,
        lease_token, claimed_at, lease_expires_at
    ) VALUES (
        v_claim_id, v_delivery_id, p_dispatch_id, p_attempt_number,
        v_authorization_id, v_reservation_id, v_quota_receipt_id,
        v_running_transition_id, v_dispatch_provider,
        v_dispatch_source_type, v_request_fingerprint, v_window_start,
        v_window_end, v_license_scope, v_license_version, v_exposure,
        p_worker_identity, p_lease_token, v_started_at,
        v_started_at + interval '5 minutes'
    )
    RETURNING ingestion_dispatch_attempt_claim.claimed_at,
              ingestion_dispatch_attempt_claim.lease_expires_at
         INTO v_started_at, v_lease_expires_at;

    INSERT INTO ingestion_dispatch_transition (
        id, ingestion_dispatch_id, state_sequence, state,
        attempt_count, worker_identity, occurred_at
    ) VALUES (
        v_running_transition_id, p_dispatch_id, v_latest_sequence + 1,
        'running', p_attempt_number, p_worker_identity, v_started_at
    );

    disposition := 'started';
    provider_call_permitted := TRUE;
    claim_id := v_claim_id;
    running_transition_id := v_running_transition_id;
    provider_use_authorization_id := v_authorization_id;
    quota_receipt_id := v_quota_receipt_id;
    provider := v_dispatch_provider;
    source_type := v_dispatch_source_type;
    request_fingerprint_sha256 := v_request_fingerprint;
    window_start := v_window_start;
    window_end := v_window_end;
    estimated_cost := v_estimated_cost;
    policy_version := v_policy_version;
    max_attempts := v_max_attempts;
    license_scope := v_license_scope;
    license_version := v_license_version;
    exposure := v_exposure;
    started_at := v_started_at;
    lease_expires_at := v_lease_expires_at;
    min_request_interval := v_min_request_interval;
    quota_floor := v_quota_floor;
    quota_max_age := v_quota_max_age;
    retry_schedule_sha256 := v_retry_schedule_sha256;
    authorization_effective_until := v_effective_until;
    RETURN NEXT;
END;
$$;

-- The runtime reads database time immediately before planning a terminal
-- result.  This seam never extends a lease and refuses stale or foreign claims.
CREATE OR REPLACE FUNCTION read_ingestion_dispatch_attempt_time(
    p_claim_id UUID,
    p_worker_identity TEXT,
    p_lease_token UUID
) RETURNS TIMESTAMPTZ
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    claim_provider TEXT;
    claim_dispatch_id UUID;
    claim_attempt_number SMALLINT;
    claim_lease_expires_at TIMESTAMPTZ;
    authorization_reviewed_at TIMESTAMPTZ;
    authorization_effective_from TIMESTAMPTZ;
    authorization_effective_until TIMESTAMPTZ;
    latest_state TEXT;
    latest_attempt_count SMALLINT;
    checked_at TIMESTAMPTZ;
BEGIN
    SELECT attempt_claim.provider
      INTO claim_provider
      FROM ingestion_dispatch_attempt_claim AS attempt_claim
     WHERE attempt_claim.id = p_claim_id
       AND attempt_claim.worker_identity = p_worker_identity
       AND attempt_claim.lease_token = p_lease_token;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'attempt time requires an exact active uncompleted claim';
    END IF;
    PERFORM lock_ingestion_provider(claim_provider);

    SELECT attempt_claim.ingestion_dispatch_id,
           attempt_claim.attempt_number, attempt_claim.lease_expires_at,
           authorization.reviewed_at, authorization.effective_from,
           authorization.effective_until
      INTO claim_dispatch_id, claim_attempt_number, claim_lease_expires_at,
           authorization_reviewed_at, authorization_effective_from,
           authorization_effective_until
      FROM ingestion_dispatch_attempt_claim AS attempt_claim
      JOIN provider_use_authorization AS authorization
        ON authorization.id = attempt_claim.provider_use_authorization_id
     WHERE attempt_claim.id = p_claim_id
       AND attempt_claim.worker_identity = p_worker_identity
       AND attempt_claim.lease_token = p_lease_token
       AND NOT EXISTS (
           SELECT 1
           FROM ingestion_dispatch_attempt_completion AS completion
           WHERE completion.attempt_claim_id = attempt_claim.id
       );
    checked_at := clock_timestamp();
    IF NOT FOUND
       OR checked_at >= claim_lease_expires_at
       OR checked_at < greatest(
           authorization_reviewed_at,
           authorization_effective_from
       )
       OR checked_at >= authorization_effective_until THEN
        RAISE EXCEPTION 'attempt time requires an exact active uncompleted claim';
    END IF;

    SELECT transition.state, transition.attempt_count
      INTO latest_state, latest_attempt_count
      FROM ingestion_dispatch_transition AS transition
     WHERE transition.ingestion_dispatch_id = claim_dispatch_id
     ORDER BY transition.state_sequence DESC
     LIMIT 1;
    IF latest_state IS DISTINCT FROM 'running'
       OR latest_attempt_count IS DISTINCT FROM claim_attempt_number
       OR EXISTS (
           SELECT 1
           FROM ingestion_dispatch_attempt_claim AS other_claim
           LEFT JOIN ingestion_dispatch_attempt_completion AS completion
             ON completion.attempt_claim_id = other_claim.id
           WHERE other_claim.provider = claim_provider
             AND other_claim.id <> p_claim_id
             AND completion.id IS NULL
       ) THEN
        RAISE EXCEPTION 'attempt time requires the sole latest provider claim';
    END IF;
    RETURN checked_at;
END;
$$;

-- Call this only after the broker accepts the dispatch envelope.  The envelope
-- itself remains exactly (dispatch_id, attempt_number); publisher identities
-- and lease tokens are repository metadata and never cross the broker.
CREATE OR REPLACE FUNCTION record_ingestion_outbox_publication(
    p_publication_claim_id UUID,
    p_publisher_identity TEXT,
    p_lease_token UUID
) RETURNS TABLE (
    disposition TEXT,
    delivery_id UUID
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    claim_publisher_identity TEXT;
    claim_lease_token UUID;
    claim_lease_expires_at TIMESTAMPTZ;
    claimed_outbox_id UUID;
    claimed_dispatch_id UUID;
    claimed_attempt_number SMALLINT;
    dispatch_provider TEXT;
    existing_delivery_id UUID;
    new_delivery_id UUID;
    delivered_time TIMESTAMPTZ;
    latest_sequence INTEGER;
BEGIN
    SELECT claim.publisher_identity, claim.lease_token,
           claim.lease_expires_at, outbox.id,
           outbox.ingestion_dispatch_id, outbox.attempt_number,
           dispatch.provider
      INTO claim_publisher_identity, claim_lease_token,
           claim_lease_expires_at, claimed_outbox_id,
           claimed_dispatch_id, claimed_attempt_number,
           dispatch_provider
      FROM ingestion_outbox_publication_claim AS claim
      JOIN ingestion_dispatch_outbox AS outbox
        ON outbox.id = claim.ingestion_dispatch_outbox_id
      JOIN ingestion_dispatch AS dispatch
        ON dispatch.id = outbox.ingestion_dispatch_id
     WHERE claim.id = p_publication_claim_id;
    IF NOT FOUND
       OR claim_publisher_identity IS DISTINCT FROM p_publisher_identity
       OR claim_lease_token IS DISTINCT FROM p_lease_token THEN
        RAISE EXCEPTION 'publication completion does not match its exact claim';
    END IF;

    PERFORM lock_ingestion_provider(dispatch_provider);
    SELECT id
      INTO existing_delivery_id
      FROM ingestion_outbox_publication_delivery
     WHERE publication_claim_id = p_publication_claim_id;
    IF existing_delivery_id IS NOT NULL THEN
        disposition := 'already_recorded';
        delivery_id := existing_delivery_id;
        RETURN NEXT;
        RETURN;
    END IF;
    IF clock_timestamp() >= claim_lease_expires_at THEN
        RAISE EXCEPTION 'expired publication claims cannot record delivery';
    END IF;

    new_delivery_id := gen_random_uuid();
    INSERT INTO ingestion_outbox_publication_delivery (
        id, publication_claim_id, ingestion_dispatch_outbox_id,
        publication_id, delivered_at
    ) VALUES (
        new_delivery_id, p_publication_claim_id, claimed_outbox_id,
        p_lease_token, clock_timestamp()
    )
    RETURNING delivered_at INTO delivered_time;

    SELECT state_sequence
      INTO latest_sequence
      FROM ingestion_dispatch_transition
     WHERE ingestion_dispatch_id = claimed_dispatch_id
     ORDER BY state_sequence DESC
     LIMIT 1;

    INSERT INTO ingestion_dispatch_transition (
        ingestion_dispatch_id, state_sequence, state,
        attempt_count, occurred_at
    ) VALUES (
        claimed_dispatch_id, latest_sequence + 1, 'queued',
        claimed_attempt_number - 1, delivered_time
    );
    disposition := 'recorded';
    delivery_id := new_delivery_id;
    RETURN NEXT;
END;
$$;

-- Complete one exact provider attempt.  Exact retries return the same immutable
-- completion; any conflicting second outcome is rejected.  `retry_wait` also
-- appends the next quota reservation and outbox row under the provider lock,
-- selecting the latest exact-license quota receipt inside this transaction.
CREATE OR REPLACE FUNCTION complete_ingestion_dispatch_attempt(
    p_claim_id UUID,
    p_worker_identity TEXT,
    p_lease_token UUID,
    p_outcome TEXT,
    p_failure_code TEXT DEFAULT NULL,
    p_dead_letter_reason TEXT DEFAULT NULL,
    p_retry_not_before_at TIMESTAMPTZ DEFAULT NULL,
    p_provider_payload_receipt_id UUID DEFAULT NULL,
    p_retry_safety TEXT DEFAULT NULL
) RETURNS TABLE (
    disposition TEXT,
    completion_id UUID
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    claim_record ingestion_dispatch_attempt_claim%ROWTYPE;
    existing_completion_id UUID;
    existing_outcome TEXT;
    existing_failure_code TEXT;
    existing_retry_safety TEXT;
    existing_dead_letter_reason TEXT;
    existing_retry_not_before_at TIMESTAMPTZ;
    existing_provider_payload_receipt_id UUID;
    v_latest_state TEXT;
    v_latest_attempt_count SMALLINT;
    v_latest_sequence INTEGER;
    v_dispatch_estimated_cost INTEGER;
    v_dispatch_max_attempts SMALLINT;
    v_authorization_effective_until TIMESTAMPTZ;
    v_latest_quota_receipt_id UUID;
    v_attempt_receipt_id UUID;
    v_completion_id UUID;
    v_completion_transition_id UUID;
    v_completed_at TIMESTAMPTZ;
BEGIN
    SELECT *
      INTO claim_record
      FROM ingestion_dispatch_attempt_claim
     WHERE id = p_claim_id;
    IF NOT FOUND
       OR claim_record.worker_identity IS DISTINCT FROM p_worker_identity
       OR claim_record.lease_token IS DISTINCT FROM p_lease_token THEN
        RAISE EXCEPTION 'attempt completion does not match its exact claim lease';
    END IF;
    PERFORM lock_ingestion_provider(claim_record.provider);

    SELECT completion.id, completion.outcome, completion.failure_code,
           completion.retry_safety, completion.dead_letter_reason,
           completion.retry_not_before_at,
           receipt.provider_payload_receipt_id
      INTO existing_completion_id, existing_outcome,
           existing_failure_code, existing_retry_safety,
           existing_dead_letter_reason,
           existing_retry_not_before_at,
           existing_provider_payload_receipt_id
      FROM ingestion_dispatch_attempt_completion AS completion
      LEFT JOIN ingestion_dispatch_attempt_receipt AS receipt
        ON receipt.id = completion.attempt_receipt_id
     WHERE completion.attempt_claim_id = p_claim_id;
    IF existing_completion_id IS NOT NULL THEN
        IF existing_outcome IS NOT DISTINCT FROM p_outcome
           AND existing_failure_code IS NOT DISTINCT FROM p_failure_code
           AND existing_retry_safety IS NOT DISTINCT FROM p_retry_safety
           AND existing_dead_letter_reason IS NOT DISTINCT FROM
                p_dead_letter_reason
           AND existing_retry_not_before_at IS NOT DISTINCT FROM
                p_retry_not_before_at
           AND existing_provider_payload_receipt_id IS NOT DISTINCT FROM
                p_provider_payload_receipt_id THEN
            disposition := 'already_committed';
            completion_id := existing_completion_id;
            RETURN NEXT;
            RETURN;
        END IF;
        RAISE EXCEPTION 'attempt already has a conflicting immutable completion';
    END IF;

    IF p_outcome NOT IN ('succeeded', 'retry_wait', 'dead_lettered')
       OR clock_timestamp() >= claim_record.lease_expires_at THEN
        RAISE EXCEPTION 'attempt completion is invalid or its lease is no longer conclusive';
    END IF;
    IF p_outcome = 'retry_wait'
       AND (
           p_retry_safety IS DISTINCT FROM 'request_not_sent'
           OR p_provider_payload_receipt_id IS NOT NULL
           OR p_failure_code IS NULL
           OR p_failure_code NOT IN (
               'provider_rate_limited',
               'provider_temporary_unavailable',
               'network_timeout',
               'storage_unavailable',
               'database_unavailable',
               'queue_unavailable',
               'internal_transient'
           )
       ) THEN
        RAISE EXCEPTION 'retry completion requires an approved failure and replay-safety proof';
    END IF;

    SELECT transition.state_sequence, transition.state,
           transition.attempt_count
      INTO v_latest_sequence, v_latest_state, v_latest_attempt_count
      FROM ingestion_dispatch_transition AS transition
     WHERE transition.ingestion_dispatch_id =
           claim_record.ingestion_dispatch_id
     ORDER BY transition.state_sequence DESC
     LIMIT 1;
    IF v_latest_state IS DISTINCT FROM 'running'
       OR v_latest_attempt_count IS DISTINCT FROM claim_record.attempt_number THEN
        RAISE EXCEPTION 'attempt completion requires its exact latest running state';
    END IF;

    SELECT dispatch.estimated_cost, dispatch.max_attempts,
           authorization.effective_until
      INTO v_dispatch_estimated_cost, v_dispatch_max_attempts,
           v_authorization_effective_until
      FROM ingestion_dispatch AS dispatch
      JOIN provider_use_authorization AS authorization
        ON authorization.id = dispatch.provider_use_authorization_id
     WHERE dispatch.id = claim_record.ingestion_dispatch_id
       FOR UPDATE OF dispatch;

    IF p_outcome = 'succeeded'
       AND p_provider_payload_receipt_id IS NULL THEN
        RAISE EXCEPTION 'successful provider attempts require an exact payload receipt';
    END IF;
    IF p_outcome = 'retry_wait' THEN
        IF claim_record.attempt_number >= v_dispatch_max_attempts THEN
            RAISE EXCEPTION 'the final reviewed attempt cannot schedule another retry';
        END IF;
        IF p_retry_not_before_at IS NULL
           OR p_retry_not_before_at + interval '5 minutes' >
                v_authorization_effective_until THEN
            RAISE EXCEPTION 'retry cannot fit inside the exact provider authorization';
        END IF;
        SELECT receipt.id
          INTO v_latest_quota_receipt_id
          FROM provider_payload_receipt AS receipt
         WHERE receipt.provider = claim_record.provider
           AND receipt.license_scope IS NOT DISTINCT FROM
                claim_record.license_scope
           AND receipt.license_version IS NOT DISTINCT FROM
                claim_record.license_version
           AND receipt.provider_quota_remaining IS NOT NULL
         ORDER BY receipt.received_at DESC, receipt.created_at DESC,
                  receipt.provider_quota_remaining ASC, receipt.id DESC
         LIMIT 1
           FOR KEY SHARE;
        IF v_latest_quota_receipt_id IS NULL THEN
            RAISE EXCEPTION 'retry completion requires a latest exact-license quota receipt';
        END IF;
    END IF;

    IF p_provider_payload_receipt_id IS NOT NULL THEN
        v_attempt_receipt_id := gen_random_uuid();
        INSERT INTO ingestion_dispatch_attempt_receipt (
            id, attempt_claim_id, ingestion_dispatch_id, attempt_number,
            provider_use_authorization_id, provider_payload_receipt_id,
            provider, source_type, request_fingerprint_sha256,
            window_start, window_end, license_scope, license_version,
            linked_at
        ) VALUES (
            v_attempt_receipt_id, claim_record.id,
            claim_record.ingestion_dispatch_id, claim_record.attempt_number,
            claim_record.provider_use_authorization_id,
            p_provider_payload_receipt_id, claim_record.provider,
            claim_record.source_type, claim_record.request_fingerprint_sha256,
            claim_record.window_start, claim_record.window_end,
            claim_record.license_scope, claim_record.license_version,
            clock_timestamp()
        );
    END IF;

    v_completion_id := gen_random_uuid();
    v_completion_transition_id := gen_random_uuid();
    INSERT INTO ingestion_dispatch_attempt_completion (
        id, attempt_claim_id, ingestion_dispatch_id, attempt_number,
        completion_transition_id, attempt_receipt_id, outcome,
        worker_identity, resolution_kind, resolver_identity,
        failure_code, retry_safety, dead_letter_reason, retry_not_before_at,
        completed_at
    ) VALUES (
        v_completion_id, claim_record.id,
        claim_record.ingestion_dispatch_id, claim_record.attempt_number,
        v_completion_transition_id, v_attempt_receipt_id, p_outcome,
        claim_record.worker_identity, 'worker', claim_record.worker_identity,
        p_failure_code, p_retry_safety, p_dead_letter_reason,
        p_retry_not_before_at,
        clock_timestamp()
    )
    RETURNING ingestion_dispatch_attempt_completion.completed_at
         INTO v_completed_at;

    IF p_outcome = 'retry_wait' THEN
        INSERT INTO ingestion_quota_reservation (
            ingestion_dispatch_id, attempt_number, reserved_credits,
            reserved_at, provider_payload_receipt_id
        ) VALUES (
            claim_record.ingestion_dispatch_id,
            claim_record.attempt_number + 1,
            v_dispatch_estimated_cost, v_completed_at,
            v_latest_quota_receipt_id
        );
        INSERT INTO ingestion_dispatch_outbox (
            ingestion_dispatch_id, attempt_number, available_at
        ) VALUES (
            claim_record.ingestion_dispatch_id,
            claim_record.attempt_number + 1,
            p_retry_not_before_at
        );
    END IF;

    INSERT INTO ingestion_dispatch_transition (
        id, ingestion_dispatch_id, state_sequence, state,
        attempt_count, worker_identity, failure_code,
        dead_letter_reason, retry_not_before_at, occurred_at
    ) VALUES (
        v_completion_transition_id, claim_record.ingestion_dispatch_id,
        v_latest_sequence + 1, p_outcome, claim_record.attempt_number,
        claim_record.worker_identity, p_failure_code,
        p_dead_letter_reason, p_retry_not_before_at, v_completed_at
    );

    disposition := 'committed';
    completion_id := v_completion_id;
    RETURN NEXT;
END;
$$;

CREATE TRIGGER ingestion_outbox_publication_claim_append_only
    BEFORE UPDATE OR DELETE ON ingestion_outbox_publication_claim
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_outbox_publication_delivery_append_only
    BEFORE UPDATE OR DELETE ON ingestion_outbox_publication_delivery
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_dispatch_attempt_claim_append_only
    BEFORE UPDATE OR DELETE ON ingestion_dispatch_attempt_claim
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_dispatch_attempt_receipt_append_only
    BEFORE UPDATE OR DELETE ON ingestion_dispatch_attempt_receipt
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_dispatch_attempt_completion_append_only
    BEFORE UPDATE OR DELETE ON ingestion_dispatch_attempt_completion
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();

CREATE TRIGGER ingestion_outbox_publication_claim_append_only_truncate
    BEFORE TRUNCATE ON ingestion_outbox_publication_claim
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_outbox_publication_delivery_append_only_truncate
    BEFORE TRUNCATE ON ingestion_outbox_publication_delivery
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_dispatch_attempt_claim_append_only_truncate
    BEFORE TRUNCATE ON ingestion_dispatch_attempt_claim
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_dispatch_attempt_receipt_append_only_truncate
    BEFORE TRUNCATE ON ingestion_dispatch_attempt_receipt
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER ingestion_dispatch_attempt_completion_append_only_truncate
    BEFORE TRUNCATE ON ingestion_dispatch_attempt_completion
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_mutation();

-- PostgreSQL functions are executable by PUBLIC by default.  These operations
-- are SECURITY INVOKER and intentionally receive no ambient privilege.  A
-- deployment must not grant these routines or their underlying table privileges
-- until a function-only/RLS boundary and role-level integration tests exist.
REVOKE ALL ON TABLE ingestion_outbox_publication_claim FROM PUBLIC;
REVOKE ALL ON TABLE ingestion_outbox_publication_delivery FROM PUBLIC;
REVOKE ALL ON TABLE ingestion_dispatch_attempt_claim FROM PUBLIC;
REVOKE ALL ON TABLE ingestion_dispatch_attempt_receipt FROM PUBLIC;
REVOKE ALL ON TABLE ingestion_dispatch_attempt_completion FROM PUBLIC;

-- 007 introduced this ordinary callable helper before the runtime privilege
-- boundary existed.  Leaving PostgreSQL's default PUBLIC EXECUTE in place
-- would let an otherwise unprivileged connection hold the provider advisory
-- lock for an entire transaction and deny admission/runtime work.
REVOKE EXECUTE ON FUNCTION lock_ingestion_provider(TEXT) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION lock_provider_payload_receipt_admission() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION lock_ingestion_dispatch_transition_provider() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION lock_ingestion_dispatch_outbox_provider() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION enforce_ingestion_dispatch_authorization() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION enforce_ingestion_quota_reservation() FROM PUBLIC;

REVOKE EXECUTE ON FUNCTION
    enforce_provider_payload_receipt_runtime_times() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    enforce_ingestion_publication_claim_insert() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    enforce_ingestion_publication_delivery_insert() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    enforce_ingestion_attempt_claim_insert() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    enforce_ingestion_attempt_receipt_insert() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    enforce_ingestion_attempt_completion_insert() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    enforce_ingestion_publication_delivery_transition_pair() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    enforce_ingestion_attempt_claim_transition_pair() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    enforce_ingestion_attempt_receipt_completion_pair() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    enforce_ingestion_attempt_completion_transition_pair() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    enforce_ingestion_runtime_transition_reverse_pair() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    claim_ingestion_outbox_publication(TEXT, UUID) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    record_ingestion_outbox_publication(UUID, TEXT, UUID) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    claim_ingestion_dispatch_attempt(UUID, SMALLINT, TEXT, UUID) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    read_ingestion_dispatch_attempt_time(UUID, TEXT, UUID) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION
    complete_ingestion_dispatch_attempt(
        UUID, TEXT, UUID, TEXT, TEXT, TEXT, TIMESTAMPTZ, UUID, TEXT
    ) FROM PUBLIC;

COMMENT ON FUNCTION claim_ingestion_outbox_publication(TEXT, UUID) IS
    'SECURITY INVOKER; deployment grants must be constrained to the publisher role.';
COMMENT ON FUNCTION record_ingestion_outbox_publication(UUID, TEXT, UUID) IS
    'SECURITY INVOKER; call only after broker acceptance; deployment grants must be constrained.';
COMMENT ON FUNCTION claim_ingestion_dispatch_attempt(UUID, SMALLINT, TEXT, UUID) IS
    'SECURITY INVOKER; only started permits a provider call and only after transaction commit.';
COMMENT ON FUNCTION read_ingestion_dispatch_attempt_time(UUID, TEXT, UUID) IS
    'SECURITY INVOKER; deployment grants must be constrained to the worker role.';
COMMENT ON FUNCTION complete_ingestion_dispatch_attempt(
    UUID, TEXT, UUID, TEXT, TEXT, TEXT, TIMESTAMPTZ, UUID, TEXT
) IS
    'SECURITY INVOKER; deployment grants must be constrained to the worker role.';
