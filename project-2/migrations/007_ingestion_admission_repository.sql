-- Exact evidence bindings for a future, still-unwired admission repository.
--
-- This migration grants no provider use and creates no runtime.  In particular,
-- provider_use_authorization is deliberately left empty.  A later reviewed
-- operation must append an exact, time-bounded authorization before a new
-- dispatch can be recorded.  Historic 006 records remain readable, but every
-- record inserted after this migration must carry the new exact bindings.

CREATE TABLE provider_use_authorization (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider TEXT NOT NULL
        CHECK (provider ~ '^[a-z][a-z0-9_-]{0,63}$'),
    license_scope TEXT NOT NULL
        CHECK (license_scope ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    license_version TEXT NOT NULL
        CHECK (license_version ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    source_type TEXT NOT NULL
        CHECK (source_type ~ '^[a-z][a-z0-9_]{0,63}$'),
    exposure TEXT NOT NULL
        CHECK (exposure IN ('private_raw', 'derived')),
    -- The caller supplies a SHA-256 over the reviewed, credential-free
    -- authorization manifest.  Contract text, signed URLs, and reviewer
    -- contact data do not belong in this operational ledger.
    authorization_manifest_sha256 CHAR(64) NOT NULL UNIQUE
        CHECK (authorization_manifest_sha256 ~ '^[0-9a-f]{64}$'),
    reviewed_at TIMESTAMPTZ NOT NULL,
    effective_from TIMESTAMPTZ NOT NULL,
    effective_until TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (
        provider,
        license_scope,
        license_version,
        source_type,
        exposure
    ),
    CONSTRAINT provider_use_authorization_finite_times CHECK (
        isfinite(reviewed_at)
        AND isfinite(effective_from)
        AND isfinite(effective_until)
    ),
    CHECK (effective_from < effective_until),
    CHECK (reviewed_at < effective_until),
    CHECK (reviewed_at <= created_at)
);

CREATE TRIGGER provider_use_authorization_created_at
    BEFORE INSERT ON provider_use_authorization
    FOR EACH ROW EXECUTE FUNCTION set_ingestion_control_created_at();

-- Physical nullability preserves already-committed 006 rows.  PostgreSQL
-- enforces NOT VALID CHECK constraints for every new or changed row, so no
-- post-007 dispatch or reservation can omit its evidence while historic rows
-- are neither rewritten nor falsely backfilled.
ALTER TABLE ingestion_dispatch
    ADD COLUMN provider_use_authorization_id UUID
        REFERENCES provider_use_authorization(id);
ALTER TABLE ingestion_dispatch
    ADD CONSTRAINT ingestion_dispatch_authorization_required_for_new_records
        CHECK (provider_use_authorization_id IS NOT NULL) NOT VALID;
CREATE INDEX ingestion_dispatch_authorization_idx
    ON ingestion_dispatch (provider_use_authorization_id);

ALTER TABLE ingestion_quota_reservation
    ADD COLUMN provider_payload_receipt_id UUID
        REFERENCES provider_payload_receipt(id);
ALTER TABLE ingestion_quota_reservation
    ADD CONSTRAINT ingestion_quota_receipt_required_for_new_records
        CHECK (provider_payload_receipt_id IS NOT NULL) NOT VALID;
CREATE INDEX ingestion_quota_reservation_receipt_idx
    ON ingestion_quota_reservation (provider_payload_receipt_id);

-- This partial index supports the lock-scoped latest-quota lookup for one
-- provider and exact license tuple.  Equal observation timestamps choose the
-- lowest remaining quota before the deterministic receipt id.  A quota-bearing
-- receipt remains evidence only; this does not poll or contact a provider.
CREATE INDEX provider_payload_receipt_quota_latest_idx
    ON provider_payload_receipt (
        provider,
        license_scope,
        license_version,
        received_at DESC,
        created_at DESC,
        provider_quota_remaining ASC,
        id DESC
    )
    WHERE provider_quota_remaining IS NOT NULL;

-- Admission is serialized per provider before any decision inputs are read.
-- The repository must call this function at the beginning of its transaction,
-- then read quota, outstanding reservations, provider activity, and existing
-- idempotency keys while retaining the lock through the bundle inserts.  The
-- insert triggers below use the same lock as a defense against bypass paths.
-- A hash collision can only serialize unrelated providers more conservatively.
CREATE OR REPLACE FUNCTION lock_ingestion_provider(
    provider_name TEXT
) RETURNS void AS $$
BEGIN
    IF provider_name IS NULL
       OR provider_name !~ '^[a-z][a-z0-9_-]{0,63}$' THEN
        RAISE EXCEPTION 'ingestion admission provider is invalid';
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended('sam-ingestion-admission:' || provider_name, 0)
    );
END;
$$ LANGUAGE plpgsql;

-- Quota receipts and admissions share the provider lock, preventing a receipt
-- append from crossing a lock-scoped admission read/commit boundary.
CREATE OR REPLACE FUNCTION lock_provider_payload_receipt_admission()
RETURNS trigger AS $$
BEGIN
    PERFORM lock_ingestion_provider(NEW.provider);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER provider_payload_receipt_admission_lock
    BEFORE INSERT ON provider_payload_receipt
    FOR EACH ROW EXECUTE FUNCTION lock_provider_payload_receipt_admission();

-- Worker-owned activity facts participate in the same provider serialization
-- boundary as admission.  This trigger name sorts before the existing
-- ingestion_dispatch_transition_integrity trigger, so it obtains the provider
-- lock before that trigger locks and validates the immutable dispatch row.
CREATE OR REPLACE FUNCTION lock_ingestion_dispatch_transition_provider()
RETURNS trigger AS $$
DECLARE
    dispatch_provider TEXT;
BEGIN
    SELECT provider
      INTO dispatch_provider
      FROM ingestion_dispatch
     WHERE id = NEW.ingestion_dispatch_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ingestion dispatch transition references an unavailable dispatch';
    END IF;

    PERFORM lock_ingestion_provider(dispatch_provider);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ingestion_dispatch_transition_admission_lock
    BEFORE INSERT ON ingestion_dispatch_transition
    FOR EACH ROW EXECUTE FUNCTION lock_ingestion_dispatch_transition_provider();

-- Retry outbox facts use the same provider-first lock order as admission and
-- transitions.  This trigger name sorts before the existing outbox created-at
-- and integrity triggers, so neither can lock the dispatch row first.
CREATE OR REPLACE FUNCTION lock_ingestion_dispatch_outbox_provider()
RETURNS trigger AS $$
DECLARE
    dispatch_provider TEXT;
BEGIN
    SELECT provider
      INTO dispatch_provider
      FROM ingestion_dispatch
     WHERE id = NEW.ingestion_dispatch_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ingestion dispatch outbox references an unavailable dispatch';
    END IF;

    PERFORM lock_ingestion_provider(dispatch_provider);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ingestion_dispatch_outbox_admission_lock
    BEFORE INSERT ON ingestion_dispatch_outbox
    FOR EACH ROW EXECUTE FUNCTION lock_ingestion_dispatch_outbox_provider();

-- A dispatch names the exact reviewed ProviderUse.  It may not silently select
-- a newer license version, reuse a derived-output grant for private evidence,
-- or enter the ledger outside that authorization's reviewed validity window.
CREATE OR REPLACE FUNCTION enforce_ingestion_dispatch_authorization()
RETURNS trigger AS $$
DECLARE
    authorization_provider TEXT;
    authorization_license_scope TEXT;
    authorization_license_version TEXT;
    authorization_source_type TEXT;
    authorization_exposure TEXT;
    authorization_reviewed_at TIMESTAMPTZ;
    authorization_effective_from TIMESTAMPTZ;
    authorization_effective_until TIMESTAMPTZ;
    authorization_checked_at TIMESTAMPTZ;
BEGIN
    PERFORM lock_ingestion_provider(NEW.provider);

    SELECT provider, license_scope, license_version, source_type, exposure,
           reviewed_at, effective_from, effective_until
      INTO authorization_provider, authorization_license_scope,
           authorization_license_version, authorization_source_type,
           authorization_exposure, authorization_reviewed_at,
           authorization_effective_from, authorization_effective_until
      FROM provider_use_authorization
     WHERE id = NEW.provider_use_authorization_id
       FOR KEY SHARE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ingestion dispatch requires an available provider authorization';
    END IF;
    IF authorization_provider IS DISTINCT FROM NEW.provider
       OR authorization_source_type IS DISTINCT FROM NEW.source_type
       OR authorization_exposure IS DISTINCT FROM 'private_raw' THEN
        RAISE EXCEPTION 'ingestion dispatch does not match its private provider authorization';
    END IF;
    authorization_checked_at := clock_timestamp();
    IF authorization_checked_at < greatest(
            authorization_reviewed_at,
            authorization_effective_from
       )
       OR authorization_checked_at >= authorization_effective_until
       OR NEW.admitted_at < greatest(
            authorization_reviewed_at,
            authorization_effective_from
       )
       OR NEW.admitted_at >= authorization_effective_until THEN
        RAISE EXCEPTION 'ingestion dispatch falls outside its provider authorization window';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ingestion_dispatch_authorization_integrity
    BEFORE INSERT ON ingestion_dispatch
    FOR EACH ROW EXECUTE FUNCTION enforce_ingestion_dispatch_authorization();

-- Replace the 006 reservation guard without weakening any of its checks.  Each
-- attempt is now tied to the exact quota-bearing provider response used for
-- admission.  All historic reservations remain conservatively outstanding;
-- neither a terminal state nor a newer receipt silently returns quota.
CREATE OR REPLACE FUNCTION enforce_ingestion_quota_reservation()
RETURNS trigger AS $$
DECLARE
    dispatch_provider TEXT;
    dispatch_admitted_at TIMESTAMPTZ;
    dispatch_estimated_cost INTEGER;
    dispatch_max_attempts SMALLINT;
    dispatch_authorization_id UUID;
    authorization_provider TEXT;
    authorization_license_scope TEXT;
    authorization_license_version TEXT;
    authorization_exposure TEXT;
    authorization_reviewed_at TIMESTAMPTZ;
    authorization_effective_from TIMESTAMPTZ;
    authorization_effective_until TIMESTAMPTZ;
    authorization_checked_at TIMESTAMPTZ;
    receipt_provider TEXT;
    receipt_license_scope TEXT;
    receipt_license_version TEXT;
    receipt_quota_remaining INTEGER;
    receipt_received_at TIMESTAMPTZ;
    receipt_created_at TIMESTAMPTZ;
    latest_quota_receipt_id UUID;
    provider_reserved_credits BIGINT;
BEGIN
    -- The first immutable read obtains the provider name.  The row cannot be
    -- changed, and the subsequent lock is acquired before any decision state
    -- or aggregate is read, preserving one lock order for all callers.
    SELECT provider
      INTO dispatch_provider
      FROM ingestion_dispatch
     WHERE id = NEW.ingestion_dispatch_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ingestion quota reservation references an unavailable dispatch';
    END IF;

    PERFORM lock_ingestion_provider(dispatch_provider);

    SELECT provider, admitted_at, estimated_cost, max_attempts,
           provider_use_authorization_id
      INTO dispatch_provider, dispatch_admitted_at, dispatch_estimated_cost,
           dispatch_max_attempts, dispatch_authorization_id
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

    SELECT provider, license_scope, license_version, exposure, reviewed_at,
           effective_from, effective_until
      INTO authorization_provider, authorization_license_scope,
           authorization_license_version, authorization_exposure,
           authorization_reviewed_at, authorization_effective_from,
           authorization_effective_until
      FROM provider_use_authorization
     WHERE id = dispatch_authorization_id
       FOR KEY SHARE;

    IF NOT FOUND
       OR authorization_provider IS DISTINCT FROM dispatch_provider
       OR authorization_exposure IS DISTINCT FROM 'private_raw' THEN
        RAISE EXCEPTION 'ingestion quota reservation requires its dispatch authorization';
    END IF;
    authorization_checked_at := clock_timestamp();
    IF authorization_checked_at < greatest(
            authorization_reviewed_at,
            authorization_effective_from
       )
       OR authorization_checked_at >= authorization_effective_until
       OR NEW.reserved_at < greatest(
            authorization_reviewed_at,
            authorization_effective_from
       )
       OR NEW.reserved_at >= authorization_effective_until THEN
        RAISE EXCEPTION 'ingestion quota reservation falls outside its provider authorization window';
    END IF;

    SELECT provider, license_scope, license_version,
           provider_quota_remaining, received_at, created_at
      INTO receipt_provider, receipt_license_scope, receipt_license_version,
           receipt_quota_remaining, receipt_received_at, receipt_created_at
      FROM provider_payload_receipt
     WHERE id = NEW.provider_payload_receipt_id
       FOR KEY SHARE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ingestion quota reservation requires an available quota receipt';
    END IF;
    IF receipt_provider IS DISTINCT FROM dispatch_provider THEN
        RAISE EXCEPTION 'ingestion quota receipt does not match the dispatch provider';
    END IF;
    IF receipt_license_scope IS DISTINCT FROM authorization_license_scope
       OR receipt_license_version IS DISTINCT FROM authorization_license_version THEN
        RAISE EXCEPTION 'ingestion quota receipt does not match the dispatch license authorization';
    END IF;
    IF receipt_quota_remaining IS NULL THEN
        RAISE EXCEPTION 'ingestion quota receipt has no trusted remaining quota';
    END IF;
    IF greatest(receipt_received_at, receipt_created_at) > NEW.reserved_at THEN
        RAISE EXCEPTION 'ingestion quota receipt cannot follow its reservation';
    END IF;
    IF receipt_received_at < greatest(
            authorization_reviewed_at,
            authorization_effective_from
       )
       OR receipt_created_at < greatest(
            authorization_reviewed_at,
            authorization_effective_from
       )
       OR receipt_received_at >= authorization_effective_until
       OR receipt_created_at >= authorization_effective_until THEN
        RAISE EXCEPTION 'ingestion quota receipt falls outside its provider authorization window';
    END IF;

    -- Selecting an older or same-time-but-larger quota response would bypass
    -- conservative accounting.  Receipt inserts take this same provider lock,
    -- so the chosen quota-bearing receipt cannot change across this check and
    -- commit.
    SELECT id
      INTO latest_quota_receipt_id
      FROM provider_payload_receipt
     WHERE provider = dispatch_provider
       AND license_scope IS NOT DISTINCT FROM authorization_license_scope
       AND license_version IS NOT DISTINCT FROM authorization_license_version
       AND provider_quota_remaining IS NOT NULL
     ORDER BY received_at DESC, created_at DESC,
              provider_quota_remaining ASC, id DESC
     LIMIT 1
       FOR KEY SHARE;

    IF latest_quota_receipt_id IS DISTINCT FROM NEW.provider_payload_receipt_id THEN
        RAISE EXCEPTION 'ingestion quota reservation requires the latest provider quota receipt';
    END IF;

    SELECT COALESCE(sum(reservation.reserved_credits), 0)
      INTO provider_reserved_credits
      FROM ingestion_quota_reservation AS reservation
      JOIN ingestion_dispatch AS dispatch
        ON dispatch.id = reservation.ingestion_dispatch_id
     WHERE dispatch.provider = dispatch_provider;

    IF provider_reserved_credits + NEW.reserved_credits
       > receipt_quota_remaining THEN
        RAISE EXCEPTION 'ingestion quota reservation exceeds conservatively available quota';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER provider_use_authorization_append_only
    BEFORE UPDATE OR DELETE ON provider_use_authorization
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER provider_use_authorization_append_only_truncate
    BEFORE TRUNCATE ON provider_use_authorization
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_mutation();

-- 004 guarded row mutation but predated the statement-level truncate guards.
CREATE TRIGGER provider_payload_receipt_append_only_truncate
    BEFORE TRUNCATE ON provider_payload_receipt
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_mutation();
