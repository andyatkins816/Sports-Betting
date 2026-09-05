-- Immutable provider-receipt ledger for future odds ingestion.
--
-- A receipt represents the exact response SAM retained from a licensed
-- provider.  It deliberately records only hashes of request shape and never a
-- provider URL with credentials, request headers, or a signed object URL.
-- New columns remain nullable on historic odds rows so this migration is safe
-- for an already-populated audit ledger.  New ingestion code must provide a
-- primary provenance record backed by a receipt before it appends a quote.

CREATE TABLE provider_payload_receipt (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider TEXT NOT NULL CHECK (length(btrim(provider)) > 0),
    source_type TEXT NOT NULL CHECK (length(btrim(source_type)) > 0),
    -- Hash only the canonical, credential-free request shape (sport, markets,
    -- region, format, and other permitted parameters), never the API key.
    request_fingerprint_sha256 CHAR(64) NOT NULL
        CHECK (request_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
    payload_sha256 CHAR(64) NOT NULL CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
    payload_uri TEXT NOT NULL CHECK (
        position('?' in payload_uri) = 0
        AND position('#' in payload_uri) = 0
        AND position(E'\n' in payload_uri) = 0
        AND position(E'\r' in payload_uri) = 0
    ),
    captured_at TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    provider_response_status INTEGER NOT NULL
        CHECK (provider_response_status BETWEEN 100 AND 599),
    payload_bytes BIGINT NOT NULL CHECK (payload_bytes >= 0),
    provider_quota_remaining INTEGER CHECK (provider_quota_remaining >= 0),
    provider_quota_used INTEGER CHECK (provider_quota_used >= 0),
    provider_quota_last INTEGER CHECK (provider_quota_last >= 0),
    schema_version TEXT NOT NULL CHECK (length(btrim(schema_version)) > 0),
    license_scope TEXT CHECK (license_scope IS NULL OR length(btrim(license_scope)) > 0),
    -- The accepted provider-contract version is part of the evidence identity;
    -- a later terms change must produce distinct, reviewable evidence.
    license_version TEXT NOT NULL CHECK (length(btrim(license_version)) > 0),
    receipt_sha256 CHAR(64) NOT NULL UNIQUE CHECK (receipt_sha256 ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (captured_at <= received_at + interval '5 minutes')
);
CREATE INDEX provider_payload_receipt_provider_received_idx
    ON provider_payload_receipt (provider, source_type, received_at DESC);
CREATE INDEX provider_payload_receipt_payload_idx
    ON provider_payload_receipt (provider, payload_sha256);

-- Keep source-level provenance tied to the durable response from which it was
-- derived.  The nullable column preserves compatibility with pre-004 records.
ALTER TABLE raw_data_provenance
    ADD COLUMN provider_payload_receipt_id UUID
        REFERENCES provider_payload_receipt(id);
-- `NOT VALID` preserves historic rows without a receipt link, while PostgreSQL
-- still enforces this requirement for every new or changed provenance fact.
ALTER TABLE raw_data_provenance
    ADD CONSTRAINT raw_data_provenance_receipt_required_for_new_records
        CHECK (provider_payload_receipt_id IS NOT NULL) NOT VALID;
CREATE INDEX raw_data_provenance_receipt_idx
    ON raw_data_provenance (provider_payload_receipt_id);

-- A bookmaker is an essential market identity.  It is nullable solely for
-- evidence written before this migration; a future repository must require it
-- when storing provider odds.
ALTER TABLE odds_snapshot
    ADD COLUMN bookmaker TEXT
        CHECK (bookmaker IS NULL OR length(btrim(bookmaker)) > 0),
    ADD COLUMN primary_provenance_id UUID
        REFERENCES raw_data_provenance(id);
-- 001 treated a provider quote ID plus timestamp as immutable.  Some feeds
-- legitimately revise a price or line while retaining both values, so that
-- legacy uniqueness rule would silently reject a correction even though the
-- content-sensitive idempotency key is different.  The idempotency key remains
-- the exact immutable identity; retain a non-unique lookup index below.
ALTER TABLE odds_snapshot
    DROP CONSTRAINT IF EXISTS odds_snapshot_provider_provider_quote_id_captured_at_key;
-- Historic snapshots remain queryable.  New snapshots must identify the
-- bookmaker and their receipt-backed primary provenance rather than silently
-- inheriting incomplete legacy evidence.
ALTER TABLE odds_snapshot
    ADD CONSTRAINT odds_snapshot_provider_evidence_required_for_new_records
        CHECK (bookmaker IS NOT NULL AND primary_provenance_id IS NOT NULL) NOT VALID;
CREATE INDEX odds_snapshot_bookmaker_time_idx
    ON odds_snapshot (event_id, bookmaker, market, selection, captured_at DESC);
CREATE INDEX odds_snapshot_provider_quote_time_idx
    ON odds_snapshot (provider, provider_quote_id, captured_at DESC);
CREATE INDEX odds_snapshot_primary_provenance_idx
    ON odds_snapshot (primary_provenance_id);

-- A raw provenance record that declares a receipt must describe exactly the
-- same retained object.  This prevents an implementation from using a valid
-- receipt ID as a label for a different payload.
CREATE OR REPLACE FUNCTION enforce_raw_provenance_receipt_integrity() RETURNS trigger AS $$
DECLARE
    receipt_provider TEXT;
    receipt_source_type TEXT;
    receipt_payload_sha256 CHAR(64);
    receipt_payload_uri TEXT;
    receipt_captured_at TIMESTAMPTZ;
    receipt_received_at TIMESTAMPTZ;
    receipt_schema_version TEXT;
    receipt_license_scope TEXT;
    linked_receipt_sha256 CHAR(64);
BEGIN
    IF NEW.provider_payload_receipt_id IS NULL THEN
        RETURN NEW;
    END IF;

    SELECT receipt.provider, receipt.source_type, receipt.payload_sha256,
           receipt.payload_uri, receipt.captured_at, receipt.received_at,
           receipt.schema_version, receipt.license_scope, receipt.receipt_sha256
      INTO receipt_provider, receipt_source_type, receipt_payload_sha256,
           receipt_payload_uri, receipt_captured_at, receipt_received_at,
           receipt_schema_version, receipt_license_scope, linked_receipt_sha256
    FROM provider_payload_receipt AS receipt
    WHERE receipt.id = NEW.provider_payload_receipt_id;

    IF receipt_provider IS NULL THEN
        RAISE EXCEPTION 'raw provenance references an unavailable provider payload receipt';
    END IF;
    IF receipt_provider <> NEW.provider
       OR receipt_source_type <> NEW.source_type
       OR receipt_payload_sha256 <> NEW.payload_sha256
       OR receipt_payload_uri <> NEW.payload_uri
       OR receipt_captured_at <> NEW.captured_at
       OR receipt_received_at <> NEW.received_at
       OR receipt_schema_version <> NEW.schema_version
       OR receipt_license_scope IS DISTINCT FROM NEW.license_scope
       OR NEW.provider_record_id <> 'receipt:' || linked_receipt_sha256 THEN
        RAISE EXCEPTION 'raw provenance must match the linked provider payload receipt';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- An odds snapshot's direct primary provenance is its auditable path back to
-- a durable receipt.  `odds_snapshot_provenance` can still hold additional
-- sources, but none can replace this source-payload identity.
CREATE OR REPLACE FUNCTION enforce_odds_snapshot_primary_provenance() RETURNS trigger AS $$
DECLARE
    provenance_provider TEXT;
    provenance_payload_sha256 CHAR(64);
    provenance_receipt_id UUID;
    provenance_received_at TIMESTAMPTZ;
BEGIN
    IF NEW.primary_provenance_id IS NULL THEN
        RETURN NEW;
    END IF;

    SELECT provider, payload_sha256, provider_payload_receipt_id, received_at
      INTO provenance_provider, provenance_payload_sha256, provenance_receipt_id,
           provenance_received_at
    FROM raw_data_provenance
    WHERE id = NEW.primary_provenance_id;

    IF provenance_provider IS NULL THEN
        RAISE EXCEPTION 'odds snapshot references unavailable primary provenance';
    END IF;
    IF provenance_receipt_id IS NULL
       OR provenance_provider <> NEW.provider
       OR provenance_payload_sha256 <> NEW.source_payload_sha256
       OR provenance_received_at <> NEW.received_at
       OR NEW.captured_at > NEW.received_at + interval '5 minutes' THEN
        RAISE EXCEPTION 'odds snapshot primary provenance must match its provider, payload digest, and receipt';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER raw_data_provenance_receipt_integrity
    BEFORE INSERT ON raw_data_provenance
    FOR EACH ROW EXECUTE FUNCTION enforce_raw_provenance_receipt_integrity();
CREATE TRIGGER odds_snapshot_primary_provenance_integrity
    BEFORE INSERT ON odds_snapshot
    FOR EACH ROW EXECUTE FUNCTION enforce_odds_snapshot_primary_provenance();
CREATE TRIGGER sports_event_append_only
    BEFORE UPDATE OR DELETE ON sports_event
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER provider_payload_receipt_append_only
    BEFORE UPDATE OR DELETE ON provider_payload_receipt
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
