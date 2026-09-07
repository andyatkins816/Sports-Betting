-- Preserve actual receipt time while recording when a historical source proves
-- an observation was available. Live ingestion writes the same timestamp to
-- both fields; historical imports retain their later local receipt separately.

SET LOCAL search_path = public, pg_temp;

ALTER TABLE provider_payload_receipt
    ADD COLUMN source_available_at TIMESTAMPTZ;

ALTER TABLE provider_payload_receipt DISABLE TRIGGER provider_payload_receipt_append_only;
UPDATE provider_payload_receipt SET source_available_at = received_at;
ALTER TABLE provider_payload_receipt ENABLE TRIGGER provider_payload_receipt_append_only;

ALTER TABLE provider_payload_receipt
    ALTER COLUMN source_available_at SET NOT NULL,
    ADD CONSTRAINT provider_payload_receipt_source_availability_valid
        CHECK (
            captured_at <= source_available_at + interval '5 minutes'
            AND source_available_at <= received_at + interval '5 minutes'
        );

ALTER TABLE odds_snapshot
    ADD COLUMN source_available_at TIMESTAMPTZ;

ALTER TABLE odds_snapshot DISABLE TRIGGER odds_snapshot_append_only;
UPDATE odds_snapshot SET source_available_at = received_at;
ALTER TABLE odds_snapshot ENABLE TRIGGER odds_snapshot_append_only;

ALTER TABLE odds_snapshot
    ALTER COLUMN source_available_at SET NOT NULL,
    ADD CONSTRAINT odds_snapshot_source_availability_valid
        CHECK (
            captured_at <= source_available_at + interval '5 minutes'
            AND source_available_at <= received_at + interval '5 minutes'
        );

ALTER TABLE event_result
    ADD COLUMN source_available_at TIMESTAMPTZ;

ALTER TABLE event_result DISABLE TRIGGER event_result_append_only;
UPDATE event_result SET source_available_at = received_at;
ALTER TABLE event_result ENABLE TRIGGER event_result_append_only;

ALTER TABLE event_result
    ALTER COLUMN source_available_at SET NOT NULL,
    ADD CONSTRAINT event_result_source_availability_valid
        CHECK (
            settled_at <= source_available_at + interval '5 minutes'
            AND source_available_at <= received_at + interval '5 minutes'
        );

-- The direct receipt is the immutable authority for a snapshot's historical
-- availability.  Keep that value identical across the receipt and derived row.
CREATE OR REPLACE FUNCTION enforce_odds_snapshot_primary_provenance() RETURNS trigger AS $$
DECLARE
    provenance_provider TEXT;
    provenance_payload_sha256 CHAR(64);
    provenance_receipt_id UUID;
    provenance_received_at TIMESTAMPTZ;
    receipt_source_available_at TIMESTAMPTZ;
BEGIN
    IF NEW.primary_provenance_id IS NULL THEN
        RETURN NEW;
    END IF;

    SELECT provenance.provider, provenance.payload_sha256,
           provenance.provider_payload_receipt_id, provenance.received_at,
           receipt.source_available_at
      INTO provenance_provider, provenance_payload_sha256,
           provenance_receipt_id, provenance_received_at,
           receipt_source_available_at
    FROM raw_data_provenance AS provenance
    LEFT JOIN provider_payload_receipt AS receipt
      ON receipt.id = provenance.provider_payload_receipt_id
    WHERE provenance.id = NEW.primary_provenance_id;

    IF provenance_provider IS NULL THEN
        RAISE EXCEPTION 'odds snapshot references unavailable primary provenance';
    END IF;
    IF provenance_receipt_id IS NULL
       OR provenance_provider <> NEW.provider
       OR provenance_payload_sha256 <> NEW.source_payload_sha256
       OR provenance_received_at <> NEW.received_at
       OR receipt_source_available_at <> NEW.source_available_at
       OR NEW.captured_at > NEW.source_available_at + interval '5 minutes'
       OR NEW.source_available_at > NEW.received_at + interval '5 minutes' THEN
        RAISE EXCEPTION 'odds snapshot primary provenance must match its provider, payload digest, receipt, and source availability';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
