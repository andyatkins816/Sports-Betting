-- Allow immutable score corrections while keeping each provider result version unique.
--
-- A provider can revise a completed score. The original event-level uniqueness
-- constraint prevented the append-only ledger from retaining that correction.

SET LOCAL search_path = public, pg_temp;

ALTER TABLE event_result
    DROP CONSTRAINT IF EXISTS event_result_event_id_provider_key;

CREATE INDEX event_result_latest_idx
    ON event_result (event_id, provider, settled_at DESC, received_at DESC);
