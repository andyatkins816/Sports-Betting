-- Store immutable model artifacts in PostgreSQL and verify their content digest.
--
-- Existing model_registry rows predate durable artifact bytes. The NOT VALID
-- constraints preserve those rows while enforcing the complete, verified
-- artifact contract for every new or changed row.

ALTER TABLE model_registry
    ADD COLUMN artifact_format TEXT,
    ADD COLUMN artifact_bytes BYTEA;

ALTER TABLE model_registry
    ADD CONSTRAINT model_registry_artifact_required_for_new_records
        CHECK (
            artifact_format IS NOT NULL
            AND length(btrim(artifact_format)) > 0
            AND artifact_bytes IS NOT NULL
            AND artifact_sha256 IS NOT NULL
        ) NOT VALID,
    ADD CONSTRAINT model_registry_artifact_sha256_verified_for_new_records
        CHECK (
            artifact_bytes IS NULL
            OR artifact_sha256 IS NULL
            OR artifact_sha256 = encode(public.digest(artifact_bytes, 'sha256'), 'hex')
        ) NOT VALID;

CREATE OR REPLACE FUNCTION forbid_model_artifact_mutation() RETURNS trigger AS $$
BEGIN
    IF OLD.artifact_format IS DISTINCT FROM NEW.artifact_format
       OR OLD.artifact_bytes IS DISTINCT FROM NEW.artifact_bytes
       OR OLD.artifact_sha256 IS DISTINCT FROM NEW.artifact_sha256 THEN
        RAISE EXCEPTION 'registered model artifact fields are immutable';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER model_registry_artifact_immutable
    BEFORE UPDATE OF artifact_format, artifact_bytes, artifact_sha256 ON model_registry
    FOR EACH ROW EXECUTE FUNCTION forbid_model_artifact_mutation();
