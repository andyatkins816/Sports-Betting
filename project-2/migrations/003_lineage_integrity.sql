-- Cross-table integrity guards for the point-in-time evidence ledger.
--
-- PostgreSQL CHECK constraints cannot see another table. These triggers keep
-- the immutable links in 001/002 honest at write time, rather than relying on
-- a worker to remember each relationship.

ALTER TABLE model_registry
    ADD COLUMN artifact_sha256 CHAR(64)
    CHECK (artifact_sha256 IS NULL OR artifact_sha256 ~ '^[0-9a-f]{64}$');

CREATE TABLE odds_snapshot_provenance (
    odds_snapshot_id UUID NOT NULL REFERENCES odds_snapshot(id),
    provenance_id UUID NOT NULL REFERENCES raw_data_provenance(id),
    PRIMARY KEY (odds_snapshot_id, provenance_id)
);

CREATE TABLE event_result_provenance (
    event_result_id UUID NOT NULL REFERENCES event_result(id),
    provenance_id UUID NOT NULL REFERENCES raw_data_provenance(id),
    PRIMARY KEY (event_result_id, provenance_id)
);

CREATE TABLE prediction_feature_vector (
    prediction_id UUID PRIMARY KEY REFERENCES prediction(id),
    vector_id UUID NOT NULL REFERENCES point_in_time_feature_vector(id)
);

CREATE OR REPLACE FUNCTION enforce_vector_event_start() RETURNS trigger AS $$
DECLARE
    canonical_start TIMESTAMPTZ;
BEGIN
    SELECT starts_at INTO canonical_start
    FROM sports_event
    WHERE id = NEW.event_id;

    IF canonical_start IS NULL OR canonical_start <> NEW.event_starts_at THEN
        RAISE EXCEPTION 'feature vector event_starts_at must equal the linked sports_event start';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION enforce_vector_provenance_time() RETURNS trigger AS $$
DECLARE
    vector_as_of TIMESTAMPTZ;
    provenance_received_at TIMESTAMPTZ;
BEGIN
    SELECT as_of INTO vector_as_of
    FROM point_in_time_feature_vector
    WHERE id = NEW.vector_id;

    SELECT received_at INTO provenance_received_at
    FROM raw_data_provenance
    WHERE id = NEW.provenance_id;

    IF vector_as_of IS NULL OR provenance_received_at IS NULL
       OR provenance_received_at > vector_as_of THEN
        RAISE EXCEPTION 'feature-vector provenance must be locally received no later than vector as_of';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION enforce_training_manifest_row_lineage() RETURNS trigger AS $$
DECLARE
    vector_event_id UUID;
    vector_as_of TIMESTAMPTZ;
    vector_contract_id UUID;
    result_event_id UUID;
    result_received_at TIMESTAMPTZ;
    manifest_cutoff TIMESTAMPTZ;
    manifest_contract_id UUID;
BEGIN
    SELECT event_id, as_of, feature_contract_id
      INTO vector_event_id, vector_as_of, vector_contract_id
    FROM point_in_time_feature_vector
    WHERE id = NEW.vector_id;

    SELECT event_id, received_at
      INTO result_event_id, result_received_at
    FROM event_result
    WHERE id = NEW.result_id;

    SELECT training_cutoff, feature_contract_id
      INTO manifest_cutoff, manifest_contract_id
    FROM training_dataset_manifest
    WHERE id = NEW.manifest_id;

    IF vector_event_id IS NULL OR result_event_id IS NULL OR manifest_cutoff IS NULL THEN
        RAISE EXCEPTION 'training manifest row references unavailable evidence';
    END IF;
    IF vector_event_id <> result_event_id THEN
        RAISE EXCEPTION 'training manifest vector and result must belong to the same event';
    END IF;
    IF vector_contract_id <> manifest_contract_id THEN
        RAISE EXCEPTION 'training manifest vector must use the manifest feature contract';
    END IF;
    IF vector_as_of >= manifest_cutoff OR result_received_at > manifest_cutoff THEN
        RAISE EXCEPTION 'training manifest cannot include vector or settled result unavailable at cutoff';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION enforce_prediction_vector_lineage() RETURNS trigger AS $$
DECLARE
    prediction_event_id UUID;
    prediction_as_of TIMESTAMPTZ;
    vector_event_id UUID;
    vector_as_of TIMESTAMPTZ;
BEGIN
    SELECT event_id, as_of INTO prediction_event_id, prediction_as_of
    FROM prediction
    WHERE id = NEW.prediction_id;

    SELECT event_id, as_of INTO vector_event_id, vector_as_of
    FROM point_in_time_feature_vector
    WHERE id = NEW.vector_id;

    IF prediction_event_id IS NULL OR vector_event_id IS NULL
       OR prediction_event_id <> vector_event_id
       OR prediction_as_of <> vector_as_of THEN
        RAISE EXCEPTION 'prediction must link to a point-in-time vector for the same event and as_of';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION enforce_model_governance_lineage() RETURNS trigger AS $$
DECLARE
    report_model_id UUID;
    artifact_digest CHAR(64);
BEGIN
    IF NEW.evaluation_report_id IS NOT NULL THEN
        SELECT registry.id INTO report_model_id
        FROM model_evaluation_report report
        JOIN model_registry registry ON registry.version = report.model_version
        WHERE report.id = NEW.evaluation_report_id;

        IF report_model_id IS NULL OR report_model_id <> NEW.model_id THEN
            RAISE EXCEPTION 'model governance decision must reference an evaluation report for the same registered model';
        END IF;
    END IF;

    IF NEW.decision = 'approved' THEN
        SELECT artifact_sha256 INTO artifact_digest
        FROM model_registry
        WHERE id = NEW.model_id;
        IF artifact_digest IS NULL THEN
            RAISE EXCEPTION 'a model cannot be approved without a verified artifact SHA-256 digest';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER point_in_time_feature_vector_event_start
    BEFORE INSERT ON point_in_time_feature_vector
    FOR EACH ROW EXECUTE FUNCTION enforce_vector_event_start();
CREATE TRIGGER feature_vector_provenance_received_before_as_of
    BEFORE INSERT ON feature_vector_provenance
    FOR EACH ROW EXECUTE FUNCTION enforce_vector_provenance_time();
CREATE TRIGGER training_manifest_row_lineage
    BEFORE INSERT ON training_manifest_row
    FOR EACH ROW EXECUTE FUNCTION enforce_training_manifest_row_lineage();
CREATE TRIGGER prediction_feature_vector_lineage
    BEFORE INSERT ON prediction_feature_vector
    FOR EACH ROW EXECUTE FUNCTION enforce_prediction_vector_lineage();
CREATE TRIGGER model_governance_decision_lineage
    BEFORE INSERT ON model_governance_decision
    FOR EACH ROW EXECUTE FUNCTION enforce_model_governance_lineage();

CREATE TRIGGER odds_snapshot_provenance_append_only
    BEFORE UPDATE OR DELETE ON odds_snapshot_provenance
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER event_result_provenance_append_only
    BEFORE UPDATE OR DELETE ON event_result_provenance
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER prediction_feature_vector_append_only
    BEFORE UPDATE OR DELETE ON prediction_feature_vector
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
