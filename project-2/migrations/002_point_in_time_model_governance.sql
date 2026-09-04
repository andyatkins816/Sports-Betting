-- Point-in-time evidence, training-manifest, and model-governance ledger.
--
-- Apply after 001_initial.sql through Alembic or another transactional migration
-- runner.  These records are append-only: a provider correction, revised
-- feature set, retraining run, or release decision creates new evidence rather
-- than rewriting what supported an earlier result.

CREATE TABLE raw_data_provenance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider TEXT NOT NULL,
    provider_record_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    payload_sha256 CHAR(64) NOT NULL CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
    -- URI references are identifiers, never signed/download URLs. Keeping
    -- query strings and fragments out of the ledger avoids persisting a
    -- provider token by accident.
    payload_uri TEXT NOT NULL CHECK (
        position('?' in payload_uri) = 0
        AND position('#' in payload_uri) = 0
        AND position(E'\n' in payload_uri) = 0
        AND position(E'\r' in payload_uri) = 0
    ),
    captured_at TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    schema_version TEXT NOT NULL,
    license_scope TEXT,
    provenance_sha256 CHAR(64) NOT NULL UNIQUE CHECK (provenance_sha256 ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (captured_at <= received_at + interval '5 minutes'),
    UNIQUE (provider, provider_record_id, payload_sha256)
);
CREATE INDEX raw_data_provenance_received_idx
    ON raw_data_provenance (provider, source_type, received_at DESC);

CREATE TABLE feature_contract (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    target_definition TEXT NOT NULL,
    definition JSONB NOT NULL,
    contract_sha256 CHAR(64) NOT NULL UNIQUE CHECK (contract_sha256 ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version)
);

CREATE TABLE point_in_time_feature_vector (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES sports_event(id),
    feature_contract_id UUID NOT NULL REFERENCES feature_contract(id),
    as_of TIMESTAMPTZ NOT NULL,
    event_starts_at TIMESTAMPTZ NOT NULL,
    features_available_at TIMESTAMPTZ NOT NULL,
    values_uri TEXT NOT NULL CHECK (
        position('?' in values_uri) = 0
        AND position('#' in values_uri) = 0
        AND position(E'\n' in values_uri) = 0
        AND position(E'\r' in values_uri) = 0
    ),
    values_sha256 CHAR(64) NOT NULL CHECK (values_sha256 ~ '^[0-9a-f]{64}$'),
    vector_sha256 CHAR(64) NOT NULL UNIQUE CHECK (vector_sha256 ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (as_of < event_starts_at),
    CHECK (features_available_at <= as_of),
    UNIQUE (event_id, feature_contract_id, as_of, vector_sha256)
);
CREATE INDEX point_in_time_feature_vector_lookup_idx
    ON point_in_time_feature_vector (event_id, feature_contract_id, as_of DESC);

CREATE TABLE feature_vector_provenance (
    vector_id UUID NOT NULL REFERENCES point_in_time_feature_vector(id),
    provenance_id UUID NOT NULL REFERENCES raw_data_provenance(id),
    PRIMARY KEY (vector_id, provenance_id)
);

CREATE TABLE training_dataset_manifest (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_contract_id UUID NOT NULL REFERENCES feature_contract(id),
    target_definition TEXT NOT NULL,
    training_cutoff TIMESTAMPTZ NOT NULL,
    split_strategy TEXT NOT NULL,
    code_revision TEXT NOT NULL,
    row_count INTEGER NOT NULL CHECK (row_count > 0),
    manifest_uri TEXT NOT NULL CHECK (
        position('?' in manifest_uri) = 0
        AND position('#' in manifest_uri) = 0
        AND position(E'\n' in manifest_uri) = 0
        AND position(E'\r' in manifest_uri) = 0
    ),
    manifest_sha256 CHAR(64) NOT NULL UNIQUE CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE training_manifest_row (
    manifest_id UUID NOT NULL REFERENCES training_dataset_manifest(id),
    vector_id UUID NOT NULL REFERENCES point_in_time_feature_vector(id),
    result_id UUID NOT NULL REFERENCES event_result(id),
    row_position INTEGER NOT NULL CHECK (row_position >= 0),
    PRIMARY KEY (manifest_id, row_position),
    UNIQUE (manifest_id, vector_id),
    UNIQUE (manifest_id, result_id)
);

CREATE TABLE model_evaluation_report (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version TEXT NOT NULL,
    dataset_manifest_id UUID NOT NULL REFERENCES training_dataset_manifest(id),
    candidate_name TEXT NOT NULL,
    schema_sha256 CHAR(64) NOT NULL CHECK (schema_sha256 ~ '^[0-9a-f]{64}$'),
    data_fingerprint_sha256 CHAR(64) NOT NULL CHECK (data_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
    evaluated_rows INTEGER NOT NULL CHECK (evaluated_rows > 0),
    fold_count INTEGER NOT NULL CHECK (fold_count > 0),
    raw_metrics JSONB NOT NULL,
    calibrated_metrics JSONB NOT NULL,
    calibration_artifact_uri TEXT,
    calibration_artifact_sha256 CHAR(64) CHECK (calibration_artifact_sha256 ~ '^[0-9a-f]{64}$'),
    report_uri TEXT NOT NULL CHECK (
        position('?' in report_uri) = 0
        AND position('#' in report_uri) = 0
        AND position(E'\n' in report_uri) = 0
        AND position(E'\r' in report_uri) = 0
    ),
    report_sha256 CHAR(64) NOT NULL UNIQUE CHECK (report_sha256 ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX model_evaluation_report_version_idx
    ON model_evaluation_report (model_version, created_at DESC);

-- Approval and retirement are facts with an actor and timestamp.  Keep them
-- separate from the candidate definition so a historic approval cannot vanish.
CREATE TABLE model_governance_decision (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES model_registry(id),
    evaluation_report_id UUID REFERENCES model_evaluation_report(id),
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected', 'retired', 'suspended')),
    reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    decided_by TEXT NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX model_governance_decision_model_idx
    ON model_governance_decision (model_id, decided_at DESC);

-- A worker writes one sanitized operational fact per state change.  The API
-- should derive its control-plane response from the latest relevant entries;
-- no browser may write these records.
CREATE TABLE operational_signal (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_type TEXT NOT NULL CHECK (signal_type IN ('provider', 'model', 'deployment', 'risk', 'incident')),
    observed_at TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source TEXT NOT NULL,
    provenance_sha256 CHAR(64) CHECK (provenance_sha256 ~ '^[0-9a-f]{64}$'),
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (observed_at <= received_at + interval '5 minutes')
);
CREATE INDEX operational_signal_current_idx
    ON operational_signal (signal_type, observed_at DESC);

CREATE TRIGGER raw_data_provenance_append_only BEFORE UPDATE OR DELETE ON raw_data_provenance
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER feature_contract_append_only BEFORE UPDATE OR DELETE ON feature_contract
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER point_in_time_feature_vector_append_only BEFORE UPDATE OR DELETE ON point_in_time_feature_vector
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER feature_vector_provenance_append_only BEFORE UPDATE OR DELETE ON feature_vector_provenance
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER training_dataset_manifest_append_only BEFORE UPDATE OR DELETE ON training_dataset_manifest
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER training_manifest_row_append_only BEFORE UPDATE OR DELETE ON training_manifest_row
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER model_evaluation_report_append_only BEFORE UPDATE OR DELETE ON model_evaluation_report
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER model_governance_decision_append_only BEFORE UPDATE OR DELETE ON model_governance_decision
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER operational_signal_append_only BEFORE UPDATE OR DELETE ON operational_signal
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
