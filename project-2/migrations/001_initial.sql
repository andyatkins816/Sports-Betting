-- SAM Analytics initial PostgreSQL schema.
-- Apply through a versioned migration runner, never with db.create_all().
-- All source observations and decisions are append-only audit records.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE sports_event (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider TEXT NOT NULL,
    provider_event_id TEXT NOT NULL,
    sport TEXT NOT NULL,
    league TEXT NOT NULL,
    starts_at TIMESTAMPTZ NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (provider, provider_event_id)
);

CREATE TABLE odds_snapshot (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES sports_event(id),
    provider TEXT NOT NULL,
    provider_quote_id TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    line NUMERIC(10, 3),
    american_odds INTEGER NOT NULL CHECK (american_odds <> 0),
    decimal_odds NUMERIC(12, 6) NOT NULL CHECK (decimal_odds > 1),
    captured_at TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_payload_sha256 CHAR(64) NOT NULL,
    idempotency_key CHAR(64) NOT NULL UNIQUE,
    UNIQUE (provider, provider_quote_id, captured_at)
);
CREATE INDEX odds_snapshot_event_market_time_idx
    ON odds_snapshot (event_id, market, selection, captured_at DESC);

CREATE TABLE model_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version TEXT NOT NULL UNIQUE,
    sport TEXT NOT NULL,
    target_definition TEXT NOT NULL,
    feature_contract_sha256 CHAR(64) NOT NULL,
    artifact_uri TEXT NOT NULL,
    training_data_cutoff TIMESTAMPTZ NOT NULL,
    validation_report JSONB NOT NULL,
    approval_status TEXT NOT NULL CHECK (approval_status IN ('candidate', 'approved', 'retired')),
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((approval_status <> 'approved') OR (approved_by IS NOT NULL AND approved_at IS NOT NULL))
);

CREATE TABLE prediction (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES sports_event(id),
    model_id UUID NOT NULL REFERENCES model_registry(id),
    as_of TIMESTAMPTZ NOT NULL,
    features_available_at TIMESTAMPTZ NOT NULL,
    home_win_probability NUMERIC(7, 6) NOT NULL CHECK (home_win_probability BETWEEN 0 AND 1),
    feature_values_uri TEXT NOT NULL,
    feature_values_sha256 CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (features_available_at <= as_of),
    UNIQUE (event_id, model_id, as_of)
);
CREATE INDEX prediction_event_asof_idx ON prediction (event_id, as_of DESC);

CREATE TABLE analytic_decision (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID NOT NULL REFERENCES prediction(id),
    odds_snapshot_id UUID NOT NULL REFERENCES odds_snapshot(id),
    as_of TIMESTAMPTZ NOT NULL,
    side TEXT NOT NULL,
    expected_roi NUMERIC(10, 6) NOT NULL,
    recommended_stake NUMERIC(14, 2) NOT NULL CHECK (recommended_stake >= 0),
    policy_version TEXT NOT NULL,
    rejection_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (prediction_id, odds_snapshot_id, policy_version)
);

CREATE TABLE event_result (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES sports_event(id),
    provider TEXT NOT NULL,
    provider_result_id TEXT NOT NULL,
    settled_at TIMESTAMPTZ NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    source_payload_sha256 CHAR(64) NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (provider, provider_result_id),
    UNIQUE (event_id, provider)
);

CREATE TABLE data_quality_incident (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    category TEXT NOT NULL,
    provider TEXT,
    event_id UUID REFERENCES sports_event(id),
    details JSONB NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ
);
CREATE INDEX data_quality_open_idx ON data_quality_incident (severity, detected_at DESC)
    WHERE resolved_at IS NULL;

CREATE OR REPLACE FUNCTION forbid_audit_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit tables are append-only; create a correcting record instead';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER odds_snapshot_append_only BEFORE UPDATE OR DELETE ON odds_snapshot
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER prediction_append_only BEFORE UPDATE OR DELETE ON prediction
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER analytic_decision_append_only BEFORE UPDATE OR DELETE ON analytic_decision
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
CREATE TRIGGER event_result_append_only BEFORE UPDATE OR DELETE ON event_result
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();
