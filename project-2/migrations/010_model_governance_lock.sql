-- Serialize model inference with append-only governance decisions.
--
-- Inference takes this transaction-scoped advisory lock before reading the
-- latest decision and retains it through prediction persistence.  Governance
-- decision inserts and registry approval changes take the same per-model lock
-- here, so a suspension or retirement cannot cross that read/commit boundary.
-- Inference needs no model_registry UPDATE privilege.  Hash collisions only
-- serialize unrelated models more conservatively.

SET LOCAL search_path = public, pg_temp;

CREATE OR REPLACE FUNCTION lock_model_governance(
    governed_model_id UUID
) RETURNS void AS $$
BEGIN
    IF governed_model_id IS NULL THEN
        RAISE EXCEPTION 'model governance lock requires a model id';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'sam-model-governance:' || governed_model_id::text,
            0
        )
    );
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE OR REPLACE FUNCTION lock_model_governance_decision_insert()
RETURNS trigger AS $$
BEGIN
    PERFORM public.lock_model_governance(NEW.model_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

-- PostgreSQL fires same-kind triggers alphabetically.  "advisory_lock" sorts
-- before the existing "lineage" trigger, establishing the serialization
-- boundary before that trigger performs its governance validation reads.
CREATE OR REPLACE TRIGGER model_governance_decision_advisory_lock
    BEFORE INSERT ON model_governance_decision
    FOR EACH ROW EXECUTE FUNCTION lock_model_governance_decision_insert();

-- The registry retains the original mutable approval fields for compatibility.
-- Serialize real changes to those fields with inference as well; listing a
-- governance column in an UPDATE without changing its value does not lock.
CREATE OR REPLACE FUNCTION lock_model_registry_governance_update()
RETURNS trigger AS $$
BEGIN
    PERFORM public.lock_model_governance(NEW.id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE OR REPLACE TRIGGER model_registry_governance_advisory_lock
    BEFORE UPDATE OF approval_status, approved_by, approved_at ON model_registry
    FOR EACH ROW
    WHEN (
        OLD.approval_status IS DISTINCT FROM NEW.approval_status
        OR OLD.approved_by IS DISTINCT FROM NEW.approved_by
        OR OLD.approved_at IS DISTINCT FROM NEW.approved_at
    )
    EXECUTE FUNCTION lock_model_registry_governance_update();
