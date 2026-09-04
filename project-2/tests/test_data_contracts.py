import hashlib
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from sam_analytics.data_contracts import (
    DataContractViolation,
    FeatureContract,
    FeatureDefinition,
    FeatureObservation,
    LabeledTrainingExample,
    PointInTimeFeatureVector,
    RawDataProvenance,
    TrainingDatasetManifest,
    build_training_dataset_manifest,
    validate_point_in_time_feature_vector,
    validate_raw_data_provenance,
    validate_training_dataset_manifest,
)


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.base = datetime(2026, 1, 15, 15, tzinfo=timezone.utc)
        self.contract = FeatureContract(
            name="nba_pregame_moneyline",
            version="2026.01",
            target_definition="home_team_wins_including_overtime",
            features=(
                FeatureDefinition(
                    name="home_rating",
                    transformation_version="rating-v2",
                    minimum=0,
                    maximum=3000,
                    max_age_seconds=86_400,
                ),
                FeatureDefinition(
                    name="injury_adjustment",
                    transformation_version="injury-v1",
                    allow_missing=True,
                ),
            ),
        )

    def _source(self, *, label="odds", received_at=None):
        received_at = received_at or self.base - timedelta(minutes=2)
        return RawDataProvenance(
            provider="licensed_provider",
            provider_record_id=f"{label}-record-1",
            source_type=label,
            payload_sha256=hashlib.sha256(label.encode("utf-8")).hexdigest(),
            payload_uri=f"s3://sam-raw/{label}/2026-01-15/record-1.json",
            captured_at=received_at - timedelta(seconds=15),
            received_at=received_at,
            schema_version="v1",
            license_scope="internal_analytics_only",
        )

    def _vector(self, *, available_at=None):
        source = self._source()
        available_at = available_at or self.base - timedelta(seconds=20)
        observations = (
            FeatureObservation(
                name="home_rating",
                value=1642.5,
                source=source,
                computed_at=self.base - timedelta(seconds=30),
                available_at=available_at,
                transformation_version="rating-v2",
            ),
            FeatureObservation(
                name="injury_adjustment",
                value=None,
                source=source,
                computed_at=self.base - timedelta(seconds=30),
                available_at=available_at,
                transformation_version="injury-v1",
                missing_reason="licensed injury feed has not published an update",
            ),
        )
        return PointInTimeFeatureVector(
            event_id="provider-event-42",
            event_starts_at=self.base + timedelta(hours=2),
            as_of=self.base,
            contract=self.contract,
            observations=observations,
        )

    def test_raw_provenance_is_content_addressed_and_immutable(self):
        source = self._source()
        report = validate_raw_data_provenance(source, now=self.base)
        self.assertTrue(report.is_valid)
        self.assertEqual(len(source.payload_sha256), 64)
        self.assertEqual(len(source.digest), 64)
        with self.assertRaises(FrozenInstanceError):
            source.provider = "different-provider"

    def test_raw_provenance_rejects_bad_digest_and_future_receipt(self):
        source = RawDataProvenance(
            provider="licensed_provider",
            provider_record_id="record-1",
            source_type="odds",
            payload_sha256="not-a-digest",
            payload_uri="s3://sam-raw/odds/record-1.json",
            captured_at=self.base + timedelta(minutes=10),
            received_at=self.base + timedelta(minutes=10),
            schema_version="v1",
        )
        codes = {issue.code for issue in validate_raw_data_provenance(source, now=self.base).errors}
        self.assertIn("INVALID_PAYLOAD_SHA256", codes)
        self.assertIn("FUTURE_PROVENANCE_TIMESTAMP", codes)

    def test_feature_vector_rejects_information_available_after_decision(self):
        vector = self._vector(available_at=self.base + timedelta(seconds=1))
        report = validate_point_in_time_feature_vector(vector, now=self.base + timedelta(minutes=1))
        codes = {issue.code for issue in report.errors}
        self.assertIn("FEATURE_AVAILABLE_AFTER_AS_OF", codes)
        with self.assertRaisesRegex(DataContractViolation, "FEATURE_AVAILABLE_AFTER_AS_OF"):
            report.require_valid("point-in-time feature vector")

    def test_feature_vector_rejects_post_start_and_stale_features(self):
        vector = self._vector(available_at=self.base - timedelta(days=2))
        stale_codes = {issue.code for issue in validate_point_in_time_feature_vector(vector).errors}
        self.assertIn("STALE_FEATURE", stale_codes)
        post_start = PointInTimeFeatureVector(
            event_id=vector.event_id,
            event_starts_at=vector.event_starts_at,
            as_of=vector.event_starts_at,
            contract=vector.contract,
            observations=vector.observations,
        )
        post_start_codes = {issue.code for issue in validate_point_in_time_feature_vector(post_start).errors}
        self.assertIn("POST_START_FEATURE_VECTOR", post_start_codes)

    def test_manifest_binds_contract_rows_results_and_raw_sources(self):
        vector = self._vector()
        settled_at = self.base + timedelta(hours=3)
        result = self._source(label="official_result", received_at=settled_at + timedelta(minutes=1))
        example = LabeledTrainingExample(
            row_id="nba-2026-01-15-42",
            vector=vector,
            target=1,
            settled_at=settled_at,
            result_source=result,
        )
        manifest = build_training_dataset_manifest(
            dataset_name="nba-pregame-training",
            dataset_version="2026.01.15",
            contract=self.contract,
            examples=[example],
            training_cutoff=settled_at + timedelta(minutes=2),
            created_at=settled_at + timedelta(minutes=3),
            code_revision="abc123",
            split_strategy="walk_forward_expanding_v1",
        )
        self.assertEqual(manifest.row_count, 1)
        self.assertEqual(manifest.feature_contract_sha256, self.contract.digest)
        self.assertEqual(len(manifest.source_digests), 2)
        self.assertTrue(manifest.verify_examples([example]))
        changed_target = LabeledTrainingExample(
            row_id=example.row_id,
            vector=example.vector,
            target=0,
            settled_at=example.settled_at,
            result_source=example.result_source,
        )
        self.assertFalse(manifest.verify_examples([changed_target]))

    def test_manifest_rejects_label_that_arrived_after_cutoff(self):
        vector = self._vector()
        settled_at = self.base + timedelta(hours=3)
        result = self._source(label="official_result", received_at=settled_at + timedelta(minutes=10))
        example = LabeledTrainingExample(
            row_id="nba-2026-01-15-42",
            vector=vector,
            target=1,
            settled_at=settled_at,
            result_source=result,
        )
        with self.assertRaisesRegex(DataContractViolation, "LABEL_AVAILABLE_AFTER_CUTOFF"):
            build_training_dataset_manifest(
                dataset_name="nba-pregame-training",
                dataset_version="2026.01.15",
                contract=self.contract,
                examples=[example],
                training_cutoff=settled_at + timedelta(minutes=2),
                created_at=settled_at + timedelta(minutes=11),
                code_revision="abc123",
                split_strategy="walk_forward_expanding_v1",
            )

    def test_manifest_rejects_nonchronological_or_duplicate_event_rows(self):
        first_vector = self._vector()
        later_vector = PointInTimeFeatureVector(
            event_id="provider-event-42",
            event_starts_at=self.base + timedelta(hours=3),
            as_of=self.base + timedelta(minutes=30),
            contract=self.contract,
            observations=first_vector.observations,
        )
        settled_at = self.base + timedelta(hours=4)
        first = LabeledTrainingExample(
            row_id="later-row",
            vector=later_vector,
            target=1,
            settled_at=settled_at,
            result_source=self._source(label="result-a", received_at=settled_at),
        )
        second = LabeledTrainingExample(
            row_id="earlier-row",
            vector=first_vector,
            target=0,
            settled_at=settled_at,
            result_source=self._source(label="result-b", received_at=settled_at),
        )
        with self.assertRaisesRegex(DataContractViolation, "NONCHRONOLOGICAL_TRAINING_ROWS"):
            build_training_dataset_manifest(
                dataset_name="nba-pregame-training",
                dataset_version="2026.01.15",
                contract=self.contract,
                examples=[first, second],
                training_cutoff=settled_at + timedelta(minutes=1),
                created_at=settled_at + timedelta(minutes=2),
                code_revision="abc123",
                split_strategy="walk_forward_expanding_v1",
            )

    def test_loaded_manifest_rejects_mutable_or_invalid_digest_fields(self):
        manifest = TrainingDatasetManifest(
            dataset_name="nba-pregame-training",
            dataset_version="2026.01.15",
            feature_contract_sha256="not-a-digest",
            target_definition="home_team_wins_including_overtime",
            training_cutoff=self.base,
            created_at=self.base + timedelta(minutes=1),
            code_revision="abc123",
            split_strategy="walk_forward_expanding_v1",
            row_digests=("not-a-digest",),
            source_digests=["not-a-tuple"],
        )
        codes = {issue.code for issue in validate_training_dataset_manifest(manifest).errors}
        self.assertIn("INVALID_FEATURE_CONTRACT_SHA256", codes)
        self.assertIn("INVALID_MANIFEST_DIGEST", codes)
        self.assertIn("MUTABLE_MANIFEST_COLLECTION", codes)


if __name__ == "__main__":
    unittest.main()
