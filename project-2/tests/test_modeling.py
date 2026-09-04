import hashlib
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sam_analytics.modeling import (
    adapt_labeled_training_examples,
    CandidateScore,
    CrossFittedCalibration,
    FeatureSchema,
    FittedProbabilityModel,
    ModelDataError,
    ModelCandidate,
    NumericFeature,
    OptionalModelDependencyError,
    OOFProbability,
    OutcomeTrainingRow,
    PredictionInput,
    ProbabilityMetrics,
    ProbabilityModelEvaluator,
    PromotionPolicy,
    RollingTimeSplitter,
    build_estimator,
    cross_fit_isotonic_calibration,
    default_model_candidates,
    evaluate_promotion,
    validate_prediction_input,
)
from sam_analytics.data_contracts import (
    FeatureContract,
    FeatureDefinition,
    FeatureObservation,
    LabeledTrainingExample,
    PointInTimeFeatureVector,
    RawDataProvenance,
)


class ModelingContractTests(unittest.TestCase):
    def setUp(self):
        self.schema = FeatureSchema(
            schema_id="pregame-v1",
            features=(
                # Explicit bounds catch common provider/unit errors early.
                # The package itself never fills a missing source field.
                NumericFeature("market_probability", minimum=0.0, maximum=1.0),
                NumericFeature("rest_days", minimum=0.0, maximum=14.0),
            ),
        )
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def _rows(self, count=8):
        rows = []
        for index in range(count):
            decision_at = self.start + timedelta(hours=index * 2)
            rows.append(
                OutcomeTrainingRow(
                    event_id="event-%d" % index,
                    event_starts_at=decision_at + timedelta(minutes=30),
                    decision_at=decision_at,
                    features_available_at=decision_at - timedelta(minutes=2),
                    label_available_at=decision_at + timedelta(hours=1),
                    source_snapshot_ids=("odds-snapshot-%d" % index,),
                    label_source_snapshot_id="result-snapshot-%d" % index,
                    features={"market_probability": 0.45 + (index % 3) * 0.05, "rest_days": float(index % 4)},
                    outcome=index % 2,
                )
            )
        return rows

    def test_feature_contract_rejects_future_and_unknown_inputs(self):
        request = PredictionInput(
            event_id="event-1",
            event_starts_at=self.start + timedelta(hours=1),
            decision_at=self.start,
            features_available_at=self.start + timedelta(seconds=1),
            source_snapshot_ids=("snapshot-1",),
            features={"market_probability": 0.5, "rest_days": 2.0},
        )
        with self.assertRaisesRegex(ModelDataError, "after the requested"):
            validate_prediction_input(request, self.schema)

        strict_request = PredictionInput(
            event_id="event-1",
            event_starts_at=self.start + timedelta(hours=1),
            decision_at=self.start,
            features_available_at=self.start,
            source_snapshot_ids=("snapshot-1",),
            features={"market_probability": 0.5, "rest_days": 2.0, "untracked": 1.0},
        )
        with self.assertRaisesRegex(ModelDataError, "unexpected"):
            validate_prediction_input(strict_request, self.schema)

        post_start_request = PredictionInput(
            event_id="event-1",
            event_starts_at=self.start,
            decision_at=self.start,
            features_available_at=self.start - timedelta(minutes=1),
            source_snapshot_ids=("snapshot-1",),
            features={"market_probability": 0.5, "rest_days": 2.0},
        )
        with self.assertRaisesRegex(ModelDataError, "before the scheduled event start"):
            validate_prediction_input(post_start_request, self.schema)

    def test_rolling_split_uses_only_prior_settled_labels(self):
        rows = self._rows()
        splitter = RollingTimeSplitter(n_splits=2, min_train_rows=4, validation_rows=2)
        folds = splitter.split(rows)
        self.assertEqual([(fold.train_indices, fold.validation_indices) for fold in folds], [((0, 1, 2, 3), (4, 5)), ((0, 1, 2, 3, 4, 5), (6, 7))])

        # A delayed result must not be silently trained on for the first fold.
        delayed = self._rows()
        delayed[3] = OutcomeTrainingRow(
            **{**delayed[3].__dict__, "label_available_at": self.start + timedelta(days=2)}
        )
        with self.assertRaisesRegex(ModelDataError, "settled rows"):
            splitter.split(delayed)

    def test_cross_fitted_calibration_does_not_use_its_own_fold_labels(self):
        records = (
            OOFProbability(0, 0, "a", self.start, self.start + timedelta(hours=1), 0.1, 0),
            OOFProbability(1, 0, "b", self.start + timedelta(minutes=1), self.start + timedelta(hours=1), 0.9, 1),
            OOFProbability(2, 1, "c", self.start + timedelta(days=1), self.start + timedelta(days=1, hours=1), 0.2, 1),
            OOFProbability(3, 1, "d", self.start + timedelta(days=1, minutes=1), self.start + timedelta(days=1, hours=1), 0.8, 0),
        )
        result = cross_fit_isotonic_calibration(records, min_calibration_rows=2)
        self.assertIsInstance(result, CrossFittedCalibration)
        # Fold zero had no earlier OOF observations, so it cannot be calibrated.
        self.assertEqual(result.probabilities[:2], (0.1, 0.9))
        self.assertEqual(result.calibrated_rows, 2)
        self.assertEqual(result.uncalibrated_rows, 2)
        self.assertIsNotNone(result.final_calibrator)

    def test_promotion_requires_coverage_and_all_probability_quality_improvements(self):
        incumbent = CandidateScore(
            "incumbent",
            ProbabilityMetrics(500, brier=0.215, logloss=0.520, expected_calibration_error=0.030),
            evaluated_rows=500,
            total_rows=600,
            fold_count=4,
        )
        challenger = CandidateScore(
            "challenger",
            ProbabilityMetrics(500, brier=0.210, logloss=0.510, expected_calibration_error=0.020),
            evaluated_rows=500,
            total_rows=600,
            fold_count=4,
        )
        policy = PromotionPolicy(
            minimum_evaluated_rows=400,
            minimum_coverage=0.75,
            maximum_brier=0.25,
            maximum_logloss=0.70,
            maximum_expected_calibration_error=0.05,
            minimum_brier_improvement=0.001,
            minimum_logloss_improvement=0.001,
            minimum_calibration_improvement=0.001,
        )
        self.assertTrue(evaluate_promotion(challenger, policy, incumbent=incumbent).approved)

        low_coverage = CandidateScore(
            "under-tested",
            ProbabilityMetrics(300, brier=0.20, logloss=0.50, expected_calibration_error=0.02),
            evaluated_rows=300,
            total_rows=600,
            fold_count=4,
        )
        decision = evaluate_promotion(low_coverage, policy)
        self.assertFalse(decision.approved)
        self.assertTrue(any("coverage" in reason for reason in decision.reasons))

    def test_evaluator_emits_only_chronological_oof_predictions_without_sklearn(self):
        class FixedProbabilityEstimator:
            classes_ = [0, 1]

            def fit(self, features, outcomes):
                self.fit_rows = len(features)
                return self

            def predict_proba(self, features):
                return [[1.0 - row[0], row[0]] for row in features]

        splitter = RollingTimeSplitter(n_splits=2, min_train_rows=4, validation_rows=2)
        evaluator = ProbabilityModelEvaluator(
            self.schema, splitter, min_calibration_rows=2, calibration_bin_count=2
        )
        candidate = ModelCandidate("test-logistic", "logistic_regression", random_state=17)
        with patch("sam_analytics.modeling.build_estimator", side_effect=lambda _: FixedProbabilityEstimator()):
            evaluation = evaluator.evaluate(candidate, self._rows())

        self.assertEqual([record.row_index for record in evaluation.oof_predictions], [4, 5, 6, 7])
        self.assertEqual(evaluation.score.evaluated_rows, 4)
        self.assertEqual(evaluation.score.fold_count, 2)
        self.assertEqual(len(evaluation.data_fingerprint), 64)
        self.assertEqual(evaluation.calibration.uncalibrated_rows, 2)
        self.assertEqual(evaluation.calibration.calibrated_rows, 2)

    def test_evidence_adapter_preserves_contract_and_raw_provenance_digests(self):
        contract = FeatureContract(
            name="nba_pregame_moneyline",
            version="v1",
            target_definition="home_team_wins",
            features=(FeatureDefinition("market_probability", "market-v1", minimum=0.0, maximum=1.0),),
        )
        feature_received_at = self.start - timedelta(minutes=10)
        feature_source = RawDataProvenance(
            provider="licensed_provider",
            provider_record_id="odds-1",
            source_type="odds",
            payload_sha256=hashlib.sha256(b"odds").hexdigest(),
            payload_uri="s3://sam-raw/odds/odds-1.json",
            captured_at=feature_received_at - timedelta(seconds=15),
            received_at=feature_received_at,
            schema_version="v1",
        )
        observation = FeatureObservation(
            name="market_probability",
            value=0.55,
            source=feature_source,
            computed_at=self.start - timedelta(minutes=5),
            available_at=self.start - timedelta(minutes=4),
            transformation_version="market-v1",
        )
        vector = PointInTimeFeatureVector(
            event_id="provider-event-1",
            event_starts_at=self.start + timedelta(hours=2),
            as_of=self.start,
            contract=contract,
            observations=(observation,),
        )
        settled_at = self.start + timedelta(hours=3)
        result_source = RawDataProvenance(
            provider="official_provider",
            provider_record_id="result-1",
            source_type="result",
            payload_sha256=hashlib.sha256(b"result").hexdigest(),
            payload_uri="s3://sam-raw/results/result-1.json",
            captured_at=settled_at,
            received_at=settled_at + timedelta(minutes=1),
            schema_version="v1",
        )
        example = LabeledTrainingExample(
            row_id="model-row-1",
            vector=vector,
            target=1,
            settled_at=settled_at,
            result_source=result_source,
        )

        dataset = adapt_labeled_training_examples(contract, [example])
        self.assertEqual(dataset.schema.schema_id, contract.digest)
        self.assertEqual(dataset.feature_contract_digest, contract.digest)
        self.assertEqual(dataset.rows[0].features, {"market_probability": 0.55})
        self.assertEqual(dataset.rows[0].event_id, "provider-event-1")
        self.assertEqual(dataset.rows[0].event_starts_at, vector.event_starts_at)
        self.assertEqual(dataset.rows[0].label_available_at, result_source.received_at)
        self.assertEqual(dataset.rows[0].source_snapshot_ids, (feature_source.digest,))
        self.assertEqual(dataset.rows[0].label_source_snapshot_id, result_source.digest)

        duplicate_event = LabeledTrainingExample(
            row_id="model-row-2",
            vector=vector,
            target=1,
            settled_at=settled_at,
            result_source=result_source,
        )
        with self.assertRaisesRegex(ModelDataError, "each event_id"):
            adapt_labeled_training_examples(contract, [example, duplicate_event])

    def test_default_splitter_makes_default_promotion_sample_feasible(self):
        folds = RollingTimeSplitter().split(self._rows(750))
        self.assertEqual(len(folds), 5)
        score = CandidateScore(
            "challenger",
            ProbabilityMetrics(500, brier=0.20, logloss=0.50, expected_calibration_error=0.02),
            evaluated_rows=sum(len(fold.validation_indices) for fold in folds),
            total_rows=750,
            fold_count=len(folds),
        )
        self.assertTrue(evaluate_promotion(score, PromotionPolicy()).approved)

    def test_fitted_model_refuses_hindsight_inference(self):
        class FixedProbabilityEstimator:
            classes_ = [0, 1]

            def predict_proba(self, features):
                return [[0.4, 0.6] for _ in features]

        release_at = self.start + timedelta(hours=1)
        model = FittedProbabilityModel(
            candidate=ModelCandidate("fixed", "logistic_regression"),
            schema=self.schema,
            estimator=FixedProbabilityEstimator(),
            calibrator=None,
            trained_at=release_at,
            released_at=release_at,
            training_rows=10,
        )
        request = PredictionInput(
            event_id="future-event",
            event_starts_at=self.start + timedelta(hours=2),
            decision_at=self.start,
            features_available_at=self.start - timedelta(minutes=1),
            source_snapshot_ids=("snapshot-1",),
            features={"market_probability": 0.5, "rest_days": 2.0},
        )
        with self.assertRaisesRegex(ModelDataError, "before its release time"):
            model.predict_probability(request)

    def test_candidate_catalog_is_deterministic_and_missing_dependencies_fail_closed(self):
        candidates = default_model_candidates(random_state=17)
        self.assertEqual([candidate.name for candidate in candidates], ["logistic_baseline", "hist_gradient_boosting", "neural_mlp"])
        self.assertTrue(all(candidate.random_state == 17 for candidate in candidates))

        candidate = ModelCandidate("logistic", "logistic_regression", random_state=17)
        with patch(
            "sam_analytics.modeling._load_sklearn_components",
            side_effect=OptionalModelDependencyError("install scikit-learn"),
        ):
            with self.assertRaisesRegex(OptionalModelDependencyError, "scikit-learn"):
                build_estimator(candidate)


if __name__ == "__main__":
    unittest.main()
