import unittest
from datetime import datetime, timedelta, timezone

from sam_analytics.backtest import ApprovedModelRelease, BacktestObservation, run_backtest
from sam_analytics.calibration import IsotonicCalibrator
from sam_analytics.metrics import brier_score, log_loss
from sam_analytics.risk import BankrollPolicy


class CalibrationAndBacktestTests(unittest.TestCase):
    def _release(self, effective_at):
        return ApprovedModelRelease(
            version="nba-1",
            artifact_sha256="a" * 64,
            effective_at=effective_at,
        )

    def test_isotonic_calibration_is_monotonic(self):
        calibrator = IsotonicCalibrator().fit([0.2, 0.4, 0.8], [1, 0, 1])
        calibrated = calibrator.predict([0.2, 0.4, 0.8])
        self.assertLessEqual(calibrated[0], calibrated[1])
        self.assertLessEqual(calibrated[1], calibrated[2])
        self.assertAlmostEqual(calibrated[0], 0.5)
        self.assertAlmostEqual(calibrated[1], 0.5)
        self.assertAlmostEqual(calibrated[2], 1.0)

    def test_isotonic_calibration_groups_identical_scores(self):
        calibrated = IsotonicCalibrator().fit([0.5, 0.5], [1, 0]).predict([0.5])
        self.assertEqual(calibrated, [0.5])

    def test_proper_scores_penalize_overconfidence(self):
        self.assertLess(brier_score([0.5], [1]), brier_score([0.01], [1]))
        self.assertLess(log_loss([0.5], [1]), log_loss([0.01], [1]))

    def test_backtest_requires_chronological_observations_and_reports_stakes(self):
        now = datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
        rows = [
            BacktestObservation(
                "a", now + timedelta(hours=1), now, now - timedelta(minutes=1),
                now - timedelta(minutes=2), "quote-a", "prediction-a",
                self._release(now - timedelta(days=1)), 0.60, 2.0, 1,
            ),
            BacktestObservation(
                "b", now + timedelta(hours=2), now + timedelta(hours=1),
                now + timedelta(minutes=59), now, "quote-b", "prediction-b",
                self._release(now - timedelta(days=1)), 0.60, 2.0, 0,
            ),
        ]
        result = run_backtest(rows, BankrollPolicy(bankroll=1000))
        self.assertEqual(result.evaluated, 2)
        self.assertEqual(result.rejected, 0)
        self.assertEqual(result.stake, 20.0)
        self.assertEqual(result.profit, 0.0)
        self.assertIn("brier_score", result.metrics)

    def test_backtest_rejects_lookahead_features(self):
        now = datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
        row = BacktestObservation(
            "a", now + timedelta(hours=1), now, now, now + timedelta(seconds=1),
            "quote-a", "prediction-a", self._release(now - timedelta(days=1)), 0.60, 2.0, 1,
        )
        with self.assertRaisesRegex(ValueError, "arrived after"):
            run_backtest([row], BankrollPolicy(bankroll=1000))

    def test_backtest_rejects_post_start_or_pre_release_decisions(self):
        now = datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
        post_start = BacktestObservation(
            "a", now, now, now - timedelta(minutes=1), now - timedelta(minutes=2),
            "quote-a", "prediction-a", self._release(now - timedelta(days=1)), 0.60, 2.0, 1,
        )
        with self.assertRaisesRegex(ValueError, "precede"):
            run_backtest([post_start], BankrollPolicy(bankroll=1000))

        pre_release = BacktestObservation(
            "a", now + timedelta(hours=1), now, now - timedelta(minutes=1),
            now - timedelta(minutes=2), "quote-a", "prediction-a",
            self._release(now + timedelta(seconds=1)), 0.60, 2.0, 1,
        )
        with self.assertRaisesRegex(ValueError, "effective after"):
            run_backtest([pre_release], BankrollPolicy(bankroll=1000))
