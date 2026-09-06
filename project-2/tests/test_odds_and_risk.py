import unittest

from sam_analytics.odds import (
    OddsValidationError,
    american_to_decimal,
    devig_two_way,
    expected_roi,
    market_consensus_two_way,
)
from sam_analytics.risk import BankrollPolicy, ExposureState, size_moneyline


class OddsAndRiskTests(unittest.TestCase):
    def test_odds_conversions_and_devig_are_explicit(self):
        self.assertAlmostEqual(american_to_decimal(-110), 1.9090909)
        self.assertAlmostEqual(american_to_decimal(150), 2.5)
        self.assertEqual(devig_two_way(2.0, 2.0), (0.5, 0.5))
        self.assertAlmostEqual(expected_roi(0.55, 2.0), 0.10)
        with self.assertRaises(OddsValidationError):
            american_to_decimal(0)

    def test_market_consensus_devigs_each_book_before_robust_aggregation(self):
        home, away = market_consensus_two_way(((2.0, 2.0), (2.2, 1.8)))
        self.assertAlmostEqual(home, 0.475)
        self.assertAlmostEqual(away, 0.525)
        self.assertAlmostEqual(home + away, 1.0)
        with self.assertRaisesRegex(OddsValidationError, "At least one complete"):
            market_consensus_two_way(())

    def test_stake_is_capped_by_policy(self):
        decision = size_moneyline(
            event_id="event-1",
            model_probability=0.60,
            decimal_odds=2.0,
            policy=BankrollPolicy(bankroll=1000),
            exposure=ExposureState(),
            quote_is_fresh=True,
            model_is_approved=True,
        )
        # 25% Kelly would be $50 here, but the 1% single-stake cap controls it.
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.stake, 10.0)

    def test_stale_or_unapproved_inputs_fail_closed(self):
        decision = size_moneyline(
            event_id="event-1",
            model_probability=0.60,
            decimal_odds=2.0,
            policy=BankrollPolicy(bankroll=1000),
            exposure=ExposureState(),
            quote_is_fresh=False,
            model_is_approved=False,
        )
        self.assertEqual(decision.status, "rejected")
        self.assertEqual(decision.stake, 0.0)
        self.assertEqual(len(decision.reasons), 2)

    def test_non_finite_bankroll_and_exposure_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            BankrollPolicy(bankroll=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            ExposureState(daily_exposure=float("inf"))
