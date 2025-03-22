import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class BettingRiskService:
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        self.confidence_weight = 0.3
        self.odds_weight = 0.25
        self.volatility_weight = 0.25
        self.historical_weight = 0.2
        logger.info("Betting Risk Service initialized")

    def analyze_predictions(self, predictions: List[Dict]) -> Dict:
        """Analyze multiple predictions to find best betting opportunities"""
        try:
            logger.info(f"Analyzing {len(predictions)} predictions")
            single_bets = []
            parlay_opportunities = []

            # Analyze each prediction for single bet opportunities
            for pred in predictions:
                risk_analysis = self.analyze_betting_risk(pred['game'], pred['prediction'])

                # Calculate value score (expected value based on odds and probability)
                home_prob = pred['prediction']['home_team_win_probability']
                away_prob = pred['prediction']['away_team_win_probability']
                point_spread = abs(pred['prediction']['point_spread'])

                # Only recommend bets with positive expected value and acceptable risk
                if risk_analysis['risk_score'] < self.risk_thresholds['high']:
                    confidence = round((1 - risk_analysis['risk_score']) * 100)

                    # Determine best betting angle
                    spread_value = self._calculate_spread_value(point_spread, home_prob, away_prob)
                    moneyline_value = self._calculate_moneyline_value(
                        pred['prediction']['odds']['money_line'],
                        home_prob,
                        away_prob
                    )

                    # Choose best bet type based on value
                    if spread_value > moneyline_value and point_spread > 0:
                        bet_type = 'Spread'
                        odds = pred['prediction']['odds']['spread']
                        value = spread_value
                    else:
                        bet_type = 'Moneyline'
                        odds = pred['prediction']['odds']['money_line']['home']
                        value = moneyline_value

                    # Generate detailed reasoning
                    reasoning = self._generate_bet_reasoning(
                        pred['game'],
                        pred['prediction'],
                        risk_analysis,
                        bet_type,
                        value
                    )

                    single_bets.append((pred, {
                        'type': bet_type,
                        'odds': odds,
                        'confidence': confidence,
                        'value_score': round(value, 2),
                        'reasoning': reasoning
                    }))

            # Sort bets by value score and confidence
            single_bets.sort(key=lambda x: (x[1]['value_score'], x[1]['confidence']), reverse=True)

            # Generate parlay opportunities from high-value bets
            if len(single_bets) >= 2:
                parlay_opportunities = self._generate_parlay_opportunities(single_bets)

            return {
                'single_bets': single_bets,
                'parlay_opportunities': parlay_opportunities
            }

        except Exception as e:
            logger.error(f"Error analyzing predictions: {e}", exc_info=True)
            return {
                'single_bets': [],
                'parlay_opportunities': []
            }

    def _calculate_spread_value(self, spread: float, home_prob: float, away_prob: float) -> float:
        """Calculate expected value for spread betting"""
        if spread == 0:
            return 0

        # Adjust probabilities based on spread
        spread_adjusted_prob = home_prob - (spread / 20)  # Simple adjustment based on spread
        fair_odds = (1 / spread_adjusted_prob) * 100
        actual_odds = -110  # Standard spread odds

        return (spread_adjusted_prob * (100 / abs(actual_odds)) - (1 - spread_adjusted_prob))

    def _calculate_moneyline_value(self, money_line: Dict, home_prob: float, away_prob: float) -> float:
        """Calculate expected value for moneyline betting"""
        home_odds = float(money_line['home'])
        away_odds = float(money_line['away'])

        # Calculate expected value for both sides
        if home_odds > 0:
            home_value = (home_prob * (home_odds/100)) - (1 - home_prob)
        else:
            home_value = (home_prob * (100/abs(home_odds))) - (1 - home_prob)

        if away_odds > 0:
            away_value = (away_prob * (away_odds/100)) - (1 - away_prob)
        else:
            away_value = (away_prob * (100/abs(away_odds))) - (1 - away_prob)

        return max(home_value, away_value)

    def _generate_bet_reasoning(self, game: Dict, prediction: Dict, risk_analysis: Dict, bet_type: str, value: float) -> str:
        """Generate detailed reasoning for a bet recommendation"""
        home_team = game['home_team']['name']
        away_team = game['away_team']['name']

        # Create detailed reasoning based on multiple factors
        reasons = []

        # Value assessment
        if value > 0.15:
            reasons.append(f"Strong value opportunity with {round(value * 100, 1)}% edge")
        elif value > 0.05:
            reasons.append(f"Moderate value with {round(value * 100, 1)}% edge")

        # Team performance insights
        home_points = game['home_team']['points_avg']
        away_points = game['away_team']['points_avg']
        if abs(home_points - away_points) > 5:
            better_team = home_team if home_points > away_points else away_team
            reasons.append(f"{better_team} shows stronger offensive performance")

        # Risk factors
        if risk_analysis['risk_level'] == 'low':
            reasons.append("Historical data suggests consistent performance patterns")
        elif risk_analysis['risk_level'] == 'medium':
            reasons.append("Moderate volatility in recent performances")

        # Combine reasons
        return " | ".join(reasons)

    def _generate_parlay_opportunities(self, single_bets: List[tuple]) -> List[Dict]:
        """Generate parlay recommendations from high-value single bets"""
        parlays = []

        # Only use bets with positive value and high confidence
        quality_bets = [bet for bet in single_bets if bet[1]['value_score'] > 0 and bet[1]['confidence'] > 70]

        # Generate 2-3 game parlays
        for i in range(min(len(quality_bets) - 1, 3)):
            parlay_bets = quality_bets[i:i+2]
            combined_odds = self._calculate_parlay_odds([bet[1]['odds'] for bet in parlay_bets])
            combined_value = sum(bet[1]['value_score'] for bet in parlay_bets) / len(parlay_bets)

            parlays.append({
                'name': f"{len(parlay_bets)}-Game Value Parlay",
                'combined_odds': combined_odds,
                'potential_payout': round(combined_odds * 1.5, 2),
                'value_score': round(combined_value, 2),
                'bets': [
                    f"{game['game']['home_team']['name']} vs {game['game']['away_team']['name']}: {bet['type']} ({bet['odds']})"
                    for game, bet in parlay_bets
                ]
            })

        # Sort parlays by value score
        parlays.sort(key=lambda x: x['value_score'], reverse=True)
        return parlays

    def _calculate_parlay_odds(self, odds_list: List[str]) -> float:
        """Calculate combined odds for a parlay"""
        try:
            # Convert American odds to decimal
            decimal_odds = []
            for odds in odds_list:
                if isinstance(odds, str):
                    odds = float(odds.replace('+', ''))
                if odds > 0:
                    decimal_odds.append(1 + (odds / 100))
                else:
                    decimal_odds.append(1 + (100 / abs(odds)))

            # Calculate combined odds
            combined = 1
            for odds in decimal_odds:
                combined *= odds

            # Convert back to American odds
            if combined > 2:
                american_odds = round((combined - 1) * 100)
            else:
                american_odds = round(-100 / (combined - 1))

            return american_odds
        except Exception as e:
            logger.error(f"Error calculating parlay odds: {e}")
            return -110  # Default to standard odds

    def analyze_betting_risk(self, game_data: Dict, prediction_data: Dict) -> Dict:
        """Analyze betting risk factors and generate insights"""
        try:
            logger.debug(f"Analyzing betting risk for game: {game_data}")

            # Calculate individual risk components
            confidence_risk = self._calculate_confidence_risk(prediction_data)
            odds_risk = self._calculate_odds_risk(game_data)
            volatility_risk = self._calculate_volatility_risk(game_data)
            historical_risk = self._calculate_historical_risk(game_data)

            # Calculate weighted overall risk
            overall_risk = (
                confidence_risk * self.confidence_weight +
                odds_risk * self.odds_weight +
                volatility_risk * self.volatility_weight +
                historical_risk * self.historical_weight
            )

            # Generate risk insights
            insights = self._generate_risk_insights(
                overall_risk,
                confidence_risk,
                odds_risk,
                volatility_risk,
                historical_risk,
                game_data,
                prediction_data
            )

            return {
                'risk_score': overall_risk,
                'risk_level': self._get_risk_level(overall_risk),
                'insights': insights
            }

        except Exception as e:
            logger.error(f"Error analyzing betting risk: {str(e)}", exc_info=True)
            return self._get_default_risk_analysis()

    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score <= self.risk_thresholds['low']:
            return 'low'
        elif risk_score <= self.risk_thresholds['medium']:
            return 'medium'
        elif risk_score <= self.risk_thresholds['high']:
            return 'high'
        return 'very_high'

    def _generate_risk_insights(self, overall_risk: float, confidence_risk: float,
                              odds_risk: float, volatility_risk: float,
                              historical_risk: float, game_data: Dict,
                              prediction_data: Dict) -> List[str]:
        """Generate detailed betting risk insights"""
        insights = []

        # Generate main insight based on risk factors
        if overall_risk < self.risk_thresholds['low']:
            insights.append(f"Strong opportunity with solid statistical backing")
        elif overall_risk < self.risk_thresholds['medium']:
            insights.append(f"Balanced risk-reward profile with moderate confidence")
        else:
            insights.append(f"High-risk bet with potential for variance")

        # Add specific risk insights
        if confidence_risk > 0.6:
            insights.append("Model shows lower confidence in prediction")
        if odds_risk > 0.7:
            insights.append("Line movement suggests market uncertainty")
        if volatility_risk > 0.6:
            insights.append("Recent performance shows high variability")
        if historical_risk > 0.7:
            insights.append("Historical matchup data indicates unpredictability")

        return insights

    def _calculate_confidence_risk(self, prediction_data: Dict) -> float:
        """Calculate risk based on prediction confidence"""
        try:
            confidence = prediction_data.get('confidence_score', 0.5)
            model_agreement = prediction_data.get('spread_analysis', {}).get('model_agreement', True)

            # Higher risk if models disagree or confidence is low
            base_risk = 1 - confidence
            if not model_agreement:
                base_risk += 0.2  # Penalty for model disagreement

            return min(max(base_risk, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error calculating confidence risk: {str(e)}")
            return 0.5

    def _calculate_odds_risk(self, game_data: Dict) -> float:
        """Calculate risk based on betting odds"""
        try:
            odds = game_data.get('odds', {})
            if not odds:
                return 0.5

            # Calculate risk based on money line disparity
            home_ml = abs(float(odds.get('money_line', {}).get('home', 0)))
            away_ml = abs(float(odds.get('money_line', {}).get('away', 0)))

            # Higher risk for closely matched teams or very disparate odds
            odds_disparity = abs(home_ml - away_ml) / max(home_ml + 1, away_ml + 1)

            return min(odds_disparity, 1.0)
        except Exception as e:
            logger.error(f"Error calculating odds risk: {str(e)}")
            return 0.5

    def _calculate_volatility_risk(self, game_data: Dict) -> float:
        """Calculate risk based on team performance volatility"""
        try:
            home_team = game_data.get('home_team', {})
            away_team = game_data.get('away_team', {})

            # Use points average and against average to calculate volatility
            home_volatility = abs(home_team.get('points_avg', 0) - home_team.get('points_against_avg', 0))
            away_volatility = abs(away_team.get('points_avg', 0) - away_team.get('points_against_avg', 0))

            # Normalize volatility to 0-1 range
            max_expected_volatility = 20  # Maximum expected point differential
            volatility_risk = (home_volatility + away_volatility) / (2 * max_expected_volatility)

            return min(max(volatility_risk, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error calculating volatility risk: {str(e)}")
            return 0.5

    def _calculate_historical_risk(self, game_data: Dict) -> float:
        """Calculate risk based on historical performance"""
        try:
            home_team = game_data.get('home_team', {})
            away_team = game_data.get('away_team', {})

            # Use win percentages to determine historical consistency
            home_win_pct = home_team.get('win_pct', 0.5)
            away_win_pct = away_team.get('win_pct', 0.5)

            # Higher risk when teams are closely matched historically
            historical_disparity = abs(home_win_pct - away_win_pct)

            return min(1 - historical_disparity, 1.0)  # Higher disparity = lower risk
        except Exception as e:
            logger.error(f"Error calculating historical risk: {str(e)}")
            return 0.5

    def _get_default_risk_analysis(self) -> Dict:
        """Return default risk analysis when calculation fails"""
        return {
            'risk_score': 0.5,
            'risk_level': 'medium',
            'insights': ["Unable to calculate detailed risk analysis"]
        }