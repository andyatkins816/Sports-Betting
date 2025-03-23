import os
import json
import logging
from typing import Dict, List
from openai import OpenAI
from datetime import datetime

logger = logging.getLogger(__name__)

class AIAnalysisService:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        logger.info("AI Analysis Service initialized")

    def analyze_predictions(self, predictions: List[Dict]) -> Dict:
        """Analyze predictions using GPT to provide betting insights"""
        try:
            # Format predictions data for analysis
            formatted_data = self._format_predictions_data(predictions)
            logger.info(f"Analyzing {len(predictions)} predictions")

            if not predictions:
                logger.warning("No predictions to analyze")
                return self._get_default_analysis()

            # Create analysis prompt
            prompt = self._create_analysis_prompt(formatted_data)
            logger.debug(f"Analysis prompt: {prompt}")

            # Get analysis from GPT
            logger.info("Requesting GPT analysis for predictions")
            response = self.client.chat.completions.create(
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an NBA betting analysis expert. Analyze these games and provide detailed betting insights.

                        For each prediction, provide:
                        1. Concrete betting recommendations with specific odds and confidence levels
                        2. Statistical analysis of key matchups and trends
                        3. Risk assessment and bankroll management advice
                        4. Value betting opportunities where our model shows an edge
                        5. Parlay recommendations based on correlated outcomes

                        Use this format for your response:

                        SINGLE BET RECOMMENDATIONS:
                        Game: [Teams]
                        Type: [Spread/Moneyline/Total]
                        Odds: [Current odds]
                        Confidence: [70-95%]
                        Analysis: [Key statistical advantages, matchup analysis]
                        Risk Assessment: [Specific factors to consider]
                        Value Rating: [7-9 if strong value exists]

                        PARLAY OPPORTUNITIES:
                        Games: [List of games]
                        Combined Odds: [+/-XXX]
                        Analysis: [Why these games correlate]
                        Expected Value: [Strong/Moderate/Limited]

                        MARKET INEFFICIENCIES:
                        • [List specific odds discrepancies]
                        • [Note overreactions to recent results]
                        • [Highlight pricing mistakes]

                        KEY INSIGHTS:
                        • [Major trends affecting these games]
                        • [Important situational spots]
                        • [Key injuries or lineup changes]

                        BETTING STRATEGY:
                        Approach: [Overall strategy for these games]
                        Bankroll: [Specific sizing recommendations]
                        Risk: [How to protect against variance]"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )

            # Parse response
            raw_response = response.choices[0].message.content
            logger.info("Received GPT response")
            logger.debug(f"Raw GPT response: {raw_response}")

            analysis = self._parse_gpt_response(raw_response)
            logger.debug(f"Parsed analysis: {analysis}")

            formatted_response = self._format_analysis_response(analysis)
            logger.debug(f"Formatted response: {formatted_response}")

            return formatted_response

        except Exception as e:
            logger.error(f"Error in AI analysis: {e}", exc_info=True)
            return self._get_default_analysis()

    def _format_predictions_data(self, predictions: List[Dict]) -> List[Dict]:
        """Format predictions data for GPT analysis"""
        formatted_data = []
        for pred in predictions:
            game = pred['game']
            prediction = pred['prediction']

            formatted_data.append({
                'matchup': f"{game['home_team']['name']} vs {game['away_team']['name']}",
                'home_team': {
                    'name': game['home_team']['name'],
                    'win_percentage': f"{game['home_team']['win_pct']*100:.1f}%",
                    'points_avg': f"{game['home_team']['points_avg']:.1f}",
                    'recent_form': 'Strong' if game['home_team']['win_pct'] > 0.6 else 'Average'
                },
                'away_team': {
                    'name': game['away_team']['name'],
                    'win_percentage': f"{game['away_team']['win_pct']*100:.1f}%",
                    'points_avg': f"{game['away_team']['points_avg']:.1f}",
                    'recent_form': 'Strong' if game['away_team']['win_pct'] > 0.6 else 'Average'
                },
                'prediction': {
                    'home_win_prob': f"{prediction['home_team_win_probability']*100:.1f}%",
                    'spread': f"{prediction['point_spread']:+.1f}",
                    'total': prediction['total_points'],
                    'confidence': f"{prediction['confidence_score']*100:.0f}%"
                }
            })

        return formatted_data

    def _create_analysis_prompt(self, data: List[Dict]) -> str:
        """Create a detailed prompt for GPT analysis"""
        return f"""
        Analyze these NBA games and provide detailed betting recommendations.
        Focus on finding value bets where our model shows an edge vs the market.

        Game Data:
        {json.dumps(data, indent=2)}

        For each prediction:
        1. Compare our projected probabilities to implied odds
        2. Identify key statistical mismatches
        3. Note any scheduling advantages/disadvantages
        4. Assess recent form and head-to-head trends
        5. Consider parlay opportunities with correlated outcomes

        Provide specific, actionable recommendations including:
        - Exact bet types and sizing
        - Clear confidence levels
        - Risk factors to consider
        - Value ratings based on edge size
        - Parlay combinations that maximize EV
        """

    def _parse_gpt_response(self, response_text: str) -> Dict:
        """Parse GPT response into structured data"""
        sections = response_text.split('\n\n')
        analysis = {
            'single_bets': [],
            'parlay_opportunities': [],
            'market_inefficiencies': [],
            'key_insights': [],
            'betting_strategy': {}
        }

        current_section = None
        current_item = {}

        for section in sections:
            section = section.strip()

            if 'SINGLE BET RECOMMENDATIONS:' in section:
                current_section = 'single_bets'
                bet = self._parse_bet_section(section)
                if bet:
                    analysis['single_bets'].append(bet)
            elif 'PARLAY OPPORTUNITIES:' in section:
                current_section = 'parlays'
                parlay = self._parse_parlay_section(section)
                if parlay:
                    analysis['parlay_opportunities'].append(parlay)
            elif 'MARKET INEFFICIENCIES:' in section:
                analysis['market_inefficiencies'] = self._extract_bullet_points(section)
            elif 'KEY INSIGHTS:' in section:
                analysis['key_insights'] = self._extract_bullet_points(section)
            elif 'BETTING STRATEGY:' in section:
                analysis['betting_strategy'] = self._parse_strategy_section(section)

        return analysis

    def _parse_bet_section(self, section: str) -> Dict:
        """Parse a single bet recommendation section"""
        lines = section.split('\n')
        bet = {
            'teams': '',
            'type': '',
            'odds': '',
            'confidence': 75,
            'reasoning': '',
            'valueRating': 7,
            'riskAssessment': ''
        }

        for line in lines:
            line = line.strip()
            if line.startswith('Game:'):
                bet['teams'] = line.replace('Game:', '').strip()
            elif line.startswith('Type:'):
                bet['type'] = line.replace('Type:', '').strip()
            elif line.startswith('Odds:'):
                bet['odds'] = line.replace('Odds:', '').strip()
            elif line.startswith('Confidence:'):
                try:
                    conf = line.replace('Confidence:', '').strip().rstrip('%')
                    bet['confidence'] = int(conf)
                except:
                    pass
            elif line.startswith('Analysis:'):
                bet['reasoning'] = line.replace('Analysis:', '').strip()
            elif line.startswith('Risk Assessment:'):
                bet['riskAssessment'] = line.replace('Risk Assessment:', '').strip()
            elif line.startswith('Value Rating:'):
                try:
                    rating = line.replace('Value Rating:', '').strip()
                    bet['valueRating'] = int(rating)
                except:
                    pass

        return bet if bet['teams'] else None

    def _parse_parlay_section(self, section: str) -> Dict:
        """Parse a parlay recommendation section"""
        lines = section.split('\n')
        parlay = {
            'name': '',
            'bets': [],
            'combinedOdds': '',
            'confidence': 70,
            'reasoning': '',
            'expectedValue': ''
        }

        for line in lines:
            line = line.strip()
            if line.startswith('Games:'):
                parlay['name'] = line.replace('Games:', '').strip()
                parlay['bets'] = [game.strip() for game in parlay['name'].split('and')]
            elif line.startswith('Combined Odds:'):
                parlay['combinedOdds'] = line.replace('Combined Odds:', '').strip()
            elif line.startswith('Analysis:'):
                parlay['reasoning'] = line.replace('Analysis:', '').strip()
            elif line.startswith('Expected Value:'):
                parlay['expectedValue'] = line.replace('Expected Value:', '').strip()

        return parlay if parlay['name'] else None

    def _extract_bullet_points(self, section: str) -> List[str]:
        """Extract bullet points from a section"""
        points = []
        lines = section.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-'):
                point = line.lstrip('•').lstrip('-').strip()
                if point:
                    points.append(point)

        return points

    def _parse_strategy_section(self, section: str) -> Dict:
        """Parse betting strategy section"""
        lines = section.split('\n')
        strategy = {
            'recommended_approach': '',
            'bankroll_management': '',
            'risk_management': ''
        }

        for line in lines:
            line = line.strip()
            if line.startswith('Approach:'):
                strategy['recommended_approach'] = line.replace('Approach:', '').strip()
            elif line.startswith('Bankroll:'):
                strategy['bankroll_management'] = line.replace('Bankroll:', '').strip()
            elif line.startswith('Risk:'):
                strategy['risk_management'] = line.replace('Risk:', '').strip()

        return strategy

    def _format_analysis_response(self, analysis: Dict) -> Dict:
        """Format analysis for frontend display"""
        return {
            'singleBets': analysis['single_bets'],
            'parlayBets': analysis['parlay_opportunities'],
            'marketInsights': analysis['market_inefficiencies'],
            'keyInsights': analysis['key_insights'],
            'bettingStrategy': analysis['betting_strategy']
        }

    def _get_default_analysis(self) -> Dict:
        """Return default analysis structure"""
        return {
            'singleBets': [],
            'parlayBets': [],
            'marketInsights': [
                "Make predictions for some games to receive detailed betting analysis",
                "Our AI will analyze odds, trends, and key matchups"
            ],
            'keyInsights': [
                "Click 'Predict' on multiple games above",
                "Then use the AI assistant icon to analyze all predictions"
            ],
            'bettingStrategy': {
                'recommended_approach': "Start by making predictions for today's key matchups",
                'bankroll_management': "Analysis will include optimal bet sizing once predictions are made",
                'risk_management': "Get detailed risk assessment after analyzing specific games"
            }
        }