import logging
from datetime import datetime, timedelta
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import os

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self):
        self.initialize_nltk()

    def initialize_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('vader_lexicon')
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def fetch_team_news(self, team_name: str, days: int = 3) -> List[Dict]:
        """Fetch recent news articles about a specific team"""
        try:
            # In production, replace with actual API calls to news services
            # This is a sample implementation
            sample_news = self._get_sample_news(team_name)
            return self._analyze_articles(sample_news)
        except Exception as e:
            logger.error(f"Error fetching news for {team_name}: {e}")
            return []

    def _get_sample_news(self, team_name: str) -> List[Dict]:
        """Generate sample news data for development"""
        current_date = datetime.now()
        return [
            {
                'title': f"{team_name} Star Player Returns from Injury",
                'content': f"The {team_name} received a major boost as their star player returned to full training after recovering from injury. The team's performance is expected to improve significantly.",
                'date': (current_date - timedelta(days=1)).strftime('%Y-%m-%d'),
                'source': 'Sports News Network'
            },
            {
                'title': f"{team_name} Coach Implements New Strategy",
                'content': f"The {team_name}'s head coach has introduced a new offensive system that showed promising results in recent practices. Players are adapting well to the changes.",
                'date': (current_date - timedelta(days=2)).strftime('%Y-%m-%d'),
                'source': 'Basketball Insider'
            }
        ]

    def _analyze_articles(self, articles: List[Dict]) -> List[Dict]:
        """Analyze sentiment and extract key information from articles"""
        analyzed_articles = []

        for article in articles:
            # Perform sentiment analysis on title and content
            title_sentiment = self.sentiment_analyzer.polarity_scores(article['title'])
            content_sentences = sent_tokenize(article['content'])
            content_sentiments = [
                self.sentiment_analyzer.polarity_scores(sent) for sent in content_sentences
            ]

            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(
                title_sentiment, content_sentiments
            )

            # Extract key information
            key_info = self._extract_key_information(article['content'])

            analyzed_articles.append({
                'title': article['title'],
                'date': article['date'],
                'source': article['source'],
                'sentiment': overall_sentiment,
                'key_information': key_info
            })

        return analyzed_articles

    def _calculate_overall_sentiment(self, title_sentiment: Dict, 
                                  content_sentiments: List[Dict]) -> Dict:
        """Calculate overall sentiment score from title and content"""
        # Weight title sentiment more heavily (30%)
        title_weight = 0.3
        content_weight = 0.7

        # Calculate average content sentiment
        content_compound = sum(s['compound'] for s in content_sentiments) / len(content_sentiments)

        # Calculate weighted compound score
        overall_score = (title_sentiment['compound'] * title_weight + 
                        content_compound * content_weight)

        return {
            'score': overall_score,
            'label': 'POSITIVE' if overall_score > 0 else 'NEGATIVE',
            'confidence': abs(overall_score)
        }

    def _extract_key_information(self, content: str) -> Dict:
        """Extract key information from article content"""
        info = {
            'injury_related': False,
            'strategy_change': False,
            'team_morale': 'neutral',
            'performance_trend': 'stable'
        }

        # Simple keyword-based analysis
        content_lower = content.lower()

        # Check for injury-related news
        if any(word in content_lower for word in ['injury', 'injured', 'recovery']):
            info['injury_related'] = True

        # Check for strategy changes
        if any(word in content_lower for word in ['strategy', 'system', 'approach']):
            info['strategy_change'] = True

        # Analyze team morale
        positive_morale = ['confident', 'optimistic', 'positive']
        negative_morale = ['concerned', 'worried', 'frustrated']

        if any(word in content_lower for word in positive_morale):
            info['team_morale'] = 'positive'
        elif any(word in content_lower for word in negative_morale):
            info['team_morale'] = 'negative'

        # Analyze performance trend
        improvement = ['improving', 'better', 'progress']
        decline = ['declining', 'worse', 'struggling']

        if any(word in content_lower for word in improvement):
            info['performance_trend'] = 'improving'
        elif any(word in content_lower for word in decline):
            info['performance_trend'] = 'declining'

        return info

    def get_team_sentiment_factor(self, team_name: str) -> float:
        """Calculate a sentiment factor for predictions based on recent news"""
        try:
            articles = self.fetch_team_news(team_name)
            if not articles:
                return 0.0

            # Calculate weighted average of sentiment scores
            total_weight = 0
            weighted_score = 0

            for i, article in enumerate(articles):
                # More recent articles have higher weight
                weight = 1 / (i + 1)
                total_weight += weight
                weighted_score += article['sentiment']['score'] * weight

            # Normalize to range [-0.1, 0.1] to adjust win probability
            sentiment_factor = (weighted_score / total_weight) * 0.1
            return round(sentiment_factor, 3)

        except Exception as e:
            logger.error(f"Error calculating sentiment factor: {e}")
            return 0.0