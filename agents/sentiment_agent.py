"""Sentiment Agent — analyzes social media and public sentiment for trading signals.

Sources:
- Twitter/X (via API or scrapers)
- Reddit (r/wallstreetbets, r/stocks, r/CryptoCurrency)
- StockTwits
- Telegram/Discord channels
- News headline sentiment

Outputs:
- sentiment_score: -1.0 (very bearish) to +1.0 (very bullish)
- sentiment_magnitude: 0.0 to 1.0 (how strong the signal is)
- key_topics: List of trending topics mentioned
- credibility_score: 0.0 to 1.0 (how reliable is this sentiment)
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .llm_agent import BaseAgent, LLMConfig, AgentMessage, AgentResponse


@dataclass
class SentimentSignal:
    """Structured sentiment analysis output."""
    sentiment_score: float       # -1.0 to +1.0
    sentiment_magnitude: float   # 0.0 to 1.0
    key_topics: List[str]
    credibility_score: float     # 0.0 to 1.0
    source_breakdown: Dict[str, float]  # source -> sentiment
    time_decay_factor: float     # 0.0 to 1.0 (freshness)
    recommendation: str          # "bullish" | "bearish" | "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentiment_score": self.sentiment_score,
            "sentiment_magnitude": self.sentiment_magnitude,
            "key_topics": self.key_topics,
            "credibility_score": self.credibility_score,
            "source_breakdown": self.source_breakdown,
            "time_decay_factor": self.time_decay_factor,
            "recommendation": self.recommendation,
        }


class SentimentAgent(BaseAgent):
    """Analyzes social media and news sentiment for trading signals."""

    def __init__(self, config: LLMConfig):
        super().__init__("SentimentAgent", config)
        self._recent_sentiments: List[SentimentSignal] = []

    def get_system_prompt(self) -> str:
        return """You are a Sentiment Analysis Agent for HFT trading.

Your role:
1. Analyze social media posts, news headlines, and forum discussions
2. Extract sentiment signals relevant to trading decisions
3. Output structured JSON with sentiment scores and confidence

Sentiment scale:
- +1.0: Extremely bullish (FOMO, euphoria, strong buy calls)
- +0.5: Moderately bullish (optimistic, positive news)
-  0.0: Neutral (mixed signals, no clear direction)
- -0.5: Moderately bearish (cautious, negative news)
- -1.0: Extremely bearish (panic, fear, strong sell calls)

Credibility factors:
- Verified accounts, institutional sources → higher credibility
- Anonymous accounts, meme posts → lower credibility
- Consensus across multiple sources → higher credibility
- Outlier opinions → lower credibility

Always output valid JSON matching this schema:
{
    "sentiment_score": float,
    "sentiment_magnitude": float,
    "key_topics": [string],
    "credibility_score": float,
    "source_breakdown": {"twitter": float, "reddit": float, ...},
    "time_decay_factor": float,
    "recommendation": "bullish" | "bearish" | "neutral"
}"""

    def parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse LLM output into SentimentSignal."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data
            except json.JSONDecodeError:
                pass

        # Fallback: return empty signal
        return {
            "sentiment_score": 0.0,
            "sentiment_magnitude": 0.0,
            "key_topics": [],
            "credibility_score": 0.0,
            "source_breakdown": {},
            "time_decay_factor": 0.0,
            "recommendation": "neutral",
        }

    def analyze_text(self, texts: List[str], symbol: str = None) -> SentimentSignal:
        """Analyze a list of text snippets (posts, headlines, etc.)."""
        if not texts:
            return self._empty_signal()

        # Concatenate texts with source markers
        combined = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))

        prompt = f"""Analyze the following social media posts and news headlines for {symbol or 'the asset'}:

{combined}

Output JSON sentiment analysis:"""

        messages = [AgentMessage(role="user", content=prompt)]
        response = self.generate(messages)

        signal = SentimentSignal(
            sentiment_score=response.structured.get("sentiment_score", 0.0),
            sentiment_magnitude=response.structured.get("sentiment_magnitude", 0.0),
            key_topics=response.structured.get("key_topics", []),
            credibility_score=response.structured.get("credibility_score", 0.0),
            source_breakdown=response.structured.get("source_breakdown", {}),
            time_decay_factor=response.structured.get("time_decay_factor", 1.0),
            recommendation=response.structured.get("recommendation", "neutral"),
        )

        self._recent_sentiments.append(signal)
        if len(self._recent_sentiments) > 100:
            self._recent_sentiments = self._recent_sentiments[-100:]

        return signal

    def analyze_reddit(self, posts: List[Dict[str, Any]]) -> SentimentSignal:
        """Analyze Reddit posts (r/wallstreetbets, r/stocks, etc.)."""
        texts = []
        for post in posts:
            title = post.get("title", "")
            body = post.get("selftext", "")
            score = post.get("score", 0)
            subreddit = post.get("subreddit", "")
            texts.append(f"[r/{subreddit}] {title}: {body} (score: {score})")
        return self.analyze_text(texts)

    def analyze_twitter(self, tweets: List[Dict[str, Any]]) -> SentimentSignal:
        """Analyze Twitter/X posts."""
        texts = []
        for tweet in tweets:
            text = tweet.get("text", "")
            author = tweet.get("author", "unknown")
            verified = tweet.get("verified", False)
            retweets = tweet.get("retweet_count", 0)
            texts.append(f"[{'✓' if verified else ' '}@{author}] {text} (RTs: {retweets})")
        return self.analyze_text(texts)

    def analyze_stocktwits(self, messages: List[Dict[str, Any]]) -> SentimentSignal:
        """Analyze StockTwits messages."""
        texts = []
        for msg in messages:
            body = msg.get("body", "")
            user = msg.get("user", {}).get("username", "unknown")
            likes = msg.get("likes", {}).get("total", 0)
            bullish = msg.get("entities", {}).get("sentiment", {}).get("basic") == "Bullish"
            texts.append(f"[@{user}] {'🐂' if bullish else '🐻'} {body} (likes: {likes})")
        return self.analyze_text(texts)

    def get_sentiment_trend(self, window: int = 10) -> Dict[str, float]:
        """Get sentiment trend over recent analyses."""
        if not self._recent_sentiments:
            return {"trend": 0.0, "volatility": 0.0, "acceleration": 0.0}

        recent = self._recent_sentiments[-window:]
        scores = [s.sentiment_score for s in recent]

        if len(scores) < 2:
            return {"trend": scores[0] if scores else 0.0, "volatility": 0.0, "acceleration": 0.0}

        # Linear trend (slope)
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        trend = numerator / denominator if denominator != 0 else 0.0

        # Volatility (std of scores)
        variance = sum((s - y_mean) ** 2 for s in scores) / n
        volatility = variance ** 0.5

        # Acceleration (change in trend)
        if len(scores) >= 4:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            first_trend = (first_half[-1] - first_half[0]) / len(first_half) if len(first_half) > 1 else 0
            second_trend = (second_half[-1] - second_half[0]) / len(second_half) if len(second_half) > 1 else 0
            acceleration = second_trend - first_trend
        else:
            acceleration = 0.0

        return {
            "trend": trend,
            "volatility": volatility,
            "acceleration": acceleration,
        }

    def _empty_signal(self) -> SentimentSignal:
        return SentimentSignal(
            sentiment_score=0.0,
            sentiment_magnitude=0.0,
            key_topics=[],
            credibility_score=0.0,
            source_breakdown={},
            time_decay_factor=0.0,
            recommendation="neutral",
        )
