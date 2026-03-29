"""News Agent — parses macroeconomic news and geopolitical events for trading signals.

Sources:
- Reuters, Bloomberg, AP News APIs
- Federal Reserve announcements (FOMC, speeches)
- Economic calendars (CPI, NFP, GDP, PMI releases)
- Geopolitical event feeds

Outputs:
- impact_score: -1.0 (very negative) to +1.0 (very positive)
- urgency: 0.0 to 1.0 (how soon to act)
- affected_sectors: List of sectors impacted
- event_type: "monetary_policy" | "economic_data" | "geopolitical" | "earnings" | "other"
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

from .llm_agent import BaseAgent, LLMConfig, AgentMessage, AgentResponse


class EventType(Enum):
    MONETARY_POLICY = "monetary_policy"
    ECONOMIC_DATA = "economic_data"
    GEOPOLITICAL = "geopolitical"
    EARNINGS = "earnings"
    REGULATORY = "regulatory"
    OTHER = "other"


@dataclass
class NewsSignal:
    """Structured news analysis output."""
    headline: str
    summary: str
    impact_score: float        # -1.0 to +1.0
    urgency: float             # 0.0 to 1.0 (act now vs. wait)
    event_type: str
    affected_sectors: List[str]
    affected_assets: List[str]
    confidence: float          # 0.0 to 1.0
    source: str
    timestamp: int             # Unix timestamp ms
    follow_up_required: bool   # True if more info needed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "impact_score": self.impact_score,
            "urgency": self.urgency,
            "event_type": self.event_type,
            "affected_sectors": self.affected_sectors,
            "affected_assets": self.affected_assets,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp,
            "follow_up_required": self.follow_up_required,
        }


class NewsAgent(BaseAgent):
    """Parses macroeconomic news and geopolitical events."""

    def __init__(self, config: LLMConfig):
        super().__init__("NewsAgent", config)
        self._recent_news: List[NewsSignal] = []
        self._economic_calendar: Dict[str, Any] = {}

    def get_system_prompt(self) -> str:
        return """You are a News Analysis Agent for HFT trading.

Your role:
1. Parse breaking news headlines and articles
2. Assess market impact and urgency
3. Classify event type and affected sectors/assets
4. Output structured JSON for automated trading decisions

Event types:
- monetary_policy: Fed announcements, rate decisions, QE/QT
- economic_data: CPI, NFP, GDP, PMI, retail sales, etc.
- geopolitical: Wars, elections, trade deals, sanctions
- earnings: Company earnings reports, guidance
- regulatory: SEC actions, antitrust, new regulations
- other: Everything else

Impact scale:
- +1.0: Extremely positive (market rally expected)
- +0.5: Moderately positive
-  0.0: Neutral / mixed
- -0.5: Moderately negative
- -1.0: Extremely negative (market crash expected)

Urgency scale:
- 1.0: Act immediately (breaking news, market-moving)
- 0.5: Monitor closely (developing story)
- 0.0: Low priority (background noise)

Always output valid JSON matching this schema:
{
    "headline": string,
    "summary": string,
    "impact_score": float,
    "urgency": float,
    "event_type": string,
    "affected_sectors": [string],
    "affected_assets": [string],
    "confidence": float,
    "source": string,
    "follow_up_required": boolean
}"""

    def parse_response(self, raw: str) -> Dict[str, Any]:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return self._empty_dict()

    def analyze_headline(self, headline: str, body: str = "", source: str = "unknown") -> NewsSignal:
        """Analyze a single news headline/article."""
        import time

        prompt = f"""Analyze this news for HFT trading impact:

Headline: {headline}
Body: {body or "No additional content"}
Source: {source}

Output JSON analysis:"""

        messages = [AgentMessage(role="user", content=prompt)]
        response = self.generate(messages)

        data = response.structured or self._empty_dict()
        signal = NewsSignal(
            headline=headline,
            summary=data.get("summary", ""),
            impact_score=data.get("impact_score", 0.0),
            urgency=data.get("urgency", 0.0),
            event_type=data.get("event_type", "other"),
            affected_sectors=data.get("affected_sectors", []),
            affected_assets=data.get("affected_assets", []),
            confidence=data.get("confidence", 0.5),
            source=source,
            timestamp=int(time.time() * 1000),
            follow_up_required=data.get("follow_up_required", False),
        )

        self._recent_news.append(signal)
        if len(self._recent_news) > 50:
            self._recent_news = self._recent_news[-50:]

        return signal

    def analyze_fomc_statement(self, text: str) -> NewsSignal:
        """Analyze FOMC statement or Fed speech."""
        prompt = f"""Analyze this Federal Reserve communication for HFT trading:

{text}

Focus on:
- Rate decision (hawkish/dovish surprise)
- Forward guidance changes
- Economic projections (dot plot)
- QE/QT announcements

Output JSON analysis:"""

        messages = [AgentMessage(role="user", content=prompt)]
        response = self.generate(messages)

        data = response.structured or self._empty_dict()
        data["event_type"] = "monetary_policy"
        data["affected_assets"] = list(set(data.get("affected_assets", []) + ["USD", "TLT", "ES=F", "GC=F"]))

        return self._news_from_dict(data, "Federal Reserve")

    def analyze_economic_data(self, indicator: str, actual: float, forecast: float, previous: float) -> NewsSignal:
        """Analyze economic data release (CPI, NFP, etc.)."""
        surprise = (actual - forecast) / max(abs(forecast), 0.01) * 100

        prompt = f"""Analyze this economic data release:

Indicator: {indicator}
Actual: {actual}
Forecast: {forecast}
Previous: {previous}
Surprise: {surprise:.2f}%

Determine market impact based on the surprise magnitude and direction.
Output JSON analysis:"""

        messages = [AgentMessage(role="user", content=prompt)]
        response = self.generate(messages)

        data = response.structured or self._empty_dict()
        data["event_type"] = "economic_data"

        # Adjust impact based on surprise
        base_impact = data.get("impact_score", 0.0)
        surprise_factor = min(abs(surprise) / 10, 1.0)  # Cap at 10% surprise
        if indicator in ["CPI", "Core CPI", "PCE"]:
            # High CPI = bad for stocks (rate hike fears)
            if actual > forecast:
                data["impact_score"] = -surprise_factor
            else:
                data["impact_score"] = surprise_factor
        elif indicator in ["Nonfarm Payrolls", "Unemployment Rate"]:
            # Complex: good economy = potential rate hikes
            data["impact_score"] = base_impact * surprise_factor

        data["affected_assets"] = list(set(data.get("affected_assets", []) + ["USD", "ES=F", "NQ=F", "ZN=F"]))

        return self._news_from_dict(data, f"Economic Data: {indicator}")

    def set_economic_calendar(self, calendar: Dict[str, Any]) -> None:
        """Set upcoming economic events for context."""
        self._economic_calendar = calendar

    def get_upcoming_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get upcoming economic events."""
        if not self._economic_calendar:
            return []

        import time
        now = time.time() * 1000
        cutoff = now + hours * 3600 * 1000

        upcoming = []
        for event in self._economic_calendar.get("events", []):
            event_time = event.get("timestamp", 0)
            if now <= event_time <= cutoff:
                upcoming.append(event)
        return upcoming

    def get_news_sentiment_trend(self, window: int = 10) -> Dict[str, float]:
        """Get news sentiment trend."""
        if not self._recent_news:
            return {"trend": 0.0, "avg_impact": 0.0, "urgency_avg": 0.0}

        recent = self._recent_news[-window:]
        impacts = [n.impact_score for n in recent]
        urgencies = [n.urgency for n in recent]

        return {
            "trend": sum(impacts) / len(impacts),
            "avg_impact": sum(impacts) / len(impacts),
            "urgency_avg": sum(urgencies) / len(urgencies),
        }

    def _news_from_dict(self, data: Dict[str, Any], source: str) -> NewsSignal:
        import time
        return NewsSignal(
            headline=data.get("headline", ""),
            summary=data.get("summary", ""),
            impact_score=data.get("impact_score", 0.0),
            urgency=data.get("urgency", 0.0),
            event_type=data.get("event_type", "other"),
            affected_sectors=data.get("affected_sectors", []),
            affected_assets=data.get("affected_assets", []),
            confidence=data.get("confidence", 0.5),
            source=source,
            timestamp=int(time.time() * 1000),
            follow_up_required=data.get("follow_up_required", False),
        )

    def _empty_dict(self) -> Dict[str, Any]:
        return {
            "headline": "",
            "summary": "",
            "impact_score": 0.0,
            "urgency": 0.0,
            "event_type": "other",
            "affected_sectors": [],
            "affected_assets": [],
            "confidence": 0.5,
            "follow_up_required": False,
        }
