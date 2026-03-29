"""Fundamental Agent — analyzes company financials and fundamentals for trading.

Data sources:
- Earnings reports (10-Q, 10-K)
- Financial statements (income, balance sheet, cash flow)
- Analyst estimates and revisions
- Insider trading filings (Form 4)
- Institutional ownership (13F)

Outputs:
- fundamental_score: -1.0 (very bearish) to +1.0 (very bullish)
- quality_metrics: Revenue growth, margin trends, balance sheet health
- valuation_metrics: P/E, EV/EBITDA, PEG vs. peers
- catalyst_events: Upcoming earnings, product launches, FDA decisions
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .llm_agent import BaseAgent, LLMConfig, AgentMessage, AgentResponse


@dataclass
class FundamentalSignal:
    """Structured fundamental analysis output."""
    symbol: str
    fundamental_score: float    # -1.0 to +1.0
    quality_score: float        # 0.0 to 1.0 (business quality)
    valuation_score: float      # 0.0 to 1.0 (1.0 = very undervalued)
    growth_score: float         # 0.0 to 1.0 (growth prospects)
    financial_health: float     # 0.0 to 1.0 (balance sheet strength)
    key_metrics: Dict[str, Any]
    peer_comparison: Dict[str, Any]
    catalyst_events: List[str]
    risks: List[str]
    recommendation: str         # "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
    confidence: float           # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "fundamental_score": self.fundamental_score,
            "quality_score": self.quality_score,
            "valuation_score": self.valuation_score,
            "growth_score": self.growth_score,
            "financial_health": self.financial_health,
            "key_metrics": self.key_metrics,
            "peer_comparison": self.peer_comparison,
            "catalyst_events": self.catalyst_events,
            "risks": self.risks,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
        }


class FundamentalAgent(BaseAgent):
    """Analyzes company financials and fundamentals."""

    def __init__(self, config: LLMConfig):
        super().__init__("FundamentalAgent", config)
        self._company_data: Dict[str, Any] = {}
        self._peer_groups: Dict[str, List[str]] = {}

    def get_system_prompt(self) -> str:
        return """You are a Fundamental Analysis Agent for HFT trading.

Your role:
1. Analyze company financial statements and earnings reports
2. Assess business quality, valuation, and growth prospects
3. Compare against peer companies
4. Identify catalyst events and risks
5. Output structured JSON for trading decisions

Scoring:
- fundamental_score: Overall -1.0 (terrible) to +1.0 (excellent)
- quality_score: Business quality 0.0-1.0 (moats, margins, ROIC)
- valuation_score: 0.0-1.0 (1.0 = very undervalued vs. peers/history)
- growth_score: 0.0-1.0 (revenue/earnings growth prospects)
- financial_health: 0.0-1.0 (balance sheet, cash flow, debt)

Recommendations:
- strong_buy: Fundamental score > 0.7, all scores high
- buy: Fundamental score > 0.3
- hold: Fundamental score -0.3 to 0.3
- sell: Fundamental score < -0.3
- strong_sell: Fundamental score < -0.7

Always output valid JSON matching this schema:
{
    "fundamental_score": float,
    "quality_score": float,
    "valuation_score": float,
    "growth_score": float,
    "financial_health": float,
    "key_metrics": {"metric": value, ...},
    "peer_comparison": {"metric": {"company": value, "peer_avg": value}, ...},
    "catalyst_events": [string],
    "risks": [string],
    "recommendation": string,
    "confidence": float
}"""

    def parse_response(self, raw: str) -> Dict[str, Any]:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return self._empty_dict()

    def analyze_earnings_report(self, symbol: str, report_text: str) -> FundamentalSignal:
        """Analyze an earnings report (10-Q/10-K or earnings release)."""
        prompt = f"""Analyze this earnings report for {symbol}:

{report_text}

Focus on:
- Revenue growth (YoY, QoQ)
- Margin trends (gross, operating, net)
- EPS vs. estimates
- Guidance (raised/lowered)
- Management commentary tone

Output JSON fundamental analysis:"""

        messages = [AgentMessage(role="user", content=prompt)]
        response = self.generate(messages)

        data = response.structured or self._empty_dict()
        return self._signal_from_dict(symbol, data)

    def analyze_financial_statements(
        self,
        symbol: str,
        income_statement: Dict[str, Any],
        balance_sheet: Dict[str, Any],
        cash_flow: Dict[str, Any],
    ) -> FundamentalSignal:
        """Analyze full financial statements."""
        prompt = f"""Analyze these financial statements for {symbol}:

Income Statement:
{json.dumps(income_statement, indent=2)}

Balance Sheet:
{json.dumps(balance_sheet, indent=2)}

Cash Flow:
{json.dumps(cash_flow, indent=2)}

Calculate and assess:
- Profitability: Gross margin, operating margin, net margin, ROE, ROIC
- Growth: Revenue growth, earnings growth
- Health: Debt/Equity, current ratio, free cash flow
- Efficiency: Asset turnover, inventory turns

Output JSON fundamental analysis:"""

        messages = [AgentMessage(role="user", content=prompt)]
        response = self.generate(messages)

        data = response.structured or self._empty_dict()
        data["key_metrics"] = {
            **income_statement,
            **balance_sheet,
            **cash_flow,
        }
        return self._signal_from_dict(symbol, data)

    def set_company_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Store company data for context."""
        self._company_data[symbol] = data

    def set_peer_group(self, symbol: str, peers: List[str]) -> None:
        """Define peer group for comparison."""
        self._peer_groups[symbol] = peers

    def quick_analysis(self, symbol: str, metrics: Dict[str, Any]) -> FundamentalSignal:
        """Quick fundamental analysis from key metrics."""
        prompt = f"""Quick fundamental analysis for {symbol}:

Key metrics:
{json.dumps(metrics, indent=2)}

Peer group: {self._peer_groups.get(symbol, "Not specified")}

Output JSON fundamental analysis:"""

        messages = [AgentMessage(role="user", content=prompt)]
        response = self.generate(messages)

        data = response.structured or self._empty_dict()
        data["key_metrics"] = metrics
        return self._signal_from_dict(symbol, data)

    def get_fundamental_trend(self, symbol: str, window: int = 5) -> Dict[str, float]:
        """Get fundamental score trend for a symbol."""
        if symbol not in self._company_data:
            return {"trend": 0.0, "revision_direction": 0.0}

        data = self._company_data[symbol]
        history = data.get("score_history", [])[-window:]
        if not history:
            return {"trend": 0.0, "revision_direction": 0.0}

        scores = [h.get("fundamental_score", 0.0) for h in history]
        trend = (scores[-1] - scores[0]) / len(scores) if len(scores) > 1 else 0.0

        return {
            "trend": trend,
            "revision_direction": 1.0 if trend > 0 else -1.0 if trend < 0 else 0.0,
        }

    def _signal_from_dict(self, symbol: str, data: Dict[str, Any]) -> FundamentalSignal:
        return FundamentalSignal(
            symbol=symbol,
            fundamental_score=data.get("fundamental_score", 0.0),
            quality_score=data.get("quality_score", 0.5),
            valuation_score=data.get("valuation_score", 0.5),
            growth_score=data.get("growth_score", 0.5),
            financial_health=data.get("financial_health", 0.5),
            key_metrics=data.get("key_metrics", {}),
            peer_comparison=data.get("peer_comparison", {}),
            catalyst_events=data.get("catalyst_events", []),
            risks=data.get("risks", []),
            recommendation=data.get("recommendation", "hold"),
            confidence=data.get("confidence", 0.5),
        )

    def _empty_dict(self) -> Dict[str, Any]:
        return {
            "fundamental_score": 0.0,
            "quality_score": 0.5,
            "valuation_score": 0.5,
            "growth_score": 0.5,
            "financial_health": 0.5,
            "key_metrics": {},
            "peer_comparison": {},
            "catalyst_events": [],
            "risks": [],
            "recommendation": "hold",
            "confidence": 0.5,
        }
