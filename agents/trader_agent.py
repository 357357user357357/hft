"""Trader Agent — synthesizes all signals and makes final trading decisions.

Orchestrates:
- Sentiment signals (social media, news)
- News signals (macroeconomic, geopolitical)
- Fundamental signals (earnings, financials)
- Technical signals (from HFT algorithms)
- Topological signals (Poincaré, Whitehead)
- Risk assessment (position sizing, stops)

Outputs:
- action: "buy" | "sell" | "hold"
- size: Position size in USDT
- stop_loss: Stop loss price
- take_profit: Take profit price
- confidence: Decision confidence 0-1
- reasoning: Human-readable explanation
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .llm_agent import BaseAgent, LLMConfig, AgentMessage, AgentResponse
from .sentiment_agent import SentimentAgent, SentimentSignal
from .news_agent import NewsAgent, NewsSignal
from .fundamental_agent import FundamentalAgent, FundamentalSignal
from .risk_agent import RiskAgent, RiskSignal


class Action(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingDecision:
    """Final trading decision from the orchestrator."""
    action: str                  # "buy" | "sell" | "hold"
    symbol: str
    size_usdt: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float            # 0.0 to 1.0
    reasoning: str
    signal_sources: Dict[str, float]  # source -> contribution
    risk_adjusted: bool
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "symbol": self.symbol,
            "size_usdt": self.size_usdt,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "signal_sources": self.signal_sources,
            "risk_adjusted": self.risk_adjusted,
            "latency_ms": self.latency_ms,
        }


@dataclass
class SignalContext:
    """Aggregated signals from all sources."""
    sentiment: Optional[SentimentSignal] = None
    news: Optional[NewsSignal] = None
    fundamental: Optional[FundamentalSignal] = None
    technical: Optional[Dict[str, Any]] = None  # From HFT algos
    topological: Optional[Dict[str, Any]] = None  # Poincaré/Whitehead
    risk: Optional[RiskSignal] = None


class TraderAgent(BaseAgent):
    """Synthesizes all signals and makes final trading decisions."""

    def __init__(self, config: LLMConfig):
        super().__init__("TraderAgent", config)

    def get_system_prompt(self) -> str:
        return """You are a Trading Decision Agent for HFT trading.

Your role:
1. Synthesize signals from sentiment, news, fundamentals, technical, and topological analysis
2. Make a clear trading decision: BUY, SELL, or HOLD
3. Set position size, stop loss, and take profit levels
4. Provide confidence score and reasoning

Decision framework:
- Strong BUY: Multiple bullish signals, low risk, high confidence
- BUY: Mostly bullish signals, acceptable risk
- HOLD: Mixed signals, high risk, or unclear direction
- SELL: Mostly bearish signals, acceptable risk
- Strong SELL: Multiple bearish signals, low risk, high confidence

Position sizing:
- Use risk agent's recommended max position size
- Scale size by confidence (higher confidence = larger position)

Always output valid JSON matching this schema:
{
    "action": "buy" | "sell" | "hold",
    "size_usdt": float,
    "entry_price": float,
    "stop_loss": float,
    "take_profit": float,
    "confidence": float,
    "reasoning": string,
    "signal_sources": {"sentiment": float, "news": float, "technical": float, ...}
}"""

    def parse_response(self, raw: str) -> Dict[str, Any]:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return self._empty_dict()

    def make_decision(
        self,
        symbol: str,
        context: SignalContext,
        current_price: float,
    ) -> TradingDecision:
        """Make a trading decision based on all available signals."""
        import time
        start = time.perf_counter()

        # Build comprehensive prompt
        prompt = self._build_prompt(symbol, context, current_price)

        messages = [AgentMessage(role="user", content=prompt)]
        response = self.generate(messages)

        data = response.structured or self._empty_dict()
        latency_ms = (time.perf_counter() - start) * 1000

        # Calculate stop loss and take profit prices
        action = data.get("action", "hold")
        entry = data.get("entry_price", current_price)
        sl_pct = data.get("stop_loss", 0.02)
        tp_pct = data.get("take_profit", 0.05)

        if action == "buy":
            stop_loss = entry * (1 - sl_pct)
            take_profit = entry * (1 + tp_pct)
        elif action == "sell":
            stop_loss = entry * (1 + sl_pct)
            take_profit = entry * (1 - tp_pct)
        else:
            stop_loss = entry
            take_profit = entry

        return TradingDecision(
            action=action,
            symbol=symbol,
            size_usdt=data.get("size_usdt", 0),
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", "No reasoning provided"),
            signal_sources=data.get("signal_sources", {}),
            risk_adjusted=context.risk is not None,
            latency_ms=latency_ms,
        )

    def _build_prompt(self, symbol: str, context: SignalContext, price: float) -> str:
        """Build comprehensive prompt from all signal sources."""
        parts = [f"Make a trading decision for {symbol} at price ${price}.\n"]

        if context.sentiment:
            s = context.sentiment
            parts.append(f"""Sentiment Analysis:
- Score: {s.sentiment_score:.2f} (magnitude: {s.sentiment_magnitude:.2f})
- Recommendation: {s.recommendation}
- Credibility: {s.credibility_score:.2f}
- Topics: {', '.join(s.key_topics) if s.key_topics else 'None'}
""")

        if context.news:
            n = context.news
            parts.append(f"""News Analysis:
- Impact: {n.impact_score:.2f}, Urgency: {n.urgency:.2f}
- Event Type: {n.event_type}
- Summary: {n.summary}
""")

        if context.fundamental:
            f = context.fundamental
            parts.append(f"""Fundamental Analysis:
- Score: {f.fundamental_score:.2f}
- Recommendation: {f.recommendation}
- Quality: {f.quality_score:.2f}, Valuation: {f.valuation_score:.2f}
""")

        if context.technical:
            t = context.technical
            parts.append(f"""Technical Signals:
- Poincaré regime: {t.get('poincare_regime', 'unknown')}
- Whitehead signal: {t.get('whitehead_signal', 'unknown')}
- HFT algo signals: {t.get('hft_signals', 'none')}
""")

        if context.risk:
            r = context.risk
            parts.append(f"""Risk Assessment:
- Risk level: {r.risk_level}
- Max position: ${r.max_position_size:.2f} ({r.position_size_pct:.1%})
- VaR 95%: {r.var_95:.2%}
""")

        parts.append("\nOutput your trading decision as JSON:")
        return "\n".join(parts)

    def _empty_dict(self) -> Dict[str, Any]:
        return {
            "action": "hold",
            "size_usdt": 0,
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "confidence": 0.5,
            "reasoning": "Insufficient signal data",
            "signal_sources": {},
        }


class TradingAgentsOrchestrator:
    """
    Orchestrates all trading agents in a multi-agent collaboration.

    Usage:
        config = LLMConfig.for_2x3090()
        orchestrator = TradingAgentsOrchestrator(config)

        # Run full analysis pipeline
        decision = await orchestrator.run_analysis("BTCUSDT", current_price=50000)

        # Or run individual agents
        sentiment = orchestrator.sentiment.analyze_text([...])
        news = orchestrator.news.analyze_headline(...)
    """

    def __init__(self, config: LLMConfig, initial_equity: float = 10000.0):
        self.config = config
        self.sentiment = SentimentAgent(config)
        self.news = NewsAgent(config)
        self.fundamental = FundamentalAgent(config)
        self.risk = RiskAgent(config, initial_equity)
        self.trader = TraderAgent(config)

        self._signal_history: Dict[str, List[SignalContext]] = {}

    def get_context(self, symbol: str) -> SignalContext:
        """Get current signal context for a symbol."""
        history = self._signal_history.get(symbol, [])
        if history:
            return history[-1]
        return SignalContext()

    async def run_analysis(
        self,
        symbol: str,
        current_price: float,
        texts: List[str] = None,
        headlines: List[Dict[str, Any]] = None,
        technical_data: Dict[str, Any] = None,
    ) -> TradingDecision:
        """Run full multi-agent analysis pipeline."""
        context = SignalContext()

        # 1. Sentiment analysis (if texts provided)
        if texts:
            context.sentiment = self.sentiment.analyze_text(texts, symbol)

        # 2. News analysis (if headlines provided)
        if headlines:
            for h in headlines:
                context.news = self.news.analyze_headline(
                    h.get("headline", ""),
                    h.get("body", ""),
                    h.get("source", "unknown"),
                )

        # 3. Technical/topological signals
        if technical_data:
            context.technical = technical_data

        # 4. Risk assessment
        signal_strength = self._aggregate_signal_strength(context)
        context.risk = self.risk.assess_trade(symbol, signal_strength)

        # 5. Store context
        if symbol not in self._signal_history:
            self._signal_history[symbol] = []
        self._signal_history[symbol].append(context)
        if len(self._signal_history[symbol]) > 100:
            self._signal_history[symbol] = self._signal_history[symbol][-100:]

        # 6. Trading decision
        decision = self.trader.make_decision(symbol, context, current_price)

        return decision

    def _aggregate_signal_strength(self, context: SignalContext) -> float:
        """Aggregate all signals into a single strength score."""
        scores = []
        weights = []

        if context.sentiment:
            scores.append(context.sentiment.sentiment_score)
            weights.append(context.sentiment.credibility_score * 0.25)

        if context.news:
            scores.append(context.news.impact_score)
            weights.append(context.news.urgency * 0.25)

        if context.fundamental:
            scores.append(context.fundamental.fundamental_score)
            weights.append(context.fundamental.confidence * 0.25)

        if context.technical:
            # Extract from Poincaré score
            poincare = context.technical.get("poincare_score", 0)
            scores.append(poincare)
            weights.append(0.25)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        return sum(s * w for s, w in zip(scores, weights)) / total_weight
