"""Risk Agent — assesses portfolio risk and validates trading decisions.

Monitors:
- Position concentration risk
- Correlation exposure
- VaR (Value at Risk)
- Drawdown limits
- Liquidity risk
- Volatility regimes

Outputs:
- risk_level: "low" | "medium" | "high" | "critical"
- max_position_size: Maximum allowed position size
- risk_adjusted_signal: Original signal adjusted for risk
- stop_loss_recommendation: Suggested stop loss level
"""

from __future__ import annotations
import json
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .llm_agent import BaseAgent, LLMConfig, AgentMessage, AgentResponse


@dataclass
class RiskSignal:
    """Structured risk assessment output."""
    risk_level: str              # "low" | "medium" | "high" | "critical"
    risk_score: float            # 0.0 to 1.0
    max_position_size: float     # USDT
    position_size_pct: float     # % of portfolio
    var_95: float                # 95% Value at Risk (%)
    max_drawdown_allowed: float  # % before forced liquidation
    correlation_warning: bool    # True if too correlated
    concentration_warning: bool  # True if too concentrated
    volatility_regime: str       # "low" | "normal" | "high"
    stop_loss_recommendation: float  # % stop loss
    take_profit_recommendation: float  # % take profit
    risk_adjusted_signal: float  # Original signal * risk adjustment

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "max_position_size": self.max_position_size,
            "position_size_pct": self.position_size_pct,
            "var_95": self.var_95,
            "max_drawdown_allowed": self.max_drawdown_allowed,
            "correlation_warning": self.correlation_warning,
            "concentration_warning": self.concentration_warning,
            "volatility_regime": self.volatility_regime,
            "stop_loss_recommendation": self.stop_loss_recommendation,
            "take_profit_recommendation": self.take_profit_recommendation,
            "risk_adjusted_signal": self.risk_adjusted_signal,
        }


class RiskAgent(BaseAgent):
    """Assesses portfolio risk and validates trading decisions."""

    def __init__(self, config: LLMConfig, initial_equity: float = 10000.0):
        super().__init__("RiskAgent", config)
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.positions: List[Dict[str, Any]] = []
        self.pnl_history: List[float] = []
        self._volatility_history: List[float] = []

    def get_system_prompt(self) -> str:
        return """You are a Risk Management Agent for HFT trading.

Your role:
1. Assess portfolio risk before each trade
2. Set position size limits based on current risk
3. Recommend stop-loss and take-profit levels
4. Adjust trading signals for risk
5. Monitor drawdown and volatility regimes

Risk levels:
- low: Risk score < 0.3, normal market conditions
- medium: Risk score 0.3-0.6, elevated volatility
- high: Risk score 0.6-0.8, stressed conditions
- critical: Risk score > 0.8, reduce positions immediately

Position sizing rules:
- Low risk: Up to 10% of portfolio per trade
- Medium risk: Up to 5% of portfolio per trade
- High risk: Up to 2% of portfolio per trade
- Critical: No new positions, reduce existing

Always output valid JSON matching this schema:
{
    "risk_level": string,
    "risk_score": float,
    "max_position_size": float,
    "position_size_pct": float,
    "var_95": float,
    "max_drawdown_allowed": float,
    "correlation_warning": boolean,
    "concentration_warning": boolean,
    "volatility_regime": string,
    "stop_loss_recommendation": float,
    "take_profit_recommendation": float,
    "risk_adjusted_signal": float
}"""

    def parse_response(self, raw: str) -> Dict[str, Any]:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return self._empty_dict()

    def assess_trade(
        self,
        symbol: str,
        signal_strength: float,  # -1.0 to +1.0 from other agents
        current_volatility: float = None,
    ) -> RiskSignal:
        """Assess risk for a potential trade."""
        # Calculate current portfolio metrics
        portfolio_vol = self._calculate_portfolio_volatility()
        current_drawdown = self._calculate_drawdown()
        var_95 = self._calculate_var_95()

        # Determine volatility regime
        if current_volatility is None:
            current_volatility = portfolio_vol

        if current_volatility < 0.01:
            vol_regime = "low"
        elif current_volatility < 0.03:
            vol_regime = "normal"
        else:
            vol_regime = "high"

        # Build context for LLM
        prompt = f"""Assess risk for this potential trade:

Symbol: {symbol}
Signal strength: {signal_strength:.2f}
Current equity: ${self.current_equity:.2f}
Initial equity: ${self.initial_equity:.2f}
Current drawdown: {current_drawdown:.2%}
Portfolio volatility: {portfolio_vol:.2%}
Volatility regime: {vol_regime}
95% VaR: {var_95:.2%}
Open positions: {len(self.positions)}

Output JSON risk assessment:"""

        messages = [AgentMessage(role="user", content=prompt)]
        response = self.generate(messages)

        data = response.structured or self._empty_dict()
        return self._signal_from_dict(data, signal_strength)

    def update_position(self, pnl_pct: float) -> None:
        """Record a completed trade's PnL."""
        self.pnl_history.append(pnl_pct)
        self.current_equity *= (1 + pnl_pct / 100.0)

        # Keep only recent history
        if len(self.pnl_history) > 1000:
            self.pnl_history = self.pnl_history[-1000:]

    def set_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Set current portfolio positions."""
        self.positions = positions

    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility from recent PnL."""
        if len(self.pnl_history) < 10:
            return 0.02  # Default 2%

        import statistics
        return statistics.stdev(self.pnl_history[-100:])

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak equity."""
        if not self.pnl_history:
            return 0.0

        # Calculate running peak
        cumulative = [self.initial_equity]
        for pnl in self.pnl_history:
            cumulative.append(cumulative[-1] * (1 + pnl / 100.0))

        peak = max(cumulative)
        current = cumulative[-1]
        return (peak - current) / peak

    def _calculate_var_95(self) -> float:
        """Calculate 95% Value at Risk."""
        if len(self.pnl_history) < 20:
            return 0.05  # Default 5%

        sorted_pnl = sorted(self.pnl_history)
        var_index = int(len(sorted_pnl) * 0.05)
        return abs(sorted_pnl[var_index])

    def get_risk_limits(self) -> Dict[str, Any]:
        """Get current risk limits."""
        drawdown = self._calculate_drawdown()
        vol = self._calculate_portfolio_volatility()

        # Position size limits based on drawdown
        if drawdown > 0.15:
            max_pct = 0.01  # 1% per trade
        elif drawdown > 0.10:
            max_pct = 0.02
        elif drawdown > 0.05:
            max_pct = 0.03
        else:
            max_pct = 0.05

        return {
            "max_position_pct": max_pct,
            "max_total_exposure": 0.20 if drawdown < 0.10 else 0.10,
            "max_daily_loss": 0.03,  # 3% daily loss limit
            "max_drawdown": 0.20,     # 20% max drawdown before halt
            "current_drawdown": drawdown,
            "portfolio_volatility": vol,
        }

    def _signal_from_dict(self, data: Dict[str, Any], original_signal: float) -> RiskSignal:
        risk_level = data.get("risk_level", "medium")

        # Position size based on risk level
        size_map = {
            "low": 0.10,
            "medium": 0.05,
            "high": 0.02,
            "critical": 0.0,
        }
        position_pct = data.get("position_size_pct", size_map.get(risk_level, 0.05))
        max_position = self.current_equity * position_pct

        # Risk adjustment to signal
        risk_score = data.get("risk_score", 0.5)
        risk_adjustment = 1.0 - risk_score  # Higher risk = lower signal
        adjusted_signal = original_signal * risk_adjustment

        return RiskSignal(
            risk_level=risk_level,
            risk_score=risk_score,
            max_position_size=max_position,
            position_size_pct=position_pct,
            var_95=data.get("var_95", 0.05),
            max_drawdown_allowed=data.get("max_drawdown_allowed", 0.20),
            correlation_warning=data.get("correlation_warning", False),
            concentration_warning=data.get("concentration_warning", False),
            volatility_regime=data.get("volatility_regime", "normal"),
            stop_loss_recommendation=data.get("stop_loss_recommendation", 0.02),
            take_profit_recommendation=data.get("take_profit_recommendation", 0.05),
            risk_adjusted_signal=adjusted_signal,
        )

    def _empty_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": "medium",
            "risk_score": 0.5,
            "max_position_size": self.current_equity * 0.05,
            "position_size_pct": 0.05,
            "var_95": 0.05,
            "max_drawdown_allowed": 0.20,
            "correlation_warning": False,
            "concentration_warning": False,
            "volatility_regime": "normal",
            "stop_loss_recommendation": 0.02,
            "take_profit_recommendation": 0.05,
            "risk_adjusted_signal": 0.0,
        }
