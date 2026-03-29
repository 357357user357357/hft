"""Integration layer: connects LLM agents with existing HFT math signals.

Bridges:
- Poincaré topology (poincare_trading.py)
- Whitehead torsion (whitehead_signal.py)
- Hecke operators (hecke_operators.py)
- HFT algorithms (Shot, DepthShot, etc.)
- Risk management (risk_management.py)

Usage:
    from agents.integration import SignalIntegrator

    integrator = SignalIntegrator()

    # Get combined signal for a symbol
    combined = integrator.get_combined_signal(
        symbol="BTCUSDT",
        prices=[...],
        trades=[...],
    )
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

# Import math modules from repo root (agents/ is not a standalone package)
from poincare_trading import poincare_analysis, TopologyReport
from whitehead_signal import whitehead_analysis, WhiteheadReport
from hecke_operators import HeckeAlgebra

from .risk_agent import RiskSignal
from .sentiment_agent import SentimentSignal
from .news_agent import NewsSignal
from .fundamental_agent import FundamentalSignal


@dataclass
class CombinedSignal:
    """Unified signal from all sources (math + LLM agents)."""
    symbol: str
    action: str                    # "buy" | "sell" | "hold"
    confidence: float              # 0.0 to 1.0
    size_usdt: float

    # Math signals
    poincare_score: float          # -1.0 to +1.0
    poincare_regime: str           # "mean-reversion" | "trending" | "neutral"
    whitehead_signal: str          # "regime_change" | "same_regime"
    hecke_l_value: complex         # L-function value
    zeta_signal: Optional[str]     # "strong" | "moderate" | "weak"

    # LLM agent signals
    sentiment_score: float
    news_impact: float
    fundamental_score: float
    risk_adjusted: bool

    # Execution
    entry_price: float
    stop_loss: float
    take_profit: float

    # Metadata
    latency_ms: float
    signal_sources: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "size_usdt": self.size_usdt,
            "poincare_score": self.poincare_score,
            "poincare_regime": self.poincare_regime,
            "whitehead_signal": self.whitehead_signal,
            "hecke_l_value": str(self.hecke_l_value),
            "zeta_signal": self.zeta_signal,
            "sentiment_score": self.sentiment_score,
            "news_impact": self.news_impact,
            "fundamental_score": self.fundamental_score,
            "risk_adjusted": self.risk_adjusted,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "latency_ms": self.latency_ms,
            "signal_sources": self.signal_sources,
        }


class SignalIntegrator:
    """Integrates mathematical signals with LLM agent signals."""

    def __init__(self, hecke_max_n: int = 20, hecke_weight: int = 2):
        self.hecke_algebra = HeckeAlgebra(max_n=hecke_max_n, weight=hecke_weight)
        self._last_poincare: Optional[TopologyReport] = None
        self._last_whitehead: Optional[WhiteheadReport] = None

    def get_combined_signal(
        self,
        symbol: str,
        prices: List[float],
        current_price: float,
        sentiment: Optional[SentimentSignal] = None,
        news: Optional[NewsSignal] = None,
        fundamental: Optional[FundamentalSignal] = None,
        risk: Optional[RiskSignal] = None,
        initial_equity: float = 10000.0,
    ) -> CombinedSignal:
        """
        Combine all signals into a unified trading decision.

        Signal weighting:
        - Poincaré topology: 25%
        - Whitehead torsion: 15%
        - Hecke L-function: 10%
        - Sentiment: 20%
        - News: 15%
        - Fundamental: 15%
        """
        import time
        start = time.perf_counter()

        # 1. Poincaré topology analysis
        poincare = poincare_analysis(prices, embed_dim=3, subsample=60)
        self._last_poincare = poincare

        # 2. Whitehead torsion analysis
        whitehead = whitehead_analysis(prices, embed_dim=3, subsample=60)
        self._last_whitehead = whitehead

        # 3. Hecke L-function analysis
        self.hecke_algebra.set_eigenvalues_from_prices(prices)
        l_value = self.hecke_algebra.l_function_value(prices, s=0.5)
        zeta_sig = self.hecke_algebra.zeta_signal(prices)

        # 4. Convert to normalized scores
        poincare_score = poincare.poincare_score  # Already -1 to +1

        # Whitehead: regime_change = uncertainty = reduce confidence
        whitehead_score = 0.5 if whitehead.signal == "same_regime" else -0.3

        # Hecke: |L(1/2)| magnitude as signal strength
        hecke_score = min(abs(l_value) / 2.0, 1.0)  # Normalize to 0-1
        if zeta_sig == "weak":
            hecke_score *= -0.5  # Near zero = weak signal

        # 5. LLM agent scores
        sentiment_score = sentiment.sentiment_score if sentiment else 0.0
        news_score = news.impact_score if news else 0.0
        fundamental_score = fundamental.fundamental_score if fundamental else 0.0

        # 6. Weighted combination
        weights = {
            "poincare": 0.25,
            "whitehead": 0.15,
            "hecke": 0.10,
            "sentiment": 0.20,
            "news": 0.15,
            "fundamental": 0.15,
        }

        scores = {
            "poincare": poincare_score,
            "whitehead": whitehead_score,
            "hecke": hecke_score * (1 if l_value.real > 0 else -1),
            "sentiment": sentiment_score,
            "news": news_score,
            "fundamental": fundamental_score,
        }

        combined_score = sum(scores[k] * weights[k] for k in weights.keys())

        # 7. Risk adjustment
        if risk:
            combined_score *= (1.0 - risk.risk_score * 0.5)  # Reduce by up to 50%
            size_usdt = risk.max_position_size
        else:
            size_usdt = initial_equity * 0.05  # Default 5%

        # 8. Action decision
        if combined_score > 0.25:
            action = "buy"
        elif combined_score < -0.25:
            action = "sell"
        else:
            action = "hold"

        # 9. Position sizing by confidence
        confidence = min(abs(combined_score) * 2, 1.0)
        size_usdt *= confidence

        # 10. Stop loss / take profit
        if action == "buy":
            stop_loss = current_price * 0.98
            take_profit = current_price * 1.05
        elif action == "sell":
            stop_loss = current_price * 1.02
            take_profit = current_price * 0.95
        else:
            stop_loss = current_price
            take_profit = current_price

        latency_ms = (time.perf_counter() - start) * 1000

        return CombinedSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            size_usdt=size_usdt,
            poincare_score=poincare_score,
            poincare_regime=poincare.regime,
            whitehead_signal=whitehead.signal,
            hecke_l_value=l_value,
            zeta_signal=zeta_sig,
            sentiment_score=sentiment_score,
            news_impact=news_score,
            fundamental_score=fundamental_score,
            risk_adjusted=risk is not None,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            latency_ms=latency_ms,
            signal_sources={k: scores[k] for k in weights.keys()},
        )

    def get_poincare_report(self) -> Optional[Dict[str, Any]]:
        """Get last Poincaré analysis report."""
        if self._last_poincare is None:
            return None
        return {
            "regime": self._last_poincare.regime,
            "poincare_score": self._last_poincare.poincare_score,
            "mean_ricci": self._last_poincare.mean_ricci,
            "neg_ricci_frac": self._last_poincare.neg_ricci_frac,
            "beta0": self._last_poincare.beta0,
            "beta1": self._last_poincare.beta1,
            "beta2": self._last_poincare.beta2,
            "simply_connected": self._last_poincare.simply_connected,
        }

    def get_whitehead_report(self) -> Optional[Dict[str, Any]]:
        """Get last Whitehead analysis report."""
        if self._last_whitehead is None:
            return None
        return {
            "signal": self._last_whitehead.signal,
            "torsion_ratio": self._last_whitehead.torsion.torsion_ratio,
            "is_simple": self._last_whitehead.torsion.is_simple,
            "beta1_changes": self._last_whitehead.torsion.beta1_changes,
            "beta2_changes": self._last_whitehead.torsion.beta2_changes,
            "num_regimes": self._last_whitehead.reeb.num_regimes,
            "poincare_check": self._last_whitehead.poincare_check,
        }


def create_llm_config_for_hardware(hardware: str = "2x3090"):
    """Get optimal LLM config for your hardware setup."""
    from .llm_agent import LLMConfig, ModelProvider

    if hardware == "2x3090":
        # 2×RTX 3090 (48GB total) — run DeepSeek-Math-V2-70B with tensor parallel
        return LLMConfig(
            provider=ModelProvider.VLLM,
            model="deepseek-ai/DeepSeek-Math-V2-70B-Instruct",
            vllm_url="http://localhost:8000/v1",
            tensor_parallel=2,
            max_tokens=512,
            temperature=0.05,
        )
    elif hardware == "cmp50hx":
        # CMP 50HX 10GB — all weights in VRAM, use smaller model
        return LLMConfig(
            provider=ModelProvider.OLLAMA,
            ollama_url="http://localhost:11434",
            ollama_model="mistral-nemo:latest",  # ~8GB Q4
            max_tokens=256,
            temperature=0.1,
        )
    elif hardware == "cloud":
        # Fallback to Claude API
        import os
        return LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-sonnet-4-6",
            anthropic_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=512,
            temperature=0.1,
        )
    else:
        raise ValueError(f"Unknown hardware: {hardware}")
