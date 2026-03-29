"""Multi-agent LLM system for TradingAgents-style analysis + HFT math integration.

Agents:
- SentimentAgent: Social media, news sentiment scoring
- NewsAgent: Macroeconomic news parsing
- FundamentalAgent: Earnings, financial statement analysis
- RiskAgent: Portfolio risk assessment
- TraderAgent: Final trade decision synthesizing all signals

Integration:
- SignalIntegrator: Combines LLM agents with Poincaré/Whitehead/Hecke math signals

Hardware configs:
- 2×RTX 3090: DeepSeek-Math-V2-70B via vLLM (tensor parallel)
- CMP 50HX 10GB: Mistral-Nemo-12B via Ollama (all weights in VRAM)
- Cloud: Claude API fallback

Usage:
    from agents import (
        LLMConfig,
        SentimentAgent,
        TradingAgentsOrchestrator,
        SignalIntegrator,
        create_llm_config_for_hardware,
    )

    # For 2×3090
    config = create_llm_config_for_hardware("2x3090")
    orchestrator = TradingAgentsOrchestrator(config)

    # For CMP 50HX
    config = create_llm_config_for_hardware("cmp50hx")

    # Full integration with math signals
    integrator = SignalIntegrator()
    signal = integrator.get_combined_signal(
        symbol="BTCUSDT",
        prices=[...],
        current_price=50000,
        sentiment=sentiment_signal,
    )
"""

from .llm_agent import LLMConfig, ModelProvider, BaseAgent, AgentMessage, AgentResponse
from .sentiment_agent import SentimentAgent
from .news_agent import NewsAgent
from .fundamental_agent import FundamentalAgent
from .risk_agent import RiskAgent
from .trader_agent import TraderAgent, TradingAgentsOrchestrator
from .integration import SignalIntegrator, CombinedSignal, create_llm_config_for_hardware

__all__ = [
    # Config
    "LLMConfig",
    "ModelProvider",
    "create_llm_config_for_hardware",
    # Base
    "BaseAgent",
    "AgentMessage",
    "AgentResponse",
    # Agents
    "SentimentAgent",
    "NewsAgent",
    "FundamentalAgent",
    "RiskAgent",
    "TraderAgent",
    "TradingAgentsOrchestrator",
    # Integration
    "SignalIntegrator",
    "CombinedSignal",
]
