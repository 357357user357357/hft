"""Tests for LLM agents with mocked responses.

These tests verify agent logic without making real API calls.
All LLM responses are mocked to ensure fast, deterministic tests.
"""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_agent import LLMConfig, ModelProvider, AgentMessage, AgentResponse
from agents.sentiment_agent import SentimentAgent, SentimentSignal
from agents.news_agent import NewsAgent, NewsSignal
from agents.risk_agent import RiskAgent, RiskSignal
from agents.fundamental_agent import FundamentalAgent, FundamentalSignal
from agents.trader_agent import TraderAgent, TradingDecision


class TestLLMConfig:
    """Test LLM configuration helpers."""

    def test_for_2x3090(self):
        config = LLMConfig.for_2x3090()
        assert config.provider == ModelProvider.VLLM
        assert "DeepSeek-Math-V2-70B" in config.model
        assert config.tensor_parallel == 2
        assert config.max_tokens == 512
        assert config.temperature == 0.05

    def test_for_cmp50hx(self):
        config = LLMConfig.for_cmp50hx()
        assert config.provider == ModelProvider.OLLAMA
        assert "mistral-nemo" in config.ollama_model
        assert config.max_tokens == 256

    def test_for_claude_api(self):
        config = LLMConfig.for_claude_api("test-key-123")
        assert config.provider == ModelProvider.ANTHROPIC
        assert config.anthropic_key == "test-key-123"
        assert config.model == "claude-sonnet-4-6"


class TestSentimentAgent:
    """Test sentiment analysis with mocked LLM responses."""

    @pytest.fixture
    def mock_config(self):
        return LLMConfig.for_claude_api("test-key")

    @pytest.fixture
    def mock_llm_response(self):
        return {
            "sentiment_score": 0.65,
            "sentiment_magnitude": 0.8,
            "recommendation": "bullish",
            "credibility_score": 0.75,
            "key_topics": ["BTC", "bullish"],
            "source_breakdown": {"twitter": 0.7},
            "time_decay_factor": 0.9,
        }

    @pytest.fixture
    def mock_agent(self, mock_config):
        with patch('agents.llm_agent.BaseAgent._init_client', return_value=Mock()):
            yield SentimentAgent(mock_config)

    def test_sentiment_agent_init(self, mock_agent):
        assert mock_agent is not None

    def test_analyze_text_mocked(self, mock_agent, mock_llm_response):
        mock_resp = AgentResponse(content="{}", structured=mock_llm_response)
        with patch.object(mock_agent, 'generate', return_value=mock_resp):
            result = mock_agent.analyze_text(["BTC breaking resistance"])

            assert isinstance(result, SentimentSignal)
            assert result.sentiment_score == 0.65
            assert result.recommendation == "bullish"

    def test_negative_sentiment(self, mock_agent):
        mock_resp = AgentResponse(content="{}", structured={
            "sentiment_score": -0.45,
            "sentiment_magnitude": 0.6,
            "recommendation": "bearish",
            "credibility_score": 0.8,
            "key_topics": ["crash"],
            "source_breakdown": {},
            "time_decay_factor": 0.9,
        })
        with patch.object(mock_agent, 'generate', return_value=mock_resp):
            result = mock_agent.analyze_text(["Market crash fears grow"])
            assert result.sentiment_score == -0.45
            assert result.recommendation == "bearish"


class TestNewsAgent:
    """Test news analysis with mocked LLM responses."""

    @pytest.fixture
    def mock_config(self):
        return LLMConfig.for_claude_api("test-key")

    @pytest.fixture
    def mock_agent(self, mock_config):
        with patch('agents.llm_agent.BaseAgent._init_client', return_value=Mock()):
            yield NewsAgent(mock_config)

    def test_news_agent_init(self, mock_agent):
        assert mock_agent is not None

    def test_analyze_headline_mocked(self, mock_agent):
        mock_resp = AgentResponse(content="{}", structured={
            "impact_score": 0.7,
            "urgency": "high",
            "event_type": "macroeconomic",
            "summary": "Fed announces rate decision",
            "affected_assets": ["BTC", "ETH", "SPX"],
        })
        with patch.object(mock_agent, 'generate', return_value=mock_resp):
            result = mock_agent.analyze_headline("Fed holds rates steady")
            assert isinstance(result, NewsSignal)
            assert result.impact_score == 0.7
            assert result.urgency == "high"


class TestRiskAgent:
    """Test risk assessment with mocked LLM responses."""

    @pytest.fixture
    def mock_config(self):
        return LLMConfig.for_claude_api("test-key")

    @pytest.fixture
    def mock_agent(self, mock_config):
        with patch('agents.llm_agent.BaseAgent._init_client', return_value=Mock()):
            yield RiskAgent(mock_config)

    def test_risk_agent_init(self, mock_agent):
        assert mock_agent is not None

    def test_assess_trade_mocked(self, mock_agent):
        mock_resp = AgentResponse(content="{}", structured={
            "risk_level": "moderate",
            "risk_score": 0.45,
            "max_position_size": 500.0,
            "recommended_leverage": 1.0,
            "risk_factors": ["Volatility spike"],
        })
        with patch.object(mock_agent, 'generate', return_value=mock_resp):
            result = mock_agent.assess_trade(
                symbol="BTCUSDT",
                signal_strength=0.6,
                current_volatility=0.25,
            )
            assert isinstance(result, RiskSignal)
            assert result.risk_score == 0.45


class TestFundamentalAgent:
    """Test fundamental analysis with mocked LLM responses."""

    @pytest.fixture
    def mock_config(self):
        return LLMConfig.for_claude_api("test-key")

    @pytest.fixture
    def mock_agent(self, mock_config):
        with patch('agents.llm_agent.BaseAgent._init_client', return_value=Mock()):
            yield FundamentalAgent(mock_config)

    def test_fundamental_agent_init(self, mock_agent):
        assert mock_agent is not None

    def test_analyze_earnings_mocked(self, mock_agent):
        mock_resp = AgentResponse(content="{}", structured={
            "fundamental_score": 0.72,
            "valuation": "undervalued",
            "growth_outlook": "positive",
            "financial_health": "strong",
        })
        with patch.object(mock_agent, 'generate', return_value=mock_resp):
            result = mock_agent.analyze_earnings_report("BTC", "Earnings beat expectations")
            assert isinstance(result, FundamentalSignal)
            assert result.fundamental_score == 0.72


class TestTraderAgent:
    """Test trader agent synthesis with mocked responses."""

    @pytest.fixture
    def mock_config(self):
        return LLMConfig.for_claude_api("test-key")

    @pytest.fixture
    def mock_agent(self, mock_config):
        with patch('agents.llm_agent.BaseAgent._init_client', return_value=Mock()):
            yield TraderAgent(mock_config)

    def test_trader_agent_init(self, mock_agent):
        assert mock_agent is not None

    def test_make_decision_mocked(self, mock_agent):
        from agents.trader_agent import SignalContext
        mock_resp = AgentResponse(content="{}", structured={
            "action": "buy",
            "symbol": "BTCUSDT",
            "size_usdt": 500.0,
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "take_profit": 52500.0,
            "confidence": 0.78,
            "reasoning": "Multiple signals align",
            "signal_sources": {"poincare": 0.6},
            "risk_adjusted": True,
        })
        with patch.object(mock_agent, 'generate', return_value=mock_resp):
            result = mock_agent.make_decision(
                symbol="BTCUSDT",
                context=SignalContext(),
                current_price=50000.0,
            )
            assert isinstance(result, TradingDecision)
            assert result.action == "buy"
            assert result.confidence == 0.78


class TestAgentMessage:
    """Test agent message formatting."""

    def test_agent_message_creation(self):
        msg = AgentMessage(role="user", content="Analyze sentiment")
        assert msg.role == "user"
        assert msg.content == "Analyze sentiment"

    def test_agent_response_creation(self):
        resp = AgentResponse(
            content="test",
            structured={"score": 0.75},
            latency_ms=150.0,
        )
        assert resp.content == "test"
        assert resp.structured["score"] == 0.75
        assert resp.latency_ms == 150.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
