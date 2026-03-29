# TradingAgents for HFT — Multi-Agent LLM System

Implementation of [TradingAgents](https://github.com/TauricResearch/TradingAgents)-style multi-agent collaboration for HFT trading, integrated with advanced mathematical signals (Poincaré topology, Whitehead torsion, Hecke operators).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TradingAgents Orchestrator                    │
├─────────────────────────────────────────────────────────────────┤
│  SentimentAgent  │  NewsAgent  │  FundamentalAgent  │  Risk   │
│  (social media)  │  (macro)    │  (earnings)        │  Agent  │
└────────┬─────────┴──────┬──────┴─────────┬──────────┴────┬────┘
         │                │                │               │
         └────────────────┴────────────────┴───────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
   ┌─────▼─────┐     ┌───────▼───────┐    ┌──────▼──────┐
   │ Poincaré  │     │ Whitehead     │    │ Hecke       │
   │ Topology  │     │ Torsion/Mapper│    │ L-function  │
   └───────────┘     └───────────────┘    └─────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Trading Decision │
                    │  (action/size/SL) │
                    └───────────────────┘
```

---

## Hardware Support

| Hardware | Model | VRAM | Inference |
|----------|-------|------|-----------|
| **CMP 50HX 10GB** | Mistral-Nemo-12B (Q4) | ~8GB | Ollama (local) |
| **2×RTX 3090** | DeepSeek-Math-V2-70B | ~45GB | vLLM (tensor parallel) |
| **Cloud** | Claude 4.6 / GPT-4 | N/A | API |

### Your Setup

**CMP 50HX 10GB (available now):**
- All weights fit in VRAM (no PCIe bottleneck)
- Use Ollama with `mistral-nemo:12b` (~8GB Q4)
- Latency: ~50-200ms per agent call

**2×RTX 3090 (when rented):**
- Tensor parallel across 2 GPUs
- Use vLLM with `DeepSeek-Math-V2-70B-Instruct`
- Latency: ~100-500ms per agent call
- Much better reasoning for complex analysis

---

## Quick Start

### 1. Install Dependencies

```bash
# For CMP 50HX (Ollama)
pip install ollama

# For 2×3090 (vLLM) — install when you rent GPUs
pip install vllm

# For cloud API
pip install anthropic openai
```

### 2. Start LLM Server

**CMP 50HX:**
```bash
# Download Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model (first time only)
ollama pull mistral-nemo:12b

# Start server
ollama serve
```

**2×3090:**
```bash
./scripts/start_vllm_2x3090.sh
```

### 3. Run Example

```bash
cd /home/nikolas/Documents/hft
PYTHONPATH=. python3 agents/example_usage.py
```

---

## Usage Examples

### Sentiment Analysis Only

```python
from agents import SentimentAgent, create_llm_config_for_hardware

config = create_llm_config_for_hardware("cmp50hx")
agent = SentimentAgent(config)

posts = [
    "BTC breaking out! $100k incoming",
    "Fed hints at rate pause, crypto rallies",
    "Technical analysis shows strong support at $48k",
]

sentiment = agent.analyze_text(posts, symbol="BTCUSDT")
print(f"Sentiment: {sentiment.sentiment_score:.2f}")
print(f"Recommendation: {sentiment.recommendation}")
```

### Full Integration (Math + LLM)

```python
from agents import SignalIntegrator, create_llm_config_for_hardware
from agents import SentimentAgent

config = create_llm_config_for_hardware("cmp50hx")
sentiment_agent = SentimentAgent(config)
integrator = SignalIntegrator()

# Get sentiment
sentiment = sentiment_agent.analyze_text(posts, "BTCUSDT")

# Get prices (from your data pipeline)
prices = [...]  # List of recent prices

# Combined signal
combined = integrator.get_combined_signal(
    symbol="BTCUSDT",
    prices=prices,
    current_price=50000,
    sentiment=sentiment,  # From LLM agent
    initial_equity=10000,
)

print(f"Action: {combined.action}")
print(f"Size: ${combined.size_usdt:.2f}")
print(f"Confidence: {combined.confidence:.2f}")
```

### Multi-Agent Orchestrator

```python
from agents import TradingAgentsOrchestrator, create_llm_config_for_hardware

config = create_llm_config_for_hardware("cmp50hx")
orchestrator = TradingAgentsOrchestrator(config)

# Run full analysis pipeline
decision = await orchestrator.run_analysis(
    symbol="BTCUSDT",
    current_price=50000,
    texts=["BTC to the moon!"],  # Social media posts
    headlines=[{"headline": "Fed pauses rates", "source": "Reuters"}],
    technical_data={"poincare_score": 0.3, "whitehead_signal": "same_regime"},
)

print(f"Decision: {decision.action}")
print(f"Size: ${decision.size_usdt}")
print(f"Stop loss: ${decision.stop_loss}")
```

---

## Agents

| Agent | Purpose | Input | Output |
|-------|---------|-------|--------|
| `SentimentAgent` | Social media/news sentiment | Text posts, tweets, Reddit | sentiment_score, recommendation |
| `NewsAgent` | Macroeconomic news analysis | Headlines, articles | impact_score, urgency, event_type |
| `FundamentalAgent` | Earnings/financial analysis | 10-Q, 10-K, metrics | fundamental_score, valuation |
| `RiskAgent` | Portfolio risk assessment | Positions, PnL history | risk_level, max_position_size |
| `TraderAgent` | Final decision synthesis | All signals above | action, size, SL, TP |

---

## Math Signals (Already Implemented)

| Signal | Source | What It Detects |
|--------|--------|-----------------|
| **Poincaré** | `poincare_trading.py` | Mean-reversion vs. trending via Ricci curvature |
| **Whitehead** | `whitehead_signal.py` | Regime changes via torsion proxy |
| **Hecke** | `hecke_operators.py` | Spectral structure via L-functions |
| **Polar** | `polar_features.py` | Geometric phase space (r, θ) |
| **Graph** | `graph_trades.py` | Pattern matching via rustworkx |

---

## What's Missing (vs. TradingAgents)

TradingAgents uses external data sources that you'll need to add:

| Source | Status | How to Add |
|--------|--------|------------|
| Twitter/X API | Not implemented | `tweepy` + API key |
| Reddit API | Not implemented | `praw` |
| News APIs | Not implemented | NewsAPI, Bloomberg, Reuters |
| Alpha Vantage | Not implemented | `yfinance` or API |
| Earnings data | Not implemented | SEC EDGAR, IEX Cloud |

The agents are designed to accept data from any source — just format it as text and pass to `analyze_text()` or `analyze_headline()`.

---

## Files Created

```
agents/
├── __init__.py              # Package exports
├── llm_agent.py             # Base agent + LLMConfig
├── sentiment_agent.py       # Social media sentiment
├── news_agent.py            # Macroeconomic news
├── fundamental_agent.py     # Earnings/financials
├── risk_agent.py            # Risk management
├── trader_agent.py          # Final decision + orchestrator
├── integration.py           # Math + LLM integration
├── example_usage.py         # Usage example
├── requirements.txt         # Dependencies
└── README.md                # This file

scripts/
├── start_vllm_2x3090.sh     # vLLM startup (2×3090)
└── start_ollama_cmp50hx.sh  # Ollama startup (CMP 50HX)
```

---

## Next Steps

1. **For CMP 50HX (now):**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull mistral-nemo:12b
   ollama serve
   PYTHONPATH=. python3 agents/example_usage.py
   ```

2. **When you rent 2×3090:**
   ```bash
   pip install vllm
   ./scripts/start_vllm_2x3090.sh
   # Update config: create_llm_config_for_hardware("2x3090")
   ```

3. **Add data sources:**
   - Twitter: `pip install tweepy`
   - Reddit: `pip install praw`
   - News: Get API key from NewsAPI.org

4. **Integrate with your HFT loop:**
   ```python
   # In your main.py or algorithm runner
   from agents import SignalIntegrator

   integrator = SignalIntegrator()

   # On each price update
   signal = integrator.get_combined_signal(
       symbol="BTCUSDT",
       prices=recent_prices,
       current_price=current_price,
   )

   if signal.action == "buy" and signal.confidence > 0.7:
       # Execute trade
       pass
   ```
