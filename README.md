# HFT Backtest

Python implementation of four trading algorithms with advanced mathematical signal processing for cryptocurrency futures backtesting. Uses Binance Vision aggTrades data.

## Features

- **Four Core Trading Algorithms**: Shot, Depth Shot, Averages, and Vector
- **Advanced Mathematical Signal Processing**: Algebraic topology, number theory, differential geometry
- **Multi-Agent LLM Integration**: Sentiment analysis and trading decision synthesis
- **Realistic Backtesting**: Slippage, fees, position sizing, and latency simulation
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar ratios, drawdown analysis

## Algorithms

### Shot
Profits from sharp price spikes (breakthroughs) followed by pullbacks. Keeps an order at a set distance from price, automatically moving it during smooth movements via a buffer (dead zone).

**Key parameters:**
- `distance_pct` — how far from price to keep the order (%)
- `buffer_pct` — dead zone width (%)
- `follow_price_delay_secs` — delay before moving order toward price
- `replace_delay_secs` — delay before moving order away from price

### Depth Shot
Like Shot, but places orders based on order book volume rather than fixed distance. Order is placed where accumulated volume in the book reaches the target value.

**Key parameters:**
- `target_volume` — target cumulative volume in quote asset (USDT)
- `min/max_distance_pct` — allowed distance range from price
- `volume_buffer` — volume corridor where order doesn't move
- TP modes: Classic (fixed %), Historic (% of 2s movement), Depth (% of breakthrough)

### Averages
Profits from corrections — when price temporarily moves against the main trend, then returns. Compares average price across two timeframes and places orders when the difference falls within a set range.

**Key parameters:**
- `long_period_secs` / `short_period_secs` — averaging periods
- `trigger_min/max_pct` — delta range to trigger (positive = downtrend, negative = uptrend)
- `order_distance_pct` — order placement distance from trigger price
- Multiple conditions with AND logic

### Vector
Detects regions with abnormal market activity — moments when price ranges increase sharply over a short period. Analyzes market in small time intervals (frames).

**Key parameters:**
- `frame_size_secs` — micro-interval size (default 0.2s)
- `time_frame_secs` — total analysis window (default 1s)
- `min_spread_size_pct` — minimum spread per frame
- `upper/lower_border_range` — boundary movement filters
- `order_distance_pct` — % of spread for order placement
- `take_profit_spread_pct` — TP as % of spread
- `detect_shot` — mode for detecting sharp spikes with pullback

## Mathematical Signal Processing

The framework includes advanced mathematical modules for signal generation:

- **Poincaré Geometry**: Hyperbolic space trading signals
- **Ricci Curvature**: Geometric market analysis
- **Persistent Homology**: Topological data analysis (via `gudhi`)
- **Hecke Operators**: Number-theoretic signal processing
- **Dirichlet Characters**: Multiplicative character analysis
- **Groebner Bases**: Polynomial ideal computations
- **Frenet-Serret Frames**: Differential geometric features
- **p-adic Analysis**: Non-archimedean market metrics
- **Quaternions**: 4D rotational analysis

## LLM Multi-Agent System

The `agents/` directory contains a multi-agent system for trading decision synthesis:

- **Sentiment Agent**: Social media sentiment analysis
- **News Agent**: News and announcement analysis
- **Fundamental Agent**: Fundamental analysis
- **Risk Agent**: Risk assessment
- **Trader Agent**: Decision synthesis from all agents
- **Integration Module**: Combines mathematical signals with LLM insights

Requires `ollama` or `vllm` for local inference, or OpenAI/Anthropic APIs.

## Data Source

Binance Vision aggTrades (futures UM monthly):
```
https://data.binance.vision/?prefix=data/futures/um/monthly/aggTrades
```

## Setup

### 1. Install Python 3.11+

```bash
# Using pyenv (recommended)
pyenv install 3.11
pyenv local 3.11

# Or use your system's Python 3.11+
python3 --version
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies (create `requirements.txt` if needed):
```
# Testing
pytest

# Optional - for mathematical modules
# gudhi  # Persistent homology

# Optional - for LLM agents
# ollama
# vllm
# openai
# anthropic
```

### 3. Download Data

```bash
# Download BTCUSDT aggTrades for January 2024
python download_data.py --symbol BTCUSDT --year 2024 --month 01

# Or manually from:
# https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip
```

### 4. Run Backtest

```bash
# Run all algorithms
python main.py --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo all

# Run specific algorithm
python main.py --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo shot
python main.py --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo depth_shot
python main.py --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo averages
python main.py --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo vector

# Sell side
python main.py --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo shot --side sell

# Limit trades for quick test
python main.py --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo all --max-trades 100000
```

## Project Structure

```
hft/
├── main.py                    # Main backtest runner (CLI)
├── hft_types.py               # Core type definitions
├── data.py                    # Data loading: AggTrade CSV/ZIP parser
├── constants.py               # Global configuration constants
├── config.py                  # JSON config loader
├── download_data.py           # Binance data downloader
│
├── algorithms/
│   ├── shot.py                # Shot algorithm
│   ├── depth_shot.rs          # Depth Shot algorithm
│   ├── averages.py            # Averages algorithm
│   ├── vector.py              # Vector algorithm
│   ├── poly_signal.py         # Polynomial signals
│   └── hecke_signal.py        # Hecke operator signals
│
├── agents/                    # LLM multi-agent system
│   ├── llm_agent.py
│   ├── sentiment_agent.py
│   ├── news_agent.py
│   ├── fundamental_agent.py
│   ├── risk_agent.py
│   ├── trader_agent.py
│   ├── integration.py
│   └── README.md
│
├── research/                  # Experimental mathematical modules
│   ├── poincare_trading.py    # Poincaré geometry signals
│   ├── ricci_curvature.py     # Ricci flow analysis
│   ├── dirichlet.py           # Dirichlet character signals
│   ├── hecke_operators.py     # Hecke operator signals
│   ├── groebner.py            # Groebner basis computations
│   ├── frenet_serret.py       # Frenet-Serret frame features
│   ├── p_adic.py              # p-adic analysis
│   ├── quaternions.py         # Quaternion features
│   ├── polar_features.py      # Polar coordinate analysis
│   └── ...
│
├── tests/                     # Test suite
│   ├── test_algorithms.py
│   ├── test_data.py
│   ├── test_math_modules.py
│   └── ...
│
├── data/                      # Historical trading data
└── scripts/                   # Utility scripts (LLM server startup)
```

## Algorithm Configuration Examples

### Shot — Buy on dip with trailing SL
```python
ShotConfig(
    side=Side.BUY,
    distance_pct=0.8,
    buffer_pct=0.3,
    follow_price_delay_secs=0.3,
    replace_delay_secs=0.3,
    order_size_usdt=100.0,
    take_profit=TakeProfitConfig(
        enabled=True,
        percentage=0.4,
        auto_price_down=AutoPriceDown(
            timer_secs=2.0,
            step_pct=0.1,
            limit_pct=0.1,
        ),
    ),
    stop_loss=StopLossConfig(
        enabled=True,
        percentage=1.2,
        delay_secs=1.0,  # Wait 1s before SL activates
        trailing=TrailingStop(spread_pct=0.3),
    ),
)
```

### Averages — Multi-timeframe buy on dip
```python
AveragesConfig(
    side=Side.BUY,
    order_distance_pct=-0.5,
    conditions=[
        AveragesCondition(
            long_period_secs=60.0,
            short_period_secs=10.0,
            trigger_min_pct=-0.5,
            trigger_max_pct=-0.1,
        ),
        AveragesCondition(
            long_period_secs=300.0,
            short_period_secs=60.0,
            trigger_min_pct=0.1,
            trigger_max_pct=0.5,
        ),
    ],
)
```

### Vector — Detect volatility bursts
```python
VectorConfig(
    side=Side.BUY,
    frame_size_secs=0.2,
    time_frame_secs=1.0,
    min_spread_size_pct=0.3,
    order_distance_pct=5.0,
    use_adaptive_order_distance=True,
    take_profit_spread_pct=80.0,
    use_adaptive_take_profit=True,
)
```

## Risk Management

All algorithms support:

- **Position Sizing**: Fixed USDT, fractional portfolio, Kelly criterion
- **Slippage Simulation**: Configurable slippage based on order size
- **Fee Calculation**: Maker/taker fees with rebate support
- **Latency Simulation**: Network delay modeling

## Stop Loss Features

All algorithms support:
- **Basic SL** — fixed % from entry price
- **Delay** — wait N seconds before SL activates (useful for Shot/DepthShot)
- **Trailing Stop** — SL follows price in profitable direction
- **Second SL** — replaces first SL when price reaches profit level
- **Spread** — additional offset for limit orders to guarantee execution

## Take Profit Features

- **Basic TP** — fixed % from entry price
- **Auto Price Down** — gradually lowers TP if price doesn't reach initial level
- **Vector TP** — % of spread size (not price)
- **Depth Shot TP** — Classic / Historic (% of 2s movement) / Depth (% of breakthrough)
- **Adaptive TP** (Vector) — adjusts based on boundary movement trend

## Performance Metrics

The backtester outputs comprehensive metrics:
- Total PnL (USDT and %)
- Win rate
- Profit factor
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Maximum drawdown
- Drawdown duration
- Average win/loss
- Largest win/loss

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_algorithms.py

# Run with coverage
pytest --cov=. --cov-report=html
```

## License

This project is released under **The Unlicense** — see [LICENSE](LICENSE) for details.

In short: **Do whatever you want with this code. No warranty. No restrictions.**
