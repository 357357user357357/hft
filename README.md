# HFT Backtest

Rust implementation of four trading algorithms using [hftbacktest](https://github.com/nkaz001/hftbacktest) framework with Binance aggTrades data.

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

## Data Source

Binance Vision aggTrades (futures UM monthly):
```
https://data.binance.vision/?prefix=data/futures/um/monthly/aggTrades
```

## Setup

### 1. Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 2. Download Data
```bash
# Download BTCUSDT aggTrades for January 2024
cargo run --bin download_data -- --symbol BTCUSDT --year 2024 --month 01

# Or manually from:
# https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip
```

### 3. Run Backtest
```bash
# Run all algorithms
cargo run --release -- --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo all

# Run specific algorithm
cargo run --release -- --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo shot
cargo run --release -- --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo depth_shot
cargo run --release -- --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo averages
cargo run --release -- --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo vector

# Sell side
cargo run --release -- --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo shot --side sell

# Limit trades for quick test
cargo run --release -- --file ./data/BTCUSDT-aggTrades-2024-01.zip --algo all --max-trades 100000
```

## Project Structure

```
src/
├── main.rs                    # Main backtest runner (CLI)
├── types.rs                   # Common types: Side, TakeProfitConfig, StopLossConfig, Position
├── data.rs                    # Data loading: AggTrade CSV/ZIP parser, SimpleOrderBook
├── hbt_integration.rs         # hftbacktest Bot/Strategy integration layer
├── algorithms/
│   ├── mod.rs
│   ├── shot.rs                # Shot algorithm
│   ├── depth_shot.rs          # Depth Shot algorithm
│   ├── averages.rs            # Averages algorithm
│   └── vector.rs              # Vector algorithm
└── bin/
    └── download_data.rs       # Data downloader binary
```

## Algorithm Configuration Examples

### Shot — Buy on dip with trailing SL
```rust
ShotConfig {
    side: Side::Buy,
    distance_pct: 0.8,
    buffer_pct: 0.3,
    follow_price_delay_secs: 0.3,
    replace_delay_secs: 0.3,
    order_size_usdt: 100.0,
    take_profit: TakeProfitConfig {
        enabled: true,
        percentage: 0.4,
        auto_price_down: Some(AutoPriceDown {
            timer_secs: 2.0,
            step_pct: 0.1,
            limit_pct: 0.1,
        }),
    },
    stop_loss: StopLossConfig {
        enabled: true,
        percentage: 1.2,
        delay_secs: 1.0,  // Wait 1s before SL activates
        trailing: Some(TrailingStop { spread_pct: 0.3 }),
        ..Default::default()
    },
}
```

### Averages — Multi-timeframe buy on dip
```rust
AveragesConfig {
    side: Side::Buy,
    order_distance_pct: -0.5,
    conditions: vec![
        AveragesCondition {
            long_period_secs: 60.0,
            short_period_secs: 10.0,
            trigger_min_pct: -0.5,
            trigger_max_pct: -0.1,
        },
        AveragesCondition {
            long_period_secs: 300.0,
            short_period_secs: 60.0,
            trigger_min_pct: 0.1,
            trigger_max_pct: 0.5,
        },
    ],
    ..Default::default()
}
```

### Vector — Detect volatility bursts
```rust
VectorConfig {
    side: Side::Buy,
    frame_size_secs: 0.2,
    time_frame_secs: 1.0,
    min_spread_size_pct: 0.3,
    order_distance_pct: 5.0,
    use_adaptive_order_distance: true,
    take_profit_spread_pct: 80.0,
    use_adaptive_take_profit: true,
    ..Default::default()
}
```

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
