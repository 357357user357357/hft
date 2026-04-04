#!/usr/bin/env python3
"""Run signal backtest on real BTC data."""

import sys
sys.path.insert(0, '.')

from data import load_agg_trades_csv
from signal_backtest import SignalBacktest, SignalConfig, compare_signals
from pathlib import Path

# Load real BTC data
data_file = Path("./data/BTCUSDT-aggTrades-2024-01-15.zip")
print(f"Loading {data_file.name}...")

trades = load_agg_trades_csv(data_file)
print(f"Loaded {len(trades):,} trades")

# Extract prices and volumes
prices = [t.price for t in trades]
volumes = [t.quantity for t in trades]

print(f"Price range: {min(prices):.2f} - {max(prices):.2f}")
print(f"Date range: {trades[0].transact_time} to {trades[-1].transact_time}")

# Downsample to 1-minute bars for faster backtesting
def resample_to_bars(prices, volumes, bar_size=60):
    """Resample tick data to bars."""
    bar_prices = []
    bar_volumes = []
    for i in range(0, len(prices), bar_size):
        end = min(i + bar_size, len(prices))
        bar_prices.append((prices[i] + prices[end-1]) / 2)  # Average price
        bar_volumes.append(sum(volumes[i:end]))
    return bar_prices, bar_volumes

print("\nResampling to 1-minute bars...")
bar_prices, bar_volumes = resample_to_bars(prices, volumes, bar_size=60)
print(f"Created {len(bar_prices)} bars")

# Run individual signal backtests
print("\n" + "="*70)
print("INDIVIDUAL SIGNAL BACKTESTS")
print("="*70)

signals_to_test = [
    ("composite", "Composite (all 19 dims)"),
    ("poincare", "Poincaré Topology"),
    ("torsion", "Whitehead Torsion"),
    ("geometry", "Frenet-Serret Geometry"),
    ("polar", "Polar Coordinates"),
    ("hurst", "Hurst Exponent"),
    ("momentum", "Momentum/RSI"),
    ("simons", "Jim Simons SDEs"),
    ("spectral", "Spectral Analysis"),
    ("fel", "FEL Semigroup"),
]

results = []
for sig_key, sig_name in signals_to_test:
    print(f"\nTesting {sig_name}...")
    config = SignalConfig(
        signal_type=sig_key,
        lookback_bars=50,
        hold_bars=10,
        threshold=0.12,
        allow_long=True,
        allow_short=True,
    )
    backtest = SignalBacktest(config)
    stats = backtest.run(bar_prices, bar_volumes)
    results.append((sig_key, sig_name, stats))

    print(f"  {sig_name:30s}: "
          f"trades={stats.total_trades:3d}  "
          f"win={stats.win_rate*100:5.1f}%  "
          f"pnl={stats.total_pnl_pct:+7.2f}%  "
          f"sharpe={stats.sharpe_ratio:6.2f}")

# Rank by Sharpe ratio
print("\n" + "="*70)
print("SIGNALS RANKED BY SHARPE RATIO")
print("="*70)

ranked = sorted(results, key=lambda x: x[2].sharpe_ratio, reverse=True)
for rank, (sig_key, sig_name, stats) in enumerate(ranked, 1):
    print(f"  {rank:2d}. {sig_name:30s} Sharpe={stats.sharpe_ratio:6.2f}  "
          f"win={stats.win_rate*100:5.1f}%  "
          f"pnl={stats.total_pnl_pct:+7.2f}%")

# Print detailed stats for top 3
print("\n" + "="*70)
print("TOP 3 SIGNALS - DETAILED STATS")
print("="*70)

for sig_key, sig_name, stats in ranked[:3]:
    stats.print_summary(f"{sig_name} ({sig_key})")

# Save results to file
output_file = Path("./signal_backtest_results.txt")
with open(output_file, "w") as f:
    f.write("SIGNAL BACKTEST RESULTS - BTCUSDT 2024-01-15\n")
    f.write("="*70 + "\n\n")
    f.write("SIGNALS RANKED BY SHARPE RATIO\n")
    f.write("="*70 + "\n")
    for rank, (sig_key, sig_name, stats) in enumerate(ranked, 1):
        f.write(f"{rank:2d}. {sig_name:30s} Sharpe={stats.sharpe_ratio:6.2f}  "
                f"win={stats.win_rate*100:5.1f}%  "
                f"pnl={stats.total_pnl_pct:+7.2f}%\n")

print(f"\nResults saved to {output_file}")
