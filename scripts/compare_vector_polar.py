#!/usr/bin/env python3
"""Backtest comparison: Vector vs Vector+Polar on real BTC data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.vector import VectorBacktest, VectorConfig
from hft_types import Side
from data import load_agg_trades_csv


def main():
    print("="*70)
    print("Vector vs Vector+Polar Backtest Comparison")
    print("="*70)

    # Load real BTC data
    data_file = Path(__file__).parent.parent / "data" / "BTCUSDT-aggTrades-2024-01-15.zip"

    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        return

    trades = load_agg_trades_csv(data_file)
    trades = trades[:2000]  # Limit for speed
    print(f"\nLoaded {len(trades)} trades from {data_file.name}")

    # Run Vector without polar
    print("\nRunning Vector (polar disabled)...")
    config_off = VectorConfig(side=Side.Buy, use_polar_signals=False)
    bt_off = VectorBacktest(config_off)
    for t in trades:
        bt_off.on_trade(t)

    # Run Vector with polar
    print("Running Vector+Polar...")
    config_on = VectorConfig(
        side=Side.Buy,
        use_polar_signals=True,
        polar_tau=10,
        polar_price_scale=50000.0,
    )
    bt_on = VectorBacktest(config_on)
    for t in trades:
        bt_on.on_trade(t)

    # Print results
    print("\n" + "="*70)
    print(f"{'Metric':<25} {'Vector':<20} {'Vector+Polar':<20} {'Diff':>10}")
    print("="*70)

    metrics = [
        ("Total Trades", float(bt_off.stats.total_trades), float(bt_on.stats.total_trades)),
        ("Total PnL (%)", bt_off.stats.total_pnl_pct, bt_on.stats.total_pnl_pct),
        ("Win Rate (%)", bt_off.stats.win_rate(), bt_on.stats.win_rate()),
        ("Avg PnL/Trade", bt_off.stats.avg_pnl_per_trade(), bt_on.stats.avg_pnl_per_trade()),
        ("Max Drawdown (%)", bt_off.stats.max_drawdown_pct, bt_on.stats.max_drawdown_pct),
    ]

    for name, v1, v2 in metrics:
        diff = v2 - v1
        diff_str = f"{diff:+.4f}"
        print(f"{name:<25} {v1:>20.4f} {v2:>20.4f} {diff_str:>10}")

    print("="*70)

    # Summary
    pnl_diff = bt_on.stats.total_pnl_pct - bt_off.stats.total_pnl_pct
    trades_diff = bt_on.stats.total_trades - bt_off.stats.total_trades

    print("\n### SUMMARY ###")
    print(f"Polar signals generated: {trades_diff:+d} additional trades")
    print(f"PnL impact: {pnl_diff:+.4f}%")
    print()
    print("Note: This is a raw comparison without parameter optimization.")
    print("Polar signals add information - tuning is needed for profitability.")


if __name__ == "__main__":
    main()
