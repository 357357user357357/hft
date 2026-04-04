#!/usr/bin/env python3
"""Signal backtest — parallel across instruments AND signals.

Runs every (instrument × signal) combination concurrently using
ProcessPoolExecutor so all CPU cores are used.

Usage:
    python run_signal_backtest.py                          # all data/*.zip
    python run_signal_backtest.py --symbol BTCUSDT ETHUSDT
    python run_signal_backtest.py --workers 8
"""

from __future__ import annotations

import argparse
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from data import load_agg_trades_csv
from signal_backtest import SignalBacktest, SignalConfig, SignalStats

SIGNALS: List[Tuple[str, str]] = [
    ("composite",  "Composite (all 19)"),
    ("poincare",   "Poincaré Topology"),
    ("torsion",    "Whitehead Torsion"),
    ("geometry",   "Frenet-Serret"),
    ("polar",      "Polar Coords"),
    ("hurst",      "Hurst Exponent"),
    ("momentum",   "Momentum/RSI"),
    ("simons",     "Simons SDEs"),
    ("spectral",   "Spectral/Hecke"),
    ("fel",        "FEL Semigroup"),
    ("order_flow", "Order Flow"),
    ("autocorr",   "Autocorrelation"),
    ("volatility", "Volatility"),
    ("quaternion", "Quaternion"),
]


# ── worker (runs in subprocess) ───────────────────────────────────────────────

def _run_one(
    symbol: str,
    zip_path: str,
    sig_key: str,
    lookback: int,
    hold: int,
    threshold: float,
    bar_size: int,
) -> Tuple[str, str, SignalStats]:
    """One (symbol, signal) combination. Runs in a worker process."""
    from data import load_agg_trades_csv
    from signal_backtest import SignalBacktest, SignalConfig

    trades = load_agg_trades_csv(zip_path)
    prices  = [t.price    for t in trades]
    volumes = [t.quantity for t in trades]

    # Resample tick data → bars
    bp, bv = [], []
    for i in range(0, len(prices), bar_size):
        end = min(i + bar_size, len(prices))
        bp.append((prices[i] + prices[end - 1]) / 2)
        bv.append(sum(volumes[i:end]))

    cfg = SignalConfig(
        signal_type=sig_key,
        lookback_bars=lookback,
        hold_bars=hold,
        threshold=threshold,
        allow_long=True,
        allow_short=True,
    )
    stats = SignalBacktest(cfg).run(bp, bv)
    return symbol, sig_key, stats


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel signal backtest across instruments")
    parser.add_argument("--symbol", nargs="+",
                        help="Symbols to test (default: all ZIPs in ./data/)")
    parser.add_argument("--data-dir",  default="./data")
    parser.add_argument("--lookback",  type=int,   default=50)
    parser.add_argument("--hold",      type=int,   default=10)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--bar-size",  type=int,   default=60,
                        help="Ticks per bar (default 60)")
    parser.add_argument("--workers",   type=int,   default=None,
                        help="Worker processes (default: cpu_count)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.symbol:
        files = {sym: data_dir / f"{sym}-aggTrades-2024-01-15.zip"
                 for sym in args.symbol}
        files = {k: v for k, v in files.items() if v.exists()}
    else:
        files = {}
        for z in sorted(data_dir.glob("*-aggTrades-*.zip")):
            sym = z.name.split("-aggTrades-")[0]
            files[sym] = z

    if not files:
        print("No data files found. Run download_data.py first.")
        sys.exit(1)

    symbols = sorted(files)
    total_jobs = len(symbols) * len(SIGNALS)
    print(f"Running {len(symbols)} instruments × {len(SIGNALS)} signals "
          f"= {total_jobs} jobs in parallel\n"
          f"Symbols: {', '.join(symbols)}\n")

    # Submit all (symbol, signal) pairs at once
    results: Dict[str, Dict[str, SignalStats]] = {s: {} for s in symbols}
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                sym, str(files[sym]),
                sig_key,
                args.lookback, args.hold, args.threshold, args.bar_size,
            ): (sym, sig_key, sig_name)
            for sym in symbols
            for sig_key, sig_name in SIGNALS
        }

        for fut in as_completed(futures):
            sym, sig_key, sig_name = futures[fut]
            completed += 1
            try:
                _, _, stats = fut.result()
                results[sym][sig_key] = stats
                print(f"  [{completed:>3}/{total_jobs}] {sym:<10} {sig_name:<22} "
                      f"trades={stats.total_trades:3d}  "
                      f"win={stats.win_rate*100:5.1f}%  "
                      f"pnl={stats.total_pnl_pct:+7.3f}%  "
                      f"sharpe={stats.sharpe_ratio:6.2f}",
                      flush=True)
            except Exception as e:
                print(f"  [{completed:>3}/{total_jobs}] {sym:<10} {sig_name:<22} ERROR: {e}",
                      flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    sig_map = dict(SIGNALS)

    print("\n\n" + "=" * 90)
    print("BEST INSTRUMENT PER SIGNAL  (by Sharpe)")
    print("=" * 90)
    for sig_key, sig_name in SIGNALS:
        row = [(sym, results[sym][sig_key])
               for sym in symbols
               if sig_key in results[sym] and results[sym][sig_key].total_trades > 0]
        if not row:
            continue
        row.sort(key=lambda x: x[1].sharpe_ratio, reverse=True)
        best_sym, s = row[0]
        print(f"  {sig_name:<22} → {best_sym:<10}  "
              f"sharpe={s.sharpe_ratio:6.2f}  pnl={s.total_pnl_pct:+7.3f}%  "
              f"win={s.win_rate*100:5.1f}%  trades={s.total_trades}")

    print("\n" + "=" * 90)
    print("BEST SIGNAL PER INSTRUMENT  (by Sharpe)")
    print("=" * 90)
    for sym in symbols:
        ranked = sorted(
            [(k, v) for k, v in results[sym].items() if v.total_trades > 0],
            key=lambda x: x[1].sharpe_ratio, reverse=True,
        )
        if not ranked:
            continue
        best_k, s = ranked[0]
        print(f"  {sym:<10} → {sig_map[best_k]:<22}  "
              f"sharpe={s.sharpe_ratio:6.2f}  pnl={s.total_pnl_pct:+7.3f}%  "
              f"win={s.win_rate*100:5.1f}%  trades={s.total_trades}")

    print("\n" + "=" * 90)
    print("ALL INSTRUMENTS — RANKED BY TOTAL PnL (sum across all signals)")
    print("=" * 90)
    ranking = []
    for sym in symbols:
        sigs = results[sym]
        total_pnl   = sum(s.total_pnl_pct   for s in sigs.values() if s.total_trades > 0)
        best_sharpe = max((s.sharpe_ratio    for s in sigs.values() if s.total_trades > 0), default=0.0)
        total_trades = sum(s.total_trades    for s in sigs.values())
        ranking.append((sym, total_pnl, best_sharpe, total_trades))
    ranking.sort(key=lambda x: x[1], reverse=True)
    for sym, pnl, sharpe, trades in ranking:
        bar = "█" * min(int(abs(pnl) * 1.5), 40)
        print(f"  {sym:<10}  pnl={pnl:+8.3f}%  best_sharpe={sharpe:6.2f}  "
              f"trades={trades:4d}  {bar}")

    # ── Per-instrument top-3 signals ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TOP 3 SIGNALS PER INSTRUMENT")
    print("=" * 90)
    for sym in symbols:
        ranked = sorted(
            [(k, v) for k, v in results[sym].items() if v.total_trades > 0],
            key=lambda x: x[1].sharpe_ratio, reverse=True,
        )[:3]
        if not ranked:
            continue
        print(f"\n  {sym}:")
        for i, (k, s) in enumerate(ranked, 1):
            print(f"    {i}. {sig_map[k]:<22}  sharpe={s.sharpe_ratio:6.2f}  "
                  f"pnl={s.total_pnl_pct:+7.3f}%  win={s.win_rate*100:5.1f}%  "
                  f"trades={s.total_trades}")


if __name__ == "__main__":
    main()
