#!/usr/bin/env python3
"""Walk-forward validation: train on N days, test on M days, roll forward.

Prevents overfitting by never testing on data used for parameter selection.

Usage:
    python walk_forward.py                              # all data/*.zip
    python walk_forward.py --symbol SOLUSDT BTCUSDT
    python walk_forward.py --train-days 5 --test-days 2
    python walk_forward.py --no-slow                    # skip expensive signals
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from data import load_agg_trades_csv
from signal_backtest import SignalBacktest, SignalConfig, SignalStats

# Signals to test
SIGNALS: List[Tuple[str, str]] = [
    ("order_flow", "Order Flow"),
    ("simons",     "Simons SDEs"),
    ("volatility", "Volatility"),
    ("polar",      "Polar Coords"),
    ("hurst",      "Hurst Exponent"),
    ("poincare",   "Poincare"),
    ("autocorr",   "Autocorrelation"),
    ("momentum",   "Momentum/RSI"),
]

THRESHOLDS = [0.08, 0.10, 0.12, 0.15, 0.20]
HOLD_BARS  = [5, 10, 15, 20]


def _find_data_files(data_dir: Path, symbol: str) -> List[Path]:
    """Find all daily zip files for a symbol, sorted by date."""
    files = sorted(data_dir.glob(f"{symbol}-aggTrades-*.zip"))
    return files


def _load_bars(zip_path: str, bar_size: int = 60) -> Tuple[List[float], List[float]]:
    """Load trades from zip and resample to bars."""
    trades = load_agg_trades_csv(zip_path)
    prices  = [t.price    for t in trades]
    volumes = [t.quantity for t in trades]
    bp, bv = [], []
    for i in range(0, len(prices), bar_size):
        end = min(i + bar_size, len(prices))
        bp.append((prices[i] + prices[end - 1]) / 2)
        bv.append(sum(volumes[i:end]))
    return bp, bv


def _optimize_on_train(
    prices: List[float], volumes: List[float],
    sig_key: str, lookback: int,
) -> Tuple[float, int, float]:
    """Grid search best (threshold, hold_bars) on training data.
    Returns (best_threshold, best_hold, best_sharpe)."""
    best_sharpe = -999.0
    best_thr    = 0.12
    best_hold   = 10

    for thr in THRESHOLDS:
        for hold in HOLD_BARS:
            stride = 5 if sig_key in ("poincare", "torsion", "composite") else 1
            cfg = SignalConfig(
                signal_type=sig_key,
                lookback_bars=lookback,
                hold_bars=hold,
                threshold=thr,
                signal_stride=stride,
                slippage_enabled=True,
            )
            try:
                stats = SignalBacktest(cfg).run(prices, volumes)
                if stats.total_trades >= 3 and stats.sharpe_ratio > best_sharpe:
                    best_sharpe = stats.sharpe_ratio
                    best_thr    = thr
                    best_hold   = hold
            except Exception:
                pass

    return best_thr, best_hold, best_sharpe


def _test_with_params(
    prices: List[float], volumes: List[float],
    sig_key: str, lookback: int, threshold: float, hold: int,
) -> SignalStats:
    """Run backtest on test data with fixed params."""
    stride = 5 if sig_key in ("poincare", "torsion", "composite") else 1
    cfg = SignalConfig(
        signal_type=sig_key,
        lookback_bars=lookback,
        hold_bars=hold,
        threshold=threshold,
        signal_stride=stride,
        slippage_enabled=True,
    )
    return SignalBacktest(cfg).run(prices, volumes)


def _run_one_fold(
    symbol: str, sig_key: str, sig_name: str,
    train_files: List[str], test_files: List[str],
    lookback: int, bar_size: int, fold_idx: int,
) -> Dict:
    """One walk-forward fold: optimize on train, evaluate on test."""
    # Load and concatenate training data
    train_p, train_v = [], []
    for f in train_files:
        bp, bv = _load_bars(f, bar_size)
        train_p.extend(bp)
        train_v.extend(bv)

    # Load test data
    test_p, test_v = [], []
    for f in test_files:
        bp, bv = _load_bars(f, bar_size)
        test_p.extend(bp)
        test_v.extend(bv)

    if len(train_p) < lookback + 20 or len(test_p) < lookback + 20:
        return {"symbol": symbol, "signal": sig_name, "fold": fold_idx,
                "error": "insufficient data"}

    # Optimize on training data
    best_thr, best_hold, train_sharpe = _optimize_on_train(
        train_p, train_v, sig_key, lookback)

    # Test with optimized params
    try:
        test_stats = _test_with_params(
            test_p, test_v, sig_key, lookback, best_thr, best_hold)
    except Exception as e:
        return {"symbol": symbol, "signal": sig_name, "fold": fold_idx,
                "error": str(e)}

    return {
        "symbol": symbol, "signal": sig_name, "fold": fold_idx,
        "train_sharpe": train_sharpe,
        "test_sharpe": test_stats.sharpe_ratio,
        "test_pnl": test_stats.total_pnl_pct,
        "test_trades": test_stats.total_trades,
        "test_winrate": test_stats.win_rate,
        "best_thr": best_thr, "best_hold": best_hold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward validation: train/test rolling window")
    parser.add_argument("--symbol",     nargs="+",
                        help="Symbols to test (default: all in data/)")
    parser.add_argument("--data-dir",   default="./data")
    parser.add_argument("--train-days", type=int, default=5,
                        help="Days per training window (default: 5)")
    parser.add_argument("--test-days",  type=int, default=2,
                        help="Days per test window (default: 2)")
    parser.add_argument("--lookback",   type=int, default=50)
    parser.add_argument("--bar-size",   type=int, default=60)
    parser.add_argument("--workers",    type=int, default=None)
    parser.add_argument("--no-slow",    action="store_true",
                        help="Skip poincare/torsion")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Discover symbols
    if args.symbol:
        symbols = args.symbol
    else:
        syms = set()
        for z in data_dir.glob("*-aggTrades-*.zip"):
            syms.add(z.name.split("-aggTrades-")[0])
        symbols = sorted(syms)

    if not symbols:
        print("No data files found. Run download_data.py first.")
        sys.exit(1)

    active_signals = [(k, n) for k, n in SIGNALS
                      if not (args.no_slow and k in ("poincare", "torsion"))]

    total_folds = 0
    jobs = []

    for sym in symbols:
        files = _find_data_files(data_dir, sym)
        if len(files) < args.train_days + args.test_days:
            print(f"  {sym}: only {len(files)} days, need {args.train_days + args.test_days} — skipping")
            continue

        # Rolling windows
        step = args.test_days
        for start in range(0, len(files) - args.train_days - args.test_days + 1, step):
            train_files = [str(f) for f in files[start:start + args.train_days]]
            test_files  = [str(f) for f in files[start + args.train_days:
                                                  start + args.train_days + args.test_days]]
            for sig_key, sig_name in active_signals:
                jobs.append((sym, sig_key, sig_name, train_files, test_files,
                             args.lookback, args.bar_size, total_folds))
                total_folds += 1

    if not jobs:
        print("No valid folds. Need more data files (run download_data.py --days 30).")
        sys.exit(1)

    workers = args.workers or min(os.cpu_count() or 4, len(jobs))
    print(f"\nWalk-forward: {len(symbols)} symbols x {len(active_signals)} signals"
          f" = {total_folds} folds")
    print(f"  Train window: {args.train_days} days, Test window: {args.test_days} days")
    print(f"  Workers: {workers}\n")

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one_fold, *job): job
            for job in jobs
        }
        for fut in as_completed(futures):
            completed += 1
            try:
                r = fut.result()
                results.append(r)
                if "error" not in r:
                    print(f"  [{completed:>3}/{total_folds}] {r['symbol']:<10} "
                          f"{r['signal']:<18} fold={r['fold']}  "
                          f"train_sh={r['train_sharpe']:+.2f}  "
                          f"test_sh={r['test_sharpe']:+.2f}  "
                          f"test_pnl={r['test_pnl']:+.3f}%  "
                          f"trades={r['test_trades']}", flush=True)
                else:
                    print(f"  [{completed:>3}/{total_folds}] {r['symbol']:<10} "
                          f"{r['signal']:<18} ERROR: {r['error']}", flush=True)
            except Exception as e:
                completed_job = futures[fut]
                print(f"  [{completed:>3}/{total_folds}] EXCEPTION: {e}", flush=True)

    # ── Summary: average OOS (out-of-sample) performance per signal ──────────
    print("\n\n" + "=" * 90)
    print("WALK-FORWARD RESULTS — Average Out-of-Sample Performance")
    print("=" * 90)

    sig_results: Dict[str, List[Dict]] = {}
    for r in results:
        if "error" in r:
            continue
        sig_results.setdefault(r["signal"], []).append(r)

    print(f"\n  {'Signal':<20} {'Folds':>5}  {'OOS Sharpe':>10}  "
          f"{'OOS PnL':>9}  {'WinRate':>7}  {'Train Sh':>9}  {'Overfit?':>8}")
    print("  " + "-" * 82)

    for sig_name in [n for _, n in active_signals]:
        folds = sig_results.get(sig_name, [])
        if not folds:
            continue
        n = len(folds)
        avg_test_sh  = sum(f["test_sharpe"] for f in folds) / n
        avg_test_pnl = sum(f["test_pnl"] for f in folds) / n
        avg_win      = sum(f["test_winrate"] for f in folds) / n
        avg_train_sh = sum(f["train_sharpe"] for f in folds) / n
        # Overfit indicator: train >> test suggests overfitting
        overfit = "YES" if avg_train_sh > 0.5 and avg_test_sh < 0 else "no"
        print(f"  {sig_name:<20} {n:>5}  {avg_test_sh:>+10.2f}  "
              f"{avg_test_pnl:>+9.3f}%  {avg_win*100:>6.1f}%  "
              f"{avg_train_sh:>+9.2f}  {overfit:>8}")

    # Per-symbol best signal
    print("\n" + "=" * 90)
    print("BEST OOS SIGNAL PER SYMBOL")
    print("=" * 90)

    for sym in symbols:
        sym_folds = [r for r in results if r.get("symbol") == sym and "error" not in r]
        if not sym_folds:
            continue
        # Group by signal, average OOS sharpe
        by_sig: Dict[str, List[float]] = {}
        for r in sym_folds:
            by_sig.setdefault(r["signal"], []).append(r["test_sharpe"])
        ranked = sorted(by_sig.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
        if ranked:
            best_name, sharpes = ranked[0]
            avg_sh = sum(sharpes) / len(sharpes)
            print(f"  {sym:<10} -> {best_name:<20}  avg OOS sharpe={avg_sh:+.2f}  "
                  f"({len(sharpes)} folds)")

    print()


if __name__ == "__main__":
    main()
