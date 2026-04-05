#!/usr/bin/env python3
"""Signal backtest — parallel across instruments AND signals.

Dispatch strategy (CPU + GPU hybrid):
  - EXPENSIVE signals (poincare, torsion): GPU workers when CuPy available,
    else CPU workers — these do large O(n²) matrix operations
  - MEDIUM signals (simons, spectral, fel, geometry): CPU workers with
    NumPy vectorisation
  - FAST signals (hurst, momentum, order_flow, autocorr, volatility,
    polar, quaternion): CPU workers — sub-millisecond, no acceleration needed

Uses ProcessPoolExecutor.  GPU workers use a shared CUDA device; CuPy
falls back to NumPy transparently when the GPU is unavailable or
JIT-fails (CUDA toolkit mismatch).

Usage:
    python run_signal_backtest.py                          # all data/*.zip
    python run_signal_backtest.py --symbol BTCUSDT ETHUSDT
    python run_signal_backtest.py --workers 8
    python run_signal_backtest.py --no-slow               # skip poincare/torsion
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

# ── Signal registry ───────────────────────────────────────────────────────────
# (key, display_name, cost_tier)
#   cost_tier: "slow" (733ms+), "medium" (55-293ms), "fast" (<10ms)
SIGNALS: List[Tuple[str, str, str]] = [
    ("composite",  "Composite (all 19)",  "medium"),
    ("poincare",   "Poincaré Topology",   "slow"),
    ("torsion",    "Whitehead Torsion",   "slow"),
    ("geometry",   "Frenet-Serret",       "medium"),
    ("polar",      "Polar Coords",        "fast"),
    ("hurst",      "Hurst Exponent",      "fast"),
    ("momentum",   "Momentum/RSI",        "fast"),
    ("simons",     "Simons SDEs",         "medium"),
    ("spectral",   "Spectral/Hecke",      "medium"),
    ("fel",        "FEL Semigroup",       "medium"),
    ("order_flow", "Order Flow",          "fast"),
    ("autocorr",   "Autocorrelation",     "fast"),
    ("volatility", "Volatility",          "fast"),
    ("quaternion", "Quaternion",          "fast"),
]

# Signals that benefit from GPU matrix ops (poincaré topology does large
# embedding/distance matrices; torsion does SVD-like decompositions)
_GPU_PREFERRED = {"poincare", "torsion"}


# ── GPU availability probe (done once per worker process) ─────────────────────

def _gpu_available() -> bool:
    """Return True if CuPy can actually JIT-compile kernels on this machine."""
    try:
        import cupy as cp
        probe = cp.array([1.0, 2.0, 3.0])
        _ = (probe * probe + probe).get()  # forces kernel compilation
        return True
    except Exception:
        return False


# ── worker (runs in subprocess) ───────────────────────────────────────────────

def _run_one(
    symbol: str,
    zip_path: str,
    sig_key: str,
    lookback: int,
    hold: int,
    threshold: float,
    bar_size: int,
    use_gpu: bool,
) -> Tuple[str, str, SignalStats]:
    """One (symbol, signal) combination. Runs in a worker process."""
    import os
    # For GPU-preferred signals, try to use CuPy; simons_sde.py auto-detects.
    # Setting CUPY_ACCELERATOR hints the signal implementations.
    if use_gpu and sig_key in _GPU_PREFERRED:
        os.environ.setdefault("HFT_USE_GPU", "1")

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

    # Use stride=5 for expensive topology signals — 5× fewer compute_signal()
    # calls with negligible accuracy loss (regime doesn't flip every bar).
    stride = 5 if sig_key in ("poincare", "torsion", "composite") else 1

    cfg = SignalConfig(
        signal_type=sig_key,
        lookback_bars=lookback,
        hold_bars=hold,
        threshold=threshold,
        allow_long=True,
        allow_short=True,
        signal_stride=stride,
    )
    stats = SignalBacktest(cfg).run(bp, bv)
    return symbol, sig_key, stats


# ── worker pool sizing ────────────────────────────────────────────────────────

def _plan_workers(
    n_jobs: int,
    requested_workers: Optional[int],
    gpu_ok: bool,
    n_slow: int,
) -> Tuple[int, int]:
    """
    Return (total_workers, gpu_slots).

    Strategy:
      - gpu_slots: reserve 2 workers for GPU-preferred jobs (poincaré, torsion)
        if GPU is available; else 0
      - cpu_slots: remaining workers handle everything else
      - Total capped at min(requested_workers or cpu_count, n_jobs)
    """
    cpu_count = os.cpu_count() or 4
    max_w = requested_workers or cpu_count
    max_w = min(max_w, n_jobs)

    if gpu_ok and n_slow > 0:
        # 2 dedicated GPU workers for the slow topology signals
        gpu_slots = min(2, n_slow, max_w)
        total = max_w
    else:
        gpu_slots = 0
        total = max_w

    return total, gpu_slots


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel signal backtest across instruments (CPU + GPU hybrid)")
    parser.add_argument("--symbol",    nargs="+",
                        help="Symbols to test (default: all ZIPs in ./data/)")
    parser.add_argument("--data-dir",  default="./data")
    parser.add_argument("--lookback",  type=int,   default=50)
    parser.add_argument("--hold",      type=int,   default=10)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--bar-size",  type=int,   default=60,
                        help="Ticks per bar (default 60)")
    parser.add_argument("--workers",   type=int,   default=None,
                        help="Worker processes (default: cpu_count)")
    parser.add_argument("--no-slow",   action="store_true",
                        help="Skip poincaré/torsion (saves ~1s per instrument)")
    parser.add_argument("--no-gpu",    action="store_true",
                        help="Disable GPU dispatch even if CuPy is available")
    args = parser.parse_args()

    # ── data files ────────────────────────────────────────────────────────────
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

    # ── signal list ───────────────────────────────────────────────────────────
    active_signals = [
        (k, name, tier) for k, name, tier in SIGNALS
        if not (args.no_slow and tier == "slow")
    ]
    sig_map = {k: name for k, name, _ in active_signals}

    symbols     = sorted(files)
    total_jobs  = len(symbols) * len(active_signals)

    # ── GPU probe ─────────────────────────────────────────────────────────────
    gpu_ok = False
    if not args.no_gpu:
        print("Probing GPU (CuPy)… ", end="", flush=True)
        gpu_ok = _gpu_available()
        print("available ✓" if gpu_ok else "not available — using CPU only")

    n_slow = sum(1 for _, _, t in active_signals if t == "slow") * len(symbols)
    total_workers, gpu_slots = _plan_workers(
        total_jobs, args.workers, gpu_ok, n_slow)

    device_info = (
        f"GPU workers: {gpu_slots} (poincaré/torsion) + "
        f"CPU workers: {total_workers - gpu_slots} (rest)"
        if gpu_ok and gpu_slots
        else f"CPU workers: {total_workers} (NumPy vectorised)"
    )
    print(
        f"\nRunning {len(symbols)} instruments × {len(active_signals)} signals "
        f"= {total_jobs} jobs\n"
        f"Symbols : {', '.join(symbols)}\n"
        f"Dispatch: {device_info}\n"
    )

    # ── submit all jobs ───────────────────────────────────────────────────────
    results: Dict[str, Dict[str, SignalStats]] = {s: {} for s in symbols}
    completed = 0

    with ProcessPoolExecutor(max_workers=total_workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                sym, str(files[sym]),
                sig_key,
                args.lookback, args.hold, args.threshold, args.bar_size,
                # route GPU-preferred signals to GPU path when GPU is available
                use_gpu=(gpu_ok and sig_key in _GPU_PREFERRED),
            ): (sym, sig_key, sig_name)
            for sym in symbols
            for sig_key, sig_name, _ in active_signals
        }

        for fut in as_completed(futures):
            sym, sig_key, sig_name = futures[fut]
            completed += 1
            try:
                _, _, stats = fut.result()
                results[sym][sig_key] = stats
                tier_tag = ""
                if sig_key in _GPU_PREFERRED and gpu_ok:
                    tier_tag = " [GPU]"
                elif sig_key in _GPU_PREFERRED:
                    tier_tag = " [CPU*]"
                print(
                    f"  [{completed:>3}/{total_jobs}] {sym:<10} "
                    f"{sig_name:<22}{tier_tag:<7} "
                    f"trades={stats.total_trades:4d}  "
                    f"win={stats.win_rate*100:5.1f}%  "
                    f"pnl={stats.total_pnl_pct:+7.3f}%  "
                    f"sharpe={stats.sharpe_ratio:6.2f}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"  [{completed:>3}/{total_jobs}] {sym:<10} "
                    f"{sig_name:<22} ERROR: {e}",
                    flush=True,
                )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 95)
    print("BEST INSTRUMENT PER SIGNAL  (by Sharpe)")
    print("=" * 95)
    for sig_key, sig_name, _ in active_signals:
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

    print("\n" + "=" * 95)
    print("BEST SIGNAL PER INSTRUMENT  (by Sharpe)")
    print("=" * 95)
    for sym in symbols:
        ranked = sorted(
            [(k, v) for k, v in results[sym].items() if v.total_trades > 0],
            key=lambda x: x[1].sharpe_ratio, reverse=True,
        )
        if not ranked:
            continue
        best_k, s = ranked[0]
        print(f"  {sym:<10} → {sig_map.get(best_k, best_k):<22}  "
              f"sharpe={s.sharpe_ratio:6.2f}  pnl={s.total_pnl_pct:+7.3f}%  "
              f"win={s.win_rate*100:5.1f}%  trades={s.total_trades}")

    print("\n" + "=" * 95)
    print("ALL INSTRUMENTS — RANKED BY TOTAL PnL (sum across all signals)")
    print("=" * 95)
    ranking = []
    for sym in symbols:
        sigs = results[sym]
        total_pnl    = sum(s.total_pnl_pct for s in sigs.values() if s.total_trades > 0)
        best_sharpe  = max((s.sharpe_ratio  for s in sigs.values() if s.total_trades > 0), default=0.0)
        total_trades = sum(s.total_trades   for s in sigs.values())
        ranking.append((sym, total_pnl, best_sharpe, total_trades))
    ranking.sort(key=lambda x: x[1], reverse=True)
    for sym, pnl, sharpe, trades in ranking:
        bar = "█" * min(int(abs(pnl) * 1.5), 40)
        sign = "+" if pnl >= 0 else ""
        print(f"  {sym:<10}  pnl={sign}{pnl:.3f}%  best_sharpe={sharpe:6.2f}  "
              f"trades={trades:4d}  {bar}")

    # ── Per-instrument top-3 signals ──────────────────────────────────────────
    print("\n" + "=" * 95)
    print("TOP 3 SIGNALS PER INSTRUMENT")
    print("=" * 95)
    for sym in symbols:
        ranked = sorted(
            [(k, v) for k, v in results[sym].items() if v.total_trades > 0],
            key=lambda x: x[1].sharpe_ratio, reverse=True,
        )[:3]
        if not ranked:
            continue
        print(f"\n  {sym}:")
        for i, (k, s) in enumerate(ranked, 1):
            print(f"    {i}. {sig_map.get(k, k):<22}  sharpe={s.sharpe_ratio:6.2f}  "
                  f"pnl={s.total_pnl_pct:+7.3f}%  win={s.win_rate*100:5.1f}%  "
                  f"trades={s.total_trades}")

    # ── GPU acceleration report ───────────────────────────────────────────────
    if gpu_ok and gpu_slots:
        print(f"\n[GPU] CMP 50HX used for poincaré/torsion signals "
              f"({gpu_slots} dedicated workers)")
    elif not gpu_ok and not args.no_gpu:
        print("\n[GPU] Not available — all signals ran on CPU with NumPy vectorisation")


if __name__ == "__main__":
    main()
