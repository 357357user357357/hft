#!/usr/bin/env python3
"""Standalone GPU MA optimizer — runs on historical data OR live price feeds.

Usage:
    # Optimize on all historical zip files and print best params per symbol
    python gpu_optimizer.py

    # Specific symbols
    python gpu_optimizer.py --symbol SOLUSDT BTCUSDT

    # More MA window combinations (slower, more accurate)
    python gpu_optimizer.py --fine

    # Save results to JSON for bybit_trader.py to load on startup
    python gpu_optimizer.py --save

Output (--save): gpu_params.json with best fast_w, slow_w, threshold per symbol.
bybit_trader.py loads this file on startup to skip warm-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))


def _load_prices(zip_path: str, bar_size: int = 10) -> List[float]:
    """Load tick prices from zip, resample to bars.
    Uses CuPy for resampling if available (faster on GPU)."""
    from data import load_agg_trades_csv
    trades = load_agg_trades_csv(zip_path)
    prices = [t.price for t in trades]

    # CuPy fast-path: reshape on GPU
    try:
        import cupy as cp
        arr = cp.array(prices, dtype=cp.float32)
        n   = len(arr)
        # Trim to multiple of bar_size, take last of each bar
        trim = (n // bar_size) * bar_size
        bars = arr[:trim].reshape(-1, bar_size)[:, -1]  # last price of each bar
        result = bars.get().tolist()
        # Append last partial bar if any
        if n % bar_size:
            result.append(prices[-1])
        return result
    except Exception:
        pass

    # Pure Python fallback
    bars = []
    for i in range(0, len(prices), bar_size):
        end = min(i + bar_size, len(prices))
        bars.append(prices[end - 1])
    return bars


def optimize_gpu(
    price_bufs: Dict[str, List[float]],
    t_target: int = 60_000,
    fine: bool = False,
) -> Optional[Dict]:
    """Run GPU MA sweep across all symbols in price_bufs.

    Args:
        price_bufs: {symbol: [prices]}
        t_target:   tensor length (larger = more GPU load)
        fine:       if True, use finer MA grid (slower, more accurate)

    Returns:
        dict with best_fw, best_sw, best_thr, sharpe, util_pct, vram_mb
        or None if torch/CUDA unavailable.
    """
    try:
        import torch
    except ImportError:
        print("torch not installed — GPU optimization unavailable")
        return None

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not torch.cuda.is_available():
        print("CUDA not available — running on CPU (slow)")

    syms = [s for s, b in price_bufs.items() if len(b) >= 30]
    if not syms:
        return None

    N = len(syms)
    print(f"  GPU optimizing {N} symbols on {dev}  (T={t_target:,})", flush=True)

    # ── Build price tensor: tile each buffer to T_TARGET ──────────────────────
    t0 = time.monotonic()
    rows = []
    for sym in syms:
        b   = price_bufs[sym]
        arr = torch.tensor(b, dtype=torch.float32, device=dev)
        reps = (t_target // max(1, len(arr))) + 2
        tiled = arr.repeat(reps)[:t_target]
        noise = torch.randn_like(tiled) * (tiled.std() * 0.001 + 1e-6)
        rows.append(tiled + noise)

    prices_t = torch.stack(rows)           # (N, T)
    T = prices_t.shape[1]

    # Returns
    ret = torch.zeros_like(prices_t)
    ret[:, 1:] = (prices_t[:, 1:] - prices_t[:, :-1]) / (prices_t[:, :-1].abs() + 1e-9)

    # Cumsum for fast rolling mean
    cs = torch.zeros(N, T + 1, device=dev)
    cs[:, 1:] = prices_t.cumsum(1)

    def rolling_mean(w: int):
        m   = (cs[:, w:] - cs[:, :T - w + 1]) / w
        pad = torch.full((N, w - 1), float("nan"), device=dev)
        return torch.cat([pad, m], 1)

    # ── MA grid ───────────────────────────────────────────────────────────────
    if fine:
        fg_r = torch.arange(2, 31, device=dev, dtype=torch.int32)   # fast: 2-30
        sg_r = torch.arange(5, 121, device=dev, dtype=torch.int32)  # slow: 5-120
    else:
        fg_r = torch.arange(2, 21, device=dev, dtype=torch.int32)   # fast: 2-20
        sg_r = torch.arange(8, 81, device=dev, dtype=torch.int32)   # slow: 8-80

    fg, sg = torch.meshgrid(fg_r, sg_r, indexing="ij")
    fg = fg.reshape(-1); sg = sg.reshape(-1)
    mask = fg < sg; fg = fg[mask]; sg = sg[mask]
    C = fg.shape[0]

    # Pre-compute all unique window MAs
    uw  = torch.unique(torch.cat([fg, sg])).tolist()
    mc  = {int(w): rolling_mean(int(w)) for w in uw}
    fgl = fg.tolist(); sgl = sg.tolist()

    thr_list = torch.logspace(-5, -3, 20, device=dev)
    BATCH    = 64

    best_s   = -999.0
    best_fw  = 5; best_sw = 20; best_thr = 0.00003
    ret_b    = ret.unsqueeze(0)   # (1, N, T)

    for bs in range(0, C, BATCH):
        be  = min(bs + BATCH, C); B = be - bs
        fma = torch.stack([mc[int(fgl[i])] for i in range(bs, be)])  # (B, N, T)
        sma = torch.stack([mc[int(sgl[i])] for i in range(bs, be)])
        gm  = torch.nan_to_num((fma - sma) / (sma.abs() + 1e-9), nan=0.0)

        for thr in thr_list.tolist():
            sig = (gm > thr).float() - (gm < -thr).float()    # (B, N, T)
            pnl = sig[:, :, :-1] * ret_b[:, :, 1:]            # (B, N, T-1)
            pfl = pnl.reshape(B, -1)
            sh  = (pfl.mean(1) / (pfl.std(1) + 1e-9)) * (252 ** 0.5)
            idx = int(sh.argmax().item())
            if sh[idx].item() > best_s:
                best_s   = sh[idx].item()
                best_fw  = int(fg[bs + idx].item())
                best_sw  = int(sg[bs + idx].item())
                best_thr = float(thr)

        del fma, sma, gm

    elapsed = time.monotonic() - t0

    # Real GPU utilization via nvidia-smi (more accurate than wall-clock proxy)
    util_pct = 0.0
    vram_mb  = 0
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            util_pct = float(parts[0].strip())
            vram_mb  = int(parts[1].strip())
    except Exception:
        vram_mb = int(torch.cuda.memory_allocated(dev) // 1024**2)

    return {
        "fast_w":    best_fw,
        "slow_w":    best_sw,
        "threshold": best_thr,
        "sharpe":    best_s,
        "elapsed_s": round(elapsed, 2),
        "util_pct":  util_pct,
        "vram_mb":   vram_mb,
    }


def optimize_per_symbol(
    price_bufs: Dict[str, List[float]],
    t_target: int = 60_000,
    fine: bool = False,
) -> Dict[str, Dict]:
    """Run GPU optimizer per symbol independently, return best params each."""
    results = {}
    for sym, prices in price_bufs.items():
        if len(prices) < 30:
            continue
        r = optimize_gpu({sym: prices}, t_target=t_target, fine=fine)
        if r:
            results[sym] = r
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU MA optimizer on historical data")
    parser.add_argument("--symbol",   nargs="+",
                        help="Symbols to optimize (default: all in data/)")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--bar-size", type=int, default=10,
                        help="Ticks per bar (default 10)")
    parser.add_argument("--t-target", type=int, default=60_000,
                        help="GPU tensor length (default 60000 ≈ 70% util)")
    parser.add_argument("--fine",  action="store_true",
                        help="Finer MA grid (2-30 fast, 5-120 slow)")
    parser.add_argument("--per-symbol", action="store_true",
                        help="Optimize each symbol independently")
    parser.add_argument("--save",  action="store_true",
                        help="Save results to gpu_params.json")
    parser.add_argument("--days",  type=int, default=0,
                        help="Use only last N days of data (0=all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ── Discover files ────────────────────────────────────────────────────────
    if args.symbol:
        syms = args.symbol
    else:
        syms = sorted({z.name.split("-aggTrades-")[0]
                       for z in data_dir.glob("*-aggTrades-*.zip")})

    if not syms:
        print("No data files found. Run download_data.py first.")
        sys.exit(1)

    print(f"\nGPU Optimizer — {len(syms)} symbols")
    print(f"  Grid: {'fine (2-30/5-120)' if args.fine else 'normal (2-20/8-80)'}")
    print(f"  T_TARGET: {args.t_target:,}")
    if args.days:
        print(f"  Using last {args.days} days only")
    print()

    # ── Load price buffers ────────────────────────────────────────────────────
    price_bufs: Dict[str, List[float]] = {}
    for sym in syms:
        files = sorted(data_dir.glob(f"{sym}-aggTrades-*.zip"))
        if args.days:
            files = files[-args.days:]
        if not files:
            print(f"  {sym}: no files found — skipping")
            continue
        prices = []
        for f in files:
            prices.extend(_load_prices(str(f), args.bar_size))
        price_bufs[sym] = prices
        print(f"  Loaded {sym}: {len(prices):,} bars from {len(files)} files")

    if not price_bufs:
        print("No price data loaded.")
        sys.exit(1)

    print()

    # ── Run optimization ──────────────────────────────────────────────────────
    t_start = time.monotonic()

    if args.per_symbol:
        print("Running per-symbol optimization...\n")
        sym_results = optimize_per_symbol(price_bufs, args.t_target, args.fine)
        print()
        print("=" * 70)
        print("RESULTS PER SYMBOL")
        print("=" * 70)
        for sym, r in sym_results.items():
            print(f"  {sym:<12}  fast={r['fast_w']:>2}  slow={r['slow_w']:>3}  "
                  f"thr={r['threshold']:.6f}  sharpe={r['sharpe']:+.2f}  "
                  f"GPU={r['util_pct']:.0f}%  {r['elapsed_s']:.1f}s")
    else:
        print("Running joint optimization (all symbols together)...\n")
        r = optimize_gpu(price_bufs, args.t_target, args.fine)
        sym_results = {sym: r for sym in price_bufs} if r else {}
        if r:
            print("=" * 70)
            print("JOINT RESULT (same params for all symbols)")
            print("=" * 70)
            print(f"  fast_w={r['fast_w']}  slow_w={r['slow_w']}  "
                  f"threshold={r['threshold']:.6f}")
            print(f"  sharpe={r['sharpe']:+.2f}  elapsed={r['elapsed_s']:.1f}s  "
                  f"GPU util={r['util_pct']:.0f}%  VRAM={r['vram_mb']}MB")

    total = time.monotonic() - t_start
    print(f"\nTotal time: {total:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.save and sym_results:
        out = {}
        for sym, r in sym_results.items():
            if r:
                out[sym] = {
                    "fast_w":    r["fast_w"],
                    "slow_w":    r["slow_w"],
                    "threshold": r["threshold"],
                    "sharpe":    round(r.get("sharpe", 0.0), 4),
                }
        save_path = Path(os.path.dirname(__file__)) / "gpu_params.json"
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {save_path}")
        print("bybit_trader.py will load these params on next startup.")


if __name__ == "__main__":
    main()
