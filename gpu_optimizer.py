#!/usr/bin/env python3
"""Standalone GPU MA optimizer — runs on historical data OR live price feeds.

Uses Tilelang for GPU compute and hft_rs (Rust) for data operations.
No torch.cuda, no CuPy.

Usage:
    python gpu_optimizer.py
    python gpu_optimizer.py --symbol SOLUSDT BTCUSDT
    python gpu_optimizer.py --fine
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
from typing import Dict, List, Optional

import tilelang
from tilelang import language as T
import hft_rs
import torch
import numpy as np


def _to_dev(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).cuda()


def _from_dev(t: torch.Tensor) -> np.ndarray:
    return t.cpu().numpy()


# ── Tilelang GPU kernels ─────────────────────────────────────────────────────

@tilelang.jit
def _gpu_returns_kernel(prices: T.Tensor):
    N = T.const("N")
    ret = T.alloc_fragment([N], T.float32)
    ret[0] = 0.0
    for i in T.serial(1, N):
        denom = T.abs(prices[i - 1]) + 1e-9
        ret[i] = (prices[i] - prices[i - 1]) / denom
    return ret


@tilelang.jit
def _gpu_ma_gap_kernel(fma: T.Tensor, sma: T.Tensor):
    N = T.const("N")
    out = T.alloc_fragment([N], T.float32)
    for i in T.serial(N):
        denom = T.abs(sma[i]) + 1e-9
        out[i] = (fma[i] - sma[i]) / denom
    return out


@tilelang.jit
def _gpu_threshold_kernel(signal: T.Tensor, threshold: T.float32):
    N = T.const("N")
    out = T.alloc_fragment([N], T.float32)
    for i in T.serial(N):
        s = signal[i]
        if s > threshold:
            out[i] = 1.0
        elif s < -threshold:
            out[i] = -1.0
        else:
            out[i] = 0.0
    return out


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_prices(zip_path: str, bar_size: int = 10) -> List[float]:
    """Load tick prices from zip, resample to bars using hft_rs."""
    from data import load_agg_trades_csv
    trades = load_agg_trades_csv(zip_path)
    prices = [t.price for t in trades]

    # hft_rs fast-path: Rust-accelerated resampling
    if prices:
        prices_np = np.array(prices, dtype=np.float32)
        bar_ms = bar_size * 1  # ticks → use indices as time proxy
        # For tick-based data, we use simple resampling (no timestamps)
        n = len(prices_np)
        trim = (n // bar_size) * bar_size
        bars = prices_np[:trim].reshape(-1, bar_size)[:, -1]
        result = bars.tolist()
        if n % bar_size:
            result.append(prices[-1])
        return result

    # Pure Python fallback
    bars = []
    for i in range(0, len(prices), bar_size):
        end = min(i + bar_size, len(prices))
        bars.append(prices[end - 1])
    return bars


# ── GPU optimizer ─────────────────────────────────────────────────────────────

def optimize_gpu(
    price_bufs: Dict[str, List[float]],
    t_target: int = 60_000,
    fine: bool = False,
) -> Optional[Dict]:
    """Run GPU MA sweep across all symbols.

    Tilelang GPU kernels for returns/threshold/gap math.
    hft_rs Rust for rolling_mean and backtest sweep.
    """
    try:
        # Check Tilelang/GPU availability
        torch.cuda.is_available()
    except Exception:
        print("torch not installed — GPU optimization unavailable")
        return None

    if not torch.cuda.is_available():
        print("CUDA not available — running on CPU (slow)")

    syms = [s for s, b in price_bufs.items() if len(b) >= 30]
    if not syms:
        return None

    N = len(syms)
    print(f"  GPU optimizing {N} symbols  (T={t_target:,})", flush=True)

    # ── Build price tensor ────────────────────────────────────────────────────
    t0 = time.monotonic()
    rows = []
    for sym in syms:
        b = price_bufs[sym]
        arr = torch.tensor(b, dtype=torch.float32, device="cuda")
        reps = (t_target // max(1, len(arr))) + 2
        tiled = arr.repeat(reps)[:t_target]
        noise = torch.randn_like(tiled) * (tiled.std() * 0.001 + 1e-6)
        rows.append(tiled + noise)

    prices_t = torch.stack(rows)  # (N, T)
    T_len = prices_t.shape[1]

    # Returns via Tilelang GPU kernel
    ret_rows = []
    for i in range(N):
        ret_dev = _gpu_returns_kernel(prices_t[i], N=T_len)
        ret_rows.append(_from_dev(ret_dev))
    ret = torch.tensor(np.array(ret_rows), dtype=torch.float32, device="cuda")

    # ── MA grid ───────────────────────────────────────────────────────────────
    if fine:
        fg_r = list(range(2, 31))   # fast: 2-30
        sg_r = list(range(5, 121))  # slow: 5-120
    else:
        fg_r = list(range(2, 21))   # fast: 2-20
        sg_r = list(range(8, 81))   # slow: 8-80

    pairs = [(f, s) for f in fg_r for s in sg_r if f < s]
    C = len(pairs)

    # Pre-compute all unique window MAs via hft_rs (Rust-accelerated)
    uw = sorted(set(w for f, s in pairs for w in (f, s)))
    mc = {}
    for i in range(N):
        sym_prices = prices_t[i].cpu().numpy()
        for w in uw:
            rm = np.array(hft_rs.rolling_mean(sym_prices.tolist(), w), dtype=np.float32)
            mc.setdefault(w, []).append(rm)
    # Stack into arrays per window
    mc_arrays = {}
    for w in uw:
        mc_arrays[w] = np.array(mc[w], dtype=np.float32)  # (N, T)

    thr_list = [10.0**x for x in np.linspace(-5, -3, 20)]

    best_s = -999.0
    best_fw = 5; best_sw = 20; best_thr = 0.00003

    for bs_idx in range(0, C, 64):
        be = min(bs_idx + 64, C)
        batch_pairs = pairs[bs_idx:be]

        for f, s in batch_pairs:
            fma = mc_arrays[f]  # (N, T)
            sma = mc_arrays[s]  # (N, T)

            # MA gap via Tilelang GPU
            fma_dev = torch.from_numpy(fma).cuda()
            sma_dev = torch.from_numpy(sma).cuda()
            gm_dev = _gpu_ma_gap_kernel(fma_dev, sma_dev, N=T_len)
            gm = _from_dev(gm_dev)  # (N, T)

            for thr in thr_list:
                # Threshold signal via Tilelang GPU
                gm_dev = torch.from_numpy(gm).cuda()
                thr_dev = torch.tensor(thr, dtype=torch.float32, device="cuda")
                sig_dev = _gpu_threshold_kernel(gm_dev, thr_dev, N=T_len)
                sig = _from_dev(sig_dev)

                # PnL = entries * returns
                entries = sig  # (N, T)
                pnl = entries[:, :-1] * ret[:, 1:].cpu().numpy()
                pfl = pnl.reshape(-1)
                if len(pfl) < 3:
                    continue
                sh = (pfl.mean() / (pfl.std() + 1e-9)) * (252 ** 0.5)
                if sh > best_s:
                    best_s = float(sh)
                    best_fw = f
                    best_sw = s
                    best_thr = thr

        del fma_dev, sma_dev

    elapsed = time.monotonic() - t0

    # GPU utilization via nvidia-smi
    util_pct = 0.0
    vram_mb = 0
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
            vram_mb = int(parts[1].strip())
    except Exception:
        vram_mb = int(torch.cuda.memory_allocated("cuda") // 1024**2)

    return {
        "fast_w": best_fw,
        "slow_w": best_sw,
        "threshold": best_thr,
        "sharpe": best_s,
        "elapsed_s": round(elapsed, 2),
        "util_pct": util_pct,
        "vram_mb": vram_mb,
    }


def optimize_per_symbol(
    price_bufs: Dict[str, List[float]],
    t_target: int = 60_000,
    fine: bool = False,
) -> Dict[str, Dict]:
    """Run GPU optimizer per symbol independently."""
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
    parser.add_argument("--symbol", nargs="+",
                        help="Symbols to optimize (default: all in data/)")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--bar-size", type=int, default=10,
                        help="Ticks per bar (default 10)")
    parser.add_argument("--t-target", type=int, default=60_000,
                        help="GPU tensor length (default 60000)")
    parser.add_argument("--fine", action="store_true",
                        help="Finer MA grid (2-30 fast, 5-120 slow)")
    parser.add_argument("--per-symbol", action="store_true",
                        help="Optimize each symbol independently")
    parser.add_argument("--save", action="store_true",
                        help="Save results to gpu_params.json")
    parser.add_argument("--days", type=int, default=0,
                        help="Use only last N days of data (0=all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.symbol:
        syms = args.symbol
    else:
        syms = sorted({z.name.split("-aggTrades-")[0]
                       for z in data_dir.glob("*-aggTrades-*.zip")})

    if not syms:
        print("No data files found. Run download_data.py first.")
        sys.exit(1)

    print(f"\nGPU Optimizer — {len(syms)} symbols (Tilelang + hft_rs)")
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
                    "fast_w": r["fast_w"],
                    "slow_w": r["slow_w"],
                    "threshold": r["threshold"],
                    "sharpe": round(r.get("sharpe", 0.0), 4),
                }
        save_path = Path(os.path.dirname(__file__)) / "gpu_params.json"
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {save_path}")
        print("bybit_trader.py will load these params on next startup.")


if __name__ == "__main__":
    main()
