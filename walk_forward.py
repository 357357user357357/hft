#!/usr/bin/env python3
"""Walk-forward validation: train on N days, test on M days, roll forward.

Prevents overfitting by never testing on data used for parameter selection.

GPU track:   MA cross-over sweep via torch (fast, runs once per fold on GPU)
CPU track:   Signal grid search via ProcessPoolExecutor (all CPU cores)
Both run in parallel; results merged in final report.

Usage:
    python walk_forward.py                              # all data/*.zip
    python walk_forward.py --symbol SOLUSDT BTCUSDT
    python walk_forward.py --train-days 5 --test-days 2
    python walk_forward.py --no-slow                    # skip expensive signals
    python walk_forward.py --no-gpu                     # CPU-only
    python walk_forward.py --no-cpu                     # GPU MA track only
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from data import load_agg_trades_csv
from signal_backtest import SignalBacktest, SignalConfig, SignalStats

# ── CPU signal track ───────────────────────────────────────────────────────────
# Walk-forward v2 winners: Simons OU, Momentum, Volatility are the only edges.
# Dead/negative signals: Hurst (0 trades), Autocorr (overfit), Order Flow (overfit),
#                        Poincare, Polar, Torsion (removed earlier).
SIGNALS: List[Tuple[str, str]] = [
    ("momentum",   "Momentum/RSI"),
    ("simons",     "Simons OU"),
    ("volatility", "Volatility"),
]
# Stubbed (kept but not deployed):
_STUBBED: List[Tuple[str, str]] = [
    ("order_flow", "Order Flow"),
    ("hurst", "Hurst"),
]

# Wider thresholds work better on 5-min bars
THRESHOLDS = [0.40, 0.55, 0.70]
HOLD_BARS  = [5, 15, 30]


def _find_data_files(data_dir: Path, symbol: str) -> List[Path]:
    """Find all daily zip files for a symbol, sorted by date."""
    return sorted(data_dir.glob(f"{symbol}-aggTrades-*.zip"))


def _resample_gpu(prices_np, qtys_np, times_np, bar_seconds: int) -> Tuple[List[float], List[float]]:
    """GPU-accelerated resampling via hft_rs (Rust). Replaces CuPy scatter ops."""
    import hft_rs
    return hft_rs.resample_bars(
        times_np.tolist(),
        prices_np.tolist(),
        qtys_np.tolist(),
        bar_seconds,
    )


def _load_bars(zip_path: str, bar_seconds: int = 300) -> Tuple[List[float], List[float]]:
    """Load trades and resample to time-based bars.

    Priority:
    1. Parquet file alongside zip (0.5s via pyarrow+hft_rs — 4.7x faster)
    2. Zip CSV with hft_rs Rust resampling (2.2s)
    3. Pure CPU fallback (4.6s)

    Run convert_to_parquet.py to pre-convert all zip files to Parquet.
    """
    import csv as _csv
    import zipfile as _zf
    import io as _io

    # ── Parquet fast-path ──────────────────────────────────────────────────────
    pq_path = Path(zip_path).with_suffix(".parquet")
    if pq_path.exists():
        try:
            import pyarrow.parquet as pq
            import numpy as np

            t = pq.read_table(str(pq_path), columns=["price", "qty", "transact_time"])
            prices_np = t["price"].to_numpy().astype("float32")
            qtys_np   = t["qty"].to_numpy().astype("float32")
            times_np  = t["transact_time"].to_numpy().astype("int64")

            try:
                return _resample_gpu(prices_np, qtys_np, times_np, bar_seconds)
            except Exception:
                pass  # fall through to CPU resample

            # CPU resample on numpy arrays (faster than list version)
            bar_ms  = bar_seconds * 1000
            t_start = int((times_np[0]  // bar_ms) * bar_ms)
            t_end   = int((times_np[-1] // bar_ms) * bar_ms)
            n_bars  = (t_end - t_start) // bar_ms + 1
            bar_idx = np.clip((times_np - t_start) // bar_ms, 0, n_bars - 1)
            vol_bars   = np.zeros(n_bars, dtype="float32")
            np.add.at(vol_bars, bar_idx, qtys_np)
            close_bars = np.zeros(n_bars, dtype="float32")
            for i in range(len(times_np)):
                close_bars[bar_idx[i]] = prices_np[i]  # last wins
            last = float(prices_np[0])
            for i in range(n_bars):
                if close_bars[i] == 0.0:
                    close_bars[i] = last
                else:
                    last = close_bars[i]
            return close_bars.tolist(), vol_bars.tolist()

        except Exception:
            pass  # fall through to zip CSV

    # ── Zip CSV fast parse + GPU resample ─────────────────────────────────────
    with _zf.ZipFile(zip_path, "r") as z:
        with z.open(z.namelist()[0]) as inner:
            text = _io.TextIOWrapper(inner, encoding="utf-8")
            reader = _csv.reader(text)
            header = next(reader)
            try:
                price_col = header.index("price")
                qty_col   = header.index("qty") if "qty" in header else header.index("quantity")
                time_col  = header.index("transact_time")
            except ValueError:
                price_col, qty_col, time_col = 1, 2, 5
            raw = [(float(r[price_col]), float(r[qty_col]), int(r[time_col])) for r in reader]

    if not raw:
        return [], []

    import numpy as _np
    prices_np = _np.array([r[0] for r in raw], dtype="float32")
    qtys_np   = _np.array([r[1] for r in raw], dtype="float32")
    times_np  = _np.array([r[2] for r in raw], dtype="int64")

    try:
        return _resample_gpu(prices_np, qtys_np, times_np, bar_seconds)
    except Exception:
        pass

    # ── Pure CPU fallback ──────────────────────────────────────────────────────
    bar_ms  = bar_seconds * 1000
    t_start = (raw[0][2]  // bar_ms) * bar_ms
    t_end   = (raw[-1][2] // bar_ms) * bar_ms
    prices_out: List[float] = []
    vols_out:   List[float] = []
    last_close = raw[0][0]
    t = t_start
    idx = 0
    n = len(raw)
    while t <= t_end:
        bc = t + bar_ms
        vol = 0.0; close = last_close
        while idx < n and raw[idx][2] < bc:
            close = raw[idx][0]; vol += raw[idx][1]; idx += 1
        prices_out.append(close); vols_out.append(vol)
        last_close = close; t += bar_ms
    return prices_out, vols_out


def _load_bars_parallel(
    file_paths: List[str],
    bar_seconds: int = 300,
    max_workers: int = 8,
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Load multiple zip files in parallel using I/O threads.

    CSV parsing is I/O-bound (zip decompression + disk read). Parallel threads
    overlap the I/O of multiple files, giving ~4-8x speedup vs sequential.
    Uses ThreadPoolExecutor so all results share memory (no fork overhead).
    """
    cache: Dict[str, Tuple[List[float], List[float]]] = {}
    lock  = threading.Lock()

    def _load_one(path: str):
        result = _load_bars(path, bar_seconds)
        with lock:
            cache[path] = result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_load_one, p): p for p in file_paths}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
                if done % 10 == 0 or done == len(file_paths):
                    print(f"  Loading... {done}/{len(file_paths)} files", flush=True)
            except Exception as e:
                path = futures[fut]
                print(f"  WARNING: failed to load {path}: {e}", flush=True)

    return cache


# ── CPU: signal grid search ────────────────────────────────────────────────────

def _optimize_on_train(
    prices: List[float], volumes: List[float],
    sig_key: str, lookback: int,
) -> Tuple[float, int, float]:
    """Grid search best (threshold, hold_bars) on training data."""
    best_sharpe = -999.0
    best_thr    = THRESHOLDS[len(THRESHOLDS) // 2]
    best_hold   = HOLD_BARS[len(HOLD_BARS) // 2]

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
    train_p: List[float], train_v: List[float],
    test_p:  List[float], test_v:  List[float],
    lookback: int, fold_idx: int,
) -> Dict:
    """One walk-forward fold: optimize on train, evaluate on test.
    Accepts pre-loaded bar arrays (caller handles data loading/caching)."""
    if len(train_p) < lookback + 20 or len(test_p) < lookback + 20:
        return {"symbol": symbol, "signal": sig_name, "fold": fold_idx,
                "error": "insufficient data", "track": "cpu"}

    best_thr, best_hold, train_sharpe = _optimize_on_train(
        train_p, train_v, sig_key, lookback)

    try:
        test_stats = _test_with_params(
            test_p, test_v, sig_key, lookback, best_thr, best_hold)
    except Exception as e:
        return {"symbol": symbol, "signal": sig_name, "fold": fold_idx,
                "error": str(e), "track": "cpu"}

    return {
        "symbol": symbol, "signal": sig_name, "fold": fold_idx,
        "train_sharpe": train_sharpe,
        "test_sharpe":  test_stats.sharpe_ratio,
        "test_pnl":     test_stats.total_pnl_pct,
        "test_trades":  test_stats.total_trades,
        "test_winrate": test_stats.win_rate,
        "best_thr": best_thr, "best_hold": best_hold,
        "track": "cpu",
    }


# ── GPU: MA sweep + Simons OU/Fourier signals per fold ────────────────────────

def _rolling_mean_gpu(cs: "torch.Tensor", T: int, w: int, device) -> "torch.Tensor":
    """Rolling mean from cumsum tensor. Returns length-T tensor."""
    import torch
    m   = (cs[w:] - cs[:T - w + 1]) / w
    pad = torch.full((w - 1,), float("nan"), device=device)
    return torch.cat([pad, m])


def _oos_sharpe_gpu(prices_t: "torch.Tensor", sig_t: "torch.Tensor", device) -> Tuple[float, float, float, int]:
    """Compute OOS Sharpe/PnL/WinRate/nTrades from price and signal tensors."""
    import torch
    T = len(prices_t)
    ret = torch.zeros(T, device=device)
    ret[1:] = (prices_t[1:] - prices_t[:-1]) / (prices_t[:-1].abs() + 1e-9)
    pnl = sig_t[:-1] * ret[1:]
    valid = ~torch.isnan(pnl)
    pnl = pnl[valid]
    if len(pnl) < 5:
        return 0.0, 0.0, 0.0, 0
    sh      = (pnl.mean() / (pnl.std() + 1e-9) * (252 ** 0.5)).item()
    total   = (pnl.sum() * 100).item()
    win_r   = float((pnl > 0).float().mean().item())
    n_trade = int(((sig_t[:-1] != 0) & (sig_t[:-1] != sig_t[1:])).sum().item())
    return sh, total, win_r, n_trade


def _gpu_ma_sweep(train_t, T: int, device) -> Tuple[int, int, float, float]:
    """GPU MA crossover sweep on train data. Returns (fast_w, slow_w, threshold, sharpe)."""
    import torch
    cs = torch.zeros(T + 1, device=device)
    cs[1:] = train_t.cumsum(0)
    ret = torch.zeros(T, device=device)
    ret[1:] = (train_t[1:] - train_t[:-1]) / (train_t[:-1].abs() + 1e-9)

    fg_r = torch.arange(2, 21, device=device, dtype=torch.int32)
    sg_r = torch.arange(8, 61, device=device, dtype=torch.int32)
    fg, sg = torch.meshgrid(fg_r, sg_r, indexing="ij")
    fg = fg.reshape(-1); sg = sg.reshape(-1)
    mask = fg < sg; fg = fg[mask]; sg = sg[mask]

    uw = torch.unique(torch.cat([fg, sg])).tolist()
    mc = {int(w): _rolling_mean_gpu(cs, T, int(w), device) for w in uw}

    thr_list = torch.logspace(-5, -3, 12, device=device)
    best_s = -999.0; best_fw = 5; best_sw = 20; best_thr = 3e-5

    fgl = fg.tolist(); sgl = sg.tolist()
    for bs in range(0, len(fgl), 128):
        be  = min(bs + 128, len(fgl))
        fma = torch.stack([mc[int(fgl[i])] for i in range(bs, be)])
        sma = torch.stack([mc[int(sgl[i])] for i in range(bs, be)])
        gm  = torch.nan_to_num((fma - sma) / (sma.abs() + 1e-9), nan=0.0)
        for thr in thr_list.tolist():
            sig = (gm > thr).float() - (gm < -thr).float()
            pnl = sig[:, :-1] * ret[1:]
            sh  = (pnl.mean(1) / (pnl.std(1) + 1e-9)) * (252 ** 0.5)
            idx = int(sh.argmax().item())
            if sh[idx].item() > best_s:
                best_s = sh[idx].item()
                best_fw = int(fg[bs + idx].item())
                best_sw = int(sg[bs + idx].item())
                best_thr = float(thr)
        del fma, sma, gm

    return best_fw, best_sw, best_thr, best_s


def _gpu_ou_signal(prices_t: "torch.Tensor", window: int, z_thr: float, device) -> "torch.Tensor":
    """OU (Simons-style) z-score signal on GPU.

    Uses rolling OLS of x[t] ~ x[t-1] to estimate OU params, then
    computes z-score of residual vs rolling mean:
      z > +thr → short (mean reversion: price above equilibrium)
      z < -thr → long  (price below equilibrium)
    Returns signal tensor (same length as prices_t).
    """
    import torch
    T = len(prices_t)
    sig = torch.zeros(T, device=device)
    if T < window + 5:
        return sig

    cs = torch.zeros(T + 1, device=device)
    cs[1:] = prices_t.cumsum(0)
    cs2 = torch.zeros(T + 1, device=device)
    cs2[1:] = (prices_t ** 2).cumsum(0)

    # Rolling mean and std using cumsum
    for i in range(window, T):
        lo = i - window
        n  = float(window)
        mu  = (cs[i] - cs[lo]) / n
        mu2 = (cs2[i] - cs2[lo]) / n
        var = mu2 - mu * mu
        std = (var.clamp(min=1e-12)).sqrt()
        z   = (prices_t[i] - mu) / (std + 1e-9)
        # Mean-reversion: high z → sell, low z → buy (Simons OU logic)
        if z > z_thr:
            sig[i] = -1.0   # short: price above equilibrium
        elif z < -z_thr:
            sig[i] = 1.0    # long:  price below equilibrium

    return sig


def _gpu_ou_sweep(train_t, T: int, device) -> Tuple[int, float, float]:
    """Sweep OU window and z-score threshold on train data. Returns (window, z_thr, sharpe)."""
    import torch
    ret = torch.zeros(T, device=device)
    ret[1:] = (train_t[1:] - train_t[:-1]) / (train_t[:-1].abs() + 1e-9)

    windows = [10, 20, 30, 50, 80]
    z_thrs  = [0.5, 0.8, 1.2, 1.5, 2.0]
    best_s = -999.0; best_w = 20; best_z = 1.0

    for w in windows:
        for z_thr in z_thrs:
            sig = _gpu_ou_signal(train_t, w, z_thr, device)
            pnl = sig[:-1] * ret[1:]
            valid = ~torch.isnan(pnl)
            pnl = pnl[valid]
            if len(pnl) < 10:
                continue
            sh = (pnl.mean() / (pnl.std() + 1e-9) * (252 ** 0.5)).item()
            if sh > best_s:
                best_s = sh; best_w = w; best_z = z_thr

    return best_w, best_z, best_s


def _gpu_fold(
    symbol: str, fold_idx: int,
    train_prices: List[float], test_prices: List[float],
    device,
) -> List[Dict]:
    """GPU fold: runs MA crossover + Simons OU signal. Returns list of results."""
    results = []
    try:
        import torch

        if len(train_prices) < 30 or len(test_prices) < 10:
            err = {"symbol": symbol, "signal": "GPU MA", "fold": fold_idx,
                   "error": "insufficient data", "track": "gpu"}
            return [err, {**err, "signal": "Simons OU (GPU)"}]

        # Tile train prices to T for MA sweep
        T = min(20_000, max(len(train_prices) * 3, 500))
        arr = torch.tensor(train_prices, dtype=torch.float32, device=device)
        reps = (T // max(1, len(arr))) + 2
        train_t = arr.repeat(reps)[:T]
        noise = torch.randn_like(train_t) * (train_t.std() * 0.001 + 1e-6)
        train_t = train_t + noise

        tp = torch.tensor(test_prices, dtype=torch.float32, device=device)

        # ── MA crossover track ─────────────────────────────────────────────────
        best_fw, best_sw, best_thr, train_sh = _gpu_ma_sweep(train_t, T, device)

        if len(test_prices) >= best_sw + 5:
            cs2 = torch.zeros(len(tp) + 1, device=device); cs2[1:] = tp.cumsum(0)
            fma2 = _rolling_mean_gpu(cs2, len(tp), best_fw, device)
            sma2 = _rolling_mean_gpu(cs2, len(tp), best_sw, device)
            gap2 = torch.nan_to_num((fma2 - sma2) / (sma2.abs() + 1e-9), nan=0.0)
            sig2 = (gap2 > best_thr).float() - (gap2 < -best_thr).float()
            oos_sh, oos_pnl, win_r, n_tr = _oos_sharpe_gpu(tp, sig2, device)
        else:
            oos_sh, oos_pnl, win_r, n_tr = 0.0, 0.0, 0.0, 0

        results.append({
            "symbol": symbol, "signal": "GPU MA", "fold": fold_idx,
            "train_sharpe": train_sh, "test_sharpe": oos_sh,
            "test_pnl": oos_pnl, "test_trades": n_tr, "test_winrate": win_r,
            "best_thr": best_thr, "best_hold": 0, "track": "gpu",
            "gpu_fw": best_fw, "gpu_sw": best_sw,
        })

        # ── Simons OU (mean-reversion) track ───────────────────────────────────
        # Use actual (non-tiled) train prices for OU calibration
        arr_real = torch.tensor(train_prices, dtype=torch.float32, device=device)
        ou_w, ou_z, ou_train_sh = _gpu_ou_sweep(arr_real, len(arr_real), device)
        sig_ou = _gpu_ou_signal(tp, ou_w, ou_z, device)
        ou_sh, ou_pnl, ou_wr, ou_n = _oos_sharpe_gpu(tp, sig_ou, device)

        results.append({
            "symbol": symbol, "signal": "Simons OU (GPU)", "fold": fold_idx,
            "train_sharpe": ou_train_sh, "test_sharpe": ou_sh,
            "test_pnl": ou_pnl, "test_trades": ou_n, "test_winrate": ou_wr,
            "best_thr": ou_z, "best_hold": 0, "track": "gpu",
            "gpu_ou_window": ou_w, "gpu_ou_zthr": ou_z,
        })

        del train_t, tp, arr_real

    except Exception as e:
        results.append({"symbol": symbol, "signal": "GPU MA", "fold": fold_idx,
                        "error": str(e), "track": "gpu"})
        results.append({"symbol": symbol, "signal": "Simons OU (GPU)", "fold": fold_idx,
                        "error": str(e), "track": "gpu"})

    return results


def _run_gpu_track(
    symbols: List[str],
    symbol_fold_arrays: Dict[str, List[Tuple[List[float], List[float], List[float], List[float]]]],
    gpu_results: List[Dict],
    lock: threading.Lock,
    total_gpu: int,
) -> None:
    """Background thread: run GPU MA + Simons OU walk-forward for all symbols/folds.
    symbol_fold_arrays: {sym: [(train_p, train_v, test_p, test_v), ...]}"""
    try:
        import torch
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        print("  [GPU] torch not available — skipping GPU track", flush=True)
        return

    print(f"  [GPU] Starting on {dev}  ({total_gpu} fold-symbols)", flush=True)
    done = 0

    for sym in symbols:
        folds = symbol_fold_arrays.get(sym, [])
        for fold_idx, (train_p, train_v, test_p, test_v) in enumerate(folds):

            fold_results = _gpu_fold(sym, fold_idx, train_p, test_p, dev)
            done += 1

            with lock:
                gpu_results.extend(fold_results)

            for r in fold_results:
                if "error" not in r:
                    extra = ""
                    if r["signal"] == "GPU MA":
                        extra = f"fast={r.get('gpu_fw','-'):<3} slow={r.get('gpu_sw','-'):<3}  "
                    elif r["signal"] == "Simons OU (GPU)":
                        extra = f"ou_w={r.get('gpu_ou_window','-'):<3} z={r.get('gpu_ou_zthr','-'):.2f}  "
                    print(f"  [GPU {done:>3}/{total_gpu}] {sym:<10} {r['signal']:<18} fold={fold_idx}  "
                          f"{extra}train={r['train_sharpe']:+.2f}  "
                          f"oos={r['test_sharpe']:+.2f}  pnl={r['test_pnl']:+.3f}%",
                          flush=True)
                else:
                    print(f"  [GPU {done:>3}/{total_gpu}] {sym:<10} {r['signal']:<18} fold={fold_idx}  "
                          f"ERROR: {r['error']}", flush=True)

    if cuda_ok:
        import torch
        torch.cuda.empty_cache()
    print(f"  [GPU] Done ({total_gpu} folds)", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward: GPU MA sweep + CPU signal grid, run in parallel")
    parser.add_argument("--symbol",     nargs="+",
                        help="Symbols to test (default: all in data/)")
    parser.add_argument("--data-dir",   default="./data")
    parser.add_argument("--train-days", type=int, default=5,
                        help="Days per training window (default: 5)")
    parser.add_argument("--test-days",  type=int, default=2,
                        help="Days per test window (default: 2)")
    parser.add_argument("--lookback",   type=int, default=30,
                        help="Lookback bars for signal computation (default: 30)")
    parser.add_argument("--bar-size",   type=int, default=300,
                        help="Bar size in seconds (default: 300 = 5-min bars)")
    parser.add_argument("--workers",    type=int, default=None,
                        help="CPU parallel workers (default: min(cpu_count, 6))")
    parser.add_argument("--no-slow",    action="store_true",
                        help="Skip poincare/torsion signals")
    parser.add_argument("--no-gpu",     action="store_true",
                        help="Disable GPU MA track")
    parser.add_argument("--no-cpu",     action="store_true",
                        help="Disable CPU signal track (GPU only)")
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

    # ── Pre-load ALL bars in parallel (I/O threads) — avoids re-parsing zip per fold ──
    sym_files: Dict[str, List[Path]] = {}
    all_file_paths: List[str] = []
    for sym in symbols:
        files = _find_data_files(data_dir, sym)
        sym_files[sym] = files
        for f in files:
            key = str(f)
            if key not in all_file_paths:
                all_file_paths.append(key)

    n_files = len(all_file_paths)
    io_workers = min(os.cpu_count() or 4, 12, n_files)
    print(f"\nPre-loading {n_files} files with {io_workers} I/O threads "
          f"(bar_size={args.bar_size}s) ...")
    bar_cache = _load_bars_parallel(all_file_paths, args.bar_size, io_workers)
    total_bars = sum(len(p) for p, v in bar_cache.values())
    print(f"  Done: {len(bar_cache)} files, {total_bars:,} total bars\n")

    # ── Build fold arrays from cache ───────────────────────────────────────────
    # symbol_fold_arrays: {sym: [(train_p, train_v, test_p, test_v), ...]}
    symbol_fold_arrays: Dict[str, List[Tuple]] = {}
    cpu_jobs = []
    total_cpu = 0

    for sym in symbols:
        files = sym_files[sym]
        if len(files) < args.train_days + args.test_days:
            print(f"  {sym}: only {len(files)} days, need "
                  f"{args.train_days + args.test_days} — skipping")
            continue

        folds_for_sym = []
        step = args.test_days
        for start in range(0, len(files) - args.train_days - args.test_days + 1, step):
            train_keys = [str(f) for f in files[start:start + args.train_days]]
            test_keys  = [str(f) for f in files[start + args.train_days:
                                                  start + args.train_days + args.test_days]]

            # Concat from cache
            train_p: List[float] = []; train_v: List[float] = []
            for k in train_keys:
                bp, bv = bar_cache[k]; train_p.extend(bp); train_v.extend(bv)
            test_p: List[float] = []; test_v: List[float] = []
            for k in test_keys:
                bp, bv = bar_cache[k]; test_p.extend(bp); test_v.extend(bv)

            folds_for_sym.append((train_p, train_v, test_p, test_v))

            for sig_key, sig_name in active_signals:
                cpu_jobs.append((sym, sig_key, sig_name,
                                 train_p, train_v, test_p, test_v,
                                 args.lookback, total_cpu))
                total_cpu += 1

        symbol_fold_arrays[sym] = folds_for_sym

    total_gpu = sum(len(v) for v in symbol_fold_arrays.values())

    if not cpu_jobs and not total_gpu:
        print("No valid folds. Need more data files (run download_data.py --days 30).")
        sys.exit(1)

    workers = args.workers or min(os.cpu_count() or 4, 4, max(1, total_cpu))

    print(f"Walk-forward hybrid: {len(symbol_fold_arrays)} symbols")
    print(f"  Train: {args.train_days} days  |  Test: {args.test_days} days  "
          f"|  Bar: {args.bar_size}s")
    if not args.no_gpu:
        print(f"  GPU track: {total_gpu} fold-symbols (MA + Simons OU on GPU)")
    if not args.no_cpu:
        print(f"  CPU track: {total_cpu} total jobs, {len(active_signals)} signals "
              f"({workers} threads)")
    print()

    all_results: List[Dict] = []
    lock = threading.Lock()
    gpu_results: List[Dict] = []

    # ── Launch GPU track in background thread ─────────────────────────────────
    gpu_thread = None
    if not args.no_gpu and total_gpu > 0:
        gpu_thread = threading.Thread(
            target=_run_gpu_track,
            args=(list(symbol_fold_arrays.keys()), symbol_fold_arrays,
                  gpu_results, lock, total_gpu),
            daemon=False,   # non-daemon: main will join() before exiting
        )
        gpu_thread.start()

    # ── Launch CPU track via threads (shared memory, no fork OOM) ─────────────
    cpu_results: List[Dict] = []
    if not args.no_cpu and cpu_jobs:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one_fold, *job): job for job in cpu_jobs}
            for fut in as_completed(futures):
                completed += 1
                try:
                    r = fut.result()
                    cpu_results.append(r)
                    if "error" not in r:
                        print(f"  [CPU {completed:>4}/{total_cpu}] {r['symbol']:<10} "
                              f"{r['signal']:<18} fold={r['fold']}  "
                              f"train={r['train_sharpe']:+.2f}  "
                              f"oos={r['test_sharpe']:+.2f}  "
                              f"pnl={r['test_pnl']:+.3f}%  "
                              f"n={r['test_trades']}", flush=True)
                    else:
                        print(f"  [CPU {completed:>4}/{total_cpu}] {r['symbol']:<10} "
                              f"{r['signal']:<18} ERROR: {r['error']}", flush=True)
                except Exception as e:
                    print(f"  [CPU {completed:>4}/{total_cpu}] EXCEPTION: {e}", flush=True)

    # Wait for GPU thread
    if gpu_thread is not None:
        gpu_thread.join()

    all_results = cpu_results + gpu_results

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 95)
    print("WALK-FORWARD RESULTS — Average Out-of-Sample Performance")
    print("=" * 95)

    # Collect all signal names (CPU + GPU tracks)
    all_sig_names = [n for _, n in active_signals]
    if not args.no_gpu:
        all_sig_names += ["GPU MA", "Simons OU (GPU)"]

    sig_results: Dict[str, List[Dict]] = {}
    for r in all_results:
        if "error" in r:
            continue
        sig_results.setdefault(r["signal"], []).append(r)

    print(f"\n  {'Signal':<20} {'Track':>5}  {'Folds':>5}  {'OOS Sharpe':>10}  "
          f"{'OOS PnL':>9}  {'WinRate':>7}  {'Train Sh':>9}  {'Overfit?':>8}")
    print("  " + "-" * 87)

    for sig_name in all_sig_names:
        folds = sig_results.get(sig_name, [])
        if not folds:
            continue
        n = len(folds)
        avg_test_sh  = sum(f["test_sharpe"] for f in folds) / n
        avg_test_pnl = sum(f["test_pnl"] for f in folds) / n
        avg_win      = sum(f["test_winrate"] for f in folds) / n
        avg_train_sh = sum(f["train_sharpe"] for f in folds) / n
        overfit = "YES" if avg_train_sh > 0.5 and avg_test_sh < 0 else "no"
        track = folds[0].get("track", "?").upper()
        print(f"  {sig_name:<20} {track:>5}  {n:>5}  {avg_test_sh:>+10.2f}  "
              f"{avg_test_pnl:>+9.3f}%  {avg_win*100:>6.1f}%  "
              f"{avg_train_sh:>+9.2f}  {overfit:>8}")

    # Per-symbol best signal
    print("\n" + "=" * 95)
    print("BEST OOS SIGNAL PER SYMBOL")
    print("=" * 95)

    for sym in symbols:
        sym_folds = [r for r in all_results if r.get("symbol") == sym and "error" not in r]
        if not sym_folds:
            continue
        by_sig: Dict[str, List[float]] = {}
        for r in sym_folds:
            by_sig.setdefault(r["signal"], []).append(r["test_sharpe"])
        ranked = sorted(by_sig.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
        if ranked:
            best_name, sharpes = ranked[0]
            avg_sh = sum(sharpes) / len(sharpes)
            print(f"  {sym:<10} -> {best_name:<20}  avg OOS sharpe={avg_sh:+.2f}  "
                  f"({len(sharpes)} folds)")

    # GPU signal detail
    gpu_ok = [r for r in gpu_results if "error" not in r]
    if gpu_ok:
        print("\n" + "=" * 95)
        print("GPU SIGNAL PARAMS SUMMARY (per symbol, averaged over folds)")
        print("=" * 95)
        # MA summary
        ma_folds = [r for r in gpu_ok if r["signal"] == "GPU MA"]
        if ma_folds:
            print("  MA Crossover:")
            by_sym: Dict[str, List[Dict]] = {}
            for r in ma_folds:
                by_sym.setdefault(r["symbol"], []).append(r)
            for sym, folds in by_sym.items():
                avg_fw  = sum(f.get("gpu_fw", 0) for f in folds) / len(folds)
                avg_sw  = sum(f.get("gpu_sw", 0) for f in folds) / len(folds)
                avg_sh  = sum(f["test_sharpe"] for f in folds) / len(folds)
                avg_pnl = sum(f["test_pnl"]    for f in folds) / len(folds)
                print(f"    {sym:<10}  fast≈{avg_fw:.0f}  slow≈{avg_sw:.0f}  "
                      f"OOS sharpe={avg_sh:+.2f}  pnl={avg_pnl:+.2f}%  ({len(folds)} folds)")
        # OU summary
        ou_folds = [r for r in gpu_ok if r["signal"] == "Simons OU (GPU)"]
        if ou_folds:
            print("  Simons OU (mean-reversion z-score):")
            by_sym2: Dict[str, List[Dict]] = {}
            for r in ou_folds:
                by_sym2.setdefault(r["symbol"], []).append(r)
            for sym, folds in by_sym2.items():
                avg_w   = sum(f.get("gpu_ou_window", 0) for f in folds) / len(folds)
                avg_z   = sum(f.get("gpu_ou_zthr", 0)   for f in folds) / len(folds)
                avg_sh  = sum(f["test_sharpe"]           for f in folds) / len(folds)
                avg_pnl = sum(f["test_pnl"]              for f in folds) / len(folds)
                print(f"    {sym:<10}  ou_w≈{avg_w:.0f}  z_thr≈{avg_z:.2f}  "
                      f"OOS sharpe={avg_sh:+.2f}  pnl={avg_pnl:+.2f}%  ({len(folds)} folds)")

    print()


if __name__ == "__main__":
    main()
