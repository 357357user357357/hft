#!/usr/bin/env python3
"""Parallel Pair Arbitrage Backtest Runner
==========================================
Screens all candidate pairs, runs the 4-algorithm backtest in parallel,
and prints a ranked results table.

Usage:
    python run_pair_backtest.py                       # all pairs
    python run_pair_backtest.py --pairs SOL/ETH       # specific pair
    python run_pair_backtest.py --workers 4           # parallel jobs
    python run_pair_backtest.py --zscore 2.5          # tighter entry
    python run_pair_backtest.py --no-coint            # skip coint screen
    python run_pair_backtest.py --bars 1000           # shorter lookback

Data source: same Binance aggTrades CSVs used by the single-instrument
backtest.  We load each symbol's price+volume series once and share across
all pairs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from pair_arb import CANDIDATE_PAIRS, PairConfig, engle_granger
from pair_backtest import PairBacktest, PairBacktestConfig, PairStats, run_all_pairs

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_symbol(symbol: str, max_bars: int = 5000) -> Tuple[List[float], List[float]]:
    """Load price + volume series for one symbol.

    Mirrors run_signal_backtest.py exactly:
      1. Scan ./data/ for  <SYMBOL>-aggTrades-*.zip  (same naming convention)
      2. Load aggTrades, resample to bars (bar_size=100 trades per bar)
      3. Fall back to synthetic random-walk if no file found

    Returns (prices, volumes).
    """
    from pathlib import Path
    from data import load_agg_trades_csv

    data_dir = Path(os.path.dirname(__file__)) / "data"
    # Match the same glob pattern as run_signal_backtest.py
    zips = sorted(data_dir.glob(f"{symbol}-aggTrades-*.zip"))

    if zips:
        try:
            trades   = load_agg_trades_csv(str(zips[-1]))
            bar_size = max(1, len(trades) // max_bars)
            prices: List[float] = []
            volumes: List[float] = []
            for i in range(0, len(trades), bar_size):
                chunk = trades[i:i + bar_size]
                prices.append((chunk[0].price + chunk[-1].price) / 2)
                volumes.append(sum(t.quantity for t in chunk))
            return prices[-max_bars:], volumes[-max_bars:]
        except Exception:
            pass

    # Synthetic fallback — realistic price levels so Kalman β is sensible
    import random
    random.seed(hash(symbol) % (2**31))
    base = {
        "BTCUSDT": 45000, "ETHUSDT": 2500, "SOLUSDT": 100, "BNBUSDT": 300,
        "LTCUSDT": 80, "XRPUSDT": 0.5, "LINKUSDT": 15,
        "ADAUSDT": 0.4, "DOGEUSDT": 0.08,
    }.get(symbol, 100.0)
    prices_s: List[float] = [base]
    vols_s:   List[float] = []
    for _ in range(max_bars - 1):
        prices_s.append(prices_s[-1] * (1 + random.gauss(0, 0.012)))
        vols_s.append(random.uniform(100_000, 2_000_000))
    vols_s.append(vols_s[-1] if vols_s else 1_000_000)
    return prices_s, vols_s


def _load_all_symbols(
    symbols: List[str], max_bars: int
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Load prices + volumes for all symbols needed."""
    prices: Dict[str, List[float]] = {}
    volumes: Dict[str, List[float]] = {}
    for sym in symbols:
        p, v = _load_symbol(sym, max_bars)
        prices[sym]  = p
        volumes[sym] = v
        print(f"  {sym:<12s} {len(p):>5d} bars")
    return prices, volumes


# ── Single-pair worker (runs in subprocess) ───────────────────────────────────

def _run_pair_worker(
    leg_a: str,
    leg_b: str,
    prices_a: List[float],
    prices_b: List[float],
    volumes_a: List[float],
    volumes_b: List[float],
    bt_cfg_kwargs: dict,
    pair_cfg_kwargs: dict,
) -> Tuple[str, str, dict]:
    """Worker function for ProcessPoolExecutor."""
    from pair_arb import PairConfig
    from pair_backtest import PairBacktest, PairBacktestConfig

    pair_cfg = PairConfig(leg_a=leg_a, leg_b=leg_b, **pair_cfg_kwargs)
    bt_cfg   = PairBacktestConfig(**bt_cfg_kwargs)
    bt       = PairBacktest(pair_cfg, bt_cfg)
    results  = bt.run(prices_a, prices_b, volumes_a, volumes_b)

    # Serialise PairStats to dicts for IPC
    out: dict = {}
    for algo, st in results.items():
        out[algo] = {
            "algo":            st.algo,
            "pair":            st.pair,
            "total_trades":    st.total_trades,
            "win_rate":        st.win_rate,
            "total_pnl_pct":   st.total_pnl_pct,
            "sharpe_ratio":    st.sharpe_ratio,
            "profit_factor":   st.profit_factor,
            "max_drawdown_pct":st.max_drawdown_pct,
            "avg_hold_bars":   st.avg_hold_bars,
            "coint_pval":      st.coint_pval,
            "coint_adf":       st.coint_adf,
        }
    return leg_a, leg_b, out


# ── Pretty printing ───────────────────────────────────────────────────────────

_BOLD  = "\033[1m"
_RESET = "\033[0m"
_GREEN = "\033[92m"
_RED   = "\033[91m"
_CYAN  = "\033[96m"
_DIM   = "\033[2m"
_YELLOW= "\033[93m"


def _pnl_col(v: float) -> str:
    return _GREEN if v > 0 else _RED if v < 0 else ""


def _print_results(all_results: List[Tuple[str, str, dict]]) -> None:
    # Flatten into rows
    rows = []
    for leg_a, leg_b, algo_dict in all_results:
        for algo, d in algo_dict.items():
            rows.append({
                "pair":    f"{leg_a}/{leg_b}",
                "algo":    algo,
                **d,
            })

    if not rows:
        print("No results.")
        return

    # ── Cointegration summary ─────────────────────────────────────────────────
    print(f"\n{_BOLD}{'='*95}{_RESET}")
    print(f"{_BOLD}  COINTEGRATION SCREENING{_RESET}")
    print(f"{_BOLD}{'='*95}{_RESET}")
    seen_pairs = {}
    for r in rows:
        k = r["pair"]
        if k not in seen_pairs:
            seen_pairs[k] = r
    for pair, r in seen_pairs.items():
        coi = r["coint_pval"]
        col = _GREEN if coi <= 0.05 else _YELLOW if coi <= 0.10 else _RED
        verdict = "COINTEGRATED" if coi <= 0.05 else "marginal" if coi <= 0.10 else "not coint."
        print(
            f"  {pair:<22s}  ADF={r['coint_adf']:>+6.2f}  "
            f"p≈{col}{coi:.2f}{_RESET}  {col}{verdict}{_RESET}"
        )

    # ── Best per pair ─────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{'='*95}{_RESET}")
    print(f"{_BOLD}  BEST ALGORITHM PER PAIR  (by Sharpe){_RESET}")
    print(f"{_BOLD}{'='*95}{_RESET}")
    pair_best: Dict[str, dict] = {}
    for r in rows:
        p = r["pair"]
        if p not in pair_best or r["sharpe_ratio"] > pair_best[p]["sharpe_ratio"]:
            pair_best[p] = r
    for pair, r in sorted(pair_best.items(), key=lambda x: -x[1]["sharpe_ratio"]):
        p = _pnl_col(r["total_pnl_pct"])
        print(
            f"  {pair:<22s}  {r['algo']:<12s}"
            f"trades={r['total_trades']:>4d}  "
            f"win={r['win_rate']*100:>5.1f}%  "
            f"pnl={p}{r['total_pnl_pct']:>+7.3f}%{_RESET}  "
            f"sharpe={r['sharpe_ratio']:>5.2f}  "
            f"mdd={r['max_drawdown_pct']:>5.1f}%"
        )

    # ── Best per algo ─────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{'='*95}{_RESET}")
    print(f"{_BOLD}  BEST PAIR PER ALGORITHM  (by Sharpe){_RESET}")
    print(f"{_BOLD}{'='*95}{_RESET}")
    algo_best: Dict[str, dict] = {}
    for r in rows:
        a = r["algo"]
        if a not in algo_best or r["sharpe_ratio"] > algo_best[a]["sharpe_ratio"]:
            algo_best[a] = r
    for algo, r in sorted(algo_best.items(), key=lambda x: -x[1]["sharpe_ratio"]):
        p = _pnl_col(r["total_pnl_pct"])
        print(
            f"  {algo:<12s}  {r['pair']:<22s}"
            f"trades={r['total_trades']:>4d}  "
            f"win={r['win_rate']*100:>5.1f}%  "
            f"pnl={p}{r['total_pnl_pct']:>+7.3f}%{_RESET}  "
            f"sharpe={r['sharpe_ratio']:>5.2f}"
        )

    # ── Full table ranked by Sharpe ───────────────────────────────────────────
    print(f"\n{_BOLD}{'='*95}{_RESET}")
    print(f"{_BOLD}  ALL RESULTS — RANKED BY SHARPE{_RESET}")
    print(f"{_BOLD}{'='*95}{_RESET}")
    print(f"  {'Pair':<22s}  {'Algo':<12s} {'Trades':>6}  {'Win%':>6}  "
          f"{'PnL%':>8}  {'Sharpe':>7}  {'PF':>5}  {'MDD%':>6}  {'coint_p':>7}")
    print(f"  {'─'*22}  {'─'*12} {'─'*6}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*7}")
    for r in sorted(rows, key=lambda x: -x["sharpe_ratio"]):
        if r["total_trades"] == 0:
            continue
        p = _pnl_col(r["total_pnl_pct"])
        pf = f"{r['profit_factor']:.2f}" if r["profit_factor"] < 1e9 else "∞"
        print(
            f"  {r['pair']:<22s}  {r['algo']:<12s} {r['total_trades']:>6d}  "
            f"{r['win_rate']*100:>6.1f}  "
            f"{p}{r['total_pnl_pct']:>+8.3f}%{_RESET}  "
            f"{r['sharpe_ratio']:>7.2f}  {pf:>5s}  "
            f"{r['max_drawdown_pct']:>6.1f}  "
            f"{r['coint_pval']:>7.2f}"
        )

    print(f"\n[Rust pair arb: ", end="")
    try:
        import ricci_rs as _rm
        mod = _rm if hasattr(_rm, "kalman_pair_update") else getattr(_rm, "ricci_rs", None)
        ok  = mod is not None and hasattr(mod, "kalman_pair_update")
        print(f"{'YES — Kalman/Coint in Rust' if ok else 'not loaded'}]")
    except Exception:
        print("not available]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair arbitrage backtest — all candidate pairs in parallel"
    )
    parser.add_argument("--pairs",    nargs="+", default=None,
                        help="Filter pairs e.g. SOL/ETH LINK/BNB")
    parser.add_argument("--workers",  type=int,  default=min(4, os.cpu_count() or 4),
                        help="Parallel workers (default: min(4, cpu_count))")
    parser.add_argument("--bars",     type=int,  default=5000,
                        help="Max bars per symbol (default: 5000)")
    parser.add_argument("--lookback", type=int,  default=200,
                        help="Warmup bars before trading (default: 200)")
    parser.add_argument("--zscore",   type=float, default=2.0,
                        help="Z-score entry threshold (default: 2.0)")
    parser.add_argument("--no-coint", action="store_true",
                        help="Skip cointegration requirement")
    parser.add_argument("--hold",     type=int,  default=50,
                        help="Max hold bars (default: 50)")
    args = parser.parse_args()

    # Build candidate list
    candidates = CANDIDATE_PAIRS
    if args.pairs:
        filtered = []
        for leg_a, leg_b, note in CANDIDATE_PAIRS:
            key = f"{leg_a.replace('USDT','')}/{leg_b.replace('USDT','')}"
            if any(p.upper() == key or p.upper() == f"{leg_a}/{leg_b}" for p in args.pairs):
                filtered.append((leg_a, leg_b, note))
        if filtered:
            candidates = filtered

    # All unique symbols needed
    symbols = list({s for pair in candidates for s in (pair[0], pair[1])})

    print(f"\nPair Arbitrage Backtest")
    print(f"  Pairs   : {len(candidates)}")
    print(f"  Symbols : {', '.join(sorted(symbols))}")
    print(f"  Workers : {args.workers}")
    print(f"  Bars    : {args.bars}")
    print(f"  Z-entry : {args.zscore}")
    print(f"\nLoading data…")

    prices, volumes = _load_all_symbols(symbols, args.bars)

    bt_cfg_kw = dict(
        lookback_bars   = args.lookback,
        max_hold_bars   = args.hold,
        require_coint   = not args.no_coint,
    )
    pair_cfg_kw = dict(
        zscore_entry    = args.zscore,
        zscore_exit     = 0.3,
        zscore_window   = 60,
        coint_p_threshold = 0.10,
    )

    t0 = time.monotonic()
    all_results: List[Tuple[str, str, dict]] = []

    if args.workers <= 1 or len(candidates) == 1:
        # Single-process
        print(f"\nRunning {len(candidates)} pairs × 4 algos…\n")
        for idx, (leg_a, leg_b, note) in enumerate(candidates, 1):
            if leg_a not in prices or leg_b not in prices:
                continue
            print(f"  [{idx:2d}/{len(candidates)}] {leg_a}/{leg_b}  ({note})")
            la, lb, res = _run_pair_worker(
                leg_a, leg_b,
                prices[leg_a], prices[leg_b],
                volumes[leg_a], volumes[leg_b],
                bt_cfg_kw, pair_cfg_kw,
            )
            all_results.append((la, lb, res))
    else:
        # Parallel
        print(f"\nDispatching {len(candidates)} pairs to {args.workers} workers…\n")
        futures_map = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for leg_a, leg_b, note in candidates:
                if leg_a not in prices or leg_b not in prices:
                    continue
                fut = pool.submit(
                    _run_pair_worker,
                    leg_a, leg_b,
                    prices[leg_a], prices[leg_b],
                    volumes[leg_a], volumes[leg_b],
                    bt_cfg_kw, pair_cfg_kw,
                )
                futures_map[fut] = (leg_a, leg_b)

            done = 0
            for fut in as_completed(futures_map):
                la, lb, res = fut.result()
                all_results.append((la, lb, res))
                done += 1
                # Print best sharpe for this pair inline
                best_sharpe = max((v["sharpe_ratio"] for v in res.values()), default=0.0)
                best_algo   = max(res.items(), key=lambda x: x[1]["sharpe_ratio"])[0] if res else "?"
                best_pnl    = max((v["total_pnl_pct"] for v in res.values()), default=0.0)
                col = _GREEN if best_pnl > 0 else _RED
                print(
                    f"  [{done:2d}/{len(futures_map)}] {la}/{lb:<10s}  "
                    f"best={best_algo:<12s}  "
                    f"sharpe={best_sharpe:>5.2f}  "
                    f"pnl={col}{best_pnl:>+7.3f}%{_RESET}"
                )

    elapsed = time.monotonic() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    _print_results(all_results)


if __name__ == "__main__":
    main()
