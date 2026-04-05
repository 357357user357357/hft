"""Pair Arbitrage Backtest
========================
Backtests a (leg_a, leg_b) pair using the spread z-score signal alongside
all four algorithms (Shot, DepthShot, Averages, Vector).

Each algorithm sees the SPREAD as a synthetic price series and makes its own
independent entry/exit decisions on top of the statistical signal.  This
means four independent P&L streams that we can compare and aggregate.

Key differences from single-instrument backtest:
  - Two price series required (leg_a + leg_b)
  - P&L is PAIR P&L: Δspread * direction, not raw price move
  - Hedge ratio β is tracked by Kalman filter (time-varying)
  - Cointegration is screened before entry

Signal → Algorithm mapping:
  shot      : fires on spread z-score breakouts (like price breakthrough)
  depth_shot: fires when spread moves sharply with high volume imbalance
  averages  : fires when short-MA spread deviates from long-MA spread
  vector    : fires on rapid spread velocity (momentum bursts)

Usage:
    from pair_backtest import PairBacktest, PairBacktestConfig
    from pair_arb import PairConfig

    pair_cfg = PairConfig(leg_a="SOLUSDT", leg_b="ETHUSDT")
    bt_cfg   = PairBacktestConfig(lookback_bars=500, hold_bars=20)
    bt = PairBacktest(pair_cfg, bt_cfg)
    results = bt.run(sol_prices, eth_prices)
    results.print_summary()
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pair_arb import (
    PairArb, PairConfig, PairSignal,
    kalman_batch, rolling_zscore, engle_granger,
    CANDIDATE_PAIRS,
    _rust_mod as _pair_rust_mod, _RUST_PAIR,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PairBacktestConfig:
    """Configuration for pair backtest."""

    # Minimum history bars before trading
    lookback_bars:   int   = 200
    # Bars to hold if z-score exit not triggered
    max_hold_bars:   int   = 50
    # Transaction cost per leg (round-trip = 2× this, both legs = 4× this)
    fee_pct:         float = 0.04   # 4 bps per side (Binance maker)
    # Slippage estimate per leg
    slippage_pct:    float = 0.02   # 2 bps

    # Which "algorithms" to test against the spread
    # shot      : z-score breakout → mean reversion (core statistical arb)
    # depth_shot: requires volume data (falls back to shot if missing)
    # averages  : short vs long moving average of spread
    # vector    : spread velocity burst detection
    run_shot:       bool = True
    run_depth_shot: bool = True
    run_averages:   bool = True
    run_vector:     bool = True

    # Averages params
    avg_short_bars: int   = 10
    avg_long_bars:  int   = 60
    avg_trigger_pct: float = 0.3   # % deviation of short avg from long avg

    # Vector (velocity) params
    vec_velocity_bars:  int   = 5     # lookback for velocity calc
    vec_min_velocity:   float = 0.5   # min z-score velocity to trigger

    # Require cointegration before entry
    require_coint: bool = True


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class PairTrade:
    """Single completed pair trade."""
    algo:          str     # "shot" | "depth_shot" | "averages" | "vector"
    entry_bar:     int
    exit_bar:      int
    side:          str     # "long_spread" | "short_spread"
    entry_zscore:  float
    exit_zscore:   float
    entry_spread:  float
    exit_spread:   float
    pnl_pct:       float   # net P&L in % of notional (both legs combined)
    hold_bars:     int
    entry_beta:    float


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class PairStats:
    """Backtest statistics for one (algo, pair) combination."""
    algo:       str
    pair:       str         # "SOLUSDT/ETHUSDT"
    trades:     List[PairTrade] = field(default_factory=list)

    total_trades:   int   = 0
    winning_trades: int   = 0
    losing_trades:  int   = 0
    win_rate:       float = 0.0
    total_pnl_pct:  float = 0.0
    avg_pnl_pct:    float = 0.0
    avg_win_pct:    float = 0.0
    avg_loss_pct:   float = 0.0
    profit_factor:  float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio:   float = 0.0
    avg_hold_bars:  float = 0.0
    coint_adf:      float = 0.0
    coint_pval:     float = 1.0

    def add_trade(self, t: PairTrade) -> None:
        self.trades.append(t)
        self.total_trades += 1
        if t.pnl_pct > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

    def compute(self) -> None:
        if not self.trades:
            return
        pnls = [t.pnl_pct for t in self.trades]
        self.win_rate      = self.winning_trades / self.total_trades
        self.total_pnl_pct = sum(pnls)
        self.avg_pnl_pct   = self.total_pnl_pct / len(pnls)
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        self.avg_win_pct   = sum(wins)   / len(wins)   if wins   else 0.0
        self.avg_loss_pct  = sum(losses) / len(losses) if losses else 0.0
        gp = sum(wins);    gl = abs(sum(losses)) if losses else 0.0
        self.profit_factor = gp / gl if gl > 0 else float("inf")
        self.avg_hold_bars = sum(t.hold_bars for t in self.trades) / len(self.trades)

        # Max drawdown
        cum = 0.0; peak = 0.0; mdd = 0.0
        for p in pnls:
            cum += p
            if cum > peak: peak = cum
            dd = (peak - cum) / peak * 100 if peak > 0 else 0.0
            if dd > mdd: mdd = dd
        self.max_drawdown_pct = mdd

        # Sharpe (trade-level, annualised assuming ~252 trades/year as proxy)
        if len(pnls) > 1:
            avg = sum(pnls) / len(pnls)
            var = sum((p-avg)**2 for p in pnls) / (len(pnls)-1)
            std = math.sqrt(var) if var > 0 else 0.0
            self.sharpe_ratio = (avg / std * math.sqrt(252)) if std > 0 else 0.0

    def print_summary(self) -> None:
        print(f"  {self.algo:<12s} {self.pair:<22s}"
              f"trades={self.total_trades:>4d}  "
              f"win={self.win_rate*100:>5.1f}%  "
              f"pnl={self.total_pnl_pct:>+7.3f}%  "
              f"sharpe={self.sharpe_ratio:>5.2f}  "
              f"avghold={self.avg_hold_bars:>5.1f}bars  "
              f"coint_p={self.coint_pval:.2f}")


# ── Main backtest class ───────────────────────────────────────────────────────

class PairBacktest:
    """Backtests one pair across all four algorithms.

    The backtest pre-computes the full Kalman β series and z-score series
    (using Rust), then simulates each algorithm's trading logic bar-by-bar.

    P&L calculation (dual-leg):
      For a LongSpread  trade: pnl = (exit_spread - entry_spread) / |entry_price_a|
      For a ShortSpread trade: pnl = (entry_spread - exit_spread) / |entry_price_a|
      Minus fees: 4 × fee_pct  (2 legs × 2 sides round trip)
    """

    def __init__(self, pair_cfg: PairConfig, bt_cfg: Optional[PairBacktestConfig] = None):
        self.pair_cfg = pair_cfg
        self.bt_cfg   = bt_cfg or PairBacktestConfig()

    def run(
        self,
        prices_a: List[float],
        prices_b: List[float],
        volumes_a: Optional[List[float]] = None,
        volumes_b: Optional[List[float]] = None,
    ) -> Dict[str, PairStats]:
        """Run full backtest.  Returns dict of algo_name → PairStats."""
        cfg    = self.bt_cfg
        pcfg   = self.pair_cfg
        n      = min(len(prices_a), len(prices_b))
        pair_label = f"{pcfg.leg_a}/{pcfg.leg_b}"

        if n < cfg.lookback_bars + 10:
            # Not enough data — return empty stats
            return {k: PairStats(algo=k, pair=pair_label) for k in
                    ("shot", "depth_shot", "averages", "vector")}

        # ── Pre-compute Kalman β and z-score (Rust-accelerated) ──────────────
        betas, _pvars, spreads = kalman_batch(
            prices_a[:n], prices_b[:n],
            pcfg.kalman_q, pcfg.kalman_r,
        )
        zscores, zmeans, zstds = rolling_zscore(
            spreads, pcfg.zscore_window,
        )

        # ── Cointegration screening ───────────────────────────────────────────
        screen_end  = min(n, cfg.lookback_bars + 200)
        beta_ols, adf_stat, coint_p = engle_granger(
            prices_a[:screen_end], prices_b[:screen_end]
        )

        # ── Fee + slippage total (both legs, round-trip) ──────────────────────
        total_cost = 4 * (cfg.fee_pct + cfg.slippage_pct) / 100.0

        # ── Cointegration gate — skip simulation entirely if pair not coint ───
        coint_blocked = cfg.require_coint and coint_p > pcfg.coint_p_threshold

        # ── Rust fast-path: run all 4 algos in one native loop ────────────────
        # pair_backtest_run() replaces the Python for-loop over each bar.
        # Falls back to Python loop automatically if Rust unavailable.
        rust_available = (
            _RUST_PAIR
            and _pair_rust_mod is not None
            and hasattr(_pair_rust_mod, "pair_backtest_run")
            and not coint_blocked
            and cfg.run_shot and cfg.run_depth_shot
            and cfg.run_averages and cfg.run_vector
        )

        results: Dict[str, PairStats] = {}

        if rust_available:
            # Single Rust call — all 4 algorithms at once.
            # Rust extension returns untyped tuple; unpack via index to keep Pyright happy.
            raw = _pair_rust_mod.pair_backtest_run(  # type: ignore[union-attr]
                zscores, spreads, prices_a[:n],
                volumes_a, volumes_b,
                cfg.lookback_bars, cfg.max_hold_bars,
                pcfg.zscore_entry, pcfg.zscore_exit, pcfg.zscore_stop,
                cfg.fee_pct, cfg.slippage_pct,
                cfg.avg_short_bars, cfg.avg_long_bars, cfg.avg_trigger_pct,
                cfg.vec_velocity_bars, cfg.vec_min_velocity,
            )
            # raw = 20-tuple: (trades, wins, pnl, sharpe, mdd) × 4 algos
            # Rust computes Sharpe (Welford online) and MDD inline — no Python needed.
            # raw = Vec<f64> of 20 values: (trades,wins,pnl,sharpe,mdd) × 4 algos
            # Rust computes Sharpe via Welford online algorithm and proper MDD.
            from typing import cast as _cast
            _r: List[float] = [float(x) for x in _cast(tuple, raw)]  # type: ignore[arg-type,misc]
            _algo_raw5: List[Tuple[str, int, int, float, float, float]] = [
                ("shot",       int(_r[0]),  int(_r[1]),  _r[2],  _r[3],  _r[4]),
                ("depth_shot", int(_r[5]),  int(_r[6]),  _r[7],  _r[8],  _r[9]),
                ("averages",   int(_r[10]), int(_r[11]), _r[12], _r[13], _r[14]),
                ("vector",     int(_r[15]), int(_r[16]), _r[17], _r[18], _r[19]),
            ]
            for algo_name, total, wins, pnl, sharpe, mdd in _algo_raw5:
                s = PairStats(algo=algo_name, pair=pair_label)
                s.coint_adf = adf_stat;  s.coint_pval = coint_p
                s.total_trades = total;  s.winning_trades = wins
                s.losing_trades = total - wins
                s.total_pnl_pct = pnl;   s.sharpe_ratio = sharpe
                s.max_drawdown_pct = mdd
                if total > 0:
                    s.win_rate    = wins / total
                    s.avg_pnl_pct = pnl / total
                    wp = pnl * s.win_rate;  lp = pnl * (1.0 - s.win_rate)
                    s.profit_factor = abs(wp / lp) if abs(lp) > 1e-12 else float("inf")
                results[algo_name] = s
            return results

        # ── Python fallback: original per-algorithm loop ──────────────────────
        algo_flags = {
            "shot":       cfg.run_shot,
            "depth_shot": cfg.run_depth_shot,
            "averages":   cfg.run_averages,
            "vector":     cfg.run_vector,
        }

        for algo_name, enabled in algo_flags.items():
            if not enabled:
                continue
            st = PairStats(algo=algo_name, pair=pair_label)
            st.coint_adf  = adf_stat
            st.coint_pval = coint_p

            # Don't trade if cointegration test fails (when required)
            if coint_blocked:
                st.compute()
                results[algo_name] = st
                continue

            in_pos     = False
            pos_side   = ""
            entry_bar  = 0
            entry_sp   = 0.0
            entry_pa   = 0.0

            for i in range(cfg.lookback_bars, n):
                z  = zscores[i]
                sp = spreads[i]
                pa = prices_a[i]

                # ── Exit check ────────────────────────────────────────────────
                if in_pos:
                    bars_held = i - entry_bar
                    exit_now  = False

                    if algo_name == "shot":
                        # Exit: mean reversion or z-score crosses back
                        if pos_side == "long_spread"  and z >= -pcfg.zscore_exit: exit_now = True
                        if pos_side == "short_spread" and z <= pcfg.zscore_exit:  exit_now = True
                        if abs(z) > pcfg.zscore_stop:                              exit_now = True
                        if bars_held >= cfg.max_hold_bars:                         exit_now = True

                    elif algo_name == "depth_shot":
                        # Uses volume imbalance to confirm exit
                        vol_a = volumes_a[i] if volumes_a else 1.0
                        vol_b = volumes_b[i] if volumes_b else 1.0
                        vol_ratio = vol_a / (vol_b + 1e-9)
                        # Strong counter-volume or z-score mean-reversion
                        if pos_side == "long_spread"  and z >= -pcfg.zscore_exit and vol_ratio < 1.5: exit_now = True
                        if pos_side == "short_spread" and z <= pcfg.zscore_exit  and vol_ratio > 0.7: exit_now = True
                        if abs(z) > pcfg.zscore_stop:  exit_now = True
                        if bars_held >= cfg.max_hold_bars: exit_now = True

                    elif algo_name == "averages":
                        # Exit when short spread MA reverts to long MA
                        if i >= cfg.avg_long_bars:
                            short_win = spreads[i-cfg.avg_short_bars+1:i+1]
                            long_win  = spreads[i-cfg.avg_long_bars+1:i+1]
                            short_avg = sum(short_win) / len(short_win)
                            long_avg  = sum(long_win)  / len(long_win)
                            delta_pct = (short_avg - long_avg) / (abs(long_avg) + 1e-9) * 100
                            if pos_side == "long_spread"  and delta_pct >= -cfg.avg_trigger_pct * 0.3: exit_now = True
                            if pos_side == "short_spread" and delta_pct <= cfg.avg_trigger_pct * 0.3:  exit_now = True
                        if bars_held >= cfg.max_hold_bars: exit_now = True

                    elif algo_name == "vector":
                        # Exit when z-score velocity reverses
                        if i >= cfg.vec_velocity_bars:
                            vel_now  = z - zscores[i - cfg.vec_velocity_bars]
                            if pos_side == "long_spread"  and vel_now >  0: exit_now = True
                            if pos_side == "short_spread" and vel_now <  0: exit_now = True
                        if bars_held >= cfg.max_hold_bars: exit_now = True

                    if exit_now:
                        # Compute P&L as spread change normalised to price_a
                        normaliser = abs(entry_pa) if abs(entry_pa) > 1e-6 else 1.0
                        if pos_side == "long_spread":
                            raw_pnl = (sp - entry_sp) / normaliser
                        else:
                            raw_pnl = (entry_sp - sp) / normaliser
                        net_pnl = raw_pnl * 100 - total_cost  # convert to %

                        t = PairTrade(
                            algo=algo_name,
                            entry_bar=entry_bar, exit_bar=i,
                            side=pos_side,
                            entry_zscore=zscores[entry_bar],
                            exit_zscore=z,
                            entry_spread=entry_sp, exit_spread=sp,
                            pnl_pct=net_pnl,
                            hold_bars=i - entry_bar,
                            entry_beta=betas[entry_bar],
                        )
                        st.add_trade(t)
                        in_pos   = False
                        pos_side = ""

                # ── Entry check ───────────────────────────────────────────────
                if not in_pos and i >= cfg.lookback_bars:

                    if algo_name == "shot":
                        # Core stat arb: enter on z-score extreme
                        if z > pcfg.zscore_entry:
                            in_pos = True; pos_side = "short_spread"
                            entry_bar = i; entry_sp = sp; entry_pa = pa
                        elif z < -pcfg.zscore_entry:
                            in_pos = True; pos_side = "long_spread"
                            entry_bar = i; entry_sp = sp; entry_pa = pa

                    elif algo_name == "depth_shot":
                        # Confirm with volume imbalance before entering
                        vol_a = volumes_a[i] if volumes_a else 1.0
                        vol_b = volumes_b[i] if volumes_b else 1.0
                        vol_ratio = vol_a / (vol_b + 1e-9)
                        # High volume in leg_a while spread elevated → conviction short spread
                        if z > pcfg.zscore_entry and vol_ratio > 1.2:
                            in_pos = True; pos_side = "short_spread"
                            entry_bar = i; entry_sp = sp; entry_pa = pa
                        elif z < -pcfg.zscore_entry and vol_ratio < 0.8:
                            in_pos = True; pos_side = "long_spread"
                            entry_bar = i; entry_sp = sp; entry_pa = pa

                    elif algo_name == "averages":
                        # Enter when short-MA spread deviates from long-MA spread
                        if i >= cfg.avg_long_bars:
                            short_win = spreads[i-cfg.avg_short_bars+1:i+1]
                            long_win  = spreads[i-cfg.avg_long_bars+1:i+1]
                            short_avg = sum(short_win) / len(short_win)
                            long_avg  = sum(long_win)  / len(long_win)
                            delta_pct = (short_avg - long_avg) / (abs(long_avg) + 1e-9) * 100
                            # Spread MA spread too high → short
                            if delta_pct >  cfg.avg_trigger_pct and z > 1.0:
                                in_pos = True; pos_side = "short_spread"
                                entry_bar = i; entry_sp = sp; entry_pa = pa
                            # Spread MA spread too low  → long
                            elif delta_pct < -cfg.avg_trigger_pct and z < -1.0:
                                in_pos = True; pos_side = "long_spread"
                                entry_bar = i; entry_sp = sp; entry_pa = pa

                    elif algo_name == "vector":
                        # Enter on rapid z-score velocity burst (impulse)
                        if i >= cfg.vec_velocity_bars:
                            vel = z - zscores[i - cfg.vec_velocity_bars]
                            # Fast move up + already elevated → short spread
                            if vel >  cfg.vec_min_velocity and z > 1.0:
                                in_pos = True; pos_side = "short_spread"
                                entry_bar = i; entry_sp = sp; entry_pa = pa
                            # Fast move down + already depressed → long spread
                            elif vel < -cfg.vec_min_velocity and z < -1.0:
                                in_pos = True; pos_side = "long_spread"
                                entry_bar = i; entry_sp = sp; entry_pa = pa

            st.compute()
            results[algo_name] = st

        return results


# ── Batch runner for multiple pairs ──────────────────────────────────────────

def run_all_pairs(
    price_data: Dict[str, List[float]],
    volume_data: Optional[Dict[str, List[float]]] = None,
    bt_cfg: Optional[PairBacktestConfig] = None,
) -> List[Tuple[str, str, Dict[str, PairStats]]]:
    """Run backtest for all candidate pairs.

    Arguments:
        price_data  – dict of symbol → price list
        volume_data – dict of symbol → volume list (optional)
        bt_cfg      – shared backtest config

    Returns list of (leg_a, leg_b, algo_results) tuples.
    """
    results = []
    for leg_a, leg_b, _note in CANDIDATE_PAIRS:
        if leg_a not in price_data or leg_b not in price_data:
            continue
        pair_cfg = PairConfig(leg_a=leg_a, leg_b=leg_b)
        bt       = PairBacktest(pair_cfg, bt_cfg)
        pa       = price_data[leg_a]
        pb       = price_data[leg_b]
        va       = volume_data.get(leg_a) if volume_data else None
        vb       = volume_data.get(leg_b) if volume_data else None
        algo_res = bt.run(pa, pb, va, vb)
        results.append((leg_a, leg_b, algo_res))
    return results


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(0)

    # Simulate cointegrated pair
    beta_true = 1.4
    n = 800
    xp = [200.0]
    for _ in range(n-1):
        xp.append(xp[-1] * (1 + random.gauss(0, 0.012)))
    ou = [0.0]
    for _ in range(n-1):
        ou.append(ou[-1] * 0.92 + random.gauss(0, 0.8))
    yp = [beta_true * xp[i] + ou[i] for i in range(n)]
    vol_a = [random.uniform(500, 2000) for _ in range(n)]
    vol_b = [random.uniform(400, 1800) for _ in range(n)]

    pair_cfg = PairConfig(leg_a="SYN_A", leg_b="SYN_B", zscore_entry=2.0)
    bt_cfg   = PairBacktestConfig(lookback_bars=150, max_hold_bars=30,
                                   require_coint=False)
    bt = PairBacktest(pair_cfg, bt_cfg)
    results = bt.run(yp, xp, vol_a, vol_b)

    print("\n=== Pair Backtest Demo (SYN_A / SYN_B) ===\n")
    for algo, st in results.items():
        st.print_summary()

    best = max(results.values(), key=lambda s: s.sharpe_ratio)
    print(f"\nBest algo: {best.algo}  sharpe={best.sharpe_ratio:.2f}  pnl={best.total_pnl_pct:+.3f}%")
