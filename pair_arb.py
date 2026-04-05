"""Pair Arbitrage Engine
======================
Statistical pairs trading for crypto futures.

The core idea: two instruments that are economically linked (e.g. SOL/ETH,
LINK/BNB) tend to move together long-term.  When their ratio (spread) drifts
too far from the historical mean, we bet on mean-reversion:
  - Spread too high → sell leg-A, buy leg-B
  - Spread too low  → buy  leg-A, sell leg-B

Why this works in crypto (even without co-location):
  • Edge is in spread deviation, not tick latency — seconds/minutes, not μs.
  • Cointegrated pairs share common stochastic trends (macro, BTC correlation).
  • Entry/exit based on z-score: statistically robust, not prediction.

Architecture
------------
  PairConfig        — all tuning knobs
  PairSpread        — live spread state: Kalman β, rolling z-score, signals
  PairSignal        — entry/exit decision (enum: Long, Short, Close, Hold)
  PairArb           — top-level engine: wraps Rust Kalman + z-score

Rust acceleration (ricci_rs)
----------------------------
  kalman_pair_update()  — single-step O(1) Kalman, called per-tick
  kalman_pair_batch()   — batch Kalman for backtest pre-computation
  pair_zscore()         — rolling z-score (full history)
  engle_granger_coint() — OLS + approximate ADF for pair screening

All four algorithms (Shot, DepthShot, Averages, Vector) are wired to the
spread signal via the standard SignalGate interface — they see the spread as
a synthetic "price" and trade it using their normal logic.

Candidate pairs (all historically cointegrated on Binance Futures):
  ("SOLUSDT",  "ETHUSDT")   — DeFi L1 correlation
  ("LINKUSDT", "BNBUSDT")   — mid-cap DeFi
  ("BNBUSDT",  "ETHUSDT")   — exchange tokens
  ("LTCUSDT",  "XRPUSDT")   — legacy altcoins
  ("ADAUSDT",  "XRPUSDT")   — same market cap tier
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

# ── Rust acceleration ─────────────────────────────────────────────────────────

_RUST_PAIR = False
_rust_mod = None
try:
    import ricci_rs as _rm
    _rust_mod = _rm if hasattr(_rm, "kalman_pair_update") else getattr(_rm, "ricci_rs", None)
    if _rust_mod is not None and hasattr(_rust_mod, "kalman_pair_update"):
        _RUST_PAIR = True
except Exception:
    pass


def _kalman_update_py(
    beta: float, p: float, y: float, x: float, q: float, r: float
) -> Tuple[float, float, float, float]:
    """Pure-Python fallback: single Kalman step for hedge ratio β.

    State model:  β_t = β_{t-1} + noise_q      (random walk on hedge ratio)
    Observation:  y_t = β_t * x_t + noise_r     (spread residual)

    Returns (beta_new, p_new, spread_new, gain).
    """
    p_pred  = p + q                       # predicted covariance
    spread  = y - beta * x                # innovation
    innov_v = p_pred * x * x + r          # innovation variance
    gain    = p_pred * x / innov_v if innov_v > 1e-30 else 0.0
    beta_n  = beta + gain * spread
    p_n     = max((1.0 - gain * x) * p_pred, 1e-12)
    return (beta_n, p_n, y - beta_n * x, gain)


def kalman_update(
    beta: float, p: float, y: float, x: float, q: float = 1e-5, r: float = 1e-3
) -> Tuple[float, float, float, float]:
    """Kalman hedge-ratio update — uses Rust if available, else Python fallback."""
    if _RUST_PAIR:
        return _rust_mod.kalman_pair_update(beta, p, y, x, q, r)  # type: ignore[union-attr]
    return _kalman_update_py(beta, p, y, x, q, r)


def kalman_batch(
    ys: List[float], xs: List[float], q: float = 1e-5, r: float = 1e-3
) -> Tuple[List[float], List[float], List[float]]:
    """Batch Kalman over full price history.  Returns (betas, p_vars, spreads)."""
    if _RUST_PAIR:
        return _rust_mod.kalman_pair_batch(ys, xs, q, r)  # type: ignore[union-attr]
    # Python fallback
    n = min(len(ys), len(xs))
    init = min(n, 20)
    sx = sum(xs[:init]); sy = sum(ys[:init])
    sxx = sum(v*v for v in xs[:init])
    sxy = sum(x*y for x, y in zip(xs[:init], ys[:init]))
    denom = sxx - sx*sx/init if init > 0 else 1.0
    beta = (sxy - sx*sy/init) / denom if abs(denom) > 1e-12 else 1.0
    p = 1.0
    betas: List[float] = []
    pvars: List[float] = []
    spreads: List[float] = []
    for i in range(n):
        beta, p, sp, _ = _kalman_update_py(beta, p, ys[i], xs[i], q, r)
        betas.append(beta); pvars.append(p); spreads.append(sp)
    return betas, pvars, spreads


def rolling_zscore(
    spreads: List[float], window: int = 60
) -> Tuple[List[float], List[float], List[float]]:
    """Rolling z-score of spread series.  Returns (zscores, means, stds)."""
    if _RUST_PAIR:
        return _rust_mod.pair_zscore(spreads, window)  # type: ignore[union-attr]
    n = len(spreads); w = max(window, 2)
    zs: List[float] = []; ms: List[float] = []; ss: List[float] = []
    for i in range(n):
        sl = spreads[max(0, i+1-w):i+1]
        m  = sum(sl) / len(sl)
        s  = math.sqrt(sum((v-m)**2 for v in sl) / len(sl))
        ms.append(m); ss.append(s)
        zs.append((spreads[i]-m)/s if i+1 >= w and s > 1e-12 else 0.0)
    return zs, ms, ss


def engle_granger(
    ys: List[float], xs: List[float]
) -> Tuple[float, float, float]:
    """Engle-Granger cointegration test.  Returns (beta_ols, adf_stat, p_approx).

    p < 0.05  → statistically cointegrated at 5% level  (trade this pair)
    p < 0.01  → very strong cointegration               (high confidence)
    p > 0.10  → not cointegrated, skip
    """
    if _RUST_PAIR:
        return _rust_mod.engle_granger_coint(ys, xs)  # type: ignore[union-attr]
    n = min(len(ys), len(xs))
    if n < 10:
        return 1.0, 0.0, 1.0
    sx = sum(xs[:n]); sy = sum(ys[:n]); nf = float(n)
    sxx = sum(v*v for v in xs[:n])
    sxy = sum(x*y for x, y in zip(xs[:n], ys[:n]))
    denom = sxx - sx*sx/nf
    beta  = (sxy - sx*sy/nf) / denom if abs(denom) > 1e-12 else 1.0
    a_int = (sy - beta*sx) / nf
    resid = [ys[i] - beta*xs[i] - a_int for i in range(n)]
    sum_e2  = sum(resid[t-1]**2        for t in range(1, n))
    sum_ede = sum(resid[t-1]*(resid[t]-resid[t-1]) for t in range(1, n))
    deltas  = [resid[t]-resid[t-1] for t in range(1, n)]
    md = sum(deltas)/(n-1) if n > 1 else 0.0
    vd = sum((d-md)**2 for d in deltas)/(max(n-2,1))
    denom2 = math.sqrt(vd) * math.sqrt(sum_e2) if vd > 0 and sum_e2 > 0 else 0.0
    adf = sum_ede / denom2 if denom2 > 1e-30 else 0.0
    tau1  = -3.43 + 3.39/nf; tau5 = -2.86 + 2.74/nf; tau10 = -2.57 + 1.99/nf
    p = 0.01 if adf < tau1 else 0.05 if adf < tau5 else 0.10 if adf < tau10 else 0.50
    return beta, adf, p


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class PairConfig:
    """All tuning knobs for one pair."""

    # Identity
    leg_a: str = "SOLUSDT"    # instrument we go long/short on
    leg_b: str = "ETHUSDT"    # hedge leg (opposite side)

    # Kalman filter
    kalman_q: float = 1e-5    # process noise  — higher = β adapts faster
    kalman_r: float = 1e-3    # obs noise      — higher = smoother β

    # Z-score thresholds
    zscore_entry:    float = 2.0   # |z| > entry  → open position
    zscore_exit:     float = 0.5   # |z| < exit   → close position
    zscore_stop:     float = 4.0   # |z| > stop   → emergency close (runaway)
    zscore_window:   int   = 60    # rolling window for mean/std

    # Minimum history before trading
    min_warmup_bars: int = 100

    # Position sizing (fraction of per-leg equity)
    position_size_pct: float = 0.05   # 5% per leg

    # Cointegration gate — only trade if p < this threshold
    coint_p_threshold: float = 0.10

    # Re-screen cointegration every N bars
    coint_rescreen_bars: int = 500

    # Signal overlay: which extra signals must align
    require_signal_alignment: bool = False
    signal_alignment_threshold: float = 0.1


# ── Signal enum ───────────────────────────────────────────────────────────────

class PairSignal(Enum):
    Hold       = "hold"        # do nothing
    LongSpread = "long_spread"  # buy A, sell B  (spread too low → expect rise)
    ShortSpread= "short_spread" # sell A, buy B  (spread too high → expect fall)
    Close      = "close"        # exit current position


# ── Live spread state (one per pair) ─────────────────────────────────────────

@dataclass
class PairSpread:
    """Tracks the live spread state between two instruments.

    Fields updated on every tick via PairArb.on_prices().
    """
    # Kalman state
    beta:      float = 1.0    # current hedge ratio β
    p_var:     float = 1.0    # Kalman error covariance

    # Spread history (ring buffer, size = zscore_window * 2)
    spreads:   Deque[float] = field(default_factory=lambda: deque(maxlen=200))

    # Latest z-score
    zscore:    float = 0.0
    spread_mean: float = 0.0
    spread_std:  float = 0.0

    # Last raw spread value
    spread_raw:  float = 0.0

    # Cointegration screening
    coint_beta:  float = 1.0
    coint_adf:   float = 0.0
    coint_pval:  float = 1.0
    coint_ok:    bool  = False
    bars_since_screen: int = 0

    # Current position
    in_position:    bool         = False
    position_side:  PairSignal   = PairSignal.Hold
    entry_zscore:   float        = 0.0
    entry_time:     float        = field(default_factory=time.monotonic)

    # Bar counter
    bars:           int          = 0

    # Prices
    price_a:        float        = 0.0
    price_b:        float        = 0.0


# ── Main engine ───────────────────────────────────────────────────────────────

class PairArb:
    """Pair arbitrage engine for one (leg_a, leg_b) pair.

    Usage (live):
        arb = PairArb(PairConfig(leg_a="SOLUSDT", leg_b="ETHUSDT"))
        signal = arb.on_prices(sol_price, eth_price)
        if signal == PairSignal.LongSpread:
            # buy SOL, sell ETH proportionally

    Usage (backtest):
        betas, pvars, spreads = arb.fit_history(sol_prices, eth_prices)
        zscores, means, stds  = arb.zscore_history(spreads)
    """

    def __init__(self, config: PairConfig):
        self.config = config
        self.state  = PairSpread()

    # ── History helpers (for backtest pre-computation) ────────────────────────

    def fit_history(
        self,
        prices_a: List[float],
        prices_b: List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        """Run Kalman filter over full price history.

        Returns (betas, p_vars, spreads) — same length as input.
        Rust-accelerated when available.
        """
        return kalman_batch(prices_a, prices_b, self.config.kalman_q, self.config.kalman_r)

    def zscore_history(
        self,
        spreads: List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        """Compute rolling z-score for a spread history.

        Returns (zscores, means, stds).
        """
        return rolling_zscore(spreads, self.config.zscore_window)

    def screen_cointegration(
        self,
        prices_a: List[float],
        prices_b: List[float],
    ) -> Tuple[float, float, float]:
        """Run Engle-Granger cointegration test on price history.

        Returns (beta_ols, adf_stat, p_value).
        p < 0.05 → pair is cointegrated, safe to trade.
        """
        return engle_granger(prices_a, prices_b)

    # ── Live tick-by-tick update ──────────────────────────────────────────────

    def on_prices(self, price_a: float, price_b: float) -> PairSignal:
        """Update spread state with new prices.  Returns trading signal.

        Call this once per bar (or per trade aggregation interval).
        The engine:
          1. Updates Kalman β  (tracks time-varying hedge ratio)
          2. Computes spread   s = price_a − β·price_b
          3. Updates rolling z-score
          4. Screens cointegration periodically
          5. Returns entry/exit signal for the 4 algorithms
        """
        st  = self.state
        cfg = self.config

        st.price_a = price_a
        st.price_b = price_b
        st.bars   += 1

        # ── Kalman update ─────────────────────────────────────────────────────
        st.beta, st.p_var, st.spread_raw, _gain = kalman_update(
            st.beta, st.p_var,
            price_a, price_b,
            cfg.kalman_q, cfg.kalman_r,
        )
        st.spreads.append(st.spread_raw)

        # ── Rolling z-score ───────────────────────────────────────────────────
        if len(st.spreads) >= 2:
            sl  = list(st.spreads)
            w   = min(cfg.zscore_window, len(sl))
            win = sl[-w:]
            m   = sum(win) / w
            s   = math.sqrt(sum((v-m)**2 for v in win) / w)
            st.spread_mean = m
            st.spread_std  = s
            if len(st.spreads) >= cfg.zscore_window and s > 1e-12:
                st.zscore = (st.spread_raw - m) / s
            else:
                st.zscore = 0.0

        # ── Periodic cointegration screen ────────────────────────────────────
        st.bars_since_screen += 1
        if st.bars_since_screen >= cfg.coint_rescreen_bars:
            if len(st.spreads) >= 50:
                # Use recent price history reconstructed from spreads + beta
                # We can't recover raw prices from spreads alone; skip full
                # OLS and just test the residuals for stationarity via ADF.
                resid = list(st.spreads)[-min(len(st.spreads), 200):]
                n = len(resid); nf = float(n)
                sum_e2  = sum(resid[t-1]**2 for t in range(1, n))
                sum_ede = sum(resid[t-1]*(resid[t]-resid[t-1]) for t in range(1, n))
                deltas  = [resid[t]-resid[t-1] for t in range(1, n)]
                md = sum(deltas)/(n-1)
                vd = sum((d-md)**2 for d in deltas)/max(n-2,1)
                denom2 = math.sqrt(vd)*math.sqrt(sum_e2) if vd > 0 and sum_e2 > 0 else 0.0
                adf = sum_ede/denom2 if denom2 > 1e-30 else 0.0
                tau5 = -2.86 + 2.74/nf
                st.coint_adf = adf
                st.coint_pval = 0.05 if adf < tau5 else 0.50
                st.coint_ok = (adf < tau5)
            st.bars_since_screen = 0

        # ── Not warm yet — no trades ──────────────────────────────────────────
        if st.bars < cfg.min_warmup_bars:
            return PairSignal.Hold

        # Cointegration gate (only applies after first screen)
        if st.bars >= cfg.coint_rescreen_bars and not st.coint_ok:
            # Close any open position if pair broke cointegration
            if st.in_position:
                st.in_position   = False
                st.position_side = PairSignal.Hold
                return PairSignal.Close
            return PairSignal.Hold

        z = st.zscore

        # ── Exit logic ────────────────────────────────────────────────────────
        if st.in_position:
            should_close = False
            # Mean reversion achieved
            if st.position_side == PairSignal.ShortSpread and z <= cfg.zscore_exit:
                should_close = True
            if st.position_side == PairSignal.LongSpread  and z >= -cfg.zscore_exit:
                should_close = True
            # Stop — spread keeps going (runaway)
            if abs(z) > cfg.zscore_stop:
                should_close = True
            if should_close:
                st.in_position   = False
                st.position_side = PairSignal.Hold
                return PairSignal.Close

        # ── Entry logic ───────────────────────────────────────────────────────
        if not st.in_position:
            if z > cfg.zscore_entry:
                # Spread too HIGH → short spread (sell A, buy B)
                st.in_position   = True
                st.position_side = PairSignal.ShortSpread
                st.entry_zscore  = z
                st.entry_time    = time.monotonic()
                return PairSignal.ShortSpread

            if z < -cfg.zscore_entry:
                # Spread too LOW → long spread (buy A, sell B)
                st.in_position   = True
                st.position_side = PairSignal.LongSpread
                st.entry_zscore  = z
                st.entry_time    = time.monotonic()
                return PairSignal.LongSpread

        return PairSignal.Hold

    # ── Status string for dashboard ───────────────────────────────────────────

    def status_line(self) -> str:
        st  = self.state
        cfg = self.config
        pos = "LONG " if st.position_side == PairSignal.LongSpread \
            else "SHORT" if st.position_side == PairSignal.ShortSpread \
            else "     "
        coint = "OK" if st.coint_ok else "??"
        return (
            f"{cfg.leg_a}/{cfg.leg_b}  "
            f"β={st.beta:+.4f}  "
            f"spread={st.spread_raw:+.4f}  "
            f"z={st.zscore:+.2f}  "
            f"pos={pos}  "
            f"coint={coint}  "
            f"bars={st.bars}"
        )


# ── Candidate pairs ───────────────────────────────────────────────────────────

#: Historically cointegrated pairs on Binance Futures.
#: Tuples of (leg_a, leg_b, notes).
CANDIDATE_PAIRS: List[Tuple[str, str, str]] = [
    ("SOLUSDT",  "ETHUSDT",  "DeFi L1 — strong BTC-beta correlation"),
    ("LINKUSDT", "BNBUSDT",  "mid-cap DeFi — exchange-correlated"),
    ("BNBUSDT",  "ETHUSDT",  "exchange tokens — co-move with CEX volume"),
    ("LTCUSDT",  "XRPUSDT",  "legacy PoW/PoS — same investor base"),
    ("ADAUSDT",  "XRPUSDT",  "same market-cap tier — retail-driven"),
    ("SOLUSDT",  "BNBUSDT",  "high-beta L1/exchange pair"),
    ("ETHUSDT",  "BNBUSDT",  "L1 vs exchange — stablecoin proxy"),
]


def make_default_pairs() -> List[PairArb]:
    """Instantiate PairArb engines for all candidate pairs with default config."""
    return [
        PairArb(PairConfig(leg_a=a, leg_b=b))
        for a, b, _ in CANDIDATE_PAIRS
    ]


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    print(f"Rust pair acceleration: {'YES' if _RUST_PAIR else 'NO (Python fallback)'}\n")

    # Simulate cointegrated pair: y = 1.5*x + OU noise
    beta_true = 1.5
    x = [100.0]
    for _ in range(499):
        x.append(x[-1] * (1 + random.gauss(0, 0.01)))
    noise = [0.0]
    for _ in range(499):
        noise.append(noise[-1] * 0.95 + random.gauss(0, 0.5))
    y = [beta_true * x[i] + noise[i] for i in range(500)]

    arb = PairArb(PairConfig(
        leg_a="SYN_A", leg_b="SYN_B",
        zscore_entry=2.0, zscore_exit=0.5,
        min_warmup_bars=60, zscore_window=60,
    ))

    signals = []
    for i in range(500):
        sig = arb.on_prices(y[i], x[i])
        if sig != PairSignal.Hold:
            signals.append((i, sig.value, arb.state.zscore))

    print(f"Fitted β ≈ {arb.state.beta:.4f}  (true = {beta_true})")
    print(f"Final z-score: {arb.state.zscore:+.3f}")
    print(f"Signals fired: {len(signals)}")
    for bar, sig, z in signals[:10]:
        print(f"  bar {bar:3d}: {sig:<12s}  z={z:+.3f}")

    # Cointegration test
    beta_ols, adf, p = engle_granger(y, x)
    print(f"\nEngle-Granger: β_ols={beta_ols:.4f}  ADF={adf:.3f}  p≈{p:.2f}")
    print(f"  → {'COINTEGRATED' if p < 0.05 else 'not cointegrated'} at 5%")
    print()
    print(arb.status_line())
