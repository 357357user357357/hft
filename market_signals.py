"""Classic financial market microstructure and statistical signals.

Implements the missing signal dimensions:
  - Hurst exponent      (mean-reversion strength, H < 0.5 vs H > 0.5)
  - Realized volatility (σ regime: calm / normal / explosive)
  - Order flow imbalance (buy vs sell volume pressure)
  - Volume profile      (VWAP, VPOC, value area)
  - Microstructure      (Amihud illiquidity, Kyle's lambda, spread proxy)
  - RSI / Momentum      (classic price momentum)
  - Autocorrelation     (lag-1 return autocorrelation)
  - Funding-rate proxy  (open-interest-weighted skew for perp futures)

All functions take plain Python lists — no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ─── Hurst Exponent ───────────────────────────────────────────────────────────

@dataclass
class HurstResult:
    """
    Hurst exponent via Rescaled Range (R/S) analysis.

    H < 0.5  → mean-reverting (anti-persistent)
    H = 0.5  → random walk
    H > 0.5  → trending (persistent)
    """
    H: float
    regime: str          # "mean_reverting" | "random_walk" | "trending"
    score: float         # +1 = strong MR, -1 = strong trend, 0 = random

    @staticmethod
    def compute(prices: List[float], min_chunk: int = 8) -> "HurstResult":
        if len(prices) < 20:
            return HurstResult(0.5, "random_walk", 0.0)

        rets = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        n = len(rets)

        rs_vals: List[Tuple[int, float]] = []
        chunk = min_chunk
        while chunk <= n // 2:
            chunks = [rets[i:i + chunk] for i in range(0, n - chunk + 1, chunk)]
            rs_list = []
            for c in chunks:
                mean_c = sum(c) / len(c)
                cumdev = []
                s = 0.0
                for v in c:
                    s += v - mean_c
                    cumdev.append(s)
                R = max(cumdev) - min(cumdev)
                std_c = math.sqrt(sum((v - mean_c) ** 2 for v in c) / len(c))
                if std_c > 0:
                    rs_list.append(R / std_c)
            if rs_list:
                rs_vals.append((chunk, sum(rs_list) / len(rs_list)))
            chunk *= 2

        if len(rs_vals) < 2:
            return HurstResult(0.5, "random_walk", 0.0)

        # OLS log(RS) ~ H * log(n)
        xs = [math.log(r[0]) for r in rs_vals]
        ys = [math.log(r[1]) for r in rs_vals]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
        den = sum((xs[i] - mx) ** 2 for i in range(len(xs)))
        H = num / den if den > 0 else 0.5
        H = max(0.0, min(1.0, H))

        if H < 0.4:
            regime = "mean_reverting"
        elif H > 0.6:
            regime = "trending"
        else:
            regime = "random_walk"

        # score: +1 = pure mean-reverting (H=0), -1 = pure trending (H=1)
        score = 1.0 - 2.0 * H
        return HurstResult(H=H, regime=regime, score=score)


# ─── Realized Volatility ──────────────────────────────────────────────────────

@dataclass
class VolatilityResult:
    """
    Rolling realized volatility (annualized) and vol regime.

    Regime classification uses a dynamic threshold based on recent history.
    """
    realized_vol: float     # annualized realized vol (e.g. 0.80 = 80%)
    regime: str             # "calm" | "normal" | "explosive"
    vol_percentile: float   # where current vol sits in its own history
    score: float            # +1 = calm (predictable), -1 = explosive

    @staticmethod
    def compute(prices: List[float], window: int = 20, annualize: float = 252.0) -> "VolatilityResult":
        if len(prices) < window + 1:
            return VolatilityResult(0.0, "normal", 0.5, 0.0)

        rets = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]

        # Rolling std
        rolling_vols = []
        for i in range(window, len(rets) + 1):
            chunk = rets[i - window:i]
            mean_c = sum(chunk) / len(chunk)
            var = sum((r - mean_c) ** 2 for r in chunk) / (len(chunk) - 1)
            rolling_vols.append(math.sqrt(var * annualize))

        if not rolling_vols:
            return VolatilityResult(0.0, "normal", 0.5, 0.0)

        current_vol = rolling_vols[-1]
        sorted_vols = sorted(rolling_vols)
        rank = sum(1 for v in sorted_vols if v <= current_vol)
        pct = rank / len(sorted_vols)

        if pct < 0.33:
            regime = "calm"
            score = 1.0
        elif pct > 0.67:
            regime = "explosive"
            score = -1.0
        else:
            regime = "normal"
            score = 0.0

        return VolatilityResult(
            realized_vol=current_vol,
            regime=regime,
            vol_percentile=pct,
            score=score,
        )


# ─── Order Flow Imbalance ─────────────────────────────────────────────────────

@dataclass
class OrderFlowResult:
    """
    Order flow imbalance from tick or bar data.

    OFI > 0 → buy pressure → bullish
    OFI < 0 → sell pressure → bearish

    When raw trade data isn't available, approximated from price+volume
    using the Lee-Ready rule: uptick → buyer-initiated, downtick → seller.
    """
    ofi: float           # raw imbalance: buy_vol - sell_vol
    ofi_ratio: float     # (buy - sell) / (buy + sell) in [-1, +1]
    buy_volume: float
    sell_volume: float
    score: float         # = ofi_ratio

    @staticmethod
    def compute(
        prices: List[float],
        volumes: Optional[List[float]] = None,
        trades: Optional[List[Tuple[float, float, str]]] = None,  # (price, qty, "buy"/"sell")
    ) -> "OrderFlowResult":
        buy_vol = 0.0
        sell_vol = 0.0

        if trades:
            for _, qty, side in trades:
                if side.lower() in ("buy", "b"):
                    buy_vol += qty
                else:
                    sell_vol += qty
        else:
            # Lee-Ready proxy: price up → buyer-initiated
            vols = volumes or [1.0] * len(prices)
            for i in range(1, len(prices)):
                v = vols[i] if i < len(vols) else 1.0
                if prices[i] >= prices[i - 1]:
                    buy_vol += v
                else:
                    sell_vol += v

        total = buy_vol + sell_vol
        ofi_ratio = (buy_vol - sell_vol) / total if total > 0 else 0.0
        return OrderFlowResult(
            ofi=buy_vol - sell_vol,
            ofi_ratio=ofi_ratio,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            score=ofi_ratio,
        )


# ─── Volume Profile (VWAP / VPOC) ─────────────────────────────────────────────

@dataclass
class VolumeProfileResult:
    """
    Volume-weighted average price (VWAP) and volume point of control (VPOC).

    VPOC = price level with the highest traded volume (fair value).
    Price above VPOC → premium, below → discount.
    """
    vwap: float
    vpoc: float              # price level with most volume
    value_area_high: float   # top of 70% volume area
    value_area_low: float    # bottom of 70% volume area
    price_vs_vwap: float     # (current_price - vwap) / vwap
    price_vs_vpoc: float     # (current_price - vpoc) / vpoc
    score: float             # negative when price far above VWAP (overbought)

    @staticmethod
    def compute(
        prices: List[float],
        volumes: Optional[List[float]] = None,
        n_bins: int = 30,
    ) -> "VolumeProfileResult":
        if not prices:
            return VolumeProfileResult(0, 0, 0, 0, 0, 0, 0)

        vols = volumes or [1.0] * len(prices)
        current = prices[-1]

        # VWAP
        total_vol = sum(vols)
        vwap = sum(p * v for p, v in zip(prices, vols)) / total_vol if total_vol > 0 else current

        # VPOC via price bins
        lo, hi = min(prices), max(prices)
        if hi <= lo:
            return VolumeProfileResult(vwap, current, current, current, 0, 0, 0)

        bin_size = (hi - lo) / n_bins
        bins: List[float] = [0.0] * n_bins
        bin_prices: List[float] = [lo + (i + 0.5) * bin_size for i in range(n_bins)]

        for p, v in zip(prices, vols):
            idx = min(int((p - lo) / bin_size), n_bins - 1)
            bins[idx] += v

        vpoc_idx = bins.index(max(bins))
        vpoc = bin_prices[vpoc_idx]

        # Value area: cumulative 70% of total volume around VPOC
        target = total_vol * 0.70
        accum = bins[vpoc_idx]
        lo_idx, hi_idx = vpoc_idx, vpoc_idx
        while accum < target and (lo_idx > 0 or hi_idx < n_bins - 1):
            lo_add = bins[lo_idx - 1] if lo_idx > 0 else 0
            hi_add = bins[hi_idx + 1] if hi_idx < n_bins - 1 else 0
            if lo_add >= hi_add and lo_idx > 0:
                lo_idx -= 1
                accum += lo_add
            elif hi_idx < n_bins - 1:
                hi_idx += 1
                accum += hi_add
            else:
                break

        val_area_low = bin_prices[lo_idx]
        val_area_high = bin_prices[hi_idx]

        pv = (current - vwap) / vwap if vwap > 0 else 0
        pp = (current - vpoc) / vpoc if vpoc > 0 else 0

        # Mean-reversion signal: price far from VWAP → expect return
        # Negative score when overbought (above VWAP), positive when oversold
        score = math.tanh(-pv * 10.0)

        return VolumeProfileResult(
            vwap=vwap,
            vpoc=vpoc,
            value_area_high=val_area_high,
            value_area_low=val_area_low,
            price_vs_vwap=pv,
            price_vs_vpoc=pp,
            score=score,
        )


# ─── Microstructure ───────────────────────────────────────────────────────────

@dataclass
class MicrostructureResult:
    """
    Market microstructure quality indicators.

    - Amihud illiquidity: |return| / volume — how much $1 of volume moves price
    - Roll spread: inferred bid-ask spread from return autocorrelation
    - Kyle's lambda proxy: price impact per unit volume
    """
    amihud: float         # illiquidity ratio (higher = less liquid)
    roll_spread: float    # inferred bid-ask spread as fraction of price
    kyle_lambda: float    # price impact per unit volume
    liquidity_score: float  # +1 = very liquid, -1 = illiquid
    score: float          # same as liquidity_score

    @staticmethod
    def compute(
        prices: List[float],
        volumes: Optional[List[float]] = None,
    ) -> "MicrostructureResult":
        if len(prices) < 5:
            return MicrostructureResult(0, 0, 0, 0, 0)

        vols = volumes or [1.0] * len(prices)
        rets = [abs(math.log(prices[i] / prices[i - 1])) for i in range(1, len(prices))]
        signed_rets = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]

        # Amihud illiquidity
        daily_amihud = []
        for r, v in zip(rets, vols[1:]):
            if v > 0:
                daily_amihud.append(r / v)
        amihud = sum(daily_amihud) / len(daily_amihud) if daily_amihud else 0

        # Roll spread: 2 * sqrt(-cov(r_t, r_{t-1}))
        if len(signed_rets) >= 2:
            cov_sum = 0.0
            n_pairs = len(signed_rets) - 1
            for i in range(n_pairs):
                cov_sum += signed_rets[i] * signed_rets[i + 1]
            cov = cov_sum / n_pairs
            roll_spread = 2.0 * math.sqrt(max(-cov, 0))
        else:
            roll_spread = 0.0

        # Kyle's lambda: regress |price change| on volume
        # λ = cov(|ΔP|, V) / var(V)
        if len(rets) >= 4:
            v_slice = [vols[i + 1] for i in range(len(rets))]
            mv = sum(v_slice) / len(v_slice)
            mr = sum(rets) / len(rets)
            cov_rv = sum((rets[i] - mr) * (v_slice[i] - mv) for i in range(len(rets)))
            var_v = sum((v - mv) ** 2 for v in v_slice)
            kyle_lambda = cov_rv / var_v if var_v > 0 else 0
        else:
            kyle_lambda = 0

        # Liquidity score: low Amihud + low Roll = liquid
        amihud_norm = math.tanh(amihud * 1000)
        roll_norm = math.tanh(roll_spread * 100)
        liquidity = 1.0 - (amihud_norm + roll_norm) / 2.0
        liquidity = max(-1.0, min(1.0, liquidity))

        return MicrostructureResult(
            amihud=amihud,
            roll_spread=roll_spread,
            kyle_lambda=kyle_lambda,
            liquidity_score=liquidity,
            score=liquidity,
        )


# ─── RSI / Momentum ───────────────────────────────────────────────────────────

@dataclass
class MomentumResult:
    """
    RSI and price momentum.

    RSI > 70 → overbought (bearish signal)
    RSI < 30 → oversold (bullish signal)
    RSI near 50 → neutral
    """
    rsi: float            # 0 to 100
    rsi_signal: str       # "overbought" | "oversold" | "neutral"
    momentum_1: float     # 1-period return
    momentum_5: float     # 5-period return
    momentum_20: float    # 20-period return
    score: float          # +1 = oversold/buy, -1 = overbought/sell

    @staticmethod
    def compute(prices: List[float], period: int = 14) -> "MomentumResult":
        if len(prices) < period + 1:
            return MomentumResult(50.0, "neutral", 0.0, 0.0, 0.0, 0.0)

        rets = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        recent = rets[-period:]

        gains = [r for r in recent if r > 0]
        losses = [-r for r in recent if r < 0]
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        if rsi > 70:
            sig = "overbought"
        elif rsi < 30:
            sig = "oversold"
        else:
            sig = "neutral"

        m1 = (prices[-1] / prices[-2] - 1) if len(prices) >= 2 else 0
        m5 = (prices[-1] / prices[-6] - 1) if len(prices) >= 6 else 0
        m20 = (prices[-1] / prices[-21] - 1) if len(prices) >= 21 else 0

        # Contrarian mean-reversion score: oversold → +1, overbought → -1
        score = (50.0 - rsi) / 50.0
        score = max(-1.0, min(1.0, score))

        return MomentumResult(
            rsi=rsi,
            rsi_signal=sig,
            momentum_1=m1,
            momentum_5=m5,
            momentum_20=m20,
            score=score,
        )


# ─── Autocorrelation ─────────────────────────────────────────────────────────

@dataclass
class AutocorrResult:
    """
    Lag-1 and lag-5 return autocorrelation.

    Positive autocorr → momentum (returns continue)
    Negative autocorr → mean-reversion (returns reverse)
    """
    lag1: float     # autocorrelation at lag 1
    lag5: float     # autocorrelation at lag 5
    lag10: float    # autocorrelation at lag 10
    score: float    # +1 = strong MR (neg autocorr), -1 = strong momentum

    @staticmethod
    def compute(prices: List[float]) -> "AutocorrResult":
        if len(prices) < 15:
            return AutocorrResult(0, 0, 0, 0)

        rets = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        mean_r = sum(rets) / len(rets)
        var_r = sum((r - mean_r) ** 2 for r in rets) / len(rets)

        def autocorr(lag: int) -> float:
            if var_r == 0 or lag >= len(rets):
                return 0.0
            c = sum((rets[i] - mean_r) * (rets[i - lag] - mean_r)
                    for i in range(lag, len(rets))) / (len(rets) - lag)
            return c / var_r

        lag1 = autocorr(1)
        lag5 = autocorr(5)
        lag10 = autocorr(10)

        # Negative autocorr → mean-reversion (positive score)
        score = -math.tanh((lag1 + 0.5 * lag5) * 3.0)
        return AutocorrResult(lag1=lag1, lag5=lag5, lag10=lag10, score=score)


# ─── Funding Rate Proxy ───────────────────────────────────────────────────────

@dataclass
class FundingProxyResult:
    """
    Funding rate proxy for perpetual futures.

    Uses price skewness + momentum divergence as a synthetic funding indicator.
    High positive funding → longs pay shorts → contrarian bearish signal.
    High negative funding → shorts pay longs → contrarian bullish signal.
    """
    proxy_rate: float    # synthetic funding rate
    signal: str          # "pay_longs" | "pay_shorts" | "neutral"
    score: float         # +1 = bearish (overheated longs), -1 = bullish

    @staticmethod
    def compute(prices: List[float]) -> "FundingProxyResult":
        if len(prices) < 20:
            return FundingProxyResult(0, "neutral", 0)

        rets = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        recent = rets[-20:]

        # Skewness of returns as funding proxy
        n = len(recent)
        mean_r = sum(recent) / n
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in recent) / n)
        if std_r == 0:
            return FundingProxyResult(0, "neutral", 0)

        skew = sum(((r - mean_r) / std_r) ** 3 for r in recent) / n
        # Positive skew = tail up = market betting on up = funding positive
        proxy_rate = skew * 0.01  # scale to typical funding range

        if skew > 0.5:
            signal = "pay_longs"   # overheated → contrarian bearish
            score = -math.tanh(skew)
        elif skew < -0.5:
            signal = "pay_shorts"  # oversold → contrarian bullish
            score = math.tanh(-skew)
        else:
            signal = "neutral"
            score = 0.0

        return FundingProxyResult(proxy_rate=proxy_rate, signal=signal, score=score)
