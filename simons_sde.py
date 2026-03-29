"""Jim Simons / Renaissance Technologies SDE signals.

The mathematical toolkit that powered Medallion Fund:

1.  Itô SDE (Geometric Brownian Motion)
        dS = μS dt + σS dW
    → Calibrate μ (drift) and σ (diffusion) from price data.
    → Detect when realized drift is statistically significant.

2.  Ornstein-Uhlenbeck (Mean Reversion)
        dX = θ(μ - X) dt + σ dW
    → Calibrate θ (speed), μ (long-run mean), σ (vol) via MLE.
    → Estimate half-life and z-score for mean-reversion trades.

3.  Jump-Diffusion (Merton)
        dS = μS dt + σS dW + S(e^J - 1) dN
    → Detect jump arrivals and size, separate from diffusion.

4.  Regime-Switching Drift (Hidden Markov)
        μ(t) ∈ {μ_bull, μ_bear}  switched by latent Markov chain
    → Two-state Viterbi-style EM to find regime and transition probs.

5.  Kalman Filter (optimal linear state estimator)
        x_k = A x_{k-1} + w_k    (state: [price, drift])
        z_k = H x_k  + v_k       (observation: price)
    → Continuously updates best estimate of true price + trend.

6.  Pairs / Spread Cointegration
        spread = P_a - β P_b
    → Engle-Granger residual stationarity → OU fit on spread.

7.  Kelly Criterion
        f* = μ / σ²   (continuous, for known μ, σ)
    → Optimal fraction of capital to bet given estimated edge.

8.  Fourier / Spectral Cycle Detection
    → Dominant frequency in return series → predictable cycle.

9.  Hidden Markov Model (Baum-Welch) – volatility regimes
    → 2-state HMM: calm-vol and high-vol, forward-backward EM.

All implementations are pure Python (no numpy/scipy required).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ─── 1. Itô GBM — calibrate μ and σ ──────────────────────────────────────────

@dataclass
class GBMResult:
    """
    Geometric Brownian Motion calibration.

    dS = μ S dt + σ S dW

    μ  > 0  → upward drift (bullish edge)
    μ  < 0  → downward drift (bearish edge)
    |μ/σ|   → signal-to-noise (Sharpe per unit time)
    """
    mu: float        # drift (annualized)
    sigma: float     # diffusion (annualized vol)
    sharpe: float    # μ / σ  (risk-adjusted edge)
    signal: str      # "bullish" | "bearish" | "neutral"
    score: float     # tanh(μ/σ * 2), in [-1, +1]

    @staticmethod
    def calibrate(prices: List[float], annualize: float = 252.0) -> "GBMResult":
        if len(prices) < 3:
            return GBMResult(0.0, 0.0, 0.0, "neutral", 0.0)

        log_rets = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        n = len(log_rets)
        mean_r = sum(log_rets) / n
        var_r  = sum((r - mean_r)**2 for r in log_rets) / (n - 1) if n > 1 else 0.0

        # GBM parameter recovery (continuous-time MLE)
        sigma  = math.sqrt(var_r * annualize)
        # E[log return per step] = (μ - σ²/2) dt  → μ = mean_r/dt + σ²/2
        mu     = mean_r * annualize + 0.5 * sigma**2
        sharpe = mu / sigma if sigma > 0 else 0.0

        if sharpe > 0.3:
            signal = "bullish"
        elif sharpe < -0.3:
            signal = "bearish"
        else:
            signal = "neutral"

        score = math.tanh(sharpe * 2.0)
        return GBMResult(mu=mu, sigma=sigma, sharpe=sharpe, signal=signal, score=score)


# ─── 2. Ornstein-Uhlenbeck — mean reversion ───────────────────────────────────

@dataclass
class OUResult:
    """
    Ornstein-Uhlenbeck process fit.

    dX = θ(μ - X) dt + σ dW

    θ    → reversion speed (larger = faster mean reversion)
    μ    → long-run mean
    σ    → noise level
    half_life = ln(2) / θ  (time to close 50% of gap)
    z_score = (current_price - μ) / σ_eq
    """
    theta: float       # reversion speed
    mu: float          # long-run mean
    sigma: float       # diffusion coefficient
    half_life: float   # bars to revert 50%
    z_score: float     # current deviation in σ units
    signal: str        # "mean_revert_long" | "mean_revert_short" | "neutral"
    score: float       # -z_score clamped, buy when price is below mean

    @staticmethod
    def calibrate(prices: List[float]) -> "OUResult":
        if len(prices) < 5:
            return OUResult(0.0, float(prices[-1]) if prices else 0.0,
                            0.0, float("inf"), 0.0, "neutral", 0.0)

        # Discrete AR(1) fit via OLS: X_t = a + b X_{t-1} + ε
        x  = prices[:-1]
        y  = prices[1:]
        n  = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        cov_xy = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
        var_x  = sum((xi - mx)**2 for xi in x) / n

        b  = cov_xy / var_x if var_x > 0 else 1.0
        a  = my - b * mx

        # OU parameter mapping (dt = 1 bar)
        theta    = -math.log(b) if b > 0 and b != 1.0 else 1e-6
        mu_ou    = a / (1.0 - b) if abs(1.0 - b) > 1e-9 else mx

        # σ from residuals
        resids   = [y[i] - (a + b * x[i]) for i in range(n)]
        mean_res = sum(resids) / n
        var_res  = sum((r - mean_res)**2 for r in resids) / (n - 1) if n > 1 else 0.0
        sigma_res = math.sqrt(var_res)

        # Equilibrium std of OU: σ_eq = σ / sqrt(2θ)
        sigma_eq = sigma_res / math.sqrt(2.0 * max(theta, 1e-6))

        half_life = math.log(2.0) / max(theta, 1e-9)
        current   = prices[-1]
        z         = (current - mu_ou) / sigma_eq if sigma_eq > 0 else 0.0

        if z < -1.5:
            signal = "mean_revert_long"   # price below mean → buy
        elif z > 1.5:
            signal = "mean_revert_short"  # price above mean → sell
        else:
            signal = "neutral"

        # score: negative z → positive score (buy when cheap)
        score = math.tanh(-z)
        return OUResult(theta=theta, mu=mu_ou, sigma=sigma_res,
                        half_life=half_life, z_score=z,
                        signal=signal, score=score)


# ─── 3. Jump-Diffusion (Merton) ───────────────────────────────────────────────

@dataclass
class JumpResult:
    """
    Merton jump-diffusion model.

    dS/S = (μ - λκ) dt + σ dW + (e^J - 1) dN
      λ  = jump intensity (jumps per bar)
      κ  = E[e^J - 1]  (expected relative jump size)
      σ  = diffusion vol (ex-jump)

    Jump detection: returns > mean ± k*std are flagged as jumps.
    """
    lambda_: float     # jump intensity estimate
    jump_mean: float   # average log-jump size
    jump_std: float    # std of log-jump size
    diffusion_sigma: float
    recent_jump: bool  # True if last bar was a jump
    recent_jump_size: float
    score: float       # -1 if recent large negative jump, +1 large positive

    @staticmethod
    def detect(prices: List[float], k_sigma: float = 3.0) -> "JumpResult":
        if len(prices) < 10:
            return JumpResult(0.0, 0.0, 0.0, 0.0, False, 0.0, 0.0)

        rets = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        mean_r = sum(rets) / len(rets)
        std_r  = math.sqrt(sum((r - mean_r)**2 for r in rets) / len(rets))

        threshold = k_sigma * std_r
        jumps     = [r for r in rets if abs(r - mean_r) > threshold]
        diffusion = [r for r in rets if abs(r - mean_r) <= threshold]

        lambda_  = len(jumps) / len(rets)
        jmean    = sum(jumps) / len(jumps) if jumps else 0.0
        jstd     = math.sqrt(sum((j - jmean)**2 for j in jumps) / len(jumps)) if len(jumps) > 1 else 0.0

        diff_mean = sum(diffusion) / len(diffusion) if diffusion else mean_r
        diff_std  = math.sqrt(sum((r - diff_mean)**2 for r in diffusion) / len(diffusion)) if diffusion else std_r

        last_ret      = rets[-1]
        recent_jump   = abs(last_ret - mean_r) > threshold
        score = math.tanh(last_ret / (threshold + 1e-9)) if recent_jump else 0.0

        return JumpResult(
            lambda_=lambda_,
            jump_mean=jmean,
            jump_std=jstd,
            diffusion_sigma=diff_std,
            recent_jump=recent_jump,
            recent_jump_size=last_ret if recent_jump else 0.0,
            score=score,
        )


# ─── 4. Regime-Switching Drift (2-state HMM) ──────────────────────────────────

@dataclass
class RegimeSwitchResult:
    """
    2-state hidden Markov model on return drift.

    States: 0 = bull (high drift), 1 = bear (low / negative drift)
    Estimated via simplified Baum-Welch (EM).
    """
    mu_bull: float
    mu_bear: float
    sigma_bull: float
    sigma_bear: float
    p_bull_to_bear: float    # transition probability
    p_bear_to_bull: float
    current_regime: str      # "bull" | "bear"
    regime_prob: float       # probability of current regime
    score: float             # +1 bull, -1 bear

    @staticmethod
    def fit(prices: List[float], n_iter: int = 10) -> "RegimeSwitchResult":
        if len(prices) < 20:
            return RegimeSwitchResult(0.0, 0.0, 0.01, 0.01, 0.1, 0.1,
                                      "neutral", 0.5, 0.0)

        rets = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        n    = len(rets)

        # Initial parameters: split by median
        med = sorted(rets)[n // 2]
        hi  = [r for r in rets if r >= med]
        lo  = [r for r in rets if r < med]

        def mean_std(xs: List[float]) -> Tuple[float, float]:
            if not xs:
                return 0.0, 0.01
            m = sum(xs) / len(xs)
            s = math.sqrt(sum((x - m)**2 for x in xs) / len(xs))
            return m, max(s, 1e-6)

        mu0, s0 = mean_std(hi)    # state 0: bull
        mu1, s1 = mean_std(lo)    # state 1: bear
        p01, p10 = 0.05, 0.05     # transition probs

        def gauss(x: float, mu: float, s: float) -> float:
            return math.exp(-0.5 * ((x - mu) / s)**2) / (s * math.sqrt(2 * math.pi))

        # EM: simplified forward pass only (Viterbi-like)
        for _ in range(n_iter):
            # E-step: compute state posteriors with forward algorithm
            alpha = [[0.0, 0.0] for _ in range(n)]
            # initial
            b0 = gauss(rets[0], mu0, s0)
            b1 = gauss(rets[0], mu1, s1)
            alpha[0][0] = 0.5 * b0
            alpha[0][1] = 0.5 * b1
            scale = alpha[0][0] + alpha[0][1]
            if scale > 0:
                alpha[0][0] /= scale
                alpha[0][1] /= scale

            for t in range(1, n):
                b0 = gauss(rets[t], mu0, s0)
                b1 = gauss(rets[t], mu1, s1)
                alpha[t][0] = (alpha[t-1][0] * (1 - p01) + alpha[t-1][1] * p10) * b0
                alpha[t][1] = (alpha[t-1][0] * p01       + alpha[t-1][1] * (1 - p10)) * b1
                sc = alpha[t][0] + alpha[t][1]
                if sc > 0:
                    alpha[t][0] /= sc
                    alpha[t][1] /= sc

            # M-step: update parameters from posteriors
            w0 = [alpha[t][0] for t in range(n)]
            w1 = [alpha[t][1] for t in range(n)]
            sw0 = sum(w0) or 1.0
            sw1 = sum(w1) or 1.0

            mu0 = sum(w0[t] * rets[t] for t in range(n)) / sw0
            mu1 = sum(w1[t] * rets[t] for t in range(n)) / sw1
            s0  = math.sqrt(sum(w0[t] * (rets[t] - mu0)**2 for t in range(n)) / sw0) or 1e-6
            s1  = math.sqrt(sum(w1[t] * (rets[t] - mu1)**2 for t in range(n)) / sw1) or 1e-6

            # Transition probs
            trans01 = sum(alpha[t][0] * p01 * gauss(rets[t], mu1, s1) for t in range(1, n))
            stay00  = sum(alpha[t][0] * (1 - p01) * gauss(rets[t], mu0, s0) for t in range(1, n))
            p01 = trans01 / (trans01 + stay00 + 1e-9)

            trans10 = sum(alpha[t][1] * p10 * gauss(rets[t], mu0, s0) for t in range(1, n))
            stay11  = sum(alpha[t][1] * (1 - p10) * gauss(rets[t], mu1, s1) for t in range(1, n))
            p10 = trans10 / (trans10 + stay11 + 1e-9)

        # ensure state 0 = bull (higher mu)
        if mu0 < mu1:
            mu0, mu1 = mu1, mu0
            s0, s1   = s1, s0
            p01, p10 = p10, p01
            for row in alpha:
                row[0], row[1] = row[1], row[0]

        prob_bull = alpha[-1][0]
        regime    = "bull" if prob_bull > 0.5 else "bear"
        score     = math.tanh((prob_bull - 0.5) * 4.0)

        return RegimeSwitchResult(
            mu_bull=mu0, mu_bear=mu1,
            sigma_bull=s0, sigma_bear=s1,
            p_bull_to_bear=p01, p_bear_to_bull=p10,
            current_regime=regime,
            regime_prob=prob_bull if regime == "bull" else (1 - prob_bull),
            score=score,
        )


# ─── 5. Kalman Filter — price + drift tracking ────────────────────────────────

@dataclass
class KalmanResult:
    """
    Kalman filter: optimal linear estimate of price + drift.

    State:  x = [level, drift]
    Model:  level_{t} = level_{t-1} + drift_{t-1} + w1
            drift_{t} = drift_{t-1}               + w2
            obs:     z_t = level_t + v_t

    Smoothed level and drift give a noise-reduced price signal.
    """
    filtered_price: float   # Kalman-smoothed price
    filtered_drift: float   # estimated drift (per bar)
    innovation: float       # z_t - predicted z_t  (surprise)
    innovation_std: float   # std of recent innovations
    score: float            # tanh(filtered_drift / innovation_std)

    @staticmethod
    def filter(prices: List[float],
               process_noise: float = 1e-4,
               obs_noise: float = 1e-2) -> "KalmanResult":
        if len(prices) < 3:
            return KalmanResult(prices[-1] if prices else 0.0, 0.0, 0.0, 1.0, 0.0)

        # State [level, drift], covariance P (2x2)
        level  = prices[0]
        drift  = 0.0
        # P as flat [p00, p01, p10, p11]
        p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

        # Process noise Q, observation noise R
        q0, q1 = process_noise, process_noise * 0.1
        r = obs_noise

        innovations: List[float] = []
        for z in prices[1:]:
            # Predict
            level_pred = level + drift
            drift_pred = drift
            # Predicted covariance P_pred = F P F^T + Q
            p00_p = p00 + p10 + p01 + p11 + q0
            p01_p = p01 + p11
            p10_p = p10 + p11
            p11_p = p11 + q1

            # Innovation
            innov = z - level_pred
            s_innov = p00_p + r   # innovation variance
            innovations.append(innov)

            # Kalman gain K = P_pred H^T / S  (H = [1, 0])
            k0 = p00_p / s_innov
            k1 = p10_p / s_innov

            # Update
            level = level_pred + k0 * innov
            drift = drift_pred + k1 * innov

            p00 = (1 - k0) * p00_p
            p01 = (1 - k0) * p01_p
            p10 = p10_p - k1 * p00_p
            p11 = p11_p - k1 * p01_p

        innov_std = math.sqrt(sum(i**2 for i in innovations) / len(innovations)) if innovations else 1.0
        last_innov = innovations[-1] if innovations else 0.0
        score = math.tanh(drift / (innov_std + 1e-9))

        return KalmanResult(
            filtered_price=level,
            filtered_drift=drift,
            innovation=last_innov,
            innovation_std=innov_std,
            score=score,
        )


# ─── 6. Spread / Pairs Cointegration ─────────────────────────────────────────

@dataclass
class SpreadResult:
    """
    Pairs cointegration and OU spread.

    spread_t = P_a_t - β P_b_t
    Fit OU on spread → z-score for pairs trade.
    """
    beta: float           # hedge ratio
    spread_mean: float
    spread_std: float
    z_score: float        # current spread z-score
    half_life: float      # bars to mean-revert
    signal: str           # "buy_spread" | "sell_spread" | "neutral"
    score: float

    @staticmethod
    def compute(prices_a: List[float], prices_b: List[float]) -> "SpreadResult":
        n = min(len(prices_a), len(prices_b))
        if n < 10:
            return SpreadResult(1.0, 0.0, 1.0, 0.0, float("inf"), "neutral", 0.0)

        a = prices_a[-n:]
        b = prices_b[-n:]

        # OLS beta: regress a on b
        ma = sum(a) / n
        mb = sum(b) / n
        cov_ab = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / n
        var_b  = sum((bi - mb)**2 for bi in b) / n
        beta   = cov_ab / var_b if var_b > 0 else 1.0

        spread = [a[i] - beta * b[i] for i in range(n)]
        ou     = OUResult.calibrate(spread)

        if ou.z_score < -1.5:
            signal = "buy_spread"   # spread cheap → buy a, sell b
        elif ou.z_score > 1.5:
            signal = "sell_spread"  # spread expensive → sell a, buy b
        else:
            signal = "neutral"

        return SpreadResult(
            beta=beta,
            spread_mean=ou.mu,
            spread_std=ou.sigma,
            z_score=ou.z_score,
            half_life=ou.half_life,
            signal=signal,
            score=ou.score,
        )


# ─── 7. Kelly Criterion ───────────────────────────────────────────────────────

@dataclass
class KellyResult:
    """
    Continuous Kelly criterion: f* = μ / σ²

    f* = fraction of capital to allocate.
    Kelly-half (f*/2) is often used in practice to reduce variance.
    """
    f_star: float       # full Kelly fraction
    f_half: float       # half-Kelly (safer)
    mu: float
    sigma: float
    edge: float         # risk-adjusted edge = Sharpe²
    score: float        # tanh(f_star)

    @staticmethod
    def compute(prices: List[float]) -> "KellyResult":
        gbm = GBMResult.calibrate(prices)
        f_star = gbm.mu / (gbm.sigma**2) if gbm.sigma > 0 else 0.0
        f_star = max(-1.0, min(2.0, f_star))  # clamp to sensible range
        edge   = gbm.sharpe**2
        return KellyResult(
            f_star=f_star,
            f_half=f_star / 2.0,
            mu=gbm.mu,
            sigma=gbm.sigma,
            edge=edge,
            score=math.tanh(f_star),
        )


# ─── 8. Fourier Cycle Detection ───────────────────────────────────────────────

@dataclass
class FourierResult:
    """
    Dominant cycle from DFT of return series.

    Identifies the period (in bars) of the strongest oscillation.
    Use for timing mean-reversion trades.
    """
    dominant_period: float   # bars per cycle
    dominant_power: float    # normalized power at dominant freq
    phase: float             # current phase [0, 2π]
    cycle_position: str      # "peak" | "trough" | "rising" | "falling"
    score: float             # +1 at trough (buy), -1 at peak (sell)

    @staticmethod
    def analyze(prices: List[float]) -> "FourierResult":
        if len(prices) < 8:
            return FourierResult(0.0, 0.0, 0.0, "unknown", 0.0)

        rets = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        n    = len(rets)
        # Remove mean
        mean_r = sum(rets) / n
        rets   = [r - mean_r for r in rets]

        # DFT (naive O(n²), fine for n < 500)
        best_power  = 0.0
        best_period = float(n)
        best_phase  = 0.0

        for k in range(2, n // 2 + 1):
            re = sum(rets[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
            im = sum(rets[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
            power = re**2 + im**2
            if power > best_power:
                best_power  = power
                best_period = n / k
                best_phase  = math.atan2(im, re)

        # Normalize power
        total_power = sum(
            sum(rets[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))**2 +
            sum(rets[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))**2
            for k in range(2, n // 2 + 1)
        ) or 1.0
        norm_power = best_power / total_power

        # Phase → cycle position (phase measured at last bar t = n-1)
        current_phase = (2 * math.pi * (n - 1) / best_period + best_phase) % (2 * math.pi)
        if current_phase < math.pi / 2:
            position, score = "rising",  0.0
        elif current_phase < math.pi:
            position, score = "peak",   -norm_power
        elif current_phase < 3 * math.pi / 2:
            position, score = "falling", 0.0
        else:
            position, score = "trough",  norm_power

        return FourierResult(
            dominant_period=best_period,
            dominant_power=norm_power,
            phase=current_phase,
            cycle_position=position,
            score=score,
        )


# ─── 9. HMM Volatility Regimes (Baum-Welch, 2-state) ─────────────────────────

@dataclass
class HMMVolResult:
    """
    2-state Hidden Markov Model on squared returns (volatility proxy).

    State 0: low-vol (calm)
    State 1: high-vol (explosive)
    """
    sigma_low: float
    sigma_high: float
    p_low_to_high: float
    p_high_to_low: float
    current_state: str    # "low_vol" | "high_vol"
    state_prob: float
    score: float          # +1 = calm, -1 = explosive

    @staticmethod
    def fit(prices: List[float], n_iter: int = 8) -> "HMMVolResult":
        if len(prices) < 20:
            return HMMVolResult(0.01, 0.05, 0.1, 0.3, "low_vol", 0.8, 0.5)

        rets = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        obs  = [r**2 for r in rets]   # squared returns as vol proxy
        n    = len(obs)
        med  = sorted(obs)[n // 2]

        # Initial: split by median
        hi_obs = [o for o in obs if o > med]
        lo_obs = [o for o in obs if o <= med]

        def mean_s(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else 1e-4

        # State 0 = low vol, state 1 = high vol
        var0 = mean_s(lo_obs)
        var1 = mean_s(hi_obs)
        p01, p10 = 0.1, 0.3   # initial transitions

        def gauss_sq(x: float, variance: float) -> float:
            s = math.sqrt(max(variance, 1e-12))
            return math.exp(-0.5 * x / max(variance, 1e-12)) / (s * math.sqrt(2 * math.pi))

        alpha = [[0.0, 0.0] for _ in range(n)]
        for _ in range(n_iter):
            # Forward pass
            b0 = gauss_sq(obs[0], var0)
            b1 = gauss_sq(obs[0], var1)
            alpha[0][0] = 0.5 * b0
            alpha[0][1] = 0.5 * b1
            sc = alpha[0][0] + alpha[0][1]
            if sc > 0:
                alpha[0][0] /= sc; alpha[0][1] /= sc

            for t in range(1, n):
                b0 = gauss_sq(obs[t], var0)
                b1 = gauss_sq(obs[t], var1)
                alpha[t][0] = (alpha[t-1][0] * (1 - p01) + alpha[t-1][1] * p10) * b0
                alpha[t][1] = (alpha[t-1][0] * p01        + alpha[t-1][1] * (1 - p10)) * b1
                sc = alpha[t][0] + alpha[t][1]
                if sc > 0:
                    alpha[t][0] /= sc; alpha[t][1] /= sc

            # M-step
            sw0 = sum(alpha[t][0] for t in range(n)) or 1.0
            sw1 = sum(alpha[t][1] for t in range(n)) or 1.0
            var0 = sum(alpha[t][0] * obs[t] for t in range(n)) / sw0
            var1 = sum(alpha[t][1] * obs[t] for t in range(n)) / sw1
            var0 = max(var0, 1e-12)
            var1 = max(var1, 1e-12)

            t01 = sum(alpha[t][0] * p01 * gauss_sq(obs[t], var1) for t in range(1, n))
            s00 = sum(alpha[t][0] * (1 - p01) * gauss_sq(obs[t], var0) for t in range(1, n))
            p01 = t01 / (t01 + s00 + 1e-9)

            t10 = sum(alpha[t][1] * p10 * gauss_sq(obs[t], var0) for t in range(1, n))
            s11 = sum(alpha[t][1] * (1 - p10) * gauss_sq(obs[t], var1) for t in range(1, n))
            p10 = t10 / (t10 + s11 + 1e-9)

        # ensure var0 < var1 (state 0 = low vol)
        if var0 > var1:
            var0, var1 = var1, var0
            p01, p10   = p10, p01
            for row in alpha:
                row[0], row[1] = row[1], row[0]

        prob_low   = alpha[-1][0]
        state      = "low_vol" if prob_low > 0.5 else "high_vol"
        score      = math.tanh((prob_low - 0.5) * 4.0)  # +1 calm, -1 explosive

        return HMMVolResult(
            sigma_low=math.sqrt(var0),
            sigma_high=math.sqrt(var1),
            p_low_to_high=p01,
            p_high_to_low=p10,
            current_state=state,
            state_prob=prob_low if state == "low_vol" else 1 - prob_low,
            score=score,
        )


# ─── Combined Simons SDE Report ──────────────────────────────────────────────

@dataclass
class SimonsSDEReport:
    """
    Full Jim Simons SDE signal suite for a single instrument.

    Composite score is a weighted average of all model scores.
    """
    gbm:            GBMResult
    ou:             OUResult
    jump:           JumpResult
    regime_switch:  RegimeSwitchResult
    kalman:         KalmanResult
    kelly:          KellyResult
    fourier:        FourierResult
    hmm_vol:        HMMVolResult
    composite:      float

    def __str__(self) -> str:
        W = 58
        bar = "─" * W
        def row(label: str, score: float, info: str) -> str:
            return f"║  {label:<16s} {score:+.3f}  {info[:31]:<31s}║"

        return "\n".join([
            f"╔{bar}╗",
            f"║  Simons SDE Report          composite={self.composite:+.3f}      ║",
            f"╠{bar}╣",
            row("GBM (Itô)",     self.gbm.score,
                f"μ={self.gbm.mu:+.3f} σ={self.gbm.sigma:.3f} {self.gbm.signal}"),
            row("Ornstein-Uhl.", self.ou.score,
                f"z={self.ou.z_score:+.2f} hl={self.ou.half_life:.1f}b {self.ou.signal}"),
            row("Jump-Diff.",    self.jump.score,
                f"λ={self.jump.lambda_:.3f} jump={self.jump.recent_jump}"),
            row("Regime-Switch", self.regime_switch.score,
                f"{self.regime_switch.current_regime} p={self.regime_switch.regime_prob:.2f}"),
            row("Kalman",        self.kalman.score,
                f"drift={self.kalman.filtered_drift:+.5f}"),
            row("Kelly f*",      self.kelly.score,
                f"f*={self.kelly.f_star:.3f} half={self.kelly.f_half:.3f}"),
            row("Fourier",       self.fourier.score,
                f"period={self.fourier.dominant_period:.1f} {self.fourier.cycle_position}"),
            row("HMM-Vol",       self.hmm_vol.score,
                f"{self.hmm_vol.current_state} p={self.hmm_vol.state_prob:.2f}"),
            f"╚{bar}╝",
        ])

    @staticmethod
    def compute(prices: List[float]) -> "SimonsSDEReport":
        gbm    = GBMResult.calibrate(prices)
        ou     = OUResult.calibrate(prices)
        jump   = JumpResult.detect(prices)
        regime = RegimeSwitchResult.fit(prices)
        kalman = KalmanResult.filter(prices)
        kelly  = KellyResult.compute(prices)
        fourier = FourierResult.analyze(prices)
        hmm    = HMMVolResult.fit(prices)

        # Weighted composite
        scores = {
            "gbm":    gbm.score,
            "ou":     ou.score,
            "jump":   jump.score,
            "regime": regime.score,
            "kalman": kalman.score,
            "kelly":  kelly.score,
            "fourier":fourier.score,
            "hmm":    hmm.score,
        }
        weights = {
            "gbm": 0.15, "ou": 0.25, "jump": 0.10,
            "regime": 0.15, "kalman": 0.15, "kelly": 0.05,
            "fourier": 0.10, "hmm": 0.05,
        }
        composite = sum(scores[k] * weights[k] for k in weights)

        return SimonsSDEReport(
            gbm=gbm, ou=ou, jump=jump, regime_switch=regime,
            kalman=kalman, kelly=kelly, fourier=fourier, hmm_vol=hmm,
            composite=composite,
        )
