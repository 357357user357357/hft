"""Polar coordinate features for HFT trading.

Transforms price-time series into polar coordinates (r, θ) to capture
geometric structure that scalar representations miss.

Key insights:
- **Radial distance (r)**: Market amplitude/expansion
- **Phase angle (θ)**: Market cycle position
- **Angular velocity (dθ/dt)**: Rotation speed (momentum direction changes)
- **Radial velocity (dr/dt)**: Expansion/contraction rate

Trading applications:
- Mean-reversion: θ cycles regularly, r oscillates
- Trending: r increases monotonically (spiral outward)
- Breakout: Sudden θ jump with r expansion
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

import constants as C


@dataclass
class PolarFeatures:
    """Polar coordinate representation of price action."""
    r: float          # radial distance (amplitude)
    theta: float      # phase angle in radians [-π, π]
    dr_dt: float      # radial velocity (expansion/contraction)
    dtheta_dt: float  # angular velocity (rotation speed)
    timestamp_ns: int


@dataclass
class PolarSignal:
    """Trading signal from polar analysis."""
    signal_type: "PolarSignalType"
    strength: float   # 0.0 to 1.0
    r: float
    theta: float
    description: str


class PolarSignalType(Enum):
    MEAN_REVERT_LONG = "mean_revert_long"
    MEAN_REVERT_SHORT = "mean_revert_short"
    TREND_LONG = "trend_long"
    TREND_SHORT = "trend_short"
    BREAKOUT_LONG = "breakout_long"
    BREAKOUT_SHORT = "breakout_short"
    HOLD = "hold"


class PolarExtractor:
    """Extracts polar features from price series."""

    def __init__(self, tau: int = C.POLAR_DEFAULT_TAU,
                 price_scale: float = C.POLAR_DEFAULT_PRICE_SCALE):
        """
        Initialize polar extractor.

        Args:
            tau: Lookback for momentum calculation (delay embedding)
            price_scale: Scaling factor to normalize price units
        """
        self.tau = tau
        self.price_scale = price_scale
        self._prev_r: Optional[float] = None
        self._prev_theta: Optional[float] = None

    def reset(self) -> None:
        """Reset internal state for new series."""
        self._prev_r = None
        self._prev_theta = None

    def extract(self, prices: List[float], timestamps: Optional[List[int]] = None) -> List[PolarFeatures]:
        """
        Extract polar features from price series.

        Args:
            prices: List of prices
            timestamps: Optional list of timestamps in nanoseconds

        Returns:
            List of PolarFeatures (length = len(prices) - tau - 1)
        """
        if len(prices) <= self.tau + 1:
            return []

        result = []
        self.reset()

        for t in range(self.tau + 1, len(prices)):
            # Phase space embedding: (price, momentum)
            x = prices[t] / self.price_scale
            y = (prices[t] - prices[t - self.tau]) / self.price_scale

            # Current polar coordinates
            r = math.sqrt(x * x + y * y)
            theta = math.atan2(y, x)

            # Velocities (need previous values)
            if self._prev_r is not None:
                dr_dt = r - self._prev_r
                dtheta_dt = self._normalize_theta_diff(theta, self._prev_theta)
            else:
                dr_dt = 0.0
                dtheta_dt = 0.0

            ts = timestamps[t] if timestamps and t < len(timestamps) else 0

            result.append(PolarFeatures(
                r=r,
                theta=theta,
                dr_dt=dr_dt,
                dtheta_dt=dtheta_dt,
                timestamp_ns=ts
            ))

            self._prev_r = r
            self._prev_theta = theta

        return result

    def _normalize_theta_diff(self, theta_new: float, theta_old: float) -> float:
        """Compute theta difference handling wraparound at ±π."""
        diff = theta_new - theta_old
        # Handle wraparound
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def extract_single(self, price: float, prev_price: float,
                       timestamp_ns: int = 0) -> Optional[PolarFeatures]:
        """
        Extract polar features for a single price update.

        Args:
            price: Current price
            prev_price: Price tau periods ago
            timestamp_ns: Current timestamp

        Returns:
            PolarFeatures or None if not enough history
        """
        x = price / self.price_scale
        y = (price - prev_price) / self.price_scale

        r = math.sqrt(x * x + y * y)
        theta = math.atan2(y, x)

        if self._prev_r is not None:
            dr_dt = r - self._prev_r
            dtheta_dt = self._normalize_theta_diff(theta, self._prev_theta)
        else:
            dr_dt = 0.0
            dtheta_dt = 0.0

        self._prev_r = r
        self._prev_theta = theta

        return PolarFeatures(
            r=r,
            theta=theta,
            dr_dt=dr_dt,
            dtheta_dt=dtheta_dt,
            timestamp_ns=timestamp_ns
        )


class PolarSignalGenerator:
    """Generates trading signals from polar features."""

    def __init__(self,
                 theta_threshold: float = C.POLAR_THETA_THRESHOLD,
                 r_trend_threshold: float = C.POLAR_R_TREND_THRESHOLD,
                 breakout_theta_jump: float = C.POLAR_BREAKOUT_THETA_JUMP,
                 lookback: int = C.POLAR_SIGNAL_LOOKBACK):
        """
        Initialize signal generator.

        Args:
            theta_threshold: Threshold for mean-reversion entry (radians)
            r_trend_threshold: Minimum dr/dt for trend detection
            breakout_theta_jump: Minimum theta change for breakout
            lookback: Number of features to analyze
        """
        self.theta_threshold = theta_threshold
        self.r_trend_threshold = r_trend_threshold
        self.breakout_theta_jump = breakout_theta_jump
        self.lookback = lookback

    def generate(self, features: List[PolarFeatures]) -> Optional[PolarSignal]:
        """
        Generate trading signal from polar features.

        Args:
            features: List of PolarFeatures (most recent last)

        Returns:
            PolarSignal or None if not enough data
        """
        if len(features) < self.lookback:
            return None

        recent = features[-self.lookback:]
        current = features[-1]

        # Check for mean-reversion (theta near 0 or π, r decreasing)
        mr_signal = self._check_mean_reversion(current, recent)
        if mr_signal:
            return mr_signal

        # Check for trending (r increasing steadily, stable theta velocity)
        trend_signal = self._check_trending(current, recent)
        if trend_signal:
            return trend_signal

        # Check for breakout (sudden theta jump)
        breakout_signal = self._check_breakout(current, recent)
        if breakout_signal:
            return breakout_signal

        return PolarSignal(
            signal_type=PolarSignalType.HOLD,
            strength=0.0,
            r=current.r,
            theta=current.theta,
            description="No clear polar pattern detected"
        )

    def _check_mean_reversion(self, current: PolarFeatures,
                               recent: List[PolarFeatures]) -> Optional[PolarSignal]:
        """Detect mean-reversion setup."""
        # Theta near 0 (bottom of cycle) or π (top of cycle)
        theta_normalized = abs(current.theta)

        if theta_normalized < self.theta_threshold:
            # Near bottom - check if r is decreasing (oversold)
            if current.dr_dt < 0:
                strength = 1.0 - (theta_normalized / self.theta_threshold)
                return PolarSignal(
                    signal_type=PolarSignalType.MEAN_REVERT_LONG,
                    strength=strength,
                    r=current.r,
                    theta=current.theta,
                    description=f"Mean-revert long: theta={current.theta:.3f}, dr/dt={current.dr_dt:.4f}"
                )

        elif abs(abs(current.theta) - math.pi) < self.theta_threshold:
            # Near top - check if r is decreasing (overbought)
            if current.dr_dt < 0:
                strength = 1.0 - (abs(abs(current.theta) - math.pi) / self.theta_threshold)
                return PolarSignal(
                    signal_type=PolarSignalType.MEAN_REVERT_SHORT,
                    strength=strength,
                    r=current.r,
                    theta=current.theta,
                    description=f"Mean-revert short: theta={current.theta:.3f}, dr/dt={current.dr_dt:.4f}"
                )

        return None

    def _check_trending(self, current: PolarFeatures,
                        recent: List[PolarFeatures]) -> Optional[PolarSignal]:
        """Detect trending market."""
        # Check if r is consistently increasing
        r_increasing = sum(1 for f in recent if f.dr_dt > self.r_trend_threshold)
        r_ratio = r_increasing / len(recent)

        if r_ratio >= 0.6:  # 60% of recent periods show expansion
            # Determine direction from theta
            if abs(current.theta) < math.pi / 4:
                # Theta in right half-plane → upward trend
                return PolarSignal(
                    signal_type=PolarSignalType.TREND_LONG,
                    strength=r_ratio,
                    r=current.r,
                    theta=current.theta,
                    description=f"Trend long: r expanding ({r_ratio:.1%}), theta={current.theta:.3f}"
                )
            else:
                # Theta in left half-plane → downward trend
                return PolarSignal(
                    signal_type=PolarSignalType.TREND_SHORT,
                    strength=r_ratio,
                    r=current.r,
                    theta=current.theta,
                    description=f"Trend short: r expanding ({r_ratio:.1%}), theta={current.theta:.3f}"
                )

        return None

    def _check_breakout(self, current: PolarFeatures,
                        recent: List[PolarFeatures]) -> Optional[PolarSignal]:
        """Detect breakout (sudden theta jump)."""
        if len(recent) < 2:
            return None

        theta_change = abs(current.dtheta_dt)

        if theta_change > self.breakout_theta_jump:
            # Large theta jump - determine direction
            if current.theta > 0 and current.dr_dt > 0:
                return PolarSignal(
                    signal_type=PolarSignalType.BREAKOUT_LONG,
                    strength=min(1.0, theta_change / math.pi),
                    r=current.r,
                    theta=current.theta,
                    description=f"Breakout long: theta jump={theta_change:.3f} rad"
                )
            elif current.theta < 0 and current.dr_dt > 0:
                return PolarSignal(
                    signal_type=PolarSignalType.BREAKOUT_SHORT,
                    strength=min(1.0, theta_change / math.pi),
                    r=current.r,
                    theta=current.theta,
                    description=f"Breakout short: theta jump={theta_change:.3f} rad"
                )

        return None


def describe_regime(features: List[PolarFeatures], lookback: int = 20) -> str:
    """
    Describe current market regime based on polar features.

    Returns human-readable regime description.
    """
    if len(features) < lookback:
        return "Insufficient data for regime analysis"

    recent = features[-lookback:]

    # Calculate statistics
    avg_dr_dt = sum(f.dr_dt for f in recent) / len(recent)
    std_dr_dt = math.sqrt(sum((f.dr_dt - avg_dr_dt)**2 for f in recent) / len(recent))

    avg_dtheta_dt = sum(f.dtheta_dt for f in recent) / len(recent)
    std_dtheta_dt = math.sqrt(sum((f.dtheta_dt - avg_dtheta_dt)**2 for f in recent) / len(recent))

    avg_r = sum(f.r for f in recent) / len(recent)
    std_r = math.sqrt(sum((f.r - avg_r)**2 for f in recent) / len(recent))

    # Regime classification
    # Mean-reversion: regular theta cycling (low std, non-zero avg angular velocity)
    if std_dtheta_dt < C.POLAR_REGIME_MR_STD_THETA and abs(avg_dtheta_dt) > C.POLAR_REGIME_MR_AVG_THETA:
        regime = "MEAN-REVERSION (cyclic)"
        description = f"Regular θ cycling (σ={std_dtheta_dt:.3f}, avg={avg_dtheta_dt:.3f}), expect oscillations"
    # Trending: r expanding steadily with low volatility
    elif avg_dr_dt > C.POLAR_REGIME_TREND_AVG_DR and std_r < avg_r * C.POLAR_REGIME_TREND_STD_R_RATIO:
        regime = "TRENDING (spiral out)"
        description = f"R expanding steadily (+{avg_dr_dt:.3f}/period), momentum persistent"
    # Volatile: high r velocity variance
    elif std_dr_dt > C.POLAR_REGIME_VOLATILE_STD_DR:
        regime = "VOLATILE (chaotic)"
        description = f"High r volatility (σ={std_dr_dt:.3f}), direction unclear"
    else:
        regime = "NEUTRAL"
        description = f"No clear pattern (dr/dt={avg_dr_dt:.3f}, dθ/dt={avg_dtheta_dt:.3f})"

    return f"{regime}: {description}"
