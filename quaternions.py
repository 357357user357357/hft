"""Quaternion-based analysis for trading and portfolios.

Quaternions extend complex numbers to 4D: q = w + xi + yj + zk

## Single-Asset Trading State (4D):
- w = price return (direction and magnitude)
- x = momentum (velocity, dP/dt)
- y = acceleration (change in momentum, d²P/dt²)
- z = volume or volatility term

## Multi-Asset Portfolio (4D):
- w = portfolio return (scalar part)
- x = asset 1 momentum (e.g., BTC)
- y = asset 2 momentum (e.g., ETH)
- z = correlation/divergence term

## Key Operations for Trading:
- Quaternion multiplication: captures rotation between states
- Conjugate: reverses rotation (hedging, mean-reversion)
- Norm: overall state magnitude
- SLERP: smooth interpolation (state prediction)
- Angle between quaternions: regime change detection
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Quaternion:
    """Quaternion: q = w + xi + yj + zk"""
    w: float  # scalar (real) part
    x: float  # i component
    y: float  # j component
    z: float  # k component

    def __repr__(self) -> str:
        return f"Quaternion({self.w:.4f}, {self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    # ── Basic Operations ──────────────────────────────────────────────────────

    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self.w - other.w,
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Hamilton product (quaternion multiplication)."""
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def __neg__(self) -> "Quaternion":
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __abs__(self) -> float:
        """Norm (magnitude) of quaternion."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def conjugate(self) -> "Quaternion":
        """Conjugate: q* = w - xi - yj - zk (reverses rotation)."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> "Quaternion":
        """Multiplicative inverse: q^(-1) = q* / |q|²"""
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_sq == 0:
            raise ValueError("Cannot invert zero quaternion")
        conj = self.conjugate()
        return Quaternion(
            conj.w / norm_sq,
            conj.x / norm_sq,
            conj.y / norm_sq,
            conj.z / norm_sq
        )

    def normalize(self) -> "Quaternion":
        """Unit quaternion (pure rotation, norm = 1)."""
        norm = abs(self)
        if norm == 0:
            return Quaternion(1, 0, 0, 0)  # Identity
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def dot(self, other: "Quaternion") -> float:
        """Dot product (4D)."""
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    def angle_to(self, other: "Quaternion") -> float:
        """Angle between two quaternions (radians)."""
        dot = self.normalize().dot(other.normalize())
        # Clamp to [-1, 1] to avoid numerical issues with acos
        dot = max(-1.0, min(1.0, dot))
        return math.acos(dot)

    # ── Portfolio-Specific Methods ───────────────────────────────────────────

    @classmethod
    def from_portfolio_returns(cls, returns: List[float],
                               weights: Optional[List[float]] = None) -> "Quaternion":
        """
        Create quaternion from asset returns.

        For 2 assets (BTC, ETH):
        - w = weighted portfolio return
        - x = BTC return (normalized)
        - y = ETH return (normalized)
        - z = correlation term (BTC * ETH)

        For 3+ assets, use first 3 and aggregate rest into z.
        """
        if len(returns) < 2:
            raise ValueError("Need at least 2 asset returns")

        if weights is None:
            weights = [1.0 / len(returns)] * len(returns)

        # Scalar part: portfolio return
        w = sum(r * wt for r, wt in zip(returns, weights))

        # Vector parts: individual asset returns (normalized)
        x = returns[0] / (abs(returns[0]) + 1e-10) if len(returns) > 0 else 0
        y = returns[1] / (abs(returns[1]) + 1e-10) if len(returns) > 1 else 0

        # Z component: correlation/divergence
        if len(returns) >= 2:
            z = returns[0] * returns[1]  # Simple correlation proxy
        else:
            z = 0

        return cls(w, x, y, z)

    @classmethod
    def from_price_changes(cls, prices: List[Tuple[str, float, float]]) -> "Quaternion":
        """
        Create quaternion from price changes.

        Args:
            prices: List of (symbol, old_price, new_price) tuples

        Example:
            [("BTC", 50000, 50500), ("ETH", 3000, 3100)]
        """
        if len(prices) < 2:
            raise ValueError("Need at least 2 assets")

        returns = [(new - old) / old for _, old, new in prices]
        return cls.from_portfolio_returns(returns)

    def to_rotation_axis_angle(self) -> Tuple[Tuple[float, float, float], float]:
        """
        Extract rotation axis and angle from unit quaternion.

        Returns:
            (axis, angle) where axis is (x, y, z) unit vector
        """
        q = self.normalize()

        # Angle: 2 * acos(w)
        angle = 2 * math.acos(max(-1, min(1, q.w)))

        # Axis: (x, y, z) / sin(angle/2)
        sin_half_angle = math.sin(angle / 2)
        if sin_half_angle < 1e-10:
            axis = (1.0, 0.0, 0.0)  # Arbitrary axis for zero rotation
        else:
            axis = (q.x / sin_half_angle, q.y / sin_half_angle, q.z / sin_half_angle)

        return axis, angle

    def slerp(self, other: "Quaternion", t: float) -> "Quaternion":
        """
        Spherical linear interpolation (SLERP) between quaternions.

        Args:
            other: Target quaternion
            t: Interpolation factor (0 = self, 1 = other)

        Returns:
            Interpolated unit quaternion
        """
        q1 = self.normalize()
        q2 = other.normalize()

        dot = q1.dot(q2)

        # If dot < 0, use conjugate for shortest path
        if dot < 0:
            q2 = -q2
            dot = -dot

        dot = max(-1, min(1, dot))
        theta = math.acos(dot)

        if theta < 1e-10:
            # Quaternions are too close, use linear interpolation
            return Quaternion(
                q1.w + t * (q2.w - q1.w),
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z),
            ).normalize()

        sin_theta = math.sin(theta)
        a = math.sin((1 - t) * theta) / sin_theta
        b = math.sin(t * theta) / sin_theta

        return Quaternion(
            a * q1.w + b * q2.w,
            a * q1.x + b * q2.x,
            a * q1.y + b * q2.y,
            a * q1.z + b * q2.z,
        )


# ── Single-Asset Trading State ───────────────────────────────────────────────

class TradingStateQuaternion:
    """
    Quaternion representation of single-asset trading state.

    q = w + xi + yj + zk where:
    - w = price return (direction + magnitude)
    - x = momentum (dP/dt)
    - y = acceleration (d²P/dt²)
    - z = volume or volatility

    This captures the FULL state of a asset in 4D, not just scalar price.
    """

    def __init__(self):
        self.history: List[Quaternion] = []
        self._prev_price: Optional[float] = None
        self._prev_momentum: Optional[float] = None

    def update(self, price: float, volume: float = 1.0) -> Quaternion:
        """
        Update trading state from new price/volume.

        Returns current state quaternion.
        """
        if self._prev_price is None:
            # First observation - initialize
            self._prev_price = price
            q = Quaternion(0, 0, 0, 0)
        else:
            # Compute state components
            ret = (price - self._prev_price) / self._prev_price
            momentum = ret  # Simple momentum
            acceleration = momentum - self._prev_momentum if self._prev_momentum is not None else 0
            z_term = volume * ret  # Volume-weighted move

            q = Quaternion(ret, momentum, acceleration, z_term)

        self._prev_price = price
        self._prev_momentum = momentum if self._prev_momentum is not None else 0
        self.history.append(q)
        return q

    def get_state_rotation(self) -> Optional[float]:
        """
        Get cumulative rotation from start state.

        Large rotation = significant state change (regime shift)
        """
        if len(self.history) < 2:
            return None

        total_angle = 0
        for i in range(1, len(self.history)):
            total_angle += self.history[i-1].angle_to(self.history[i])

        return total_angle

    def predict_next_state(self, steps: int = 1) -> Quaternion:
        """
        Predict future state by extrapolating recent trend.

        Uses SLERP to extrapolate from last two states.
        """
        if len(self.history) < 2:
            return Quaternion(0, 0, 0, 0)

        q1 = self.history[-2]
        q2 = self.history[-1]

        # Extrapolate: go beyond q2 in same direction
        # t > 1 means "continue the rotation"
        return q1.slerp(q2, 1.0 + steps * 0.1)

    def detect_state_regime(self) -> str:
        """
        Classify current market regime from quaternion components.

        Returns: "trending", "mean_reverting", "volatile", or "quiet"
        """
        if not self.history:
            return "unknown"

        q = self.history[-1]

        # Regime classification based on quaternion components
        if abs(q.z) > 0.1:  # High volume/volatility
            if abs(q.w) > 0.05:
                return "trending"  # Strong directional move
            else:
                return "volatile"  # High volume, no direction
        elif abs(q.y) > 0.02:  # High acceleration
            return "mean_reverting"  # Momentum changing fast
        else:
            return "quiet"

    def get_mean_reversion_signal(self) -> Optional[float]:
        """
        Generate mean-reversion signal from quaternion conjugate.

        The conjugate q* = w - xi - yj - zk represents the "reverse rotation".
        If current state is far from conjugate, mean-reversion is likely.

        Returns:
            Signal strength (-1 to 1), or None if insufficient data
        """
        if len(self.history) < 2:
            return None

        current = self.history[-1]
        conjugate = current.conjugate()

        # Angle to conjugate measures "how reversible" current state is
        angle = current.angle_to(conjugate)

        # Normalize to [-1, 1] range
        # Small angle = state is self-similar (trending)
        # Large angle = state is far from reverse (mean-reversion likely)
        signal = (angle - math.pi/2) / (math.pi/2)

        return max(-1, min(1, signal))


# ── Multi-Asset Portfolio Analysis ───────────────────────────────────────────

class QuaternionPortfolio:
    """Quaternion-based multi-asset portfolio analysis."""

    def __init__(self, asset_names: List[str]):
        self.asset_names = asset_names
        self.history: List[Quaternion] = []
        self.price_history: List[dict] = []

    def add_snapshot(self, prices: dict) -> Quaternion:
        """
        Add price snapshot and compute quaternion.

        Args:
            prices: Dict of {symbol: price}

        Returns:
            Computed quaternion
        """
        if not self.price_history:
            # First snapshot, just store
            self.price_history.append(prices.copy())
            q = Quaternion(0, 0, 0, 0)
        else:
            # Compute returns from previous snapshot
            prev = self.price_history[-1]
            price_changes = [
                (name, prev.get(name, 1), prices.get(name, 1))
                for name in self.asset_names
            ]
            q = Quaternion.from_price_changes(price_changes)

        self.history.append(q)
        self.price_history.append(prices.copy())
        return q

    def get_rotation_trend(self, window: int = 10) -> Optional[float]:
        """
        Get average rotation angle over recent window.

        High rotation = high volatility/regime change
        Low rotation = stable market

        Returns:
            Average rotation angle (radians), or None if insufficient data
        """
        if len(self.history) < window + 1:
            return None

        angles = []
        for i in range(len(self.history) - window, len(self.history)):
            q1 = self.history[i]
            q2 = self.history[i + 1] if i + 1 < len(self.history) else q1
            angles.append(q1.angle_to(q2))

        return sum(angles) / len(angles)

    def get_correlation_structure(self) -> Optional[Quaternion]:
        """
        Get aggregate correlation structure from history.

        Returns:
            Average quaternion representing typical asset relationships
        """
        if not self.history:
            return None

        # Average all quaternions
        w_avg = sum(q.w for q in self.history) / len(self.history)
        x_avg = sum(q.x for q in self.history) / len(self.history)
        y_avg = sum(q.y for q in self.history) / len(self.history)
        z_avg = sum(q.z for q in self.history) / len(self.history)

        return Quaternion(w_avg, x_avg, y_avg, z_avg)

    def detect_regime_change(self, threshold: float = 0.5) -> List[int]:
        """
        Detect regime changes (large quaternion rotations).

        Args:
            threshold: Minimum angle (radians) to count as regime change

        Returns:
            List of indices where regime changes occurred
        """
        changes = []
        for i in range(1, len(self.history)):
            angle = self.history[i-1].angle_to(self.history[i])
            if angle > threshold:
                changes.append(i)
        return changes

    def get_portfolio_magnitude(self) -> List[float]:
        """Get norm (magnitude) of each quaternion in history."""
        return [abs(q) for q in self.history]


# ── Utility Functions ────────────────────────────────────────────────────────

def slerp_path(quaternions: List[Quaternion], steps: int = 100) -> List[Quaternion]:
    """
    Create smooth interpolation path through quaternion sequence.

    Useful for visualizing portfolio state evolution.
    """
    if len(quaternions) < 2:
        return quaternions

    path = []
    for i in range(len(quaternions) - 1):
        for t in range(steps):
            interpolated = quaternions[i].slerp(quaternions[i+1], t / steps)
            path.append(interpolated)

    path.append(quaternions[-1])
    return path
