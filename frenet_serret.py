"""Frenet-Serret frames for price curve analysis.

The Frenet-Serret framework describes the geometry of curves in 3D space
using an orthonormal moving frame:

    T (tangent)     = direction of motion
    N (normal)      = direction of curvature (turning)
    B (binormal)    = T × N (twist direction)

With two scalar invariants:
    κ (curvature)   = how fast the curve bends (|dT/ds|)
    τ (torsion)     = how fast the curve twists out of plane (|dB/ds|)

## Trading Interpretation

For a price curve parameterized by time:
- **T (tangent)**: Price direction/momentum vector
- **N (normal)**: Mean-reversion force direction
- **B (binormal)**: Regime change indicator (planar → 3D motion)
- **κ (curvature)**: How "curved" the price path is
  - High κ = strong mean-reversion (circular motion)
  - Low κ = trending (straight line)
- **τ (torsion)**: How much the curve twists
  - High τ = regime change, chaotic motion
  - Low τ = stable 2D pattern

## Parameterization

We embed price in 3D as:
    r(t) = (P(t), P(t-τ), V(t))

where:
- P(t) = price at time t
- P(t-τ) = delayed price (momentum proxy)
- V(t) = volume or volatility

This creates a space curve suitable for Frenet-Serret analysis.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Vector3D:
    """3D vector for Frenet-Serret computations."""
    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3D":
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> "Vector3D":
        if abs(scalar) < 1e-10:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector3D":
        return Vector3D(-self.x, -self.y, -self.z)

    def dot(self, other: "Vector3D") -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3D") -> "Vector3D":
        """Cross product."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        """Euclidean norm."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Vector3D":
        """Unit vector."""
        n = self.norm()
        if n < 1e-10:
            return Vector3D(0, 0, 0)
        return self / n

    def __repr__(self) -> str:
        return f"Vector3D({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


@dataclass
class FrenetFrame:
    """Frenet-Serret frame at a point on the curve."""
    T: Vector3D  # Tangent (direction of motion)
    N: Vector3D  # Normal (direction of curvature)
    B: Vector3D  # Binormal (twist direction)
    curvature: float  # κ (how fast curve bends)
    torsion: float    # τ (how fast curve twists)
    t: float          # Parameter value (time)


class FrenetSerretAnalyzer:
    """
    Frenet-Serret frame analysis for price curves.

    Embeds price data in 3D and computes the moving frame
    to extract geometric trading signals.
    """

    def __init__(self, delay: int = 5):
        """
        Initialize analyzer.

        Args:
            delay: Time delay for embedding (τ in P(t-τ))
        """
        self.delay = delay
        self.curve: List[Vector3D] = []
        self.frames: List[FrenetFrame] = []
        self.times: List[float] = []

    def add_point(self, price: float, volume: float = 1.0,
                  timestamp: Optional[float] = None,
                  price_history: Optional[List[float]] = None) -> Optional[Vector3D]:
        """
        Add a point to the curve.

        Args:
            price: Current price
            volume: Volume or volatility
            timestamp: Time (defaults to index)
            price_history: List of historical prices for delay embedding

        Returns:
            3D point added, or None if insufficient history
        """
        if price_history is None or len(price_history) < self.delay:
            return None

        delayed_price = price_history[-self.delay]
        point = Vector3D(price, delayed_price, volume)
        self.curve.append(point)
        self.times.append(timestamp if timestamp is not None else len(self.curve))

        return point

    def compute_frame(self, index: int) -> Optional[FrenetFrame]:
        """
        Compute Frenet frame at curve index.

        Uses finite differences for derivatives:
        - r'(t) ≈ (r(t+h) - r(t-h)) / 2h
        - r''(t) ≈ (r(t+h) - 2r(t) + r(t-h)) / h²

        Returns:
            FrenetFrame or None if insufficient neighbors
        """
        if index < 1 or index >= len(self.curve) - 1:
            return None

        h = 1.0  # Step size

        r = self.curve[index]
        r_prev = self.curve[index - 1]
        r_next = self.curve[index + 1]

        # First derivative (velocity)
        r_prime = (r_next - r_prev) / (2 * h)

        # Second derivative (acceleration)
        r_double = (r_next - r * 2 + r_prev) / (h * h)

        # Tangent: T = r' / |r'|
        T = r_prime.normalize()
        if T.norm() < 1e-10:
            return None

        # Curvature: κ = |r' × r''| / |r'|³
        cross = r_prime.cross(r_double)
        curvature = cross.norm() / (r_prime.norm() ** 3) if r_prime.norm() > 1e-10 else 0

        # Normal: N = (r'' - (r''·T)T) / |r'' - (r''·T)T|
        # This is the component of r'' perpendicular to T
        r_double_perp = r_double - T * r_double.dot(T)
        N = r_double_perp.normalize()

        # Binormal: B = T × N
        B = T.cross(N)

        # Torsion: τ = (r' × r'') · r''' / |r' × r''|²
        # Need third derivative
        if index < 2 or index >= len(self.curve) - 2:
            torsion = 0.0
        else:
            r_triple = (self.curve[index + 2] - self.curve[index - 2]) / (4 * h)
            triple_prod = cross.dot(r_triple)
            cross_norm_sq = cross.norm() ** 2
            torsion = triple_prod / cross_norm_sq if cross_norm_sq > 1e-10 else 0.0

        frame = FrenetFrame(
            T=T, N=N, B=B,
            curvature=curvature,
            torsion=torsion,
            t=self.times[index]
        )

        if len(self.frames) <= index:
            self.frames.append(frame)
        else:
            self.frames[index] = frame

        return frame

    def compute_all_frames(self) -> List[FrenetFrame]:
        """Compute frames for all points in curve."""
        frames = []
        for i in range(len(self.curve)):
            frame = self.compute_frame(i)
            if frame:
                frames.append(frame)
        return frames

    # ── Trading Signals ──────────────────────────────────────────────────────

    def get_curvature_signal(self, window: int = 10) -> Optional[str]:
        """
        Generate signal from curvature analysis.

        High curvature → mean-reversion likely
        Low curvature → trending

        Returns:
            "mean_revert", "trending", or None
        """
        if len(self.frames) < window:
            return None

        recent_curvatures = [f.curvature for f in self.frames[-window:]]
        avg_k = sum(recent_curvatures) / len(recent_curvatures)

        # Thresholds (tunable)
        if avg_k > 0.5:
            return "mean_revert"
        elif avg_k < 0.1:
            return "trending"
        return None

    def get_torsion_signal(self, threshold: float = 0.3) -> Optional[str]:
        """
        Generate signal from torsion analysis.

        High torsion → regime change, 3D motion
        Low torsion → stable 2D pattern

        Returns:
            "regime_change", "stable", or None
        """
        if not self.frames:
            return None

        recent_torsion = self.frames[-1].torsion

        if abs(recent_torsion) > threshold:
            return "regime_change"
        else:
            return "stable"

    def get_mean_reversion_strength(self) -> Optional[float]:
        """
        Compute mean-reversion strength from normal vector.

        The normal N points toward the "center" of the curve.
        Strong N.z component → strong mean-reversion force.

        Returns:
            Strength in [-1, 1], or None
        """
        if not self.frames:
            return None

        N = self.frames[-1].N
        # Project N onto price axis (x-component)
        # Negative means pointing back toward mean
        return -N.x  # Positive = mean-reversion signal

    def detect_inflection_points(self, window: int = 5) -> List[int]:
        """
        Detect inflection points (curvature sign changes).

        Inflection points indicate trend reversals.

        Returns:
            List of indices where inflection occurred
        """
        inflections = []

        for i in range(window, len(self.frames)):
            prev_k = self.frames[i - window].curvature
            curr_k = self.frames[i].curvature

            # Sign change in curvature derivative
            if (curr_k - prev_k) * (self.frames[i-1].curvature - self.frames[i-window].curvature) < 0:
                inflections.append(i)

        return inflections

    def get_frame_alignment(self) -> Optional[float]:
        """
        Measure how aligned recent frames are.

        High alignment = consistent direction (trending)
        Low alignment = chaotic (mean-reversion or regime change)

        Returns:
            Alignment score [0, 1], or None
        """
        if len(self.frames) < 3:
            return None

        # Average tangent of recent frames
        recent_T = [f.T for f in self.frames[-5:]]
        avg_T = Vector3D(
            sum(t.x for t in recent_T) / len(recent_T),
            sum(t.y for t in recent_T) / len(recent_T),
            sum(t.z for t in recent_T) / len(recent_T),
        ).normalize()

        # Dot product with each recent tangent
        alignments = [t.dot(avg_T) for t in recent_T]
        return sum(alignments) / len(alignments)

    # ── Summary Statistics ───────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Get summary statistics of the curve."""
        if not self.frames:
            return {"error": "No frames computed"}

        curvatures = [f.curvature for f in self.frames]
        torsions = [f.torsion for f in self.frames]

        return {
            "num_points": len(self.curve),
            "num_frames": len(self.frames),
            "avg_curvature": sum(curvatures) / len(curvatures),
            "max_curvature": max(curvatures),
            "min_curvature": min(curvatures),
            "avg_torsion": sum(torsions) / len(torsions),
            "max_torsion": max(torsions),
            "current_regime": self.get_torsion_signal(),
            "curvature_signal": self.get_curvature_signal(),
            "mean_reversion_strength": self.get_mean_reversion_strength(),
            "frame_alignment": self.get_frame_alignment(),
        }


# ── Utility Functions ────────────────────────────────────────────────────────

def analyze_price_series(prices: List[float], volumes: Optional[List[float]] = None,
                         delay: int = 5) -> FrenetSerretAnalyzer:
    """
    Convenience function to analyze a price series.

    Args:
        prices: List of prices
        volumes: Optional list of volumes (defaults to 1.0)
        delay: Embedding delay

    Returns:
        FrenetSerretAnalyzer with all points added and frames computed
    """
    analyzer = FrenetSerretAnalyzer(delay=delay)

    if volumes is None:
        volumes = [1.0] * len(prices)

    for i, (price, volume) in enumerate(zip(prices, volumes)):
        history = prices[:i+1]
        analyzer.add_point(price, volume, float(i), history)

    analyzer.compute_all_frames()
    return analyzer


def curvature_to_signal(curvature: float) -> str:
    """Convert curvature value to trading signal."""
    if curvature > 0.5:
        return "STRONG_MEAN_REVERT"
    elif curvature > 0.2:
        return "WEAK_MEAN_REVERT"
    elif curvature > 0.1:
        return "NEUTRAL"
    elif curvature > 0.05:
        return "WEAK_TREND"
    else:
        return "STRONG_TREND"
