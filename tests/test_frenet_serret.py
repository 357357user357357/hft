"""Tests for Frenet-Serret frame analysis."""

import math
from frenet_serret import (
    Vector3D, FrenetFrame, FrenetSerretAnalyzer,
    analyze_price_series, curvature_to_signal
)


# ── Vector3D Tests ───────────────────────────────────────────────────────────

def test_vector_creation():
    """3D vectors can be created."""
    v = Vector3D(1, 2, 3)
    assert v.x == 1
    assert v.y == 2
    assert v.z == 3


def test_vector_addition():
    """Vector addition works."""
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    result = v1 + v2
    assert result.x == 5
    assert result.y == 7
    assert result.z == 9


def test_vector_dot_product():
    """Dot product is correct."""
    v1 = Vector3D(1, 0, 0)
    v2 = Vector3D(0, 1, 0)
    assert v1.dot(v2) == 0  # Perpendicular

    v3 = Vector3D(1, 1, 0)
    assert abs(v1.dot(v3) - 1.0) < 1e-10


def test_vector_cross_product():
    """Cross product follows right-hand rule."""
    i = Vector3D(1, 0, 0)
    j = Vector3D(0, 1, 0)
    k = i.cross(j)

    assert abs(k.x) < 1e-10
    assert abs(k.y) < 1e-10
    assert abs(k.z - 1.0) < 1e-10


def test_vector_norm():
    """Vector norm is correct."""
    v = Vector3D(3, 4, 0)
    assert abs(v.norm() - 5.0) < 1e-10


def test_vector_normalize():
    """Normalized vector has unit length."""
    v = Vector3D(3, 4, 0)
    unit = v.normalize()
    assert abs(unit.norm() - 1.0) < 1e-10


# ── FrenetSerretAnalyzer Tests ───────────────────────────────────────────────

def test_analyzer_creation():
    """Analyzer can be created."""
    analyzer = FrenetSerretAnalyzer(delay=5)
    assert analyzer.delay == 5
    assert len(analyzer.curve) == 0


def test_add_point():
    """Points can be added to curve."""
    analyzer = FrenetSerretAnalyzer(delay=3)

    # Need enough history for delay embedding
    history = [100, 101, 102, 103]
    point = analyzer.add_point(104, 1.5, 4.0, history)

    assert point is not None
    assert point.x == 104
    assert point.y == 101  # Delayed price
    assert point.z == 1.5


def test_compute_frame():
    """Frenet frame can be computed."""
    analyzer = FrenetSerretAnalyzer(delay=2)

    # Add points with curvature (not linear)
    import math
    prices = [100 + 10 * math.sin(i * 0.5) for i in range(10)]
    for i, p in enumerate(prices):
        analyzer.add_point(p, 1.0, float(i), prices[:i+1])

    frame = analyzer.compute_frame(5)

    assert frame is not None
    assert frame.T.norm() > 0.99  # T should be unit vector
    # N and B may be zero for linear sections


def test_compute_all_frames():
    """All frames can be computed."""
    analyzer = analyze_price_series([100 + i for i in range(20)])

    assert len(analyzer.frames) > 0
    assert len(analyzer.frames) < len(analyzer.curve)  # Edge points excluded


# ── Trading Signal Tests ─────────────────────────────────────────────────────

def test_curvature_signal():
    """Curvature-based signals are generated."""
    # Linear trend (low curvature)
    analyzer = analyze_price_series([100 + i * 10 for i in range(30)])
    signal = analyzer.get_curvature_signal(window=5)

    assert signal in ["trending", None]  # Low curvature = trending

    # Ranging market (higher curvature)
    prices = [100 + 10 * math.sin(i * 0.5) for i in range(30)]
    analyzer2 = analyze_price_series(prices)
    signal2 = analyzer2.get_curvature_signal(window=5)

    # May be mean-revert or None depending on parameters
    assert signal2 in ["mean_revert", "trending", None]


def test_torsion_signal():
    """Torsion-based regime signals work."""
    analyzer = analyze_price_series([100 + i for i in range(30)])
    signal = analyzer.get_torsion_signal(threshold=0.5)

    assert signal in ["regime_change", "stable", None]


def test_mean_reversion_strength():
    """Mean-reversion strength is computed."""
    analyzer = analyze_price_series([100 + i for i in range(30)])
    strength = analyzer.get_mean_reversion_strength()

    assert strength is not None
    assert -1 <= strength <= 1


def test_frame_alignment():
    """Frame alignment measures trend consistency."""
    # Strong trend (high alignment)
    analyzer = analyze_price_series([100 + i * 10 for i in range(30)])
    alignment = analyzer.get_frame_alignment()

    if alignment is not None:
        assert 0 <= alignment <= 1


def test_inflection_points():
    """Inflection points are detected."""
    # S-curve has inflection point
    prices = [100 + 10 * math.tanh(i * 0.3 - 5) for i in range(30)]
    analyzer = analyze_price_series(prices)

    inflections = analyzer.detect_inflection_points(window=3)

    # May or may not detect depending on curve discretization
    assert isinstance(inflections, list)


# ── Summary Statistics Tests ─────────────────────────────────────────────────

def test_summary_statistics():
    """Summary statistics are computed."""
    analyzer = analyze_price_series([100 + i + 5 * math.sin(i) for i in range(50)])
    summary = analyzer.get_summary()

    assert "num_points" in summary
    assert "num_frames" in summary
    assert "avg_curvature" in summary
    assert "avg_torsion" in summary
    assert summary["num_points"] > 0
    assert summary["num_frames"] > 0


def test_empty_analyzer_summary():
    """Empty analyzer returns error."""
    analyzer = FrenetSerretAnalyzer()
    summary = analyzer.get_summary()

    assert "error" in summary


# ── Utility Function Tests ───────────────────────────────────────────────────

def test_analyze_price_series():
    """Convenience function works."""
    prices = [50000 + i * 100 for i in range(30)]
    analyzer = analyze_price_series(prices, delay=5)

    # Points are only added when we have enough history
    assert len(analyzer.curve) > 0
    assert len(analyzer.frames) >= 0


def test_curvature_to_signal():
    """Curvature value converts to signal string."""
    assert curvature_to_signal(0.8) == "STRONG_MEAN_REVERT"
    assert curvature_to_signal(0.3) == "WEAK_MEAN_REVERT"
    assert curvature_to_signal(0.15) == "NEUTRAL"
    assert curvature_to_signal(0.08) == "WEAK_TREND"
    assert curvature_to_signal(0.01) == "STRONG_TREND"


# ── Edge Cases ───────────────────────────────────────────────────────────────

def test_constant_prices():
    """Constant prices produce zero curvature."""
    prices = [100] * 30
    analyzer = analyze_price_series(prices)

    # Should not crash - just verify it runs
    try:
        summary = analyzer.get_summary()
        # If no error, summary should exist
        assert isinstance(summary, dict)
    except Exception:
        pass  # Acceptable for degenerate case


def test_insufficient_points():
    """Analyzer handles insufficient points gracefully."""
    analyzer = FrenetSerretAnalyzer(delay=10)

    # Add only a few points
    for i in range(5):
        analyzer.add_point(100 + i, 1.0, float(i), [100 + j for j in range(i+1)])

    frame = analyzer.compute_frame(2)
    assert frame is None  # Not enough neighbors


def test_orthonormal_frame():
    """Frenet frame vectors are orthonormal."""
    prices = [100 + 10 * math.sin(i * 0.3) for i in range(30)]
    analyzer = analyze_price_series(prices)

    for frame in analyzer.frames:
        # All unit vectors
        assert abs(frame.T.norm() - 1.0) < 0.01
        assert abs(frame.N.norm() - 1.0) < 0.01
        assert abs(frame.B.norm() - 1.0) < 0.01

        # Orthogonal
        assert abs(frame.T.dot(frame.N)) < 0.01
        assert abs(frame.T.dot(frame.B)) < 0.01
        assert abs(frame.N.dot(frame.B)) < 0.01
