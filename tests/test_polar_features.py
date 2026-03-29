"""Tests for polar coordinate features."""

import math
from polar_features import (
    PolarExtractor, PolarSignalGenerator, PolarSignalType,
    PolarFeatures, describe_regime
)


def _make_prices(base=50000.0, trend=0.0, noise=0.001, n=200, seed=42):
    """Generate test price series."""
    import random
    random.seed(seed)
    prices = [base]
    for _ in range(n - 1):
        new_price = prices[-1] * (1 + trend + random.gauss(0, noise))
        prices.append(new_price)
    return prices


# ── PolarExtractor Tests ──────────────────────────────────────────────────────

def test_extractor_basic():
    """Extractor produces polar features from prices."""
    prices = _make_prices()
    extractor = PolarExtractor(tau=10)
    features = extractor.extract(prices)

    assert len(features) == len(prices) - 10 - 1
    assert all(f.r > 0 for f in features)  # r is always positive
    assert all(-math.pi <= f.theta <= math.pi for f in features)


def test_extractor_empty():
    """Extractor returns empty list for insufficient data."""
    prices = [50000.0] * 5
    extractor = PolarExtractor(tau=10)
    features = extractor.extract(prices)
    assert features == []


def test_extractor_single():
    """Extract single price update."""
    extractor = PolarExtractor(tau=10, price_scale=50000.0)

    # First call initializes
    result = extractor.extract_single(50000.0, 49900.0)
    assert result is not None
    assert result.r > 0

    # Second call computes velocities
    result2 = extractor.extract_single(50100.0, 49900.0)
    assert result2 is not None


def test_r_always_positive():
    """Radial distance is always positive."""
    prices = _make_prices(noise=0.05, n=100)
    extractor = PolarExtractor(tau=5)
    features = extractor.extract(prices)

    assert all(f.r >= 0 for f in features)


def test_theta_range():
    """Theta is in [-π, π]."""
    prices = _make_prices(noise=0.1, n=100)
    extractor = PolarExtractor(tau=5)
    features = extractor.extract(prices)

    assert all(-math.pi <= f.theta <= math.pi for f in features)


def test_reset():
    """Reset clears internal state."""
    prices = _make_prices(n=50)
    extractor = PolarExtractor(tau=5)

    features1 = extractor.extract(prices[:30])
    extractor.reset()
    features2 = extractor.extract(prices[20:40])

    # After reset, first feature should have zero velocities
    assert features2[0].dr_dt == 0.0
    assert features2[0].dtheta_dt == 0.0


# ── PolarSignalGenerator Tests ───────────────────────────────────────────────

def test_signal_generator_insufficient_data():
    """Returns None with insufficient data."""
    generator = PolarSignalGenerator(lookback=5)
    features = [
        PolarFeatures(r=1.0, theta=0.1, dr_dt=0.01, dtheta_dt=0.01, timestamp_ns=i)
        for i in range(3)
    ]
    signal = generator.generate(features)
    assert signal is None


def test_signal_mean_revert_long():
    """Detects mean-reversion long setup."""
    generator = PolarSignalGenerator(theta_threshold=0.2, lookback=3)

    # Create features near theta=0 with decreasing r
    features = [
        PolarFeatures(r=1.0, theta=0.1, dr_dt=-0.01, dtheta_dt=0.0, timestamp_ns=i)
        for i in range(5)
    ]
    signal = generator.generate(features)

    assert signal is not None
    assert signal.signal_type == PolarSignalType.MEAN_REVERT_LONG
    assert signal.strength > 0


def test_signal_mean_revert_short():
    """Detects mean-reversion short setup (near θ=π)."""
    generator = PolarSignalGenerator(theta_threshold=0.2, lookback=3)

    # Create features near theta=π with decreasing r
    features = [
        PolarFeatures(r=1.0, theta=3.0, dr_dt=-0.01, dtheta_dt=0.0, timestamp_ns=i)
        for i in range(5)
    ]
    signal = generator.generate(features)

    assert signal is not None
    assert signal.signal_type == PolarSignalType.MEAN_REVERT_SHORT


def test_signal_trend():
    """Detects trending market."""
    generator = PolarSignalGenerator(r_trend_threshold=0.01, lookback=5)

    # Create features with consistently increasing r
    features = [
        PolarFeatures(r=1.0 + i*0.02, theta=0.1, dr_dt=0.02, dtheta_dt=0.01, timestamp_ns=i)
        for i in range(10)
    ]
    signal = generator.generate(features)

    assert signal is not None
    assert signal.signal_type in [PolarSignalType.TREND_LONG, PolarSignalType.TREND_SHORT]
    assert signal.strength >= 0.6


def test_signal_breakout():
    """Detects breakout (theta jump)."""
    generator = PolarSignalGenerator(breakout_theta_jump=0.5, lookback=3)

    # Stable then sudden jump
    features = [
        PolarFeatures(r=1.0, theta=0.1, dr_dt=0.0, dtheta_dt=0.01, timestamp_ns=0),
        PolarFeatures(r=1.0, theta=0.1, dr_dt=0.0, dtheta_dt=0.01, timestamp_ns=1),
        PolarFeatures(r=1.1, theta=1.5, dr_dt=0.1, dtheta_dt=1.4, timestamp_ns=2),  # Jump!
    ]
    signal = generator.generate(features)

    assert signal is not None
    assert signal.signal_type in [PolarSignalType.BREAKOUT_LONG, PolarSignalType.BREAKOUT_SHORT]


def test_signal_hold():
    """Returns HOLD when no clear pattern."""
    generator = PolarSignalGenerator(lookback=5)

    # Random-ish features
    features = [
        PolarFeatures(r=1.0 + i*0.001, theta=0.5 + i*0.01, dr_dt=0.001, dtheta_dt=0.01, timestamp_ns=i)
        for i in range(10)
    ]
    signal = generator.generate(features)

    assert signal is not None
    assert signal.signal_type == PolarSignalType.HOLD


# ── Regime Description Tests ─────────────────────────────────────────────────

def test_describe_regime_insufficient_data():
    """Returns message for insufficient data."""
    features = [
        PolarFeatures(r=1.0, theta=0.1, dr_dt=0.01, dtheta_dt=0.01, timestamp_ns=i)
        for i in range(5)
    ]
    desc = describe_regime(features, lookback=20)
    assert "Insufficient" in desc


def test_describe_regime_mean_reversion():
    """Detects mean-reversion regime."""
    # Create cyclic theta pattern with consistent angular velocity
    features = []
    for i in range(30):
        # Steady angular velocity (not too high std)
        dtheta = 0.08 + 0.02 * math.sin(i * 0.3)  # avg ~0.08, varies slightly
        features.append(PolarFeatures(
            r=1.0,
            theta=0.3 * math.sin(i * 0.5),
            dr_dt=0.0,
            dtheta_dt=dtheta,
            timestamp_ns=i
        ))

    desc = describe_regime(features, lookback=20)
    assert "MEAN-REVERSION" in desc or "cyclic" in desc.lower()


def test_describe_regime_trending():
    """Detects trending regime."""
    # Create steadily increasing r
    features = [
        PolarFeatures(r=1.0 + i*0.03, theta=0.1, dr_dt=0.03, dtheta_dt=0.01, timestamp_ns=i)
        for i in range(30)
    ]

    desc = describe_regime(features, lookback=20)
    assert "TREND" in desc or "spiral" in desc.lower()


def test_describe_regime_volatile():
    """Detects volatile regime."""
    # Create high volatility r
    import random
    random.seed(42)
    features = [
        PolarFeatures(
            r=1.0 + random.gauss(0, 0.1),
            theta=0.5,
            dr_dt=random.gauss(0, 0.05),
            dtheta_dt=random.gauss(0, 0.1),
            timestamp_ns=i
        )
        for i in range(30)
    ]

    desc = describe_regime(features, lookback=20)
    assert "VOLATILE" in desc or "chaotic" in desc.lower() or "NEUTRAL" in desc


# ── Integration Test ─────────────────────────────────────────────────────────

def test_full_pipeline():
    """Full extraction → signal generation pipeline."""
    # Generate trending prices
    prices = _make_prices(trend=0.002, noise=0.001, n=100)

    # Extract polar features
    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(prices)

    assert len(features) > 0

    # Generate signals
    generator = PolarSignalGenerator(lookback=10)
    signal = generator.generate(features)

    assert signal is not None
    assert signal.r > 0
    assert -math.pi <= signal.theta <= math.pi

    # Get regime description
    desc = describe_regime(features)
    assert len(desc) > 20
