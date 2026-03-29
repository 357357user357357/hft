"""Tests for quaternion-based trading analysis."""

import math
from quaternions import Quaternion, TradingStateQuaternion, QuaternionPortfolio


# ── Quaternion Basic Tests ───────────────────────────────────────────────────

def test_quaternion_creation():
    """Quaternion can be created."""
    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    assert q.w == 1.0
    assert q.x == 2.0
    assert q.y == 3.0
    assert q.z == 4.0


def test_quaternion_addition():
    """Quaternion addition works."""
    q1 = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(5, 6, 7, 8)
    result = q1 + q2
    assert result.w == 6
    assert result.x == 8
    assert result.y == 10
    assert result.z == 12


def test_quaternion_multiplication():
    """Hamilton product works correctly."""
    # i * j = k
    i = Quaternion(0, 1, 0, 0)
    j = Quaternion(0, 0, 1, 0)
    k = i * j
    assert abs(k.w) < 1e-10
    assert abs(k.x) < 1e-10
    assert abs(k.y) < 1e-10
    assert abs(k.z - 1) < 1e-10


def test_quaternion_norm():
    """Norm calculation is correct."""
    q = Quaternion(1, 2, 2, 0)
    norm = abs(q)
    assert abs(norm - 3.0) < 1e-10  # sqrt(1+4+4) = 3


def test_quaternion_conjugate():
    """Conjugate reverses vector part."""
    q = Quaternion(1, 2, 3, 4)
    conj = q.conjugate()
    assert conj.w == 1
    assert conj.x == -2
    assert conj.y == -3
    assert conj.z == -4


def test_quaternion_normalize():
    """Normalized quaternion has unit norm."""
    q = Quaternion(3, 0, 0, 0)
    unit = q.normalize()
    assert abs(abs(unit) - 1.0) < 1e-10
    assert abs(unit.w - 1.0) < 1e-10


def test_quaternion_angle():
    """Angle between quaternions is correct."""
    q1 = Quaternion(1, 0, 0, 0)  # Identity
    q2 = Quaternion(0, 1, 0, 0)  # 180 degree rotation
    angle = q1.angle_to(q2)
    assert abs(angle - math.pi/2) < 0.1  # Approximately 90 degrees


# ── Trading State Quaternion Tests ───────────────────────────────────────────

def test_trading_state_update():
    """Trading state quaternion updates correctly."""
    ts = TradingStateQuaternion()

    # First update initializes
    q1 = ts.update(100.0, 1.0)
    assert q1.w == 0  # No return on first observation

    # Second update computes state
    q2 = ts.update(101.0, 1.5)
    assert q2.w > 0  # Positive return
    assert q2.x > 0  # Positive momentum


def test_trading_state_regime():
    """Regime detection works."""
    ts = TradingStateQuaternion()

    # Quiet market (no change)
    ts.update(100.0, 1.0)
    ts.update(100.0, 1.0)
    regime = ts.detect_state_regime()
    assert regime == "quiet"

    # Trending (strong move with volume)
    ts.update(105.0, 10.0)
    regime = ts.detect_state_regime()
    assert regime in ["trending", "volatile"]


def test_trading_state_rotation():
    """State rotation accumulates."""
    ts = TradingStateQuaternion()

    for i in range(10):
        ts.update(100 + i, 1.0)

    rotation = ts.get_state_rotation()
    assert rotation is not None
    assert rotation > 0


def test_mean_reversion_signal():
    """Mean-reversion signal is generated."""
    ts = TradingStateQuaternion()

    # Create some history
    for price in [100, 101, 102, 101, 100]:
        ts.update(price, 1.0)

    signal = ts.get_mean_reversion_signal()
    assert signal is not None
    assert -1 <= signal <= 1


# ── Quaternion Portfolio Tests ───────────────────────────────────────────────

def test_portfolio_from_returns():
    """Portfolio quaternion from returns."""
    q = Quaternion.from_portfolio_returns([0.01, -0.02], [0.5, 0.5])
    assert q.w < 0  # Net negative return
    assert q.x > 0  # First asset positive


def test_portfolio_from_prices():
    """Portfolio quaternion from price changes."""
    q = Quaternion.from_price_changes([
        ("BTC", 50000, 50500),  # +1%
        ("ETH", 3000, 3100),    # +3.3%
    ])
    assert q.w > 0  # Portfolio up
    assert q.x > 0  # BTC up
    assert q.y > 0  # ETH up


def test_portfolio_rotation_trend():
    """Portfolio rotation trend is computed."""
    port = QuaternionPortfolio(["BTC", "ETH"])

    # Add snapshots
    port.add_snapshot({"BTC": 50000, "ETH": 3000})
    port.add_snapshot({"BTC": 50500, "ETH": 3100})
    port.add_snapshot({"BTC": 51000, "ETH": 3200})

    rotation = port.get_rotation_trend(window=2)
    assert rotation is not None
    assert rotation >= 0


def test_portfolio_regime_changes():
    """Regime changes are detected."""
    port = QuaternionPortfolio(["BTC", "ETH"])

    # Stable period
    for i in range(10):
        port.add_snapshot({"BTC": 50000 + i*10, "ETH": 3000 + i*5})

    # Sudden change
    port.add_snapshot({"BTC": 45000, "ETH": 2500})

    changes = port.detect_regime_change(threshold=0.3)
    assert len(changes) >= 1  # Should detect the crash
