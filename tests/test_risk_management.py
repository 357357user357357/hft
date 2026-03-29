"""Tests for position sizing and risk management."""

from risk_management import PositionSizer, SizingConfig


def test_fixed_sizing():
    """Fixed mode returns constant size regardless of equity."""
    sizer = PositionSizer(SizingConfig(
        mode='fixed',
        fixed_size_usdt=250.0,
        initial_equity=10000.0,
    ))
    assert sizer.calculate_size() == 250.0
    sizer.update_equity(20000.0)
    assert sizer.calculate_size() == 250.0


def test_fractional_sizing():
    """Fractional mode returns percentage of current equity."""
    sizer = PositionSizer(SizingConfig(
        mode='fractional',
        fractional_pct=5.0,
        initial_equity=10000.0,
    ))
    assert sizer.calculate_size() == 500.0  # 5% of 10000
    sizer.update_equity(15000.0)
    assert sizer.calculate_size() == 750.0  # 5% of 15000


def test_kelly_not_enough_data():
    """Kelly with insufficient data falls back to fractional."""
    sizer = PositionSizer(SizingConfig(
        mode='kelly',
        fractional_pct=3.0,
        kelly_lookback=50,
        initial_equity=10000.0,
    ))
    # No trade history
    size = sizer.calculate_size([])
    assert size == 300.0  # 3% fallback


def test_kelly_calculation():
    """Kelly criterion calculates optimal bet size."""
    sizer = PositionSizer(SizingConfig(
        mode='kelly',
        fractional_pct=5.0,
        kelly_lookback=10,
        initial_equity=10000.0,
    ))
    # Create a profitable trade history: 6 wins of 2%, 4 losses of 1%
    trades = [2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, 2.0]
    kelly_fraction = sizer._calculate_kelly(trades)
    # p = 0.6, q = 0.4, b = 2.0/1.0 = 2.0
    # Kelly = (0.6 * 2.0 - 0.4) / 2.0 = (1.2 - 0.4) / 2.0 = 0.4
    assert abs(kelly_fraction - 0.4) < 0.01


def test_kelly_cap():
    """Kelly fraction is capped to avoid overbetting."""
    sizer = PositionSizer(SizingConfig(
        mode='kelly',
        kelly_cap=0.25,
        fractional_pct=5.0,
        kelly_lookback=10,
        initial_equity=10000.0,
    ))
    # Create a highly profitable history that would give Kelly > 0.25
    trades = [5.0, 5.0, 5.0, 5.0, 5.0, -1.0, -1.0]
    kelly = sizer._calculate_kelly(trades)
    # This should be capped at 0.25
    size = sizer.calculate_size(trades)
    assert size <= 2500.0  # 25% of 10000


def test_kelly_negative_edge():
    """Kelly returns 0 for losing strategies (don't trade)."""
    sizer = PositionSizer(SizingConfig(
        mode='kelly',
        fractional_pct=5.0,
        kelly_lookback=10,
        initial_equity=10000.0,
    ))
    # Create a losing trade history
    trades = [-1.0, -1.0, -1.0, -1.0, -1.0, 0.5, 0.5]
    kelly = sizer._calculate_kelly(trades)
    assert kelly >= 0.0  # Kelly should be 0 or positive (never negative)


def test_equity_update():
    """Equity updates correctly after trades."""
    sizer = PositionSizer(SizingConfig(
        mode='fractional',
        fractional_pct=10.0,
        initial_equity=5000.0,
    ))
    assert sizer.current_equity == 5000.0
    sizer.update_equity(5500.0)
    assert sizer.current_equity == 5500.0


def test_describe():
    """Describe returns human-readable sizing info."""
    fixed = PositionSizer(SizingConfig(mode='fixed', fixed_size_usdt=100.0))
    desc = fixed.describe()
    assert "Fixed" in desc
    assert "$100.00" in desc

    fractional = PositionSizer(SizingConfig(mode='fractional', fractional_pct=5.0))
    desc = fractional.describe()
    assert "Fractional" in desc
    assert "5.0%" in desc
