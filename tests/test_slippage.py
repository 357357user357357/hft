"""Tests for slippage and latency simulation."""

from hft_types import Side
from slippage import SlippageModel, SlippageConfig


def test_disabled_returns_exact_price():
    model = SlippageModel(SlippageConfig(enabled=False, fixed_bps=10.0))
    assert model.apply(100.0, Side.Buy) == 100.0
    assert model.apply(100.0, Side.Sell) == 100.0


def test_zero_slippage():
    model = SlippageModel(SlippageConfig(fixed_bps=0.0, random_bps=0.0))
    assert model.apply(100.0, Side.Buy) == 100.0


def test_fixed_bps_buy_fills_higher():
    model = SlippageModel(SlippageConfig(fixed_bps=10.0))  # 10 bps = 0.1%
    fill = model.apply(100.0, Side.Buy)
    assert fill > 100.0
    assert abs(fill - 100.10) < 0.001


def test_fixed_bps_sell_fills_lower():
    model = SlippageModel(SlippageConfig(fixed_bps=10.0))
    fill = model.apply(100.0, Side.Sell)
    assert fill < 100.0
    assert abs(fill - 99.90) < 0.001


def test_volume_impact():
    model = SlippageModel(SlippageConfig(
        fixed_bps=0.0,
        volume_impact_bps_per_lot=5.0,
        lot_size_usdt=100.0,
    ))
    # 200 USDT = 2 lots → 10 bps impact
    fill = model.apply(100.0, Side.Buy, order_size_usdt=200.0)
    assert abs(fill - 100.10) < 0.001


def test_latency_buffer():
    model = SlippageModel(SlippageConfig(latency_ms=50))
    model.push_price(1000, 100.0)
    model.push_price(1025, 101.0)
    model.push_price(1050, 102.0)
    model.push_price(1075, 103.0)
    model.push_price(1100, 104.0)

    # At t=1100, looking back 50ms → price at t=1050 → 102.0
    delayed = model.delayed_price(1100)
    assert delayed == 102.0


def test_latency_buffer_not_enough_data():
    model = SlippageModel(SlippageConfig(latency_ms=100))
    model.push_price(50, 100.0)
    # Looking back 100ms from t=50 → need t=-50 → None
    assert model.delayed_price(50) is None


def test_describe():
    model = SlippageModel(SlippageConfig(fixed_bps=2.0, latency_ms=10))
    desc = model.describe()
    assert "2.0bps" in desc
    assert "10ms" in desc

    disabled = SlippageModel(SlippageConfig(enabled=False))
    assert "disabled" in disabled.describe()
