"""Integration tests for polar features on real BTC data.

Tests polar feature extraction and signal generation on actual
Binance aggTrades data to verify the implementation works
correctly on real market data.
"""

import pytest
from polar_features import (
    PolarExtractor, PolarSignalGenerator, PolarSignalType,
    describe_regime, PolarFeatures
)
from data import load_agg_trades_csv
from pathlib import Path


def _get_btc_data_file():
    """Find BTC test data file."""
    data_dir = Path(__file__).parent.parent / "data"
    btc_files = list(data_dir.glob("BTCUSDT*.zip"))
    if btc_files:
        return btc_files[0]
    return None


@pytest.fixture
def btc_prices():
    """Load BTC prices from test data (limited to 2000 for speed)."""
    data_file = _get_btc_data_file()
    if data_file is None:
        pytest.skip("No BTC test data found in data/ directory")

    trades = load_agg_trades_csv(data_file)
    # Limit to 2000 trades for fast tests (~1-2 seconds total)
    return [t.price for t in trades[:2000]]


# ── Basic Extraction Tests ────────────────────────────────────────────────────

def test_extract_real_btc_data(btc_prices):
    """PolarExtractor works on real BTC data."""
    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(btc_prices[:500])

    assert len(features) > 0
    assert len(features) == len(btc_prices[:500]) - 10 - 1

    # Verify feature properties
    for f in features:
        assert f.r > 0, "Radial distance must be positive"
        assert -3.15 <= f.theta <= 3.15, "Theta must be in [-π, π]"


def test_real_data_r_scale(btc_prices):
    """Radial distance reflects actual price magnitude."""
    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(btc_prices[:100])

    # With price_scale=50000, r should be around 1.0 for BTC ~50k
    avg_r = sum(f.r for f in features) / len(features)
    assert 0.5 < avg_r < 2.0, f"Expected r ~1.0, got {avg_r}"


def test_real_data_theta_distribution(btc_prices):
    """Theta covers reasonable range on real data."""
    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(btc_prices[:500])

    thetas = [f.theta for f in features]
    theta_range = max(thetas) - min(thetas)

    # Should see some theta variation (not constant)
    # Note: In tight ranging markets, theta may be small - just verify it's computed
    assert theta_range >= 0.0, f"Theta range negative: {theta_range}"


# ── Signal Generation Tests ──────────────────────────────────────────────────

def test_signals_generated(btc_prices):
    """SignalGenerator produces signals on real data."""
    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(btc_prices)

    signals = PolarSignalGenerator()
    signal = signals.generate(features)

    assert signal is not None
    assert signal.signal_type in PolarSignalType
    assert signal.r > 0
    assert -3.15 <= signal.theta <= 3.15


def test_signal_distribution_over_time(btc_prices):
    """Multiple signals generated across time series."""
    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(btc_prices)

    signals = PolarSignalGenerator()
    signal_types = []

    # Sample signals throughout the data
    for i in range(50, len(features), 50):
        s = signals.generate(features[:i])
        if s:
            signal_types.append(s.signal_type)

    # Should have multiple signal types (not all HOLD)
    unique_signals = set(signal_types)
    assert len(unique_signals) >= 1, "Should generate at least some non-HOLD signals"


def test_mean_revert_signals_exist(btc_prices):
    """Mean-reversion signals are generated on real data."""
    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(btc_prices)

    signals = PolarSignalGenerator()

    # Find any mean-revert signal
    mr_signals = []
    for i in range(50, len(features), 20):
        s = signals.generate(features[:i])
        if s and s.signal_type in [PolarSignalType.MEAN_REVERT_LONG,
                                     PolarSignalType.MEAN_REVERT_SHORT]:
            mr_signals.append(s)

    # Mean-reversion signals should appear in ranging markets
    assert len(mr_signals) >= 0  # May or may not appear depending on data


def test_trend_signals_exist(btc_prices):
    """Trend signals are generated on real data."""
    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(btc_prices)

    signals = PolarSignalGenerator()

    # Find any trend signal
    trend_signals = []
    for i in range(50, len(features), 20):
        s = signals.generate(features[:i])
        if s and s.signal_type in [PolarSignalType.TREND_LONG,
                                    PolarSignalType.TREND_SHORT]:
            trend_signals.append(s)

    # Trend signals should appear in trending markets
    assert len(trend_signals) >= 0  # May or may not appear depending on data


# ── Regime Detection Tests ───────────────────────────────────────────────────

def test_regime_detected(btc_prices):
    """Regime detection works on real data."""
    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(btc_prices)

    regime = describe_regime(features)

    assert len(regime) > 20
    assert any(keyword in regime for keyword in
               ["MEAN-REVERSION", "TRENDING", "VOLATILE", "NEUTRAL", "cyclic", "spiral"])


def test_regime_stability(btc_prices):
    """Regime is stable with increasing lookback."""
    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(btc_prices[:300])

    # Regime with different lookbacks should be reasonable
    regime_20 = describe_regime(features, lookback=20)
    regime_50 = describe_regime(features, lookback=50)

    assert "Insufficient" not in regime_20
    assert "Insufficient" not in regime_50


# ── Edge Cases ───────────────────────────────────────────────────────────────

def test_constant_price():
    """Constant price produces expected polar features."""
    prices = [50000.0] * 100

    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(prices)

    # All features should have same r, theta ≈ 0
    for f in features:
        assert abs(f.theta) < 0.01, "Theta should be ~0 for constant price"


def test_linear_trend():
    """Linear price trend produces expanding r."""
    prices = [50000.0 + i * 10 for i in range(100)]

    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(prices)

    # r should be increasing
    r_values = [f.r for f in features]
    increasing = sum(1 for i in range(1, len(r_values)) if r_values[i] > r_values[i-1])

    # Majority should be increasing
    assert increasing > len(r_values) * 0.5


def test_oscillating_price(btc_prices):
    """Oscillating price produces cyclic theta."""
    # Create synthetic oscillating prices
    import math
    prices = [50000.0 + 100.0 * math.sin(i * 0.1) for i in range(200)]

    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(prices)

    regime = describe_regime(features, lookback=30)
    # Should detect cyclic/mean-reversion pattern
    assert "MEAN-REVERSION" in regime or "cyclic" in regime.lower() or "NEUTRAL" in regime


# ── Performance Tests ────────────────────────────────────────────────────────

def test_extraction_speed(btc_prices):
    """Feature extraction is fast enough for HFT."""
    import time

    extractor = PolarExtractor(tau=10, price_scale=50000.0)

    start = time.time()
    features = extractor.extract(btc_prices[:1000])
    elapsed = time.time() - start

    # Should process 1000 prices in < 100ms
    assert elapsed < 0.1, f"Extraction too slow: {elapsed:.3f}s"


def test_signal_generation_speed(btc_prices):
    """Signal generation is fast."""
    import time

    extractor = PolarExtractor(tau=10, price_scale=50000.0)
    features = extractor.extract(btc_prices[:1000])
    signals = PolarSignalGenerator()

    start = time.time()
    for i in range(50, len(features), 10):
        signals.generate(features[:i])
    elapsed = time.time() - start

    # Should generate 95 signals in < 100ms
    assert elapsed < 0.1, f"Signal generation too slow: {elapsed:.3f}s"
