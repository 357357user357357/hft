"""Tests for regime-aware parameter switching."""

from regime_detector import RegimeDetector, RegimeConfig, MarketRegime
from data import AggTrade


def _make_trades(prices, start_time=1700000000000, interval_ms=100):
    """Create AggTrades from price list."""
    return [
        AggTrade(
            agg_trade_id=i,
            price=p,
            quantity=0.1,
            first_trade_id=i,
            last_trade_id=i,
            transact_time=start_time + i * interval_ms,
            is_buyer_maker=(i % 2 == 0),
        )
        for i, p in enumerate(prices)
    ]


def test_regime_detector_initialization():
    """RegimeDetector initializes with default config."""
    detector = RegimeDetector()
    assert detector._last_regime is None
    assert detector._last_report is None


def test_regime_not_enough_data():
    """Returns NEUTRAL when insufficient trades."""
    detector = RegimeDetector()
    trades = _make_trades([50000.0] * 50)  # Only 50 trades
    regime = detector.detect_regime(trades)
    assert regime == MarketRegime.NEUTRAL


def test_regime_constant_price():
    """Constant price should detect as mean-reversion (no movement)."""
    detector = RegimeDetector()
    prices = [50000.0] * 200
    trades = _make_trades(prices)
    regime = detector.detect_regime(trades)
    # Constant price = no genus = mean-reversion
    assert regime in [MarketRegime.MEAN_REVERSION, MarketRegime.NEUTRAL]


def test_regime_random_walk():
    """Random walk should produce some regime (not crash)."""
    import random
    random.seed(42)
    prices = [50000.0]
    for _ in range(300):
        prices.append(prices[-1] * (1 + random.gauss(0, 0.001)))

    detector = RegimeDetector()
    trades = _make_trades(prices)
    regime = detector.detect_regime(trades)
    assert regime in [MarketRegime.MEAN_REVERSION, MarketRegime.TRENDING, MarketRegime.NEUTRAL]


def test_get_shot_params_all_regimes():
    """Shot params available for all regimes."""
    detector = RegimeDetector()
    for regime in MarketRegime:
        params = detector.get_shot_params(regime)
        assert params.distance_pct > 0
        assert params.tp_pct > 0
        assert params.sl_pct > 0


def test_get_averages_params_all_regimes():
    """Averages params available for all regimes."""
    detector = RegimeDetector()
    for regime in MarketRegime:
        params = detector.get_averages_params(regime)
        assert params.long_period_secs > 0
        assert params.short_period_secs > 0


def test_get_vector_params_all_regimes():
    """Vector params available for all regimes."""
    detector = RegimeDetector()
    for regime in MarketRegime:
        params = detector.get_vector_params(regime)
        assert params.frame_size_secs > 0
        assert params.max_orders > 0


def test_get_depth_shot_params_all_regimes():
    """Depth Shot params available for all regimes."""
    detector = RegimeDetector()
    for regime in MarketRegime:
        params = detector.get_depth_shot_params(regime)
        assert params.target_volume > 0
        assert params.sl_pct > 0


def test_regime_param_differences():
    """Different regimes have meaningfully different params."""
    detector = RegimeDetector()

    mr_params = detector.get_shot_params(MarketRegime.MEAN_REVERSION)
    trend_params = detector.get_shot_params(MarketRegime.TRENDING)

    # Mean-reversion should have tighter params than trending
    assert mr_params.distance_pct < trend_params.distance_pct
    assert mr_params.tp_pct < trend_params.tp_pct
    assert mr_params.sl_pct < trend_params.sl_pct


def test_describe():
    """Describe returns human-readable regime info."""
    detector = RegimeDetector()
    for regime in MarketRegime:
        desc = detector.describe(regime)
        assert len(desc) > 20
        assert regime.value in desc or "MEAN" in desc or "TREND" in desc or "NEUTRAL" in desc


def test_custom_config():
    """Custom config thresholds are applied."""
    config = RegimeConfig(
        low_genus_threshold=3,
        high_genus_threshold=50,
        k_ratio_trending=100.0,
    )
    detector = RegimeDetector(config)
    assert detector.config.low_genus_threshold == 3
    assert detector.config.high_genus_threshold == 50
