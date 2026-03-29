"""Tests for trading algorithms — Shot, DepthShot, Averages, Vector.

Covers:
  - Reproducibility: same config + same data → identical PnL
  - Property tests: buy TP > entry, sell TP < entry, etc.
  - Basic smoke tests: algorithms run without errors on synthetic data
"""

from data import AggTrade, build_synthetic_book
from hft_types import Side, TakeProfitConfig, StopLossConfig, BacktestStats
from algorithms.shot import ShotBacktest, ShotConfig
from algorithms.depth_shot import DepthShotBacktest, DepthShotConfig, Depth
from algorithms.averages import AveragesBacktest, AveragesConfig, AveragesCondition
from algorithms.vector import VectorBacktest, VectorConfig, BorderRange, ShotDirection


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trades(prices, start_time=1700000000000, interval_ms=100):
    """Create a list of AggTrades from price list."""
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


def _spike_then_pullback(base=50000.0, n=200, spike_pct=2.0, tp_pct=0.5):
    """Generate prices: stable → spike up → partial pullback → stable."""
    prices = [base] * 50
    # Gradual spike up
    for i in range(30):
        prices.append(base * (1 + spike_pct / 100.0 * i / 30))
    spike_top = prices[-1]
    # Pullback
    for i in range(30):
        prices.append(spike_top * (1 - tp_pct / 200.0 * i / 30))
    # Stable at new level
    while len(prices) < n:
        prices.append(prices[-1])
    return prices


def _run_shot(prices, side=Side.Buy, **config_overrides):
    """Run Shot backtest on prices, return (backtest, trades)."""
    cfg = ShotConfig(side=side, **config_overrides)
    bt = ShotBacktest(cfg)
    for trade in _make_trades(prices):
        bt.on_trade(trade)
    return bt


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_shot_reproducibility():
    """Same config + same data → identical PnL every time."""
    prices = _spike_then_pullback()
    results = []
    for _ in range(3):
        bt = _run_shot(prices)
        results.append(bt.stats.total_pnl_pct)
    assert results[0] == results[1] == results[2]


def test_shot_reproducibility_sell():
    """Reproducibility also holds for sell side."""
    prices = _spike_then_pullback()
    # Invert for sell: drop then recover
    inv_prices = [100000.0 - p + 50000.0 for p in prices]
    results = []
    for _ in range(3):
        bt = _run_shot(inv_prices, side=Side.Sell)
        results.append(bt.stats.total_pnl_pct)
    assert results[0] == results[1] == results[2]


# ── Property: TP/SL price correctness ────────────────────────────────────────

def test_shot_buy_tp_above_entry():
    """For buy trades, every TP exit price should be above entry."""
    prices = _spike_then_pullback()
    bt = _run_shot(prices, side=Side.Buy, distance_pct=0.5)
    for r in bt.results:
        if r.exit_reason.value == "TakeProfit":
            assert r.exit_price >= r.entry_price, (
                f"Buy TP exit {r.exit_price} < entry {r.entry_price}"
            )


def test_shot_sell_tp_below_entry():
    """For sell trades, every TP exit price should be below entry."""
    # Price drops then bounces
    prices = [50000.0] * 20
    for i in range(40):
        prices.append(50000.0 - 30 * i)
    for i in range(40):
        prices.append(prices[-1] + 10 * i)
    prices.extend([prices[-1]] * 100)

    bt = _run_shot(prices, side=Side.Sell, distance_pct=0.5)
    for r in bt.results:
        if r.exit_reason.value == "TakeProfit":
            assert r.exit_price <= r.entry_price, (
                f"Sell TP exit {r.exit_price} > entry {r.entry_price}"
            )


# ── Smoke tests ───────────────────────────────────────────────────────────────

def test_shot_smoke():
    """Shot backtest runs without errors on flat prices (may have 0 trades)."""
    prices = [50000.0 + i * 0.01 for i in range(300)]
    bt = _run_shot(prices)
    assert bt.stats.total_trades >= 0


def test_averages_smoke():
    """Averages backtest runs without errors."""
    prices = [50000.0 + (i % 50) * 10.0 for i in range(500)]
    trades = _make_trades(prices)

    cfg = AveragesConfig(
        side=Side.Buy,
        conditions=[
            AveragesCondition(long_period_secs=2.0, short_period_secs=0.5,
                              trigger_min_pct=0.01, trigger_max_pct=0.1),
        ],
    )
    bt = AveragesBacktest(cfg)
    for t in trades:
        bt.on_trade(t)
    assert bt.stats.total_trades >= 0


def test_averages_reproducibility():
    """Averages: same config → same result."""
    prices = [50000.0 + (i % 50) * 10.0 for i in range(500)]
    trades = _make_trades(prices)

    results = []
    for _ in range(2):
        cfg = AveragesConfig(
            side=Side.Buy,
            conditions=[
                AveragesCondition(long_period_secs=2.0, short_period_secs=0.5,
                                  trigger_min_pct=0.01, trigger_max_pct=0.1),
            ],
        )
        bt = AveragesBacktest(cfg)
        for t in trades:
            bt.on_trade(t)
        results.append(bt.stats.total_pnl_pct)
    assert results[0] == results[1]


def test_depth_shot_smoke():
    """DepthShot backtest runs without errors with synthetic book."""
    prices = [50000.0 + i * 5.0 for i in range(200)]
    trades = _make_trades(prices)

    cfg = DepthShotConfig(side=Side.Buy)
    bt = DepthShotBacktest(cfg)

    recent = []
    for t in trades:
        recent.append(t)
        if len(recent) > 50:
            recent = recent[-50:]
        book = build_synthetic_book(recent, levels=10, tick_size=0.1)
        bt.on_trade(t, book)
    assert bt.stats.total_trades >= 0


def test_vector_smoke():
    """Vector backtest runs without errors."""
    prices = [50000.0 + (i % 100) * 2.0 for i in range(300)]
    trades = _make_trades(prices)

    cfg = VectorConfig(
        side=Side.Buy,
        upper_border_range=BorderRange(enabled=True, min_pct=0.05, max_pct=0.15),
        lower_border_range=BorderRange(enabled=True, min_pct=0.05, max_pct=0.15),
        shot_direction=ShotDirection.Both,
    )
    bt = VectorBacktest(cfg)
    for t in trades:
        bt.on_trade(t)
    assert bt.stats.total_trades >= 0


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_shot_single_trade():
    """Algorithm handles a single trade without crashing."""
    bt = _run_shot([50000.0])
    assert bt.stats.total_trades == 0


def test_shot_constant_price():
    """No trades should trigger if price never moves."""
    bt = _run_shot([50000.0] * 500)
    # With constant price, the order sits at distance% below
    # but price never reaches it → 0 trades
    assert bt.stats.total_trades == 0


# ── Vector + Polar Integration ────────────────────────────────────────────────

def test_vector_with_polar_signals():
    """Vector algorithm works with polar signals enabled."""
    from algorithms.vector import VectorConfig, VectorBacktest

    prices = [50000.0 + (i % 100) * 2.0 for i in range(300)]
    trades = _make_trades(prices)

    # Config with polar enabled
    cfg = VectorConfig(
        side=Side.Buy,
        use_polar_signals=True,
        polar_tau=10,
        polar_price_scale=50000.0,
    )
    bt = VectorBacktest(cfg)
    for t in trades:
        bt.on_trade(t)

    # Should run without errors
    assert bt.stats.total_trades >= 0
    # Polar extractor should be initialized
    assert bt.state.polar_extractor is not None
    assert bt.state.polar_signal_gen is not None


def test_vector_polar_disabled_by_default():
    """Polar signals are disabled by default."""
    from algorithms.vector import VectorConfig, VectorBacktest

    cfg = VectorConfig()
    assert cfg.use_polar_signals == False

    bt = VectorBacktest(cfg)
    assert bt.state.polar_extractor is None
