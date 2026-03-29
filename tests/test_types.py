"""Tests for hft_types — Position, BacktestStats, Side."""

import math

from hft_types import (
    Side, TakeProfitConfig, StopLossConfig,
    Position, TradeResult, ExitReason, BacktestStats,
)


# ── Side ──────────────────────────────────────────────────────────────────────

def test_side_opposite():
    assert Side.Buy.opposite() == Side.Sell
    assert Side.Sell.opposite() == Side.Buy


# ── Position ──────────────────────────────────────────────────────────────────

def test_position_pnl_buy():
    pos = Position.new(
        side=Side.Buy, entry_price=100.0, size_usdt=1000.0,
        entry_time_ns=0, tp_price=101.0, sl_price=99.0, tp_pct=1.0,
    )
    assert pos.pnl_pct(101.0) == 1.0     # +1%
    assert pos.pnl_pct(99.0) == -1.0     # -1%
    assert pos.pnl_pct(100.0) == 0.0


def test_position_pnl_sell():
    pos = Position.new(
        side=Side.Sell, entry_price=100.0, size_usdt=1000.0,
        entry_time_ns=0, tp_price=99.0, sl_price=101.0, tp_pct=1.0,
    )
    assert pos.pnl_pct(99.0) == 1.0      # price down = profit for short
    assert pos.pnl_pct(101.0) == -1.0


# ── BacktestStats ─────────────────────────────────────────────────────────────

def _trade(pnl_pct):
    return TradeResult(
        side=Side.Buy, entry_price=100.0, exit_price=100.0 + pnl_pct,
        size_usdt=100.0, entry_time_ns=0, exit_time_ns=1,
        exit_reason=ExitReason.TakeProfit if pnl_pct > 0 else ExitReason.StopLoss,
        pnl_pct=pnl_pct,
    )


def test_stats_empty():
    s = BacktestStats()
    assert s.total_trades == 0
    assert s.win_rate() == 0.0
    assert s.avg_pnl_per_trade() == 0.0


def test_stats_single_winner():
    s = BacktestStats()
    s.record(_trade(2.0))
    assert s.total_trades == 1
    assert s.winning_trades == 1
    assert s.losing_trades == 0
    assert s.win_rate() == 100.0
    assert s.total_pnl_pct == 2.0


def test_stats_mixed_trades():
    s = BacktestStats()
    s.record(_trade(3.0))
    s.record(_trade(-1.0))
    s.record(_trade(2.0))
    s.record(_trade(-0.5))
    assert s.total_trades == 4
    assert s.winning_trades == 2
    assert s.losing_trades == 2
    assert s.win_rate() == 50.0
    assert abs(s.total_pnl_pct - 3.5) < 1e-10
    assert abs(s.avg_pnl_per_trade() - 0.875) < 1e-10


def test_stats_profit_factor():
    s = BacktestStats()
    s.record(_trade(6.0))
    s.record(_trade(-2.0))
    s.record(_trade(-1.0))
    # profit_factor = gross_profit / abs(gross_loss) = 6 / 3 = 2.0
    assert abs(s.profit_factor() - 2.0) < 1e-10


def test_stats_profit_factor_no_losses():
    s = BacktestStats()
    s.record(_trade(1.0))
    assert s.profit_factor() == math.inf


def test_stats_max_drawdown():
    """Test max drawdown calculation using equity curve (compounding)."""
    s = BacktestStats()
    # Starting equity = 100
    s.record(_trade(5.0))   # equity = 100 * 1.05 = 105, peak = 105
    s.record(_trade(-3.0))  # equity = 105 * 0.97 = 101.85, dd = 3.15
    s.record(_trade(-1.0))  # equity = 101.85 * 0.99 = 100.83, dd = 4.17
    s.record(_trade(2.0))   # equity = 100.83 * 1.02 = 102.85, dd = 2.15
    # Max drawdown = peak - lowest point = 105 - 100.83 = 4.17
    assert abs(s.max_drawdown_pct - 4.17) < 0.1


def test_stats_fee_adjusted():
    s = BacktestStats()
    s.record(_trade(1.0))
    s.record(_trade(1.0))
    # 2 trades * 2 sides * 0.05% fee = 0.2%
    assert abs(s.fee_adjusted_pnl(0.05) - 1.8) < 1e-10


# ── Property: buy TP price > entry price ──────────────────────────────────────

def test_buy_tp_above_entry():
    """For a buy trade, TP target must be above entry."""
    tp_cfg = TakeProfitConfig(enabled=True, percentage=0.5)
    entry = 50000.0
    tp_price = entry * (1 + tp_cfg.percentage / 100.0)
    assert tp_price > entry


def test_sell_tp_below_entry():
    """For a sell trade, TP target must be below entry."""
    tp_cfg = TakeProfitConfig(enabled=True, percentage=0.5)
    entry = 50000.0
    tp_price = entry * (1 - tp_cfg.percentage / 100.0)
    assert tp_price < entry
