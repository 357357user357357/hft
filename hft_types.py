"""Common types shared across all algorithms"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import logging
import math

logger = logging.getLogger("hft")


class Side(Enum):
    Buy = "Buy"
    Sell = "Sell"

    def opposite(self) -> "Side":
        return Side.Sell if self == Side.Buy else Side.Buy


@dataclass
class AutoPriceDown:
    """Gradually lowers TP if price doesn't reach initial level"""
    timer_secs: float
    step_pct: float
    limit_pct: float


@dataclass
class TakeProfitConfig:
    enabled: bool = True
    percentage: float = 0.5
    auto_price_down: Optional[AutoPriceDown] = None


@dataclass
class TrailingStop:
    """SL follows price in profitable direction"""
    spread_pct: float


@dataclass
class SecondStopLoss:
    """Replaces first SL when price reaches profit level"""
    trigger_distance_pct: float
    percentage: float
    trailing: Optional[TrailingStop]
    spread_pct: float


@dataclass
class StopLossConfig:
    enabled: bool = True
    percentage: float = 1.0
    spread_pct: float = 0.1
    delay_secs: float = 0.0
    trailing: Optional[TrailingStop] = None
    second_sl: Optional[SecondStopLoss] = None


@dataclass
class Position:
    side: Side
    entry_price: float
    size_usdt: float
    entry_time_ns: int
    take_profit_price: float
    stop_loss_price: float
    best_price: float
    second_sl_active: bool
    tp_last_step_time_ns: int
    tp_current_pct: float

    @classmethod
    def new(cls, side: Side, entry_price: float, size_usdt: float,
            entry_time_ns: int, tp_price: float, sl_price: float, tp_pct: float) -> "Position":
        return cls(
            side=side,
            entry_price=entry_price,
            size_usdt=size_usdt,
            entry_time_ns=entry_time_ns,
            take_profit_price=tp_price,
            stop_loss_price=sl_price,
            best_price=entry_price,
            second_sl_active=False,
            tp_last_step_time_ns=entry_time_ns,
            tp_current_pct=tp_pct,
        )

    def pnl_pct(self, current_price: float) -> float:
        if self.side == Side.Buy:
            return (current_price - self.entry_price) / self.entry_price * 100.0
        else:
            return (self.entry_price - current_price) / self.entry_price * 100.0


class ExitReason(Enum):
    TakeProfit = "TakeProfit"
    StopLoss = "StopLoss"


@dataclass
class TradeResult:
    side: Side
    entry_price: float
    exit_price: float
    size_usdt: float
    entry_time_ns: int
    exit_time_ns: int
    exit_reason: ExitReason
    pnl_pct: float
    # Fee and slippage tracking
    entry_fee_pct: float = 0.0
    exit_fee_pct: float = 0.0
    slippage_pct: float = 0.0

    @property
    def net_pnl_pct(self) -> float:
        """PnL after deducting fees and slippage."""
        return self.pnl_pct - self.entry_fee_pct - self.exit_fee_pct - self.slippage_pct


class BacktestStats:
    """
    Backtest statistics with advanced metrics.

    Tracks:
      - Basic: total trades, win rate, PnL, drawdown
      - Advanced: Sharpe ratio, Sortino ratio, max DD duration, equity curve
      - Fee-aware: net PnL after fees and slippage
    """

    def __init__(self):
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_pnl_pct: float = 0.0
        self.max_drawdown_pct: float = 0.0
        self.peak_pnl_pct: float = 0.0
        self.gross_profit_pct: float = 0.0
        self.gross_loss_pct: float = 0.0
        # Advanced metrics
        self.equity_curve: List[float] = [100.0]  # Starting equity = 100
        self.trade_pnls: List[float] = []  # Individual trade PnLs for Sharpe/Sortino
        self.peak_timestamp: int = 0
        self.max_drawdown_duration_ns: int = 0
        self.current_drawdown_start: Optional[int] = None
        # Fee tracking
        self.total_fees_pct: float = 0.0
        self.total_slippage_pct: float = 0.0

    def record(self, result: TradeResult, current_timestamp_ns: Optional[int] = None) -> None:
        net_pnl = result.net_pnl_pct
        self.total_trades += 1
        self.trade_pnls.append(net_pnl)
        self.total_pnl_pct += net_pnl
        self.total_fees_pct += result.entry_fee_pct + result.exit_fee_pct
        self.total_slippage_pct += result.slippage_pct

        if net_pnl > 0.0:
            self.winning_trades += 1
            self.gross_profit_pct += net_pnl
        else:
            self.losing_trades += 1
            self.gross_loss_pct += net_pnl

        # Equity curve
        new_equity = self.equity_curve[-1] * (1 + net_pnl / 100.0)
        self.equity_curve.append(new_equity)

        # Drawdown tracking
        if new_equity > self.peak_pnl_pct:
            self.peak_pnl_pct = new_equity
            self.peak_timestamp = current_timestamp_ns or 0
            self.current_drawdown_start = None
        else:
            if self.current_drawdown_start is None:
                self.current_drawdown_start = current_timestamp_ns
            drawdown = self.peak_pnl_pct - new_equity
            if drawdown > self.max_drawdown_pct:
                self.max_drawdown_pct = drawdown
            if self.current_drawdown_start is not None and current_timestamp_ns is not None:
                dd_duration = current_timestamp_ns - self.current_drawdown_start
                if dd_duration > self.max_drawdown_duration_ns:
                    self.max_drawdown_duration_ns = dd_duration

    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100.0

    def avg_pnl_per_trade(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl_pct / self.total_trades

    def profit_factor(self) -> float:
        if self.gross_loss_pct == 0.0:
            return math.inf
        return self.gross_profit_pct / abs(self.gross_loss_pct)

    def sharpe_ratio(self, risk_free_rate: float = 0.0, annualization: float = 252 * 24 * 365 * 100_000_000) -> float:
        """
        Annualized Sharpe ratio.
        Assumes risk-free rate is 0 by default (typical for crypto HFT).
        annualization: convert ns-based std to annualized (default ~1 year in 100ns units)
        """
        if len(self.trade_pnls) < 2:
            return 0.0
        avg_pnl = sum(self.trade_pnls) / len(self.trade_pnls)
        variance = sum((p - avg_pnl) ** 2 for p in self.trade_pnls) / (len(self.trade_pnls) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0
        if std_dev == 0:
            return 0.0
        # Annualize: assuming trades are roughly evenly spaced
        return (avg_pnl / std_dev) * math.sqrt(len(self.trade_pnls))

    def sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Annualized Sortino ratio (uses downside deviation instead of total std).
        """
        if len(self.trade_pnls) < 2:
            return 0.0
        avg_pnl = sum(self.trade_pnls) / len(self.trade_pnls)
        downside_sq = sum((p - avg_pnl) ** 2 for p in self.trade_pnls if p < avg_pnl)
        downside_dev = math.sqrt(downside_sq / len(self.trade_pnls)) if downside_sq > 0 else 0
        if downside_dev == 0:
            return 0.0
        return (avg_pnl / downside_dev) * math.sqrt(len(self.trade_pnls))

    def max_drawdown_duration_secs(self) -> float:
        """Return maximum drawdown duration in seconds."""
        return self.max_drawdown_duration_ns / 1_000_000_000.0

    def calmar_ratio(self) -> float:
        """Calmar ratio = annualized return / max drawdown."""
        if self.max_drawdown_pct == 0:
            return 0.0
        # Simplified: just use total PnL / max DD
        return self.total_pnl_pct / self.max_drawdown_pct

    def fee_adjusted_pnl(self, fee_pct: float) -> float:
        """Legacy method for backwards compatibility."""
        return self.total_pnl_pct - self.total_trades * 2.0 * fee_pct

    def print_summary(self, algo_name: str) -> None:
        pf = self.profit_factor()
        pf_str = f"{pf:.3f}" if not math.isinf(pf) else "inf"
        sharpe = self.sharpe_ratio()
        sortino = self.sortino_ratio()
        calmar = self.calmar_ratio()
        max_dd_duration = self.max_drawdown_duration_secs()

        logger.info("=== %s Backtest Results ===", algo_name)
        logger.info("Total trades:       %d", self.total_trades)
        logger.info("Winning trades:     %d", self.winning_trades)
        logger.info("Losing trades:      %d", self.losing_trades)
        logger.info("Win rate:           %.2f%%", self.win_rate())
        logger.info("Total PnL (gross):  %+.4f%%", self.total_pnl_pct + self.total_fees_pct + self.total_slippage_pct)
        logger.info("Total PnL (net):    %+.4f%%", self.total_pnl_pct)
        logger.info("  - Fees paid:      %.4f%%", self.total_fees_pct)
        logger.info("  - Slippage paid:  %.4f%%", self.total_slippage_pct)
        logger.info("Avg PnL/trade:      %+.4f%%", self.avg_pnl_per_trade())
        logger.info("Profit factor:      %s", pf_str)
        logger.info("Max drawdown:       %.4f%%", self.max_drawdown_pct)
        logger.info("Max DD duration:    %.1fs", max_dd_duration)
        logger.info("Sharpe ratio:       %.3f", sharpe)
        logger.info("Sortino ratio:      %.3f", sortino)
        logger.info("Calmar ratio:       %.3f", calmar)

        if self.total_trades > 0:
            logger.info("")
            logger.info("Fee sensitivity (gross PnL):")
            gross_pnl = self.total_pnl_pct + self.total_fees_pct + self.total_slippage_pct
            logger.info("  Bybit VIP maker 0%%:         %+.4f%%", gross_pnl - self.total_trades * 2.0 * 0.0)
            logger.info("  Bybit standard maker 0.01%%: %+.4f%%", gross_pnl - self.total_trades * 2.0 * 0.01)
            logger.info("  Binance maker 0.02%%:        %+.4f%%", gross_pnl - self.total_trades * 2.0 * 0.02)
            logger.info("  Binance taker 0.05%%:        %+.4f%%", gross_pnl - self.total_trades * 2.0 * 0.05)
            logger.info("  Bybit/OKX taker 0.06%%:      %+.4f%%", gross_pnl - self.total_trades * 2.0 * 0.06)
            breakeven_fee = gross_pnl / (self.total_trades * 2.0)
            logger.info("  Break-even fee:              %.4f%%/side", breakeven_fee)
