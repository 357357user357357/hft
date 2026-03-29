"""Averages Algorithm

Profits from corrections — when price temporarily moves against the main
trend, then returns back. Compares average price across two timeframes
(long and short) and places orders when the difference falls within a set range.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from hft_types import (
    Side, TakeProfitConfig, StopLossConfig,
    Position, TradeResult, ExitReason, BacktestStats
)
from data import AggTrade
from slippage import SlippageModel


@dataclass
class AveragesCondition:
    long_period_secs: float
    short_period_secs: float
    trigger_min_pct: float
    trigger_max_pct: float

    def is_triggered(self, delta_pct: float) -> bool:
        return self.trigger_min_pct <= delta_pct <= self.trigger_max_pct


@dataclass
class AveragesGrid:
    count: int
    distance_pct: float
    size_increase_pct: float
    relative_to_first: bool


@dataclass
class AveragesConfig:
    side: Side = Side.Buy
    order_distance_pct: float = -1.0
    conditions: List[AveragesCondition] = field(default_factory=lambda: [
        AveragesCondition(
            long_period_secs=60.0, short_period_secs=10.0,
            trigger_min_pct=-0.5, trigger_max_pct=-0.1
        )
    ])
    order_size_usdt: float = 100.0
    cancel_delay_secs: Optional[float] = 60.0
    do_not_trigger_if_active: bool = False
    restart_delay_secs: float = 0.0
    take_profit: TakeProfitConfig = field(default_factory=lambda: TakeProfitConfig(
        enabled=True, percentage=0.3, auto_price_down=None
    ))
    stop_loss: StopLossConfig = field(default_factory=lambda: StopLossConfig(
        enabled=True, percentage=1.0, spread_pct=0.1, delay_secs=0.0,
        trailing=None, second_sl=None
    ))
    grid: Optional[AveragesGrid] = None


class RollingAverage:
    def __init__(self, period_secs: float):
        self.period_ns = int(period_secs * 1e9)
        self.data: deque = deque()  # (ts_ns, price)

    def push(self, ts_ns: int, price: float) -> None:
        self.data.append((ts_ns, price))
        cutoff = ts_ns - self.period_ns
        while self.data and self.data[0][0] < cutoff:
            self.data.popleft()

    def average(self) -> Optional[float]:
        if not self.data:
            return None
        return sum(p for _, p in self.data) / len(self.data)

    def is_ready(self) -> bool:
        return len(self.data) >= 2


@dataclass
class PendingOrder:
    price: float
    size_usdt: float
    placed_at_ns: int


class AveragesState:
    def __init__(self, config: AveragesConfig):
        self.averages = [
            (RollingAverage(c.long_period_secs), RollingAverage(c.short_period_secs))
            for c in config.conditions
        ]
        self.pending_orders: List[PendingOrder] = []
        self.position: Optional[Position] = None
        self.last_trigger_ns: Optional[int] = None
        self.sl_placed_at: Optional[int] = None


class AveragesBacktest:
    def __init__(self, config: AveragesConfig, slippage: Optional[SlippageModel] = None,
                 maker_fee_pct: float = 0.02, taker_fee_pct: float = 0.05,
                 use_taker_fee: bool = True):
        self.config = config
        self.state = AveragesState(config)
        self.results: List[TradeResult] = []
        self.stats = BacktestStats()
        self.slippage = slippage
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.use_taker_fee = use_taker_fee
        self._last_entry_slippage: float = 0.0

    def on_trade(self, trade: AggTrade) -> None:
        price = trade.price
        ts_ns = trade.timestamp_ns()

        for long_avg, short_avg in self.state.averages:
            long_avg.push(ts_ns, price)
            short_avg.push(ts_ns, price)

        if self.state.position is not None:
            self._check_position(price, ts_ns)
            if self.config.do_not_trigger_if_active:
                return

        self._check_pending_orders(price, ts_ns)

        if self.state.last_trigger_ns is not None:
            elapsed = (ts_ns - self.state.last_trigger_ns) / 1e9
            if elapsed < self.config.restart_delay_secs:
                return

        if self.config.do_not_trigger_if_active and self.state.pending_orders:
            return

        # Check all conditions (AND logic)
        all_triggered = True
        for i, cond in enumerate(self.config.conditions):
            long_avg, short_avg = self.state.averages[i]
            if not long_avg.is_ready() or not short_avg.is_ready():
                all_triggered = False
                break
            long_val = long_avg.average()
            short_val = short_avg.average()
            if long_val is None or short_val is None:
                all_triggered = False
                break
            delta = (long_val - short_val) / long_val * 100.0
            if not cond.is_triggered(delta):
                all_triggered = False
                break

        if all_triggered and not self.state.pending_orders:
            self._place_orders(price, ts_ns)

    def _place_orders(self, trigger_price: float, ts_ns: int) -> None:
        self.state.last_trigger_ns = ts_ns

        first_order_price = trigger_price * (1.0 + self.config.order_distance_pct / 100.0)

        orders = [PendingOrder(
            price=first_order_price,
            size_usdt=self.config.order_size_usdt,
            placed_at_ns=ts_ns,
        )]

        if self.config.grid is not None:
            grid = self.config.grid
            prev_price = first_order_price
            prev_size = self.config.order_size_usdt

            for n in range(1, grid.count + 1):
                next_price = prev_price * (1.0 + grid.distance_pct / 100.0)
                if grid.relative_to_first:
                    next_size = self.config.order_size_usdt * (1.0 + (n - 1) * grid.size_increase_pct / 100.0)
                else:
                    next_size = prev_size * (1.0 + grid.size_increase_pct / 100.0)

                orders.append(PendingOrder(
                    price=next_price,
                    size_usdt=next_size,
                    placed_at_ns=ts_ns,
                ))
                prev_price = next_price
                prev_size = next_size

        self.state.pending_orders = orders

    def _check_pending_orders(self, price: float, ts_ns: int) -> None:
        filled_orders = []
        remaining = []

        for order in self.state.pending_orders:
            if self.config.cancel_delay_secs is not None:
                elapsed = (ts_ns - order.placed_at_ns) / 1e9
                if elapsed >= self.config.cancel_delay_secs:
                    continue

            filled = (price <= order.price if self.config.side == Side.Buy
                      else price >= order.price)

            if filled:
                filled_orders.append(order)
            else:
                remaining.append(order)

        self.state.pending_orders = remaining

        if filled_orders and self.state.position is None:
            order = filled_orders[0]
            self._open_position(order.price, order.size_usdt, ts_ns)

    def _open_position(self, signal_price: float, size_usdt: float, ts_ns: int) -> None:
        # Apply slippage to get realistic fill price
        fill_price = signal_price
        slippage_pct = 0.0
        if self.slippage is not None:
            fill_price = self.slippage.apply(signal_price, self.config.side, size_usdt)
            slippage_pct = abs(fill_price - signal_price) / signal_price * 100.0

        tp_pct = self.config.take_profit.percentage
        sl_pct = self.config.stop_loss.percentage

        if self.config.side == Side.Buy:
            tp_price = fill_price * (1.0 + tp_pct / 100.0)
            sl_price = fill_price * (1.0 - sl_pct / 100.0)
        else:
            tp_price = fill_price * (1.0 - tp_pct / 100.0)
            sl_price = fill_price * (1.0 + sl_pct / 100.0)

        self.state.position = Position.new(
            self.config.side, fill_price, size_usdt,
            ts_ns, tp_price, sl_price, tp_pct
        )
        self.state.sl_placed_at = ts_ns
        self._last_entry_slippage = slippage_pct

    def _check_position(self, price: float, ts_ns: int) -> None:
        pos = self.state.position
        if pos is None:
            return

        # Trailing stop
        if self.config.stop_loss.trailing is not None:
            trail_pct = self.config.stop_loss.trailing.spread_pct / 100.0
            if pos.side == Side.Buy:
                if price > pos.best_price:
                    move_amount = price - pos.best_price
                    if move_amount / pos.best_price >= trail_pct:
                        pos.stop_loss_price += move_amount
                        pos.best_price = price
            else:
                if price < pos.best_price:
                    move_amount = pos.best_price - price
                    if move_amount / pos.best_price >= trail_pct:
                        pos.stop_loss_price -= move_amount
                        pos.best_price = price

        # Auto price down TP
        if self.config.take_profit.auto_price_down is not None:
            apd = self.config.take_profit.auto_price_down
            elapsed = (ts_ns - pos.tp_last_step_time_ns) / 1e9
            if elapsed >= apd.timer_secs:
                new_pct = max(pos.tp_current_pct - apd.step_pct, apd.limit_pct)
                if new_pct != pos.tp_current_pct:
                    pos.tp_current_pct = new_pct
                    if pos.side == Side.Buy:
                        pos.take_profit_price = pos.entry_price * (1.0 + new_pct / 100.0)
                    else:
                        pos.take_profit_price = pos.entry_price * (1.0 - new_pct / 100.0)
                    pos.tp_last_step_time_ns = ts_ns

        placed_at = self.state.sl_placed_at if self.state.sl_placed_at is not None else pos.entry_time_ns
        sl_active = (ts_ns - placed_at) / 1e9 >= self.config.stop_loss.delay_secs

        tp_hit = (price >= pos.take_profit_price if pos.side == Side.Buy
                  else price <= pos.take_profit_price)
        sl_hit = sl_active and (
            price <= pos.stop_loss_price if pos.side == Side.Buy
            else price >= pos.stop_loss_price
        )

        if tp_hit or sl_hit:
            exit_reason = ExitReason.TakeProfit if tp_hit else ExitReason.StopLoss
            exit_price = pos.take_profit_price if tp_hit else pos.stop_loss_price
            pnl_pct = pos.pnl_pct(exit_price)

            result = TradeResult(
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size_usdt=pos.size_usdt,
                entry_time_ns=pos.entry_time_ns,
                exit_time_ns=ts_ns,
                exit_reason=exit_reason,
                pnl_pct=pnl_pct,
            )
            self.stats.record(result)
            self.results.append(result)
            self.state.position = None
            self.state.sl_placed_at = None

    def run(self, trades: List[AggTrade]) -> None:
        for trade in trades:
            self.on_trade(trade)
