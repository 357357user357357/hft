"""Depth Shot Algorithm

Like Shot, but places orders based on order book volume rather than fixed
distance from price. The order is placed where accumulated volume in the
book reaches the target value.
"""

import bisect
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from hft_types import (
    Side, StopLossConfig, AutoPriceDown,
    Position, TradeResult, ExitReason, BacktestStats
)
from data import AggTrade, SimpleOrderBook
from slippage import SlippageModel


class DepthTpMode:
    pass


@dataclass
class Classic(DepthTpMode):
    percentage: float


@dataclass
class Historic(DepthTpMode):
    percentage: float


@dataclass
class Depth(DepthTpMode):
    percentage: float


def calculate_tp(mode: DepthTpMode, side: Side, entry_price: float,
                 price_2s_ago: float, price_before_breakthrough: float) -> float:
    if isinstance(mode, Classic):
        if side == Side.Buy:
            return entry_price * (1.0 + mode.percentage / 100.0)
        else:
            return entry_price * (1.0 - mode.percentage / 100.0)
    elif isinstance(mode, Historic):
        distance = abs(entry_price - price_2s_ago)
        pullback = distance * mode.percentage / 100.0
        if side == Side.Buy:
            return entry_price + pullback
        else:
            return entry_price - pullback
    else:  # Depth
        breakthrough = abs(entry_price - price_before_breakthrough)
        pullback = breakthrough * mode.percentage / 100.0
        if side == Side.Buy:
            return entry_price + pullback
        else:
            return entry_price - pullback


@dataclass
class DepthShotConfig:
    side: Side = Side.Buy
    target_volume: float = 50_000.0
    min_distance_pct: float = 0.5
    max_distance_pct: float = 2.0
    volume_buffer: float = 4_000.0
    min_buffer_pct: float = 0.2
    max_buffer_pct: float = 2.0
    stop_if_out_of_range: bool = False
    follow_price_delay_secs: float = 0.5
    replace_delay_secs: float = 0.5
    order_size_usdt: float = 100.0
    take_profit_mode: DepthTpMode = field(default_factory=lambda: Depth(percentage=50.0))
    auto_price_down: Optional[AutoPriceDown] = None
    stop_loss: StopLossConfig = field(default_factory=lambda: StopLossConfig(
        enabled=True, percentage=1.5, spread_pct=0.1, delay_secs=1.0,
        trailing=None, second_sl=None
    ))


class DepthShotState:
    def __init__(self):
        self.order_price: Optional[float] = None
        self.order_volume: Optional[float] = None
        self.far_boundary_crossed_at: Optional[int] = None
        self.near_boundary_crossed_at: Optional[int] = None
        self.position: Optional[Position] = None
        self.sl_placed_at: Optional[int] = None
        self.price_history: deque = deque(maxlen=10000)
        self.price_before_breakthrough: float = 0.0

    def price_2s_ago(self, current_ts_ns: int) -> float:
        if not self.price_history:
            return 0.0
        target_ts = current_ts_ns - 2_000_000_000
        buf = list(self.price_history)
        timestamps = [ts for ts, _ in buf]
        idx = bisect.bisect_right(timestamps, target_ts) - 1
        if idx >= 0:
            return buf[idx][1]
        return buf[0][1]

    def push_price(self, ts_ns: int, price: float) -> None:
        self.price_history.append((ts_ns, price))
        cutoff = ts_ns - 5_000_000_000
        while self.price_history and self.price_history[0][0] < cutoff:
            self.price_history.popleft()


class DepthShotBacktest:
    def __init__(self, config: DepthShotConfig, slippage: Optional[SlippageModel] = None,
                 maker_fee_pct: float = 0.02, taker_fee_pct: float = 0.05,
                 use_taker_fee: bool = True):
        self.config = config
        self.state = DepthShotState()
        self.results: List[TradeResult] = []
        self.stats = BacktestStats()
        self.slippage = slippage
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.use_taker_fee = use_taker_fee
        self._last_entry_slippage: float = 0.0

    def on_trade(self, trade: AggTrade, book: SimpleOrderBook) -> None:
        price = trade.price
        ts_ns = trade.timestamp_ns()

        self.state.push_price(ts_ns, price)

        if self.state.position is not None:
            self._check_position(price, ts_ns)
            return

        # Check if existing order was hit first
        if self.state.order_price is not None:
            order_price = self.state.order_price
            order_hit = (price <= order_price if self.config.side == Side.Buy
                         else price >= order_price)
            if order_hit:
                history = list(self.state.price_history)
                self.state.price_before_breakthrough = (
                    history[-2][1] if len(history) >= 2 else price
                )
                self._open_position(price, ts_ns)
                return

        # Find order price from book volume
        if self.config.side == Side.Buy:
            target_order_price = book.bid_price_at_volume(self.config.target_volume)
        else:
            target_order_price = book.ask_price_at_volume(self.config.target_volume)

        if target_order_price is None:
            return  # Not enough volume in book

        # Check distance constraints
        if self.config.side == Side.Buy:
            distance_pct = (price - target_order_price) / price * 100.0
        else:
            distance_pct = (target_order_price - price) / price * 100.0

        if distance_pct < self.config.min_distance_pct or distance_pct > self.config.max_distance_pct:
            if self.config.stop_if_out_of_range:
                self.state.order_price = None
                self.state.order_volume = None
            else:
                clamped_dist = max(self.config.min_distance_pct,
                                   min(self.config.max_distance_pct, distance_pct))
                if self.config.side == Side.Buy:
                    clamped_price = price * (1.0 - clamped_dist / 100.0)
                else:
                    clamped_price = price * (1.0 + clamped_dist / 100.0)
                self.state.order_price = clamped_price
                self.state.order_volume = self.config.target_volume
            return

        if self.state.order_price is None:
            self.state.order_price = target_order_price
            self.state.order_volume = self.config.target_volume
            return

        order_price = self.state.order_price
        order_volume = self.state.order_volume if self.state.order_volume is not None else self.config.target_volume

        order_hit = (price <= order_price if self.config.side == Side.Buy
                     else price >= order_price)
        if order_hit:
            history = list(self.state.price_history)
            self.state.price_before_breakthrough = (
                history[-2][1] if len(history) >= 2 else price
            )
            self._open_position(price, ts_ns)
            return

        new_order_price = target_order_price

        if self.config.side == Side.Buy:
            price_moved_far = new_order_price > order_price * (1.0 + self.config.min_buffer_pct / 100.0)
            price_moved_near = new_order_price < order_price * (1.0 - self.config.min_buffer_pct / 100.0)
        else:
            price_moved_far = new_order_price < order_price * (1.0 - self.config.min_buffer_pct / 100.0)
            price_moved_near = new_order_price > order_price * (1.0 + self.config.min_buffer_pct / 100.0)

        if price_moved_far:
            if self.state.far_boundary_crossed_at is None:
                self.state.far_boundary_crossed_at = ts_ns
                self.state.near_boundary_crossed_at = None
            elapsed = (ts_ns - self.state.far_boundary_crossed_at) / 1e9
            if elapsed >= self.config.follow_price_delay_secs:
                self.state.order_price = new_order_price
                self.state.order_volume = self.config.target_volume
                self.state.far_boundary_crossed_at = None
        elif price_moved_near:
            if self.state.near_boundary_crossed_at is None:
                self.state.near_boundary_crossed_at = ts_ns
                self.state.far_boundary_crossed_at = None
            elapsed = (ts_ns - self.state.near_boundary_crossed_at) / 1e9
            if elapsed >= self.config.replace_delay_secs:
                self.state.order_price = new_order_price
                self.state.order_volume = self.config.target_volume
                self.state.near_boundary_crossed_at = None
        else:
            self.state.far_boundary_crossed_at = None
            self.state.near_boundary_crossed_at = None

    def _open_position(self, signal_price: float, ts_ns: int) -> None:
        # Apply slippage to get realistic fill price
        fill_price = signal_price
        slippage_pct = 0.0
        if self.slippage is not None:
            fill_price = self.slippage.apply(signal_price, self.config.side, self.config.order_size_usdt)
            slippage_pct = abs(fill_price - signal_price) / signal_price * 100.0

        price_2s_ago = self.state.price_2s_ago(ts_ns)
        price_before = self.state.price_before_breakthrough

        tp_price = calculate_tp(
            self.config.take_profit_mode,
            self.config.side,
            fill_price,
            price_2s_ago,
            price_before,
        )

        sl_pct = self.config.stop_loss.percentage
        if self.config.side == Side.Buy:
            sl_price = fill_price * (1.0 - sl_pct / 100.0)
        else:
            sl_price = fill_price * (1.0 + sl_pct / 100.0)

        tp_pct = abs(tp_price - fill_price) / fill_price * 100.0

        self.state.position = Position.new(
            self.config.side, fill_price, self.config.order_size_usdt,
            ts_ns, tp_price, sl_price, tp_pct
        )
        self.state.order_price = None
        self.state.order_volume = None
        self.state.sl_placed_at = ts_ns
        self._last_entry_slippage = slippage_pct

    def _check_position(self, price: float, ts_ns: int) -> None:
        pos = self.state.position
        if pos is None:
            return

        # Auto price down TP
        if self.config.auto_price_down is not None:
            apd = self.config.auto_price_down
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
            signal_exit_price = pos.take_profit_price if tp_hit else pos.stop_loss_price

            # Apply slippage to exit price
            exit_price = signal_exit_price
            exit_slippage_pct = 0.0
            if self.slippage is not None:
                exit_price = self.slippage.apply(signal_exit_price, pos.side.opposite(), pos.size_usdt)
                exit_slippage_pct = abs(exit_price - signal_exit_price) / signal_exit_price * 100.0

            pnl_pct = pos.pnl_pct(exit_price)

            fee_pct = self.taker_fee_pct if self.use_taker_fee else self.maker_fee_pct

            result = TradeResult(
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size_usdt=pos.size_usdt,
                entry_time_ns=pos.entry_time_ns,
                exit_time_ns=ts_ns,
                exit_reason=exit_reason,
                pnl_pct=pnl_pct,
                entry_fee_pct=fee_pct,
                exit_fee_pct=fee_pct,
                slippage_pct=self._last_entry_slippage + exit_slippage_pct,
            )
            self.stats.record(result, ts_ns)
            self.results.append(result)
            self.state.position = None
            self.state.sl_placed_at = None
            self._last_entry_slippage = 0.0

    def run(self, trades: List[AggTrade],
            book_builder: Callable[[List[AggTrade]], SimpleOrderBook]) -> None:
        window = 100
        for i in range(len(trades)):
            start = max(0, i - window)
            book = book_builder(trades[start:i + 1])
            self.on_trade(trades[i], book)
