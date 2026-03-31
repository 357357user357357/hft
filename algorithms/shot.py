"""Shot Algorithm

Profits from sharp price spikes (breakthroughs) followed by pullbacks.
Keeps an order at a set distance from price, automatically moving it during
smooth movements. When a breakthrough occurs, position opens and quickly
closes on the pullback.

Logic:
1. Place order at distance % from current price
2. Buffer (dead zone) around price — order doesn't move while price is inside
3. Price crosses far buffer boundary → order moves toward price (after follow_price_delay)
4. Price crosses near buffer boundary → order moves away from price (after replace_delay)
5. Order fills on breakthrough → open position → TP/SL
"""

from dataclasses import dataclass, field
from typing import List, Optional

from hft_types import (
    Side, TakeProfitConfig, StopLossConfig, AutoPriceDown,
    Position, TradeResult, ExitReason, BacktestStats
)
from data import AggTrade
from slippage import SlippageModel
from risk_management import PositionSizer
from signal_gate import SignalGate


@dataclass
class ShotConfig:
    side: Side = Side.Buy
    distance_pct: float = 1.0
    buffer_pct: float = 0.4
    follow_price_delay_secs: float = 0.5
    replace_delay_secs: float = 0.5
    order_size_usdt: float = 100.0
    take_profit: TakeProfitConfig = field(default_factory=lambda: TakeProfitConfig(
        enabled=True, percentage=0.5, auto_price_down=None
    ))
    stop_loss: StopLossConfig = field(default_factory=lambda: StopLossConfig(
        enabled=True, percentage=1.0, spread_pct=0.1, delay_secs=1.0,
        trailing=None, second_sl=None
    ))


class ShotState:
    def __init__(self):
        self.order_price: Optional[float] = None
        self.buffer_center: Optional[float] = None
        self.far_boundary_crossed_at: Optional[int] = None
        self.near_boundary_crossed_at: Optional[int] = None
        self.position: Optional[Position] = None
        self.sl_placed_at: Optional[int] = None


class ShotBacktest:
    def __init__(self, config: ShotConfig, slippage: Optional[SlippageModel] = None,
                 maker_fee_pct: float = 0.02, taker_fee_pct: float = 0.05,
                 use_taker_fee: bool = True, position_sizer: Optional[PositionSizer] = None,
                 signal_gate: Optional[SignalGate] = None):
        self.config = config
        self.state = ShotState()
        self.results: List[TradeResult] = []
        self.stats = BacktestStats()
        self.slippage = slippage
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.use_taker_fee = use_taker_fee
        self.position_sizer = position_sizer
        self.signal_gate = signal_gate
        self._last_entry_slippage: float = 0.0
        self._last_entry_ts: Optional[int] = None

    def on_trade(self, trade: AggTrade) -> None:
        price = trade.price
        ts_ns = trade.timestamp_ns()

        # Feed the signal gate
        if self.signal_gate is not None:
            self.signal_gate.on_trade(price, trade.quantity, ts_ns)

        if self.state.position is not None:
            self._check_position(price, ts_ns)
            return

        if self.state.order_price is None:
            self._place_initial_order(price, ts_ns)
            return

        order_price = self.state.order_price
        buffer_center = self.state.buffer_center if self.state.buffer_center is not None else price

        half_buffer = buffer_center * self.config.buffer_pct / 100.0 / 2.0
        far_boundary = buffer_center + half_buffer
        near_boundary = buffer_center - half_buffer

        order_hit = (price <= order_price if self.config.side == Side.Buy
                     else price >= order_price)

        if order_hit:
            self._open_position(price, ts_ns)
            return

        if self.config.side == Side.Buy:
            if price > far_boundary:
                if self.state.far_boundary_crossed_at is None:
                    self.state.far_boundary_crossed_at = ts_ns
                    self.state.near_boundary_crossed_at = None
                elapsed = (ts_ns - self.state.far_boundary_crossed_at) / 1e9
                if elapsed >= self.config.follow_price_delay_secs:
                    self._update_order(price, ts_ns)
            elif price < near_boundary:
                if self.state.near_boundary_crossed_at is None:
                    self.state.near_boundary_crossed_at = ts_ns
                    self.state.far_boundary_crossed_at = None
                elapsed = (ts_ns - self.state.near_boundary_crossed_at) / 1e9
                if elapsed >= self.config.replace_delay_secs:
                    self._update_order(price, ts_ns)
            else:
                self.state.far_boundary_crossed_at = None
                self.state.near_boundary_crossed_at = None
        else:  # Sell
            if price < near_boundary:
                if self.state.far_boundary_crossed_at is None:
                    self.state.far_boundary_crossed_at = ts_ns
                    self.state.near_boundary_crossed_at = None
                elapsed = (ts_ns - self.state.far_boundary_crossed_at) / 1e9
                if elapsed >= self.config.follow_price_delay_secs:
                    self._update_order(price, ts_ns)
            elif price > far_boundary:
                if self.state.near_boundary_crossed_at is None:
                    self.state.near_boundary_crossed_at = ts_ns
                    self.state.far_boundary_crossed_at = None
                elapsed = (ts_ns - self.state.near_boundary_crossed_at) / 1e9
                if elapsed >= self.config.replace_delay_secs:
                    self._update_order(price, ts_ns)
            else:
                self.state.far_boundary_crossed_at = None
                self.state.near_boundary_crossed_at = None

    def _place_initial_order(self, price: float, ts_ns: int) -> None:
        if self.config.side == Side.Buy:
            order_price = price * (1.0 - self.config.distance_pct / 100.0)
        else:
            order_price = price * (1.0 + self.config.distance_pct / 100.0)
        self.state.order_price = order_price
        self.state.buffer_center = price
        self.state.far_boundary_crossed_at = None
        self.state.near_boundary_crossed_at = None

    def _update_order(self, price: float, ts_ns: int) -> None:
        if self.config.side == Side.Buy:
            order_price = price * (1.0 - self.config.distance_pct / 100.0)
        else:
            order_price = price * (1.0 + self.config.distance_pct / 100.0)
        self.state.order_price = order_price
        self.state.buffer_center = price
        self.state.far_boundary_crossed_at = None
        self.state.near_boundary_crossed_at = None

    def _open_position(self, signal_price: float, ts_ns: int) -> None:
        # Signal gate check
        gate_confidence = 1.0
        if self.signal_gate is not None:
            allowed, gate_confidence = self.signal_gate.should_enter(self.config.side)
            if not allowed:
                return

        # Calculate position size using sizer if available
        if self.position_sizer is not None:
            size_usdt = self.position_sizer.calculate_size(
                [r.net_pnl_pct for r in self.results]
            )
        else:
            size_usdt = self.config.order_size_usdt

        # Scale by signal gate confidence
        size_usdt *= gate_confidence

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
        self.state.order_price = None
        self.state.buffer_center = None
        self.state.sl_placed_at = ts_ns
        # Store slippage for later fee calculation
        self._last_entry_slippage = slippage_pct
        self._last_entry_ts = ts_ns

    def _check_position(self, price: float, ts_ns: int) -> None:
        pos = self.state.position
        if pos is None:
            return

        # Update trailing stop
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

        # Check SL delay
        if self.config.stop_loss.delay_secs > 0.0:
            placed_at = self.state.sl_placed_at if self.state.sl_placed_at is not None else pos.entry_time_ns
            sl_active = (ts_ns - placed_at) / 1e9 >= self.config.stop_loss.delay_secs
        else:
            sl_active = True

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

            # Calculate fees (taker for both entry and exit in HFT)
            entry_fee_pct = self.taker_fee_pct
            exit_fee_pct = self.taker_fee_pct

            result = TradeResult(
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size_usdt=pos.size_usdt,
                entry_time_ns=pos.entry_time_ns,
                exit_time_ns=ts_ns,
                exit_reason=exit_reason,
                pnl_pct=pnl_pct,
                entry_fee_pct=entry_fee_pct,
                exit_fee_pct=exit_fee_pct,
                slippage_pct=getattr(self, '_last_entry_slippage', 0.0) + exit_slippage_pct,
            )
            self.stats.record(result, ts_ns)
            self.results.append(result)

            # Update position sizer equity
            if self.position_sizer is not None:
                new_equity = self.position_sizer.current_equity * (1 + result.net_pnl_pct / 100.0)
                self.position_sizer.update_equity(new_equity)

            self.state.position = None
            self.state.sl_placed_at = None
            self._last_entry_slippage = 0.0

    def run(self, trades: List[AggTrade]) -> None:
        for trade in trades:
            self.on_trade(trade)
