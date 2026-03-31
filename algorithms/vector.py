"""Vector Algorithm

Detects regions with abnormal market activity — moments when price ranges
increase sharply over a short period. Allows quickly entering such movements
and capturing profits on short-term market impulses.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import constants as C
from hft_types import (
    Side, StopLossConfig, AutoPriceDown,
    Position, TradeResult, ExitReason, BacktestStats
)
from data import AggTrade
from slippage import SlippageModel
from polar_features import PolarExtractor, PolarSignalGenerator, PolarSignalType
from signal_gate import SignalGate


@dataclass
class Frame:
    min_price: float = 0.0
    max_price: float = 0.0
    trade_count: int = 0
    quote_volume: float = 0.0
    start_ts_ns: int = 0
    end_ts_ns: int = 0

    def spread(self) -> float:
        return self.max_price - self.min_price

    def spread_pct(self) -> float:
        if self.min_price == 0.0:
            return 0.0
        return self.spread() / self.min_price * 100.0

    def upper_boundary(self) -> float:
        return self.max_price

    def lower_boundary(self) -> float:
        return self.min_price


@dataclass
class BorderRange:
    enabled: bool = False
    min_pct: float = 0.0
    max_pct: float = 0.5

    def check(self, change_pct: float) -> bool:
        if not self.enabled:
            return True
        return self.min_pct <= change_pct <= self.max_pct


class ShotDirection(Enum):
    Up = "Up"
    Down = "Down"
    Both = "Both"  # detect shots in either direction


@dataclass
class VectorConfig:
    side: Side = Side.Buy
    frame_size_secs: float = 0.2
    time_frame_secs: float = 1.0
    min_spread_size_pct: float = 0.5
    upper_border_range: BorderRange = field(default_factory=BorderRange)
    lower_border_range: BorderRange = field(default_factory=BorderRange)
    min_trades_per_frame: int = 2
    min_quote_asset_volume: float = 10_000.0
    order_distance_pct: float = 5.0
    use_adaptive_order_distance: bool = False
    order_lifetime_secs: float = 1.0
    max_orders: int = 3
    order_frequency_secs: float = 0.1
    detect_shot: bool = False
    detect_shot_pullback_pct: float = 80.0
    shot_direction: ShotDirection = ShotDirection.Down
    take_profit_spread_pct: float = 90.0
    use_adaptive_take_profit: bool = False
    auto_price_down: Optional[AutoPriceDown] = None
    stop_loss: StopLossConfig = field(default_factory=lambda: StopLossConfig(
        enabled=True, percentage=1.0, spread_pct=0.1, delay_secs=0.0,
        trailing=None, second_sl=None
    ))
    order_size_usdt: float = 100.0
    # Polar integration
    use_polar_signals: bool = False
    polar_tau: int = C.POLAR_DEFAULT_TAU
    polar_price_scale: float = C.POLAR_VECTOR_DEFAULT_PRICE_SCALE


@dataclass
class PendingOrder:
    price: float
    placed_at_ns: int


class VectorState:
    def __init__(self, config: VectorConfig = None):
        self.current_frame: Optional[Frame] = None
        self.current_frame_start_ns: int = 0
        self.frames: deque = deque()
        self.pending_orders: List[PendingOrder] = []
        self.positions: List[Position] = []
        self.last_order_placed_ns: Optional[int] = None
        self.sl_placed_at: List[int] = []
        # Polar integration
        self.polar_extractor: Optional[PolarExtractor] = None
        self.polar_signal_gen: Optional[PolarSignalGenerator] = None
        if config is not None and config.use_polar_signals:
            self.polar_extractor = PolarExtractor(
                tau=config.polar_tau,
                price_scale=config.polar_price_scale
            )
            self.polar_signal_gen = PolarSignalGenerator()
        self._polar_prices: List[float] = []

    def frames_needed(self, config: VectorConfig) -> int:
        import math
        return math.ceil(config.time_frame_secs / config.frame_size_secs)

    def update_polar(self, price: float) -> Optional[PolarSignalType]:
        """Update polar features and return signal if available."""
        if self.polar_extractor is None or self.polar_signal_gen is None:
            return None

        self._polar_prices.append(price)
        features = self.polar_extractor.extract(self._polar_prices)

        if len(features) >= 5:
            signal = self.polar_signal_gen.generate(features)
            if signal and signal.signal_type != PolarSignalType.HOLD:
                return signal.signal_type
        return None


class VectorBacktest:
    def __init__(self, config: VectorConfig, slippage: Optional[SlippageModel] = None,
                 maker_fee_pct: float = C.VECTOR_DEFAULT_MAKER_FEE_PCT,
                 taker_fee_pct: float = C.VECTOR_DEFAULT_TAKER_FEE_PCT,
                 position_sizer: Optional["PositionSizer"] = None,
                 signal_gate: Optional[SignalGate] = None):
        self.config = config
        self.state = VectorState(config)  # Pass config for polar init
        self.results: List[TradeResult] = []
        self.stats = BacktestStats()
        self.slippage = slippage
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.position_sizer = position_sizer
        self.signal_gate = signal_gate
        self._last_entry_slippage: float = 0.0
        self._last_polar_signal: Optional[PolarSignalType] = None

    def on_trade(self, trade: AggTrade) -> None:
        price = trade.price
        ts_ns = trade.timestamp_ns()
        quote_vol = trade.quote_volume()

        # Feed the signal gate
        if self.signal_gate is not None:
            self.signal_gate.on_trade(price, trade.quantity, ts_ns)

        # Update polar features (if enabled)
        if self.config.use_polar_signals:
            polar_signal = self.state.update_polar(price)
            if polar_signal is not None:
                self._last_polar_signal = polar_signal

        self._update_frame(price, quote_vol, ts_ns)
        self._check_pending_orders(price, ts_ns)
        self._check_positions(price, ts_ns)

        frames_needed = self.state.frames_needed(self.config)
        if len(self.state.frames) < frames_needed:
            return

        active_count = len(self.state.pending_orders) + len(self.state.positions)
        if active_count >= self.config.max_orders:
            return

        if self.state.last_order_placed_ns is not None:
            elapsed = (ts_ns - self.state.last_order_placed_ns) / 1e9
            if elapsed < self.config.order_frequency_secs:
                return

        # Use polar signal if enabled and available
        if self.config.use_polar_signals and self._last_polar_signal is not None:
            self._handle_polar_signal(price, ts_ns)
        elif self.config.detect_shot:
            self._check_detect_shot(price, ts_ns)
        else:
            self._check_vector_signal(price, ts_ns)

    def _update_frame(self, price: float, quote_vol: float, ts_ns: int) -> None:
        frame_size_ns = int(self.config.frame_size_secs * 1e9)

        if self.state.current_frame is None:
            self.state.current_frame = Frame(
                min_price=price, max_price=price,
                trade_count=1, quote_volume=quote_vol,
                start_ts_ns=ts_ns, end_ts_ns=ts_ns,
            )
            self.state.current_frame_start_ns = ts_ns
            return

        frame = self.state.current_frame
        elapsed = ts_ns - self.state.current_frame_start_ns

        if elapsed >= frame_size_ns:
            frame.end_ts_ns = ts_ns
            completed = Frame(
                min_price=frame.min_price, max_price=frame.max_price,
                trade_count=frame.trade_count, quote_volume=frame.quote_volume,
                start_ts_ns=frame.start_ts_ns, end_ts_ns=frame.end_ts_ns,
            )
            self.state.frames.append(completed)

            # Remove old frames outside time window
            time_window_ns = int(self.config.time_frame_secs * 1e9)
            keep_window_ns = time_window_ns + frame_size_ns
            while self.state.frames and (ts_ns - self.state.frames[0].start_ts_ns > keep_window_ns):
                self.state.frames.popleft()

            self.state.current_frame = Frame(
                min_price=price, max_price=price,
                trade_count=1, quote_volume=quote_vol,
                start_ts_ns=ts_ns, end_ts_ns=ts_ns,
            )
            self.state.current_frame_start_ns = ts_ns
        else:
            if price < frame.min_price:
                frame.min_price = price
            if price > frame.max_price:
                frame.max_price = price
            frame.trade_count += 1
            frame.quote_volume += quote_vol
            frame.end_ts_ns = ts_ns

    def _check_vector_signal(self, price: float, ts_ns: int) -> None:
        all_frames = list(self.state.frames)
        if not all_frames:
            return

        frames_needed = self.state.frames_needed(self.config)
        frames = all_frames[-frames_needed:]

        if len(frames) < frames_needed:
            return

        last_frame = frames[-1]
        last_spread = last_frame.spread()

        all_ok = all(
            f.spread_pct() >= self.config.min_spread_size_pct
            and f.trade_count >= self.config.min_trades_per_frame
            and f.quote_volume >= self.config.min_quote_asset_volume
            for f in frames
        )

        if not all_ok:
            return

        if len(frames) >= 2:
            for i in range(1, len(frames)):
                prev = frames[i - 1]
                curr = frames[i]

                if self.config.upper_border_range.enabled and last_spread > 0.0:
                    change = (curr.upper_boundary() - prev.upper_boundary()) / last_spread * 100.0
                    if not self.config.upper_border_range.check(change):
                        return

                if self.config.lower_border_range.enabled and last_spread > 0.0:
                    change = (curr.lower_boundary() - prev.lower_boundary()) / last_spread * 100.0
                    if not self.config.lower_border_range.check(change):
                        return

        order_price = self._calculate_order_price(last_frame, last_spread, frames)
        self._place_order(order_price, ts_ns)

    def _handle_polar_signal(self, price: float, ts_ns: int) -> None:
        """Handle polar signal for order placement."""
        if self._last_polar_signal is None:
            return

        # Polar signal to order logic
        should_place_order = False

        if self.config.side == Side.Buy:
            # Long signals: mean_revert_long, trend_long, breakout_long
            if self._last_polar_signal in [
                PolarSignalType.MEAN_REVERT_LONG,
                PolarSignalType.TREND_LONG,
                PolarSignalType.BREAKOUT_LONG
            ]:
                should_place_order = True
        else:  # Side.Sell
            # Short signals: mean_revert_short, trend_short, breakout_short
            if self._last_polar_signal in [
                PolarSignalType.MEAN_REVERT_SHORT,
                PolarSignalType.TREND_SHORT,
                PolarSignalType.BREAKOUT_SHORT
            ]:
                should_place_order = True

        if should_place_order:
            # Use last frame for order price calculation
            if self.state.frames:
                last_frame = self.state.frames[-1]
                spread = last_frame.spread()
                order_price = self._calculate_order_price(
                    last_frame, spread, list(self.state.frames)
                )
                self._place_order(order_price, ts_ns)

    def _check_detect_shot(self, price: float, ts_ns: int) -> None:
        if not self.state.frames:
            return
        last_frame = self.state.frames[-1]

        if last_frame.spread_pct() < self.config.min_spread_size_pct:
            return
        if last_frame.trade_count < self.config.min_trades_per_frame:
            return
        if last_frame.quote_volume < self.config.min_quote_asset_volume:
            return

        spread = last_frame.spread()
        pullback_threshold = self.config.detect_shot_pullback_pct / 100.0

        move = last_frame.max_price - last_frame.min_price
        if self.config.shot_direction == ShotDirection.Up:
            pullback = last_frame.max_price - price
            shot_detected = move > 0.0 and pullback / move >= pullback_threshold
        elif self.config.shot_direction == ShotDirection.Down:
            pullback = price - last_frame.min_price
            shot_detected = move > 0.0 and pullback / move >= pullback_threshold
        else:  # Both — detect shot in either direction
            pullback_up = last_frame.max_price - price
            pullback_down = price - last_frame.min_price
            shot_detected = move > 0.0 and (
                pullback_up / move >= pullback_threshold
                or pullback_down / move >= pullback_threshold
            )

        if not shot_detected:
            return

        order_price = self._calculate_order_price(last_frame, spread, [last_frame])
        self._place_order(order_price, ts_ns)

    def _calculate_order_price(self, last_frame: Frame, spread: float,
                                frames: List[Frame]) -> float:
        if self.config.side == Side.Buy:
            base_price = last_frame.lower_boundary() + spread * self.config.order_distance_pct / 100.0
        else:
            base_price = last_frame.upper_boundary() - spread * self.config.order_distance_pct / 100.0

        if not self.config.use_adaptive_order_distance or len(frames) < 2:
            return base_price

        avg_change = self._calculate_avg_boundary_change(frames)
        return base_price + avg_change

    def _calculate_avg_boundary_change(self, frames: List[Frame]) -> float:
        if len(frames) < 2:
            return 0.0
        total_change = 0.0
        n = len(frames) - 1
        for i in range(1, len(frames)):
            if self.config.side == Side.Buy:
                change = frames[i].lower_boundary() - frames[i - 1].lower_boundary()
            else:
                change = frames[i].upper_boundary() - frames[i - 1].upper_boundary()
            total_change += change
        return total_change / n

    def _place_order(self, order_price: float, ts_ns: int) -> None:
        self.state.pending_orders.append(PendingOrder(price=order_price, placed_at_ns=ts_ns))
        self.state.last_order_placed_ns = ts_ns

    def _check_pending_orders(self, price: float, ts_ns: int) -> None:
        remaining = []
        filled_prices = []

        for order in self.state.pending_orders:
            elapsed = (ts_ns - order.placed_at_ns) / 1e9
            if elapsed > self.config.order_lifetime_secs:
                continue

            filled = (price <= order.price if self.config.side == Side.Buy
                      else price >= order.price)

            if filled:
                filled_prices.append(order.price)
            else:
                remaining.append(order)

        self.state.pending_orders = remaining

        for fill_price in filled_prices:
            self._open_position(fill_price, ts_ns)

    def _open_position(self, signal_price: float, ts_ns: int) -> None:
        if not self.state.frames:
            return
        last_frame = self.state.frames[-1]
        spread = last_frame.spread()
        frames = list(self.state.frames)

        # Signal gate check
        gate_confidence = 1.0
        if self.signal_gate is not None:
            allowed, gate_confidence = self.signal_gate.should_enter(self.config.side)
            if not allowed:
                return

        # Apply slippage to get realistic fill price
        fill_price = signal_price
        slippage_pct = 0.0
        if self.slippage is not None:
            fill_price = self.slippage.apply(signal_price, self.config.side, self.config.order_size_usdt)
            slippage_pct = abs(fill_price - signal_price) / signal_price * 100.0

        if self.config.side == Side.Buy:
            tp_base = last_frame.lower_boundary() + spread * self.config.take_profit_spread_pct / 100.0
        else:
            tp_base = last_frame.upper_boundary() - spread * self.config.take_profit_spread_pct / 100.0

        if self.config.use_adaptive_take_profit and len(frames) >= 2:
            avg_change = self._calculate_avg_boundary_change(frames)
            tp_price = tp_base + avg_change
        else:
            tp_price = tp_base

        sl_pct = self.config.stop_loss.percentage
        if self.config.side == Side.Buy:
            sl_price = fill_price * (1.0 - sl_pct / 100.0)
        else:
            sl_price = fill_price * (1.0 + sl_pct / 100.0)

        tp_pct = abs(tp_price - fill_price) / fill_price * 100.0

        # Scale size by gate confidence
        size_usdt = self.config.order_size_usdt * gate_confidence
        if self.position_sizer is not None:
            size_usdt = self.position_sizer.calculate_size(
                [r.net_pnl_pct for r in self.results]
            ) * gate_confidence

        pos = Position.new(
            self.config.side, fill_price, size_usdt,
            ts_ns, tp_price, sl_price, tp_pct
        )
        self.state.positions.append(pos)
        self.state.sl_placed_at.append(ts_ns)
        self._last_entry_slippage = slippage_pct

    def _check_positions(self, price: float, ts_ns: int) -> None:
        remaining_positions = []
        remaining_sl_times = []

        for pos, sl_placed_at in zip(self.state.positions, self.state.sl_placed_at):
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

            sl_active = (ts_ns - sl_placed_at) / 1e9 >= self.config.stop_loss.delay_secs

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

                result = TradeResult(
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    size_usdt=pos.size_usdt,
                    entry_time_ns=pos.entry_time_ns,
                    exit_time_ns=ts_ns,
                    exit_reason=exit_reason,
                    pnl_pct=pnl_pct,
                    entry_fee_pct=self.taker_fee_pct,
                    exit_fee_pct=self.taker_fee_pct,
                    slippage_pct=self._last_entry_slippage + exit_slippage_pct,
                )
                self.stats.record(result, ts_ns)
                self.results.append(result)
            else:
                remaining_positions.append(pos)
                remaining_sl_times.append(sl_placed_at)

        self.state.positions = remaining_positions
        self.state.sl_placed_at = remaining_sl_times
        if not self.state.positions:
            self._last_entry_slippage = 0.0

    def run(self, trades: List[AggTrade]) -> None:
        for trade in trades:
            self.on_trade(trade)
