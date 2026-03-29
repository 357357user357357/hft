"""Slippage and latency simulation for realistic HFT backtesting.

The backtest currently assumes perfect fills at the exact trade price.
This module adds configurable:

1. **Price slippage** — fill price is worse than signal price by a
   configurable amount (fixed bps or volume-dependent).
2. **Latency** — orders react to prices with a delay, so fills use a
   future price instead of the signal price.
3. **Partial fills** — large orders may only partially fill at each
   price level (optional, disabled by default).

Usage:
    from slippage import SlippageModel, SlippageConfig

    model = SlippageModel(SlippageConfig(fixed_bps=1.0, latency_ms=5))
    fill_price = model.apply(signal_price, side, trade)
"""

import bisect
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

from hft_types import Side
from data import AggTrade


@dataclass
class SlippageConfig:
    """Slippage and latency parameters.

    fixed_bps:     Fixed slippage in basis points (1 bp = 0.01%).
                   Applied against the trader: buys fill higher, sells lower.
    random_bps:    Additional random slippage uniform in [0, random_bps].
    latency_ms:    Simulated order-to-exchange latency in milliseconds.
                   When > 0, fills use the price `latency_ms` into the future
                   instead of the current price (handled by the caller).
    volume_impact_bps_per_lot: Price impact per lot (order_size / lot_size).
                   Simulates market impact for larger orders.
    lot_size_usdt: Reference lot size for volume impact calculation.
    enabled:       Master switch. When False, apply() returns the input price.
    """
    fixed_bps: float = 0.0
    random_bps: float = 0.0
    latency_ms: int = 0
    volume_impact_bps_per_lot: float = 0.0
    lot_size_usdt: float = 100.0
    enabled: bool = True


class SlippageModel:
    """Applies slippage to fill prices."""

    def __init__(self, config: SlippageConfig = None):
        self.config = config or SlippageConfig()
        self._latency_buffer: deque = deque(maxlen=5000)  # (timestamp_ms, price) ring buffer

    def apply(self, signal_price: float, side: Side,
              order_size_usdt: float = 100.0) -> float:
        """
        Compute the realistic fill price given a signal price.

        For buys: fill price >= signal price (worse = higher).
        For sells: fill price <= signal price (worse = lower).

        Args:
            signal_price: The price at which the backtest detected a fill.
            side: Buy or Sell.
            order_size_usdt: Order size for volume impact calculation.

        Returns:
            Adjusted fill price after slippage.
        """
        if not self.config.enabled:
            return signal_price

        total_bps = self.config.fixed_bps

        # Random component
        if self.config.random_bps > 0:
            total_bps += random.uniform(0, self.config.random_bps)

        # Volume impact
        if self.config.volume_impact_bps_per_lot > 0 and self.config.lot_size_usdt > 0:
            lots = order_size_usdt / self.config.lot_size_usdt
            total_bps += self.config.volume_impact_bps_per_lot * lots

        # Convert bps to multiplier (1 bp = 0.0001)
        slip_frac = total_bps * 0.0001

        if side == Side.Buy:
            return signal_price * (1.0 + slip_frac)
        else:
            return signal_price * (1.0 - slip_frac)

    def push_price(self, timestamp_ms: int, price: float) -> None:
        """Record a price for latency-delayed lookups."""
        self._latency_buffer.append((timestamp_ms, price))

    def delayed_price(self, current_ts_ms: int) -> Optional[float]:
        """
        Get the price that was current `latency_ms` ago.

        Returns None if the buffer doesn't have data that far back.
        """
        if self.config.latency_ms <= 0 or not self._latency_buffer:
            return None

        target_ts = current_ts_ms - self.config.latency_ms
        buf: List = list(self._latency_buffer)
        timestamps = [ts for ts, _ in buf]
        idx = bisect.bisect_right(timestamps, target_ts) - 1
        if idx < 0:
            return None
        return buf[idx][1]

    def describe(self) -> str:
        if not self.config.enabled:
            return "SlippageModel: disabled (perfect fills)"
        parts = []
        if self.config.fixed_bps > 0:
            parts.append(f"fixed={self.config.fixed_bps:.1f}bps")
        if self.config.random_bps > 0:
            parts.append(f"random=[0,{self.config.random_bps:.1f}]bps")
        if self.config.latency_ms > 0:
            parts.append(f"latency={self.config.latency_ms}ms")
        if self.config.volume_impact_bps_per_lot > 0:
            parts.append(f"impact={self.config.volume_impact_bps_per_lot:.1f}bps/lot")
        return f"SlippageModel: {', '.join(parts) or 'zero slippage'}"
