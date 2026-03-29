"""Position sizing and risk management for HFT backtesting.

Implements:
1. Fixed position sizing (constant USDT per trade)
2. Fractional sizing (percentage of current equity)
3. Kelly criterion sizing (optimal bet sizing based on win rate and payoff)

Usage:
    from risk_management import PositionSizer, SizingConfig

    sizer = PositionSizer(SizingConfig(mode='kelly', initial_equity=10000.0))
    size = sizer.calculate_size(current_equity, trade_history)
"""

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SizingConfig:
    """Position sizing configuration.

    mode: 'fixed' | 'fractional' | 'kelly'
    fixed_size_usdt: Constant size for 'fixed' mode
    fractional_pct: Percentage of equity for 'fractional' mode (e.g., 5.0 = 5%)
    kelly_cap: Maximum Kelly fraction (e.g., 0.25 = 25% of equity)
    kelly_lookback: Number of trades for Kelly calculation
    initial_equity: Starting equity in USDT
    """
    mode: str = 'fractional'
    fixed_size_usdt: float = 100.0
    fractional_pct: float = 5.0
    kelly_cap: float = 0.25
    kelly_lookback: int = 50
    initial_equity: float = 10000.0


class PositionSizer:
    """Calculates position sizes based on risk management rules."""

    def __init__(self, config: SizingConfig):
        self.config = config
        self.current_equity = config.initial_equity

    def update_equity(self, new_equity: float) -> None:
        """Update current equity after a trade."""
        self.current_equity = new_equity

    def calculate_size(self, trade_history: Optional[List[float]] = None) -> float:
        """
        Calculate position size in USDT based on current mode.

        Args:
            trade_history: List of PnL percentages for Kelly calculation

        Returns:
            Position size in USDT
        """
        if self.config.mode == 'fixed':
            return self.config.fixed_size_usdt

        elif self.config.mode == 'fractional':
            return self.current_equity * (self.config.fractional_pct / 100.0)

        elif self.config.mode == 'kelly':
            kelly_fraction = self._calculate_kelly(trade_history or [])
            capped_fraction = min(kelly_fraction, self.config.kelly_cap)
            return self.current_equity * capped_fraction

        else:
            raise ValueError(f"Unknown sizing mode: {self.config.mode}")

    def _calculate_kelly(self, trade_history: List[float]) -> float:
        """
        Calculate Kelly criterion fraction.

        Kelly formula: f* = (p * b - q) / b
        where:
            p = win probability
            q = loss probability = 1 - p
            b = win/loss ratio (average win / average loss)

        Returns:
            Kelly fraction (0.0 to 1.0)
        """
        if len(trade_history) < 2:
            # Not enough data, use minimum position
            return self.config.fractional_pct / 100.0

        # Use only recent trades for Kelly calculation
        recent = trade_history[-self.config.kelly_lookback:]

        wins = [t for t in recent if t > 0]
        losses = [t for t in recent if t <= 0]

        if not wins or not losses:
            # Can't calculate Kelly without both wins and losses
            return self.config.fractional_pct / 100.0

        # Win probability
        p = len(wins) / len(recent)
        q = 1 - p

        # Average win/loss ratio
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        if avg_loss == 0:
            return self.config.kelly_cap

        b = avg_win / avg_loss

        # Kelly fraction
        kelly = (p * b - q) / b

        # Kelly can be negative (don't trade) or > 1 (capped)
        return max(0.0, min(kelly, 1.0))

    def describe(self) -> str:
        """Return human-readable description of current sizing."""
        if self.config.mode == 'fixed':
            return f"Fixed sizing: ${self.config.fixed_size_usdt:.2f} per trade"
        elif self.config.mode == 'fractional':
            return f"Fractional sizing: {self.config.fractional_pct}% of equity (${self.current_equity:.2f})"
        else:
            kelly = self._calculate_kelly([])
            return f"Kelly sizing: {kelly*100:.1f}% of equity (cap: {self.config.kelly_cap*100:.0f}%)"
