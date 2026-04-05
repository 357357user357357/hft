"""Signal backtesting framework.

Backtest the mathematical signals themselves (not just the 4 algorithms).
Answers: "Does the Poincaré regime detector actually predict future moves?"

Usage:
    from signal_backtest import SignalBacktest, SignalConfig

    config = SignalConfig(
        signal_type="composite",  # composite, poincare, hurst, simons, etc.
        lookback_bars=100,
        hold_bars=10,
        threshold=0.2,
    )
    backtest = SignalBacktest(config)
    stats = backtest.run(prices=[...], volumes=[...])
    stats.print_summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import math
import time

from instrument_index import InstrumentIndexer, InstrumentScorecard
from regime_detector import RegimeDetector, MarketRegime


@dataclass
class SignalConfig:
    """Configuration for signal backtesting."""
    # Which signal to test
    signal_type: str = "composite"  # composite, poincare, hurst, simons, geometry, polar, etc.

    # Entry/exit rules
    lookback_bars: int = 100        # Bars of history for signal computation
    hold_bars: int = 10             # Bars to hold after entry
    threshold: float = 0.2          # Signal threshold for entry (|score| > threshold)

    # Position sizing
    position_size_pct: float = 0.05  # 5% of equity per trade

    # Long/short
    allow_long: bool = True
    allow_short: bool = True

    # Stride: recompute signal every N bars instead of every bar.
    # Topology signals (poincaré, torsion) are expensive — stride=5 gives
    # 5× speedup with negligible accuracy loss since regime doesn't change
    # bar-by-bar.  Fast signals (hurst, momentum) default stride=1.
    signal_stride: int = 1


@dataclass
class SignalTrade:
    """Single signal trade."""
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    signal_score: float
    signal_type: str
    side: str  # "long" or "short"
    pnl_pct: float
    hold_bars: int


@dataclass
class SignalStats:
    """Backtest statistics for signal testing."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[SignalTrade] = field(default_factory=list)

    def record_trade(self, trade: SignalTrade) -> None:
        self.trades.append(trade)
        self.total_trades += 1

        if trade.pnl_pct > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

    def compute_metrics(self) -> None:
        if not self.trades:
            return

        # Win rate
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

        # Total and average PnL
        pnls = [t.pnl_pct for t in self.trades]
        self.total_pnl_pct = sum(pnls)
        self.avg_pnl_pct = self.total_pnl_pct / len(pnls)

        # Avg win / avg loss
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        self.avg_win_pct = sum(wins) / len(wins) if wins else 0.0
        self.avg_loss_pct = sum(losses) / len(losses) if losses else 0.0

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        self.max_drawdown_pct = max_dd

        # Sharpe ratio (annualized, assuming daily bars)
        if len(pnls) > 1:
            avg = sum(pnls) / len(pnls)
            variance = sum((p - avg) ** 2 for p in pnls) / (len(pnls) - 1)
            std = math.sqrt(variance) if variance > 0 else 0.0
            self.sharpe_ratio = (avg / std * math.sqrt(252)) if std > 0 else 0.0

    def print_summary(self, title: str = "Signal Backtest") -> None:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"  Total trades:      {self.total_trades}")
        print(f"  Win rate:          {self.win_rate*100:.1f}%")
        print(f"  Total PnL:         {self.total_pnl_pct:+.2f}%")
        print(f"  Avg PnL/trade:     {self.avg_pnl_pct:+.3f}%")
        print(f"  Avg win:           {self.avg_win_pct:+.3f}%")
        print(f"  Avg loss:          {self.avg_loss_pct:+.3f}%")
        print(f"  Profit factor:     {self.profit_factor:.2f}")
        print(f"  Max drawdown:      {self.max_drawdown_pct:.2f}%")
        print(f"  Sharpe ratio:      {self.sharpe_ratio:.2f}")
        print(f"{'='*60}\n")


class SignalBacktest:
    """
    Backtest a single signal against historical data.

    The signal generates entries when |score| > threshold:
    - score > +threshold → go long
    - score < -threshold → go short

    Exit after hold_bars or when signal flips sign.
    """

    def __init__(self, config: SignalConfig):
        self.config = config
        self._indexer = InstrumentIndexer()
        self._detector = RegimeDetector()
        self._stats = SignalStats()

    def run(self, prices: List[float], volumes: Optional[List[float]] = None) -> SignalStats:
        """Run signal backtest."""
        if len(prices) < self.config.lookback_bars + self.config.hold_bars:
            raise ValueError(f"Need at least {self.config.lookback_bars + self.config.hold_bars} bars")

        self._stats = SignalStats()
        volumes = volumes or [1.0] * len(prices)

        in_position = False
        position_side = None  # "long" or "short"
        entry_bar = 0
        entry_price = 0.0
        entry_score = 0.0
        score = 0.0  # cached signal value (reused across stride steps)
        stride = max(1, self.config.signal_stride)

        for i in range(self.config.lookback_bars, len(prices)):
            # Recompute signal only every `stride` bars to save time on
            # expensive signals (poincaré, torsion).  Regime doesn't flip
            # every tick so reusing the last value is fine.
            if (i - self.config.lookback_bars) % stride == 0:
                window_prices = prices[max(0, i - self.config.lookback_bars):i + 1]
                window_volumes = volumes[max(0, i - self.config.lookback_bars):i + 1]
                try:
                    score = self._indexer.compute_signal(
                        self.config.signal_type, window_prices, window_volumes
                    )
                except Exception:
                    score = 0.0

            # Check for exit
            if in_position:
                bars_held = i - entry_bar
                should_exit = False

                # Exit after hold period
                if bars_held >= self.config.hold_bars:
                    should_exit = True

                # Exit if signal flips against position
                if position_side == "long" and score < -self.config.threshold:
                    should_exit = True
                if position_side == "short" and score > self.config.threshold:
                    should_exit = True

                if should_exit:
                    exit_price = prices[i]
                    if position_side == "long":
                        pnl = (exit_price - entry_price) / entry_price * 100
                    else:
                        pnl = (entry_price - exit_price) / entry_price * 100

                    trade = SignalTrade(
                        entry_bar=entry_bar,
                        exit_bar=i,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        signal_score=entry_score,
                        signal_type=self.config.signal_type,
                        side=position_side,
                        pnl_pct=pnl,
                        hold_bars=bars_held,
                    )
                    self._stats.record_trade(trade)

                    in_position = False
                    position_side = None

            # Check for entry (if not in position)
            if not in_position:
                if score > self.config.threshold and self.config.allow_long:
                    in_position = True
                    position_side = "long"
                    entry_bar = i
                    entry_price = prices[i]
                    entry_score = score

                elif score < -self.config.threshold and self.config.allow_short:
                    in_position = True
                    position_side = "short"
                    entry_bar = i
                    entry_price = prices[i]
                    entry_score = score

        self._stats.compute_metrics()
        return self._stats

    def _extract_score(self, card: InstrumentScorecard) -> float:
        """Extract the relevant score from the scorecard."""
        signal = self.config.signal_type.lower()

        if signal == "composite":
            return card.composite
        elif signal == "poincare":
            return card.topology.score
        elif signal == "torsion":
            return card.torsion.score
        elif signal == "algebraic":
            return card.algebraic.score * card.algebraic.direction
        elif signal == "geometry":
            return card.geometry.score
        elif signal == "polar":
            return card.polar.score
        elif signal == "hurst":
            return card.hurst.score
        elif signal == "volatility":
            return card.volatility.score
        elif signal == "momentum":
            return card.momentum.score
        elif signal == "simons":
            return card.simons.score
        elif signal == "order_flow":
            return card.order_flow.score
        elif signal == "spectral":
            return card.spectral.score
        elif signal == "fel":
            return card.fel.score
        elif signal == "quaternion":
            return card.quaternion.score
        else:
            return card.composite


def compare_signals(prices: List[float], volumes: Optional[List[float]] = None) -> Dict[str, SignalStats]:
    """Compare all signal dimensions side by side."""
    signals = [
        "composite", "poincare", "torsion", "algebraic", "geometry",
        "polar", "hurst", "momentum", "simons", "spectral", "fel"
    ]

    results = {}
    for sig in signals:
        config = SignalConfig(signal_type=sig, lookback_bars=100, hold_bars=10, threshold=0.15)
        backtest = SignalBacktest(config)
        stats = backtest.run(prices, volumes)
        results[sig] = stats
        print(f"{sig:15s}: win_rate={stats.win_rate*100:5.1f}%  "
              f"total_pnl={stats.total_pnl_pct:+7.2f}%  "
              f"sharpe={stats.sharpe_ratio:6.2f}")

    return results


if __name__ == "__main__":
    # Demo with synthetic data
    import random
    random.seed(42)

    # Generate trending + mean-reverting synthetic series
    prices = [100.0]
    for i in range(500):
        # OU-like mean reversion + trend
        drift = 0.0002 * i  # slight trend
        mean_rev = -0.01 * (prices[-1] - 100)  # mean reversion to 100
        noise = random.gauss(0, 0.02)
        ret = drift + mean_rev + noise
        prices.append(prices[-1] * (1 + ret))

    volumes = [random.uniform(100, 1000) for _ in range(len(prices))]

    print("\n=== Signal Backtest Demo ===\n")

    # Test individual signals
    config = SignalConfig(signal_type="composite", lookback_bars=60, hold_bars=10, threshold=0.15)
    backtest = SignalBacktest(config)
    stats = backtest.run(prices, volumes)
    stats.print_summary("Composite Signal")

    # Compare all signals
    print("\n=== Comparing All Signals ===\n")
    compare_signals(prices, volumes)
