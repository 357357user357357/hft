"""Signal gate: connects the 19-dimension scorecard to the backtest loop.

Maintains a rolling price window, periodically recomputes the instrument
scorecard, and provides a should_enter() check that algorithms call before
opening positions.  Also exposes the detected regime for parameter switching.

Usage:
    from signal_gate import SignalGate

    gate = SignalGate("BTCUSDT")

    # In your trade loop:
    gate.on_trade(price, volume, timestamp_ns)

    # Before opening a position:
    allow, confidence = gate.should_enter(Side.Buy)
    if not allow:
        return  # signal says don't enter

    # For regime-adaptive parameters:
    regime = gate.regime
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from hft_types import Side
from instrument_index import InstrumentIndexer, InstrumentScorecard
from regime_detector import RegimeDetector, MarketRegime

logger = logging.getLogger("hft")


@dataclass
class GateConfig:
    """Configuration for the signal gate."""
    # How often to recompute the scorecard (seconds)
    eval_interval_secs: float = 5.0

    # Rolling window size (number of prices to keep)
    window_size: int = 500

    # Minimum prices before first evaluation
    min_prices: int = 60

    # Composite score thresholds
    # Buy allowed when composite > buy_threshold
    buy_threshold: float = -0.10
    # Sell allowed when composite < sell_threshold
    sell_threshold: float = 0.10

    # Hold zone: if |composite| < hold_zone, reduce confidence
    hold_zone: float = 0.05

    # Whether to block entries entirely on bad signals, or just reduce size
    hard_block: bool = False

    # Enable regime detection for parameter adaptation
    regime_enabled: bool = True

    # How often to re-detect regime (seconds)
    regime_interval_secs: float = 30.0


class SignalGate:
    """
    Rolling signal gate for the backtest loop.

    Call on_trade() for every incoming trade.
    Call should_enter() before opening a position.
    Read .regime for current market regime.
    Read .scorecard for the latest full scorecard.
    """

    def __init__(self, symbol: str, config: Optional[GateConfig] = None):
        self.symbol = symbol
        self.config = config or GateConfig()

        self._indexer = InstrumentIndexer()
        self._detector = RegimeDetector()

        # Rolling data
        self._prices: Deque[float] = deque(maxlen=self.config.window_size)
        self._volumes: Deque[float] = deque(maxlen=self.config.window_size)

        # State
        self._last_eval_ns: int = 0
        self._last_regime_ns: int = 0
        self._scorecard: Optional[InstrumentScorecard] = None
        self._regime: MarketRegime = MarketRegime.NEUTRAL
        self._trade_count: int = 0

    @property
    def scorecard(self) -> Optional[InstrumentScorecard]:
        return self._scorecard

    @property
    def regime(self) -> MarketRegime:
        return self._regime

    @property
    def composite(self) -> float:
        if self._scorecard is None:
            return 0.0
        return self._scorecard.composite

    def on_trade(self, price: float, volume: float, timestamp_ns: int) -> None:
        """Feed a new trade into the gate. Call this for every AggTrade."""
        self._prices.append(price)
        self._volumes.append(volume)
        self._trade_count += 1

        # Check if it's time to re-evaluate
        elapsed_secs = (timestamp_ns - self._last_eval_ns) / 1e9
        if (len(self._prices) >= self.config.min_prices
                and elapsed_secs >= self.config.eval_interval_secs):
            self._evaluate(timestamp_ns)

        # Check regime re-detection
        if self.config.regime_enabled:
            regime_elapsed = (timestamp_ns - self._last_regime_ns) / 1e9
            if (len(self._prices) >= self.config.min_prices
                    and regime_elapsed >= self.config.regime_interval_secs):
                self._detect_regime(timestamp_ns)

    def should_enter(self, side: Side) -> Tuple[bool, float]:
        """
        Check whether the signal gate allows opening a position.

        Returns:
            (allowed, confidence_multiplier)
            - allowed: True if entry is permitted
            - confidence_multiplier: 0.0 to 1.0 scaling for position size
        """
        if self._scorecard is None:
            # No scorecard yet — allow entry with default confidence
            return True, 1.0

        composite = self._scorecard.composite

        if side == Side.Buy:
            if self.config.hard_block and composite < self.config.buy_threshold:
                return False, 0.0
            # Scale confidence: higher composite = more confident for buys
            confidence = self._composite_to_confidence(composite, side)
            return True, confidence

        else:  # Sell
            if self.config.hard_block and composite > self.config.sell_threshold:
                return False, 0.0
            confidence = self._composite_to_confidence(composite, side)
            return True, confidence

    def force_evaluate(self) -> Optional[InstrumentScorecard]:
        """Force an immediate scorecard evaluation (e.g., at start of backtest)."""
        if len(self._prices) < self.config.min_prices:
            return None
        self._evaluate(0)
        return self._scorecard

    def _evaluate(self, timestamp_ns: int) -> None:
        """Recompute the instrument scorecard."""
        prices = list(self._prices)
        volumes = list(self._volumes)

        try:
            self._scorecard = self._indexer.update(
                self.symbol, prices, volumes=volumes
            )
            self._last_eval_ns = timestamp_ns
            logger.debug(
                "SignalGate %s: composite=%+.3f regime=%s (%.1fms)",
                self.symbol, self._scorecard.composite,
                self._regime.value, self._scorecard.compute_ms,
            )
        except Exception:
            logger.debug("SignalGate eval failed", exc_info=True)

    def _detect_regime(self, timestamp_ns: int) -> None:
        """Re-detect market regime from price data."""
        prices = list(self._prices)
        if len(prices) < 100:
            return

        try:
            # Build minimal AggTrade-like objects for RegimeDetector
            from data import AggTrade
            fake_trades = [
                AggTrade(
                    agg_trade_id=i,
                    price=p,
                    quantity=self._volumes[i] if i < len(self._volumes) else 1.0,
                    first_trade_id=0,
                    last_trade_id=0,
                    transact_time=0,
                    is_buyer_maker=False,
                )
                for i, p in enumerate(prices[-200:])
            ]
            self._regime = self._detector.detect_regime(fake_trades, min_trades=100)
            self._last_regime_ns = timestamp_ns
        except Exception:
            logger.debug("Regime detection failed", exc_info=True)

    def _composite_to_confidence(self, composite: float, side: Side) -> float:
        """Map composite score to a 0.0–1.0 confidence multiplier."""
        if side == Side.Buy:
            # For buys: positive composite = high confidence
            # composite of +0.5 → confidence 1.0
            # composite of  0.0 → confidence 0.5
            # composite of -0.5 → confidence 0.1
            raw = 0.5 + composite
        else:
            # For sells: negative composite = high confidence
            raw = 0.5 - composite

        # Clamp to [0.1, 1.0] — never fully zero (let the algo decide)
        return max(0.1, min(1.0, raw))
