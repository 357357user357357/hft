"""Regime-aware parameter switching for HFT algorithms.

Uses FelSemigroupSignal to detect market regime from price data,
then switches algorithm parameters to match the detected regime.

Regimes:
- **mean-reversion**: Low genus, small generators → price oscillates
  Best for: Averages, Shot (tight TP/SL)

- **trending**: High genus, large K-ratio → persistent directional move
  Best for: Vector, Depth Shot (wider TP, trailing stops)

- **neutral**: Intermediate structure → no clear pattern
  Best for: Reduced position sizing, wider stops

Usage:
    from regime_detector import RegimeDetector, RegimeConfig

    detector = RegimeDetector(RegimeConfig())
    regime = detector.detect_regime(trades)
    params = detector.get_shot_params(regime)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger("hft")

from fel_signal import FelSemigroupSignal, FelReport
from data import AggTrade


class MarketRegime(Enum):
    """Detected market regime."""
    MEAN_REVERSION = "mean-reversion"  # Oscillating, range-bound
    TRENDING = "trending"              # Persistent directional move
    NEUTRAL = "neutral"                # No clear pattern


@dataclass
class RegimeConfig:
    """Regime detection thresholds."""
    # Genus thresholds
    low_genus_threshold: int = 5       # Below = mean-reversion
    high_genus_threshold: int = 20     # Above = potentially trending

    # K-ratio threshold for trending (K_2 / K_0)
    k_ratio_trending: float = 30.0

    # Minimum generator for trending
    min_generator_trending: int = 5

    # Discrepancy threshold for mean-reversion
    discrepancy_threshold: float = 0.5


@dataclass
class ShotParams:
    """Shot algorithm parameters for a regime."""
    distance_pct: float
    buffer_pct: float
    tp_pct: float
    sl_pct: float
    follow_delay_secs: float
    replace_delay_secs: float


@dataclass
class AveragesParams:
    """Averages algorithm parameters for a regime."""
    long_period_secs: float
    short_period_secs: float
    trigger_min_pct: float
    trigger_max_pct: float
    order_distance_pct: float
    tp_pct: float
    sl_pct: float


@dataclass
class VectorParams:
    """Vector algorithm parameters for a regime."""
    frame_size_secs: float
    min_spread_pct: float
    order_distance_pct: float
    tp_spread_pct: float
    sl_pct: float
    max_orders: int


@dataclass
class DepthShotParams:
    """Depth Shot algorithm parameters for a regime."""
    target_volume: float
    min_distance_pct: float
    max_distance_pct: float
    tp_percentage: float
    sl_pct: float


class RegimeDetector:
    """Detects market regime and provides regime-optimal parameters."""

    # Pre-defined parameter sets for each regime
    SHOT_PARAMS = {
        MarketRegime.MEAN_REVERSION: ShotParams(
            distance_pct=0.05,      # Tighter entries
            buffer_pct=0.03,
            tp_pct=0.04,            # Quick profits
            sl_pct=0.10,            # Tight stops
            follow_delay_secs=0.03, # Faster reaction
            replace_delay_secs=0.03,
        ),
        MarketRegime.TRENDING: ShotParams(
            distance_pct=0.12,      # Wider entries (catch bigger moves)
            buffer_pct=0.06,
            tp_pct=0.15,            # Let profits run
            sl_pct=0.25,            # Wider stops
            follow_delay_secs=0.08, # Slower reaction
            replace_delay_secs=0.08,
        ),
        MarketRegime.NEUTRAL: ShotParams(
            distance_pct=0.08,
            buffer_pct=0.04,
            tp_pct=0.06,
            sl_pct=0.15,
            follow_delay_secs=0.05,
            replace_delay_secs=0.05,
        ),
    }

    AVERAGES_PARAMS = {
        MarketRegime.MEAN_REVERSION: AveragesParams(
            long_period_secs=20.0,   # Faster MAs
            short_period_secs=3.0,
            trigger_min_pct=-0.08,
            trigger_max_pct=-0.015,
            order_distance_pct=-0.03,
            tp_pct=0.03,
            sl_pct=0.08,
        ),
        MarketRegime.TRENDING: AveragesParams(
            long_period_secs=60.0,   # Slower MAs
            short_period_secs=10.0,
            trigger_min_pct=-0.15,
            trigger_max_pct=-0.03,
            order_distance_pct=-0.08,
            tp_pct=0.08,
            sl_pct=0.15,
        ),
        MarketRegime.NEUTRAL: AveragesParams(
            long_period_secs=30.0,
            short_period_secs=5.0,
            trigger_min_pct=-0.1,
            trigger_max_pct=-0.02,
            order_distance_pct=-0.05,
            tp_pct=0.05,
            sl_pct=0.1,
        ),
    }

    VECTOR_PARAMS = {
        MarketRegime.MEAN_REVERSION: VectorParams(
            frame_size_secs=0.3,
            min_spread_pct=0.03,
            order_distance_pct=8.0,
            tp_spread_pct=70.0,
            sl_pct=0.03,
            max_orders=1,
        ),
        MarketRegime.TRENDING: VectorParams(
            frame_size_secs=0.8,
            min_spread_pct=0.08,
            order_distance_pct=15.0,
            tp_spread_pct=90.0,
            sl_pct=0.08,
            max_orders=3,
        ),
        MarketRegime.NEUTRAL: VectorParams(
            frame_size_secs=0.5,
            min_spread_pct=0.05,
            order_distance_pct=10.0,
            tp_spread_pct=80.0,
            sl_pct=0.05,
            max_orders=2,
        ),
    }

    DEPTH_SHOT_PARAMS = {
        MarketRegime.MEAN_REVERSION: DepthShotParams(
            target_volume=30.0,      # Smaller targets
            min_distance_pct=0.03,
            max_distance_pct=0.3,
            tp_percentage=40.0,
            sl_pct=0.10,
        ),
        MarketRegime.TRENDING: DepthShotParams(
            target_volume=80.0,      # Larger targets
            min_distance_pct=0.08,
            max_distance_pct=0.8,
            tp_percentage=70.0,
            sl_pct=0.20,
        ),
        MarketRegime.NEUTRAL: DepthShotParams(
            target_volume=50.0,
            min_distance_pct=0.05,
            max_distance_pct=0.5,
            tp_percentage=50.0,
            sl_pct=0.15,
        ),
    }

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._fel_signal = FelSemigroupSignal()
        self._last_regime: Optional[MarketRegime] = None
        self._last_report: Optional[FelReport] = None

    def detect_regime(self, trades: List[AggTrade],
                      min_trades: int = 100) -> MarketRegime:
        """
        Detect market regime from trade data.

        Args:
            trades: List of AggTrade objects
            min_trades: Minimum trades needed for reliable detection

        Returns:
            Detected market regime
        """
        if len(trades) < min_trades:
            return MarketRegime.NEUTRAL

        prices = [t.price for t in trades[-min_trades:]]

        # ── Primary: direct statistical regime test ───────────────────────
        # Lag-1 autocorrelation of returns: negative = mean-reverting,
        # positive = trending.  More robust than FEL on short windows.
        try:
            rets = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            n    = len(rets)
            mean_r = sum(rets) / n
            lag1_cov = sum((rets[i] - mean_r) * (rets[i-1] - mean_r)
                           for i in range(1, n))
            var_r    = sum((r - mean_r)**2 for r in rets)
            autocorr = lag1_cov / var_r if var_r > 0 else 0.0
        except Exception:
            autocorr = 0.0

        # ── Secondary: FEL semigroup (enriches borderline cases) ──────────
        try:
            report = self._fel_signal.eval(prices)
            self._last_report = report
            fel_regime = report.regime   # "mean-reversion" | "trending" | "neutral"
        except Exception:
            logger.debug("FEL eval failed in regime detection", exc_info=True)
            fel_regime = "neutral"

        # ── Decision: autocorrelation is primary, FEL breaks ties ─────────
        if autocorr < -0.05:
            regime = MarketRegime.MEAN_REVERSION
        elif autocorr > 0.05:
            regime = MarketRegime.TRENDING
        else:
            # Borderline — use FEL
            if fel_regime == "trending":
                regime = MarketRegime.TRENDING
            elif fel_regime == "mean-reversion":
                regime = MarketRegime.MEAN_REVERSION
            else:
                regime = MarketRegime.NEUTRAL

        self._last_regime = regime
        return regime

    def get_shot_params(self, regime: MarketRegime) -> ShotParams:
        """Get Shot algorithm parameters for regime."""
        return self.SHOT_PARAMS[regime]

    def get_averages_params(self, regime: MarketRegime) -> AveragesParams:
        """Get Averages algorithm parameters for regime."""
        return self.AVERAGES_PARAMS[regime]

    def get_vector_params(self, regime: MarketRegime) -> VectorParams:
        """Get Vector algorithm parameters for regime."""
        return self.VECTOR_PARAMS[regime]

    def get_depth_shot_params(self, regime: MarketRegime) -> DepthShotParams:
        """Get Depth Shot algorithm parameters for regime."""
        return self.DEPTH_SHOT_PARAMS[regime]

    def describe(self, regime: MarketRegime) -> str:
        """Return human-readable regime description."""
        descriptions = {
            MarketRegime.MEAN_REVERSION: (
                "MEAN-REVERSION: Low genus, oscillating price. "
                "Use tight entries, quick profits, smaller stops."
            ),
            MarketRegime.TRENDING: (
                "TRENDING: High genus, persistent moves. "
                "Use wider entries, let profits run, wider stops."
            ),
            MarketRegime.NEUTRAL: (
                "NEUTRAL: No clear pattern. "
                "Use standard parameters, reduced sizing."
            ),
        }
        return descriptions[regime]
