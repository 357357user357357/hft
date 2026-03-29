"""Multi-dimensional instrument index.

Computes a full scorecard for each financial instrument across every
mathematical and microstructure signal domain, so you can rank/compare
instruments and see which signals are firing at a glance.

Index dimensions (14 total)
---------------------------
Math / topology
  topology      : Poincaré score, regime, Betti numbers
  torsion       : Whitehead torsion ratio, regime-change flag
  algebraic     : Hecke L-value magnitude, zeta signal
  geometry      : Frenet-Serret curvature / torsion
  polar         : Radial expansion, angular velocity, cycle phase
  number_theory : p-adic valuation spread (price roughness)
  graph         : Price-transition graph density
  spectral      : Multi-scale Hankel spectral score (spectral_window.py)
  fel           : Numerical semigroup / Erdős gap analysis (fel_signal.py)
  quaternion    : 4-D trading state rotation (quaternions.py)

Classic finance / microstructure
  hurst         : Hurst exponent (H < 0.5 = MR, H > 0.5 = trend)
  volatility    : Realized vol regime (calm / normal / explosive)
  order_flow    : Buy vs sell volume imbalance (Lee-Ready proxy)
  volume_profile: VWAP / VPOC deviation
  microstructure: Amihud illiquidity, Roll spread, Kyle's λ
  momentum      : RSI + multi-period price momentum
  autocorr      : Lag-1/5/10 return autocorrelation
  funding       : Funding-rate proxy (skewness-based)

Usage
-----
    from instrument_index import InstrumentIndexer

    indexer = InstrumentIndexer()
    card = indexer.update("BTCUSDT", prices=[...], volumes=[...])
    print(card)

    ranking = indexer.rank_by("composite")
    for symbol, score in ranking:
        print(f"{symbol:12s}  composite={score:+.3f}")
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Repo math modules ─────────────────────────────────────────────────────────
from poincare_trading import poincare_analysis
from whitehead_signal import whitehead_analysis
from hecke_operators import HeckeAlgebra
from frenet_serret import FrenetSerretAnalyzer, analyze_price_series as frenet_analyze, curvature_to_signal
from polar_features import PolarExtractor, PolarSignalGenerator, describe_regime
from p_adic import PAdicNumber, padic_from_integer
from spectral_window import spectral_report
from hecke_operators import sieve_primes
from fel_signal import FelSemigroupSignal, FelSignalConfig
from quaternions import TradingStateQuaternion
from simons_sde import SimonsSDEReport

# ── New classic finance signals ───────────────────────────────────────────────
from market_signals import (
    HurstResult,
    VolatilityResult,
    OrderFlowResult,
    VolumeProfileResult,
    MicrostructureResult,
    MomentumResult,
    AutocorrResult,
    FundingProxyResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dimension index dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopologyIndex:
    score: float
    regime: str
    simply_connected: bool
    beta0: int
    beta1: int
    beta2: int
    mean_ricci: float
    neg_ricci_fraction: float


@dataclass
class TorsionIndex:
    score: float
    signal: str
    torsion_ratio: float
    num_regimes: int


@dataclass
class AlgebraicIndex:
    score: float
    l_value_abs: float
    zeta_signal: str
    direction: int          # +1 bullish, -1 bearish


@dataclass
class GeometryIndex:
    curvature: float
    torsion: float
    curvature_signal: str
    score: float


@dataclass
class PolarIndex:
    r: float
    theta: float
    angular_velocity: float   # dtheta_dt
    radial_velocity: float    # dr_dt
    regime: str
    score: float


@dataclass
class NumberTheoryIndex:
    roughness: float
    prime_used: int
    mean_valuation: float
    score: float


@dataclass
class GraphIndex:
    node_count: int
    edge_count: int
    density: float
    score: float


@dataclass
class SpectralIndex:
    """Multi-scale Hankel spectral index (spectral_window.py)."""
    overall_score: float
    num_significant: int
    dominant_scale: Optional[int]
    score: float


@dataclass
class FelIndex:
    """Numerical semigroup / Erdős gap analysis index."""
    signal_strength: str    # "strong" | "moderate" | "weak" | "none"
    regime: str
    genus: int
    ek_z_score: float
    score: float


@dataclass
class QuaternionIndex:
    """4-D trading state quaternion index."""
    state_rotation: float   # cumulative rotation (large = regime shift)
    regime: str             # "trending" | "mean_reverting" | "volatile" | "quiet"
    score: float


# ── Classic finance dimensions ────────────────────────────────────────────────

@dataclass
class HurstIndex:
    H: float
    regime: str
    score: float


@dataclass
class VolatilityIndex:
    realized_vol: float
    regime: str
    vol_percentile: float
    score: float


@dataclass
class OrderFlowIndex:
    ofi_ratio: float
    buy_volume: float
    sell_volume: float
    score: float


@dataclass
class VolumeProfileIndex:
    vwap: float
    vpoc: float
    price_vs_vwap: float
    price_vs_vpoc: float
    score: float


@dataclass
class MicrostructureIndex:
    amihud: float
    roll_spread: float
    kyle_lambda: float
    score: float


@dataclass
class MomentumIndex:
    rsi: float
    rsi_signal: str
    momentum_20: float
    score: float


@dataclass
class AutocorrIndex:
    lag1: float
    lag5: float
    score: float


@dataclass
class FundingIndex:
    proxy_rate: float
    signal: str
    score: float


@dataclass
class SimonsIndex:
    """Jim Simons SDE suite composite index."""
    gbm_score: float         # Itô GBM drift signal
    ou_score: float          # Ornstein-Uhlenbeck mean reversion
    ou_z_score: float        # OU z-score (deviation from equilibrium)
    ou_half_life: float      # bars to close 50% of gap
    regime: str              # "bull" | "bear" (HMM regime switch)
    kalman_drift: float      # Kalman-estimated drift per bar
    hmm_vol_state: str       # "low_vol" | "high_vol"
    kelly_f_star: float      # Kelly optimal fraction
    fourier_period: float    # dominant cycle length (bars)
    fourier_position: str    # "peak" | "trough" | "rising" | "falling"
    score: float             # composite of all 8 models


# ─────────────────────────────────────────────────────────────────────────────
# Master scorecard
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstrumentScorecard:
    """Complete multi-dimensional scorecard for one instrument."""
    symbol: str
    timestamp: float

    # Math / topology
    topology: TopologyIndex
    torsion: TorsionIndex
    algebraic: AlgebraicIndex
    geometry: GeometryIndex
    polar: PolarIndex
    number_theory: NumberTheoryIndex
    graph: GraphIndex
    spectral: SpectralIndex
    fel: FelIndex
    quaternion: QuaternionIndex

    # Classic finance
    hurst: HurstIndex
    volatility: VolatilityIndex
    order_flow: OrderFlowIndex
    volume_profile: VolumeProfileIndex
    microstructure: MicrostructureIndex
    momentum: MomentumIndex
    autocorr: AutocorrIndex
    funding: FundingIndex
    simons: SimonsIndex

    composite: float
    compute_ms: float

    # ── display ───────────────────────────────────────────────────────────────
    def __str__(self) -> str:
        W = 58
        bar = "─" * W

        def row(label: str, score: float, info: str) -> str:
            label_part = f"  {label:<16s} {score:+.3f}"
            info_part = f"  {info[:26]:<26s}"
            line = label_part + info_part
            # pad to exactly W chars inside the box
            inner = line.ljust(W)
            return f"║{inner}║"

        lines = [
            f"╔{bar}╗",
            f"║  {self.symbol:<20s}  composite = {self.composite:+.3f}{'':>13s}║",
            f"╠{bar}╣",
            f"║{'  ── Math / Topology ──':<{W}}║",
            row("topology",    self.topology.score,    self.topology.regime),
            row("torsion",     self.torsion.score,     self.torsion.signal),
            row("algebraic",   self.algebraic.score * self.algebraic.direction,
                               self.algebraic.zeta_signal),
            row("geometry",    self.geometry.score,    self.geometry.curvature_signal),
            row("polar",       self.polar.score,       self.polar.regime[:26]),
            row("number_theory", self.number_theory.score,
                               f"rough={self.number_theory.roughness:.3f}"),
            row("graph",       self.graph.score,       f"density={self.graph.density:.3f}"),
            row("spectral",    self.spectral.score,    f"sig={self.spectral.num_significant}"),
            row("fel",         self.fel.score,         self.fel.signal_strength),
            row("quaternion",  self.quaternion.score,  self.quaternion.regime),
            f"╠{bar}╣",
            f"║{'  ── Classic Finance ──':<{W}}║",
            row("hurst",       self.hurst.score,       f"H={self.hurst.H:.3f} {self.hurst.regime}"),
            row("volatility",  self.volatility.score,  self.volatility.regime),
            row("order_flow",  self.order_flow.score,  f"OFI={self.order_flow.ofi_ratio:+.3f}"),
            row("vol_profile", self.volume_profile.score,
                               f"vs_vwap={self.volume_profile.price_vs_vwap:+.4f}"),
            row("microstr.",   self.microstructure.score,
                               f"illiq={self.microstructure.amihud:.2e}"),
            row("momentum",    self.momentum.score,    f"RSI={self.momentum.rsi:.1f} {self.momentum.rsi_signal}"),
            row("autocorr",    self.autocorr.score,    f"lag1={self.autocorr.lag1:+.3f}"),
            row("funding",     self.funding.score,     self.funding.signal),
            f"╠{bar}╣",
            f"║{'  ── Simons SDE Models ──':<{W}}║",
            row("SDE/GBM",     self.simons.gbm_score,
                               f"kelly_f*={self.simons.kelly_f_star:.2f}"),
            row("OU",          self.simons.ou_score,
                               f"z={self.simons.ou_z_score:+.2f} hl={self.simons.ou_half_life:.1f}b"),
            row("Kalman/HMM",  self.simons.score,
                               f"{self.simons.regime} {self.simons.hmm_vol_state}"),
            row("Fourier",     self.simons.score,
                               f"T={self.simons.fourier_period:.1f}b {self.simons.fourier_position}"),
            f"╠{bar}╣",
            f"║  computed in {self.compute_ms:.1f} ms{'':<{max(0, W - 19 - len(str(round(self.compute_ms, 1))))}}║",
            f"╚{bar}╝",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "composite": self.composite,
            "compute_ms": self.compute_ms,
        }
        for field_name in [
            "topology", "torsion", "algebraic", "geometry", "polar",
            "number_theory", "graph", "spectral", "fel", "quaternion",
            "hurst", "volatility", "order_flow", "volume_profile",
            "microstructure", "momentum", "autocorr", "funding", "simons",
        ]:
            d[field_name] = asdict(getattr(self, field_name))
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Dimension weights  (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────
_WEIGHTS: Dict[str, float] = {
    # math
    "topology":      0.08,
    "torsion":       0.05,
    "algebraic":     0.05,
    "geometry":      0.05,
    "polar":         0.05,
    "number_theory": 0.03,
    "graph":         0.03,
    "spectral":      0.04,
    "fel":           0.03,
    "quaternion":    0.04,
    # classic finance
    "hurst":         0.07,
    "volatility":    0.05,
    "order_flow":    0.07,
    "volume_profile":0.04,
    "microstructure":0.03,
    "momentum":      0.07,
    "autocorr":      0.07,
    "funding":       0.04,
    "simons":        0.11,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, sum(_WEIGHTS.values())


# ─────────────────────────────────────────────────────────────────────────────
# Indexer
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentIndexer:
    """
    Maintains a live scorecard for every tracked symbol.

    Call ``update(symbol, prices, volumes, trades)`` whenever new data arrives.
    Call ``rank_by(dimension)`` to get a sorted leaderboard.
    Call ``compare(symbols)`` for a side-by-side table.
    """

    def __init__(self, hecke_max_n: int = 20, hecke_weight: int = 2):
        self._hecke = HeckeAlgebra(max_n=hecke_max_n, weight=hecke_weight)
        self._frenet = FrenetSerretAnalyzer()
        self._polar_extractor = PolarExtractor()
        self._polar_signal_gen = PolarSignalGenerator()
        self._fel = FelSemigroupSignal(FelSignalConfig())
        self._quat: Dict[str, TradingStateQuaternion] = {}
        self._cards: Dict[str, InstrumentScorecard] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def update(
        self,
        symbol: str,
        prices: List[float],
        volumes: Optional[List[float]] = None,
        trades: Optional[List[Tuple[float, float, str]]] = None,
        embed_dim: int = 3,
        subsample: int = 60,
    ) -> InstrumentScorecard:
        """Recompute all dimension indexes for *symbol* and cache the result."""
        t0 = time.perf_counter()
        if len(prices) < 20:
            raise ValueError(f"Need at least 20 prices, got {len(prices)}")

        # ── math signals ──────────────────────────────────────────────────────
        topology    = self._compute_topology(prices, embed_dim, subsample)
        torsion     = self._compute_torsion(prices, embed_dim, subsample)
        algebraic   = self._compute_algebraic(prices)
        geometry    = self._compute_geometry(prices, volumes)
        polar       = self._compute_polar(prices)
        nt          = self._compute_number_theory(prices)
        graph       = self._compute_graph(prices, trades)
        spectral    = self._compute_spectral(prices)
        fel         = self._compute_fel(prices)
        quaternion  = self._compute_quaternion(symbol, prices, volumes)

        # ── Simons SDE models ─────────────────────────────────────────────────
        simons_idx  = self._compute_simons(prices)

        # ── classic finance signals ───────────────────────────────────────────
        hurst       = HurstResult.compute(prices)
        volatility  = VolatilityResult.compute(prices)
        order_flow  = OrderFlowResult.compute(prices, volumes, trades)
        vol_profile = VolumeProfileResult.compute(prices, volumes)
        microstr    = MicrostructureResult.compute(prices, volumes)
        momentum    = MomentumResult.compute(prices)
        autocorr    = AutocorrResult.compute(prices)
        funding     = FundingProxyResult.compute(prices)

        # ── wrap classic results into index types ─────────────────────────────
        hurst_idx   = HurstIndex(H=hurst.H, regime=hurst.regime, score=hurst.score)
        vol_idx     = VolatilityIndex(realized_vol=volatility.realized_vol,
                                      regime=volatility.regime,
                                      vol_percentile=volatility.vol_percentile,
                                      score=volatility.score)
        ofi_idx     = OrderFlowIndex(ofi_ratio=order_flow.ofi_ratio,
                                     buy_volume=order_flow.buy_volume,
                                     sell_volume=order_flow.sell_volume,
                                     score=order_flow.score)
        vp_idx      = VolumeProfileIndex(vwap=vol_profile.vwap,
                                         vpoc=vol_profile.vpoc,
                                         price_vs_vwap=vol_profile.price_vs_vwap,
                                         price_vs_vpoc=vol_profile.price_vs_vpoc,
                                         score=vol_profile.score)
        ms_idx      = MicrostructureIndex(amihud=microstr.amihud,
                                          roll_spread=microstr.roll_spread,
                                          kyle_lambda=microstr.kyle_lambda,
                                          score=microstr.score)
        mom_idx     = MomentumIndex(rsi=momentum.rsi,
                                    rsi_signal=momentum.rsi_signal,
                                    momentum_20=momentum.momentum_20,
                                    score=momentum.score)
        ac_idx      = AutocorrIndex(lag1=autocorr.lag1, lag5=autocorr.lag5,
                                    score=autocorr.score)
        fund_idx    = FundingIndex(proxy_rate=funding.proxy_rate,
                                   signal=funding.signal,
                                   score=funding.score)

        # ── composite score ───────────────────────────────────────────────────
        raw_scores = {
            "topology":      topology.score,
            "torsion":       torsion.score,
            "algebraic":     algebraic.score * algebraic.direction,
            "geometry":      geometry.score,
            "polar":         polar.score,
            "number_theory": nt.score,
            "graph":         graph.score,
            "spectral":      spectral.score,
            "fel":           fel.score,
            "quaternion":    quaternion.score,
            "hurst":         hurst_idx.score,
            "volatility":    vol_idx.score,
            "order_flow":    ofi_idx.score,
            "volume_profile":vp_idx.score,
            "microstructure":ms_idx.score,
            "momentum":      mom_idx.score,
            "autocorr":      ac_idx.score,
            "funding":       fund_idx.score,
            "simons":        simons_idx.score,
        }
        composite = sum(raw_scores[k] * _WEIGHTS[k] for k in _WEIGHTS)

        card = InstrumentScorecard(
            symbol=symbol,
            timestamp=time.time(),
            topology=topology,
            torsion=torsion,
            algebraic=algebraic,
            geometry=geometry,
            polar=polar,
            number_theory=nt,
            graph=graph,
            spectral=spectral,
            fel=fel,
            quaternion=quaternion,
            hurst=hurst_idx,
            volatility=vol_idx,
            order_flow=ofi_idx,
            volume_profile=vp_idx,
            microstructure=ms_idx,
            momentum=mom_idx,
            autocorr=ac_idx,
            funding=fund_idx,
            simons=simons_idx,
            composite=composite,
            compute_ms=(time.perf_counter() - t0) * 1000,
        )
        self._cards[symbol] = card
        return card

    def get(self, symbol: str) -> Optional[InstrumentScorecard]:
        return self._cards.get(symbol)

    def all_cards(self) -> Dict[str, InstrumentScorecard]:
        return dict(self._cards)

    def rank_by(self, dimension: str = "composite", descending: bool = True) -> List[Tuple[str, float]]:
        result = [(sym, self._extract_score(card, dimension))
                  for sym, card in self._cards.items()]
        result.sort(key=lambda x: x[1], reverse=descending)
        return result

    def compare(self, symbols: List[str]) -> str:
        dims = list(_WEIGHTS.keys()) + ["composite"]
        header = f"{'Dimension':<16s}" + "".join(f"{s:>12s}" for s in symbols)
        sep = "─" * (16 + 12 * len(symbols))
        rows = [sep, header, sep]
        for dim in dims:
            row = f"{dim:<16s}"
            for sym in symbols:
                card = self._cards.get(sym)
                row += f"{'N/A':>12s}" if card is None else f"{self._extract_score(card, dim):>+12.3f}"
            rows.append(row)
        rows.append(sep)
        return "\n".join(rows)

    # ── math dimension computations ───────────────────────────────────────────

    def _compute_topology(self, prices: List[float], embed_dim: int, subsample: int) -> TopologyIndex:
        try:
            r = poincare_analysis(prices, embed_dim=embed_dim, subsample=subsample)
            return TopologyIndex(score=r.poincare_score, regime=r.regime,
                                 simply_connected=r.simply_connected,
                                 beta0=r.beta0, beta1=r.beta1, beta2=r.beta2,
                                 mean_ricci=r.mean_ricci,
                                 neg_ricci_fraction=r.neg_ricci_frac)
        except Exception:
            return TopologyIndex(0.0, "neutral", True, 1, 0, 0, 0.0, 0.0)

    def _compute_torsion(self, prices: List[float], embed_dim: int, subsample: int) -> TorsionIndex:
        try:
            r = whitehead_analysis(prices, embed_dim=embed_dim, subsample=subsample)
            score = 0.5 if r.signal == "same_regime" else -0.3
            return TorsionIndex(score=score, signal=r.signal,
                                torsion_ratio=r.torsion.torsion_ratio,
                                num_regimes=r.reeb.num_regimes)
        except Exception:
            return TorsionIndex(0.0, "unknown", 0.0, 0)

    def _compute_algebraic(self, prices: List[float]) -> AlgebraicIndex:
        try:
            self._hecke.set_eigenvalues_from_prices(prices)
            l_val = self._hecke.l_function_value(prices, s=0.5)
            zeta_sig = self._hecke.zeta_signal(prices)
            l_abs = abs(l_val)
            score = min(l_abs / 2.0, 1.0)
            if zeta_sig == "weak":
                score *= 0.3
            direction = 1 if l_val.real >= 0 else -1
            return AlgebraicIndex(score=score, l_value_abs=l_abs,
                                  zeta_signal=zeta_sig or "weak", direction=direction)
        except Exception:
            return AlgebraicIndex(0.0, 0.0, "weak", 1)

    def _compute_geometry(self, prices: List[float], volumes: Optional[List[float]]) -> GeometryIndex:
        try:
            result = frenet_analyze(prices, volumes=volumes)
            if not result.frames:
                return GeometryIndex(0.0, 0.0, "low", 0.0)
            n = min(20, len(result.frames))
            recent = result.frames[-n:]
            mean_kappa = sum(f.curvature for f in recent) / n
            mean_tau   = sum(f.torsion   for f in recent) / n
            sig = curvature_to_signal(mean_kappa)
            kappa_norm  = math.tanh(mean_kappa * 5.0)
            tau_penalty = math.tanh(mean_tau   * 3.0)
            score = max(-1.0, min(1.0, kappa_norm * (1.0 - 0.5 * tau_penalty)))
            return GeometryIndex(curvature=mean_kappa, torsion=mean_tau,
                                 curvature_signal=sig, score=score)
        except Exception:
            return GeometryIndex(0.0, 0.0, "low", 0.0)

    def _compute_polar(self, prices: List[float]) -> PolarIndex:
        try:
            features = self._polar_extractor.extract(prices)
            if not features:
                return PolarIndex(0.0, 0.0, 0.0, 0.0, "neutral", 0.0)
            latest = features[-1]
            regime = describe_regime(features, lookback=min(20, len(features)))
            av = abs(latest.dtheta_dt)   # angular velocity
            rv = latest.dr_dt            # radial velocity
            score = max(-1.0, min(1.0, math.tanh(av * 2.0) - math.tanh(abs(rv) * 2.0)))
            return PolarIndex(r=latest.r, theta=latest.theta,
                              angular_velocity=av, radial_velocity=rv,
                              regime=regime, score=score)
        except Exception:
            return PolarIndex(0.0, 0.0, 0.0, 0.0, "neutral", 0.0)

    def _compute_number_theory(self, prices: List[float]) -> NumberTheoryIndex:
        try:
            prime, precision = 5, 8
            diffs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            scale = 1e4
            vals: List[int] = []
            for d in diffs[-50:]:
                n = max(1, int(d * scale))
                pn = padic_from_integer(n, prime, precision)
                v = next((i for i, c in enumerate(pn.coeffs) if c != 0), precision)
                vals.append(v)
            mean_v = sum(vals) / len(vals)
            roughness = mean_v / max(max(vals), 1)
            return NumberTheoryIndex(roughness=roughness, prime_used=prime,
                                     mean_valuation=mean_v,
                                     score=max(-1.0, min(1.0, 1.0 - 2.0 * roughness)))
        except Exception:
            return NumberTheoryIndex(0.5, 5, 0.0, 0.0)

    def _compute_graph(self, prices: List[float],
                       trades: Optional[List[Tuple[float, float, str]]]) -> GraphIndex:
        try:
            n_bins = 20
            lo, hi = min(prices), max(prices)
            if hi <= lo:
                return GraphIndex(0, 0, 0.0, 0.0)
            bin_size = (hi - lo) / n_bins
            bins = [min(int((p - lo) / bin_size), n_bins - 1) for p in prices]
            edges: Set[Tuple[int, int]] = set()
            nodes: Set[int] = set()
            for i in range(1, len(bins)):
                a, b = bins[i-1], bins[i]
                nodes.add(a); nodes.add(b)
                if a != b:
                    edges.add((a, b))
            nc = len(nodes)
            ec = len(edges)
            density = ec / (nc * (nc - 1)) if nc > 1 else 0.0
            score = max(-1.0, min(1.0, density * 2.0 - 0.5))
            return GraphIndex(node_count=nc, edge_count=ec, density=density, score=score)
        except Exception:
            return GraphIndex(0, 0, 0.0, 0.0)

    def _compute_spectral(self, prices: List[float]) -> SpectralIndex:
        try:
            primes = sieve_primes(min(len(prices) // 4, 30))
            if len(primes) < 2:
                return SpectralIndex(0.0, 0, None, 0.0)
            report = spectral_report(prices, primes)
            score = report.overall_score * 2.0 - 1.0   # 0..1 → -1..+1
            score = max(-1.0, min(1.0, score))
            return SpectralIndex(overall_score=report.overall_score,
                                 num_significant=report.num_significant,
                                 dominant_scale=report.dominant_scale,
                                 score=score)
        except Exception:
            return SpectralIndex(0.0, 0, None, 0.0)

    def _compute_fel(self, prices: List[float]) -> FelIndex:
        try:
            rpt = self._fel.eval(prices)
            if rpt is None:
                return FelIndex("none", "neutral", 0, 0.0, 0.0)
            strength_map = {"strong": 1.0, "moderate": 0.5, "weak": 0.2, "none": 0.0}
            base = strength_map.get(rpt.signal_strength, 0.0)
            score = base if rpt.regime == "mean-reversion" else -base
            return FelIndex(signal_strength=rpt.signal_strength,
                            regime=rpt.regime, genus=rpt.genus,
                            ek_z_score=rpt.ek_z_score, score=score)
        except Exception:
            return FelIndex("none", "neutral", 0, 0.0, 0.0)

    def _compute_quaternion(self, symbol: str, prices: List[float],
                            volumes: Optional[List[float]]) -> QuaternionIndex:
        try:
            if symbol not in self._quat:
                self._quat[symbol] = TradingStateQuaternion()
            q_state = self._quat[symbol]
            vols = volumes or [1.0] * len(prices)
            for p, v in zip(prices, vols):
                q_state.update(p, v)
            regime = q_state.detect_state_regime()
            rotation = q_state.get_state_rotation() or 0.0
            # High rotation = regime shift = uncertainty (negative score)
            score = math.tanh(-rotation * 0.1)
            return QuaternionIndex(state_rotation=rotation, regime=regime, score=score)
        except Exception:
            return QuaternionIndex(0.0, "quiet", 0.0)

    def _compute_simons(self, prices: List[float]) -> SimonsIndex:
        try:
            rpt = SimonsSDEReport.compute(prices)
            return SimonsIndex(
                gbm_score=rpt.gbm.score,
                ou_score=rpt.ou.score,
                ou_z_score=rpt.ou.z_score,
                ou_half_life=rpt.ou.half_life,
                regime=rpt.regime_switch.current_regime,
                kalman_drift=rpt.kalman.filtered_drift,
                hmm_vol_state=rpt.hmm_vol.current_state,
                kelly_f_star=rpt.kelly.f_star,
                fourier_period=rpt.fourier.dominant_period,
                fourier_position=rpt.fourier.cycle_position,
                score=rpt.composite,
            )
        except Exception:
            return SimonsIndex(0.0, 0.0, 0.0, float("inf"), "neutral",
                               0.0, "low_vol", 0.0, 0.0, "unknown", 0.0)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_score(card: InstrumentScorecard, dimension: str) -> float:
        mapping = {
            "composite":      card.composite,
            "topology":       card.topology.score,
            "torsion":        card.torsion.score,
            "algebraic":      card.algebraic.score * card.algebraic.direction,
            "geometry":       card.geometry.score,
            "polar":          card.polar.score,
            "number_theory":  card.number_theory.score,
            "graph":          card.graph.score,
            "spectral":       card.spectral.score,
            "fel":            card.fel.score,
            "quaternion":     card.quaternion.score,
            "hurst":          card.hurst.score,
            "volatility":     card.volatility.score,
            "order_flow":     card.order_flow.score,
            "volume_profile": card.volume_profile.score,
            "microstructure": card.microstructure.score,
            "momentum":       card.momentum.score,
            "autocorr":       card.autocorr.score,
            "funding":        card.funding.score,
            "simons":         card.simons.score,
        }
        if dimension not in mapping:
            raise ValueError(f"Unknown dimension: {dimension!r}. "
                             f"Valid: {list(mapping.keys())}")
        return mapping[dimension]
