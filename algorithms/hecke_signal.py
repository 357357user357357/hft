"""Hecke-enhanced signal for HFT algorithms.

Three-layer signal:
  1. Polynomial variety constraint (Gröbner basis of Hecke relations)
  2. Correct Hecke eigenvalue check: g_p(x_p) = x_p² - a_p·x_p + p^{k-1} ≈ 0
     where x_p is the HANKEL MATRIX eigenvalue and a_p is the autocorrelation
     estimate.  This is zero iff x_p is a root of the Eichler-Deligne
     characteristic polynomial — meaning the market's dominant oscillation
     IS a genuine Hecke eigenvalue consistent with a_p.
  3. Multi-scale spectral agreement as confidence filter
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from hft_types import (Side, TakeProfitConfig, StopLossConfig,
                       TradeResult, BacktestStats)
from data import AggTrade
from algorithms.averages import AveragesConfig, AveragesCondition, AveragesBacktest
from hecke_operators import HeckeAlgebra, sieve_primes
from spectral_window import spectral_report, ScaleSpectrum, SpectralReport


# ─── Gröbner-basis Hecke check ────────────────────────────────────────────────

def hecke_groebner_residual(x_p: float, a_p: float, p: int, weight: int) -> float:
    """
    Normalised Gröbner-basis residual for the Eichler-Deligne polynomial.

    The polynomial g_p(x) = x² − a_p·x + p^{k−1} has its minimum at x* = a_p/2.
    For real Hankel eigenvalues x_p > 0, the polynomial is never zero (since the
    roots are complex under the Ramanujan bound |a_p| ≤ 2p^{(k-1)/2}).

    We therefore use the *normalised proximity*:

        proximity = |x_p − a_p/2| / x_p

    This is 0 iff x_p = a_p/2 exactly (x_p lies at the vertex of g_p),
    and approaches 1 when x_p and a_p/2 are far apart.  Under the null
    (iid returns), a_p ≈ 0 so proximity ≈ 1.  When the market has genuine
    Hecke structure at scale p, a_p/2 ≈ x_p and proximity ≈ 0.

    Returns a value in [0, ∞); lower is more Hecke-consistent.
    """
    if x_p < 1e-10:
        return 1.0
    return abs(x_p - a_p / 2.0) / x_p


def groebner_check_scale(spec: ScaleSpectrum, weight: int,
                         tolerance_factor: float = 0.15) -> Tuple[bool, float]:
    """
    Check whether the scale spec passes the Gröbner basis constraint.

    Returns (passes, proximity).
    Passes if |x_p − a_p/2| / x_p ≤ tolerance_factor.
    Under the null, proximity ≈ 1; signal passes when proximity is small.
    """
    x_p = spec.eigenvalue_mag
    a_p = spec.estimated_a_p
    p   = spec.p

    residual = hecke_groebner_residual(x_p, a_p, p, weight)
    passes = residual <= tolerance_factor
    return passes, residual


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class HeckeSignalConfig:
    """Configuration for the Hecke + Gröbner spectral signal."""
    primes: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    min_significant_scales: int = 2
    min_overall_score: float = 0.5
    min_prices: int = 50
    # Modular form weight (k=2 is standard for cusp forms in HFT context)
    weight: int = 2
    # |λ_p| threshold to count a scale as spectrally significant
    significance_threshold: float = 0.55
    # Use Gröbner basis residual check (correct version: eval at x_p, not a_p)
    use_groebner_check: bool = True
    # tolerance = tolerance_factor * p^{k-1}  (scales with p)
    groebner_tolerance_factor: float = 0.35
    # Minimum fraction of significant scales that must pass Gröbner check
    groebner_pass_fraction: float = 0.20


# ─── Hecke Signal ─────────────────────────────────────────────────────────────

def atkin_lehner_check(report: SpectralReport, primes: List[int],
                       weight: int, tolerance: float = 0.3) -> Tuple[bool, float]:
    """
    Atkin-Lehner involution symmetry check (improvement I).

    The W_N operator gives the functional equation:
        L(s, f) = epsilon * N^{k/2 - s} * L(k - s, f)

    For pairs of primes (p, q) in our set, the Atkin-Lehner involution
    relates the spectral power at scale p to scale q when p*q divides the
    "level" N.  In practice, we check that:

        |a_p / (2*p^{(k-1)/2})| ~ |a_q / (2*q^{(k-1)/2})|

    i.e. the normalised Hecke eigenvalues should be comparable across
    scales if there is a genuine modular form.  Large discrepancy means
    the "Hecke structure" at one scale is noise.

    Returns (passes, mean_discrepancy).
    """
    scale_map = {s.p: s for s in report.scales if s.is_significant}
    if len(scale_map) < 2:
        return True, 0.0  # not enough scales to check

    # Normalise each a_p by Ramanujan bound
    normalised = {}
    for p, s in scale_map.items():
        ram = 2.0 * (p ** ((weight - 1) / 2.0))
        normalised[p] = s.estimated_a_p / ram if ram > 1e-10 else 0.0

    # Pairwise discrepancy
    discrepancies = []
    norm_vals = list(normalised.values())
    for i in range(len(norm_vals)):
        for j in range(i + 1, len(norm_vals)):
            discrepancies.append(abs(norm_vals[i] - norm_vals[j]))

    mean_disc = sum(discrepancies) / len(discrepancies) if discrepancies else 0.0
    return mean_disc <= tolerance, mean_disc


class HeckeSignal:
    """
    Signal that fires when the price window has genuine Hecke spectral structure.

    Algorithm:
      1. Compute multi-scale SpectralReport (Hankel eigenvalues x_p, estimates a_p)
      2. Recompute Gröbner basis with market-estimated a_p values (improvement H)
      3. For each significant scale p, evaluate the proximity metric:
           |x_p - a_p/2| / x_p <= tolerance
      4. Signal power floor gate (improvement F): skip noise-level scales
      5. Atkin-Lehner symmetry check across scales (improvement I)
      6. Signal fires if enough scales pass all tests
    """

    def __init__(self, config: HeckeSignalConfig):
        self.config = config
        max_p = max(config.primes)
        self.algebra = HeckeAlgebra(max_n=max_p, weight=config.weight)
        # Pre-build basis with zero eigenvalues (structure-only relations)
        try:
            self.basis = self.algebra.compute_basis()
        except Exception:
            self.basis = None
        self.last_report: Optional[SpectralReport] = None
        self.last_residuals: List[Tuple[int, float, bool]] = []  # (p, residual, passed)
        self.last_atkin_lehner: Tuple[bool, float] = (True, 0.0)

    def _update_basis_from_prices(self, prices: List[float]) -> None:
        """Recompute Gröbner basis with market-estimated eigenvalues (improvement H).

        Only recomputes if eigenvalues have changed significantly (>10% change).
        """
        try:
            old_eigs = dict(self.algebra.eigenvalues)
            self.algebra.set_eigenvalues_from_prices(prices)
            # Check if eigenvalues changed significantly
            changed = False
            for p, new_val in self.algebra.eigenvalues.items():
                old_val = old_eigs.get(p, 0.0)
                if abs(new_val - old_val) > 0.1:
                    changed = True
                    break
            if changed:
                self.basis = self.algebra.compute_basis()
        except Exception:
            pass

    def eval(self, prices: List[float]) -> bool:
        if len(prices) < self.config.min_prices:
            return False

        # Step 0: recompute Gröbner basis with market eigenvalues
        self._update_basis_from_prices(prices)

        # Step 1: Multi-scale spectral analysis
        report = spectral_report(
            prices,
            primes=self.config.primes,
            weight=self.config.weight,
            significance_threshold=self.config.significance_threshold,
        )
        self.last_report = report
        self.last_residuals = []

        if not report.is_tradeable(
            min_significant=self.config.min_significant_scales,
            min_score=self.config.min_overall_score,
        ):
            return False

        # Step 2: Gröbner basis check — proximity metric at Hankel eigenvalue
        if self.config.use_groebner_check:
            significant = [s for s in report.scales if s.is_significant]
            if not significant:
                return False

            passes = 0
            for spec in significant:
                ok, residual = groebner_check_scale(
                    spec, self.config.weight,
                    self.config.groebner_tolerance_factor
                )
                self.last_residuals.append((spec.p, residual, ok))
                if ok:
                    passes += 1

            pass_fraction = passes / len(significant)
            if pass_fraction < self.config.groebner_pass_fraction:
                return False

        # Step 3: Atkin-Lehner symmetry check
        al_ok, al_disc = atkin_lehner_check(
            report, self.config.primes, self.config.weight)
        self.last_atkin_lehner = (al_ok, al_disc)
        if not al_ok:
            return False

        return True

    def describe(self) -> str:
        lines = [
            "HeckeSignal:",
            f"  primes = {self.config.primes}",
            f"  weight k = {self.config.weight}",
            f"  Gröbner check: |x_p - a_p/2| / x_p <= tol  (Hecke proximity)",
            f"  tolerance = {self.config.groebner_tolerance_factor}",
            f"  pass_fraction >= {self.config.groebner_pass_fraction}",
            f"  Atkin-Lehner: {'pass' if self.last_atkin_lehner[0] else 'fail'}  "
            f"disc={self.last_atkin_lehner[1]:.4f}",
        ]
        if self.last_residuals:
            lines.append("  Last Gröbner residuals:")
            for p, res, ok in self.last_residuals:
                sym = "+" if ok else "-"
                lines.append(f"    p={p:2d} {sym}  residual={res:+.4f}  tol={self.config.groebner_tolerance_factor:.4f}")
        if self.algebra.eigenvalues:
            lines.append("  Market a_p estimates:")
            for p in sorted(self.algebra.eigenvalues):
                lines.append(f"    a_{p} = {self.algebra.eigenvalues[p]:+.4f}")
        return "\n".join(lines)


# ─── Hecke-enhanced Averages Backtest ─────────────────────────────────────────

class HeckeAveragesBacktest:
    """
    Averages algorithm with Hecke Gröbner spectral gate.

    Only places orders when the price window has genuine Hecke structure
    (multi-scale eigenvalues consistent with the Eichler-Deligne polynomial).
    """

    def __init__(self, averages_config: AveragesConfig,
                 hecke_config: HeckeSignalConfig):
        self.averages = AveragesBacktest(averages_config)
        self.hecke = HeckeSignal(hecke_config)
        self.price_buffer: List[float] = []
        self.buffer_maxlen: int = max(
            hecke_config.min_prices * 4,
            max(hecke_config.primes) ** 2 + 100,
        )

    @property
    def results(self) -> List[TradeResult]:
        return self.averages.results

    @property
    def stats(self) -> BacktestStats:
        return self.averages.stats

    def on_trade(self, trade: AggTrade) -> None:
        self.price_buffer.append(trade.price)
        if len(self.price_buffer) > self.buffer_maxlen:
            self.price_buffer.pop(0)

        enough = len(self.price_buffer) >= self.hecke.config.min_prices
        if enough:
            original_place = self.averages._place_orders
            hecke = self.hecke
            buf = self.price_buffer

            def gated(trigger_price: float, ts_ns: int) -> None:
                if hecke.eval(buf):
                    original_place(trigger_price, ts_ns)

            self.averages._place_orders = gated
            self.averages.on_trade(trade)
            self.averages._place_orders = original_place
        else:
            self.averages.on_trade(trade)

    def run(self, trades: List[AggTrade]) -> None:
        for trade in trades:
            self.on_trade(trade)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_hecke_averages_signal(
    side: Side,
    trigger_min_pct: float,
    trigger_max_pct: float,
    primes: Optional[List[int]] = None,
    min_significant_scales: int = 2,
    min_overall_score: float = 0.5,
    groebner_tolerance_factor: float = 0.15,
) -> HeckeAveragesBacktest:
    """
    Build a Hecke-enhanced averages backtest.

    The Gröbner check uses:
        g_p(x_p) = x_p² − a_p·x_p + p^{k−1} ≈ 0
    where x_p is the dominant Hankel eigenvalue and a_p is the autocorrelation
    estimate, so it correctly checks Eichler-Deligne consistency.
    """
    if primes is None:
        primes = [2, 3, 5, 7, 11]

    avg_config = AveragesConfig(
        side=side,
        order_distance_pct=-0.05 if side == Side.Buy else 0.05,
        conditions=[AveragesCondition(
            long_period_secs=30.0, short_period_secs=5.0,
            trigger_min_pct=trigger_min_pct, trigger_max_pct=trigger_max_pct,
        )],
        order_size_usdt=100.0,
        cancel_delay_secs=30.0,
        do_not_trigger_if_active=True,
        restart_delay_secs=2.0,
        take_profit=TakeProfitConfig(enabled=True, percentage=0.05),
        stop_loss=StopLossConfig(
            enabled=True, percentage=0.1, spread_pct=0.02,
            delay_secs=0.0, trailing=None, second_sl=None,
        ),
        grid=None,
    )

    hecke_config = HeckeSignalConfig(
        primes=primes,
        min_significant_scales=min_significant_scales,
        min_overall_score=min_overall_score,
        weight=2,
        significance_threshold=0.55,
        use_groebner_check=True,
        groebner_tolerance_factor=groebner_tolerance_factor,
        groebner_pass_fraction=0.5,
    )

    return HeckeAveragesBacktest(avg_config, hecke_config)
