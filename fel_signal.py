"""Fel Semigroup Signal — numerical semigroups for spectral trading.

Connects three mathematical frameworks:

1. **Numerical semigroups** (Fel's conjecture, proved by Chen–Ono et al.):
   Given generators d₁,...,dₘ (detected periodicities), the semigroup
   S = ⟨d₁,...,dₘ⟩ and its gap set Δ = ℕ \\ S encode which scales have
   periodic structure (S) and which don't (Δ).  Fel's formula gives
   exact invariants K_p relating generators to gaps.

2. **Erdős–Kac theorem** (1940, proved):
   The number of distinct prime factors ω(n) is normally distributed:
       (ω(n) − log log n) / √(log log n)  →  N(0,1)
   Applied here: gives the null distribution for how many spectral
   scales should appear significant by pure chance.  More significant
   scales than Erdős–Kac predicts → real signal.

3. **Erdős discrepancy theorem** (Erdős 1932 conjecture, proved by Tao 2015):
   For any ±1 sequence x₁, x₂, ..., and any d ∈ ℕ:
       sup_N |Σ_{k=1}^{N} x_{kd}| = ∞
   Applied here: periodic patterns in returns along arithmetic
   progressions of step d MUST eventually produce large partial sums.
   The discrepancy growth rate bounds how quickly a tradeable pattern
   must emerge at each scale.

4. **Erdős–Gallai theorem** (1960, proved):
   A non-increasing sequence d₁ ≥ ... ≥ dₙ is a valid graph degree
   sequence iff Σdᵢ is even AND ∀k: Σ_{i≤k} dᵢ ≤ k(k-1) + Σ_{i>k} min(dᵢ,k).
   Applied here: validates that the k-NN graph from Poincaré topology
   has a geometrically realisable structure.
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

# ─── SageMath for semigroup computations ──────────────────────────────────────
_SAGE_AVAILABLE = False
try:
    _sage_bin = os.path.expanduser("~/miniforge3/envs/sage/bin")
    if os.path.isdir(_sage_bin) and _sage_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _sage_bin + ":" + os.environ.get("PATH", "")

    from sage.all import gcd as _sage_gcd, factorial, binomial
    _SAGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Numerical Semigroup
# ═══════════════════════════════════════════════════════════════════════════════

def _gcd_list(vals: List[int]) -> int:
    """GCD of a list of positive integers."""
    from math import gcd
    result = vals[0]
    for v in vals[1:]:
        result = gcd(result, v)
    return result


class NumericalSemigroup:
    """
    Numerical semigroup S = ⟨d₁, ..., dₘ⟩.

    S is the set of all non-negative integer linear combinations of the
    generators.  The gap set Δ = ℕ \\ S is finite iff gcd(d₁,...,dₘ) = 1.

    The Frobenius number F(S) = max(Δ) is the largest non-representable
    integer (the "coin problem").
    """

    def __init__(self, generators: List[int]):
        # Remove duplicates and sort
        gens = sorted(set(g for g in generators if g > 0))
        if len(gens) < 1:
            raise ValueError("Need at least one positive generator")

        g = _gcd_list(gens)
        if g != 1:
            raise ValueError(f"gcd of generators must be 1, got gcd={g} "
                             f"for generators {gens}")

        self.generators = gens
        self.m = len(gens)  # embedding dimension

        # Compute the semigroup up to the Frobenius number + 1
        # using dynamic programming (coin change)
        self._compute_membership()

    def _compute_membership(self) -> None:
        """Compute gap set via DP (reachability)."""
        # Upper bound on Frobenius number: d1*dm - d1 - dm (Sylvester for m=2)
        # For m > 2, use d1*d2 as safe upper bound
        d1 = self.generators[0]
        d2 = self.generators[1] if self.m >= 2 else self.generators[0]
        upper = d1 * d2  # safe bound (true Frobenius ≤ this for gcd=1)

        reachable = [False] * (upper + 1)
        reachable[0] = True
        for n in range(1, upper + 1):
            for g in self.generators:
                if g <= n and reachable[n - g]:
                    reachable[n] = True
                    break

        self.gaps: List[int] = [n for n in range(upper + 1) if not reachable[n]]
        self.frobenius = self.gaps[-1] if self.gaps else 0
        self.genus = len(self.gaps)  # number of gaps = genus

        # Membership set (for quick lookup)
        self._member_set: Set[int] = {n for n in range(upper + 1) if reachable[n]}

    def __contains__(self, n: int) -> bool:
        if n < 0:
            return False
        if n > self.frobenius:
            return True  # all integers > Frobenius are in S
        return n in self._member_set

    # ── Gap power sums G_r(S) = Σ_{g∈Δ} g^r ─────────────────────────────────

    def gap_power_sum(self, r: int) -> float:
        """G_r(S) = Σ_{g ∈ Δ} g^r — the r-th moment of the gap distribution."""
        return sum(g ** r for g in self.gaps)

    # ── Hilbert numerator Q_S(z) ──────────────────────────────────────────────

    def hilbert_numerator_coeffs(self) -> Dict[int, int]:
        """
        Compute the Hilbert numerator Q_S(z) where
            H_S(z) = Q_S(z) / ∏(1 - z^{dᵢ})

        Uses Q_S = P_S · H_S directly, where:
          P_S(z) = ∏(1 - z^{dᵢ})   (product polynomial)
          H_S(z) = Σ_{s∈S} z^s      (Hilbert series, indicator of membership)

        Q_S is always a polynomial (finite degree), so truncating H_S at
        degree = deg(P) + Frobenius is sufficient.

        Returns dict mapping power -> coefficient.
        """
        # Product polynomial P_S(z) = ∏(1 - z^{dᵢ})
        P: Dict[int, int] = {0: 1}
        for d in self.generators:
            new_P: Dict[int, int] = {}
            for exp, coeff in P.items():
                new_P[exp] = new_P.get(exp, 0) + coeff
                new_P[exp + d] = new_P.get(exp + d, 0) - coeff
            P = {k: v for k, v in new_P.items() if v != 0}

        deg_P = max(P.keys()) if P else 0
        max_deg = deg_P + self.frobenius + 1

        # Q_S[k] = Σ_j P[j] · H_S[k-j] = Σ_j P[j] · [k-j ∈ S]
        Q: Dict[int, int] = {}
        for k in range(max_deg + 1):
            val = 0
            for j, pj in P.items():
                r = k - j
                if r >= 0 and r in self:
                    val += pj
            if val != 0:
                Q[k] = val

        return Q

    # ── Alternating power sums C_n(S) ─────────────────────────────────────────

    def alternating_power_sum(self, n: int) -> float:
        """
        C_n(S) = Σ c_j · j^n

        where Q_S(z) = 1 - Σ c_j z^j (writing Q in terms of its non-constant
        coefficients with sign flip).
        """
        Q = self.hilbert_numerator_coeffs()
        # Q(z) = 1 - Σ c_j z^j  →  c_j = -Q[j] for j > 0
        # But Q may have Q[0] = 1, so c_j = -(Q[j]) for j > 0
        # Actually Q already has the right signs.
        # C_n = Σ_{j>0} (-Q[j]) · j^n  if we define Q = 1 + Σ q_j z^j
        # More carefully: write Q_S(z) = Σ q_j z^j where q_0=1.
        # Then c_j = -q_j for j > 0.
        # C_n = Σ c_j j^n = Σ_{j>0} (-q_j) j^n = -Σ_{j>0} q_j j^n
        result = 0.0
        for j, qj in Q.items():
            if j > 0:
                result -= qj * (j ** n)
        return result

    # ── K-invariants via Fel's formula ────────────────────────────────────────

    def K_invariant(self, p: int) -> float:
        """
        K_p(S) = (-1)^m · p! / ((m+p)! · π_m) · C_{m+p}(S)

        where π_m = ∏ d_i  and  m = embedding dimension.

        Equivalently (Fel's conjecture, proved):
            K_p = Σ_{r=0}^p C(p,r) T_{p-r}(σ) G_r(S) + 2^{p+1}/(p+1) T_{p+1}(δ)
        """
        pi_m = 1
        for d in self.generators:
            pi_m *= d

        C_mp = self.alternating_power_sum(self.m + p)
        sign = (-1) ** self.m

        # p! / (m+p)!
        fact_ratio = 1.0
        for k in range(p + 1, self.m + p + 1):
            fact_ratio /= k

        return sign * fact_ratio / pi_m * C_mp

    # ── T_n(σ) and T_n(δ) — universal symmetric polynomials ──────────────────

    def _power_sums(self, max_n: int = 8) -> List[float]:
        """Power sums p_k = Σ d_i^k of the generators."""
        return [sum(d ** k for d in self.generators) for k in range(max_n + 1)]

    def T_sigma(self, n: int) -> float:
        """
        T_n(σ) = n! · [t^n] A(t)   where  A(t) = ∏ (e^{dᵢt} - 1)/(dᵢt).

        Computed via the recurrence on power sums p_k = Σ dᵢ^k.
        T_0 = 1, T_1 = p₁/2, T_2 = (3p₁² - p₂)/12, etc.
        """
        ps = self._power_sums(n + 1)
        if n == 0:
            return 1.0
        if n == 1:
            return ps[1] / 2.0
        if n == 2:
            return (3 * ps[1]**2 - ps[2]) / 12.0
        if n == 3:
            return (ps[1]**3 - ps[1]*ps[2]) / 8.0
        if n == 4:
            return (15*ps[1]**4 - 30*ps[1]**2*ps[2] + 3*ps[2]**2 + 8*ps[1]*ps[3] - ps[4]) / 240.0

        # General case via exponential generating function (numerical)
        # A(t) = ∏ (e^{dᵢt} - 1)/(dᵢt)
        # Compute coefficients by polynomial multiplication
        from fractions import Fraction
        max_terms = n + 2
        # Start with [1]
        coeffs = [Fraction(1)]
        for d in self.generators:
            # Multiply by (e^{dt} - 1)/(dt) = Σ (dt)^k / (k+1)! = Σ d^k t^k / (k+1)!
            factor = [Fraction(d**k, math.factorial(k + 1)) for k in range(max_terms)]
            new_coeffs = [Fraction(0)] * max_terms
            for i in range(min(len(coeffs), max_terms)):
                for j in range(min(len(factor), max_terms - i)):
                    new_coeffs[i + j] += coeffs[i] * factor[j]
            coeffs = new_coeffs

        if n < len(coeffs):
            return float(coeffs[n]) * math.factorial(n)
        return 0.0

    def T_delta(self, n: int) -> float:
        """
        T_n(δ) = (n!/2^n) · [t^n] B(t)   where  B(t) = t/(e^t - 1) · A(t).

        Incorporates Bernoulli numbers into the generator structure.
        """
        from fractions import Fraction
        max_terms = n + 2

        # B(t) = t/(e^t - 1) = Σ B_k t^k / k!  (Bernoulli numbers)
        bernoulli_coeffs = [Fraction(0)] * max_terms
        # Compute Bernoulli numbers B_0=1, B_1=-1/2, B_2=1/6, ...
        B = [Fraction(0)] * max_terms
        B[0] = Fraction(1)
        for m in range(1, max_terms):
            # Standard recurrence: Σ_{k=0}^{m} C(m+1, k) B_k = 0
            # → B_m = -1/(m+1) · Σ_{k=0}^{m-1} C(m+1, k) B_k
            s = Fraction(0)
            for k in range(m):
                # C(m+1, k) via multiplicative formula
                binom = Fraction(1)
                for j in range(1, k + 1):
                    binom = binom * Fraction(m + 1 - j + 1, j)
                s += binom * B[k]
            B[m] = -s / Fraction(m + 1)
        for k in range(max_terms):
            bernoulli_coeffs[k] = B[k] / Fraction(math.factorial(k))

        # A(t) coefficients
        a_coeffs = [Fraction(1)]
        for d in self.generators:
            factor = [Fraction(d**k, math.factorial(k + 1)) for k in range(max_terms)]
            new_coeffs = [Fraction(0)] * max_terms
            for i in range(min(len(a_coeffs), max_terms)):
                for j in range(min(len(factor), max_terms - i)):
                    new_coeffs[i + j] += a_coeffs[i] * factor[j]
            a_coeffs = new_coeffs

        # B(t) = bernoulli · A(t) — convolve
        b_coeffs = [Fraction(0)] * max_terms
        for i in range(min(len(bernoulli_coeffs), max_terms)):
            for j in range(min(len(a_coeffs), max_terms - i)):
                b_coeffs[i + j] += bernoulli_coeffs[i] * a_coeffs[j]

        if n < len(b_coeffs):
            return float(b_coeffs[n]) * math.factorial(n) / (2 ** n)
        return 0.0

    def verify_fel(self, p: int) -> Tuple[float, float]:
        """
        Verify Fel's formula for K_p:

            K_p = Σ_{r=0}^p C(p,r) T_{p-r}(σ) G_r(S) + 2^{p+1}/(p+1) T_{p+1}(δ)

        Returns (K_p from definition, K_p from Fel's formula).
        """
        K_direct = self.K_invariant(p)

        # Fel's RHS
        fel_sum = 0.0
        for r in range(p + 1):
            binom = math.comb(p, r)
            T_pr = self.T_sigma(p - r)
            G_r = self.gap_power_sum(r)
            fel_sum += binom * T_pr * G_r

        fel_sum += (2 ** (p + 1)) / (p + 1) * self.T_delta(p + 1)

        return K_direct, fel_sum

    def describe(self) -> str:
        gens_str = ", ".join(str(g) for g in self.generators)
        lines = [
            f"NumericalSemigroup S = ⟨{gens_str}⟩",
            f"  Embedding dimension m = {self.m}",
            f"  Frobenius number F = {self.frobenius}",
            f"  Genus (# gaps) = {self.genus}",
            f"  Gaps Δ = {self.gaps[:20]}{'...' if len(self.gaps) > 20 else ''}",
            f"  G_0 = {self.gap_power_sum(0):.0f}  (= genus)",
            f"  G_1 = {self.gap_power_sum(1):.0f}  (sum of gaps)",
            f"  G_2 = {self.gap_power_sum(2):.0f}  (sum of squares)",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Erdős theorems for trading
# ═══════════════════════════════════════════════════════════════════════════════

def erdos_kac_null(n_scales: int, n_prices: int) -> Tuple[float, float, float]:
    """
    Erdős–Kac theorem (1940, proved): expected number of significant
    spectral scales under the null hypothesis of iid returns.

    The number of distinct prime factors ω(n) of a random integer near N
    has mean log(log(N)) and std √(log(log(N))).  Analogously, the number
    of "significant" spectral scales in a price window of length N should
    follow the same distribution under the null (no structure).

    Returns (expected, std, z_score) where z_score = (observed - expected) / std.
    A z_score > 2 means more significant scales than chance → real signal.
    """
    if n_prices < 3:
        return 0.0, 1.0, 0.0

    log_log_n = math.log(max(math.log(max(n_prices, 2)), 1.0))
    expected = log_log_n
    std = math.sqrt(max(log_log_n, 0.1))
    z_score = (n_scales - expected) / std if std > 1e-10 else 0.0

    return expected, std, z_score


def erdos_discrepancy_bound(returns_sign: List[int], d: int) -> Tuple[float, float]:
    """
    Erdős discrepancy theorem (Erdős 1932, proved by Tao 2015):

    For any ±1 sequence x₁, x₂, ..., and any step d:
        sup_N |Σ_{k=1}^{N} x_{kd}| = ∞

    This means periodic patterns at scale d MUST eventually produce
    large partial sums.  We compute the current discrepancy and
    normalise by √N (which is the expected growth rate under randomness).

    Args:
        returns_sign: sequence of +1/-1 (sign of returns)
        d: arithmetic progression step (period to test)

    Returns (raw_discrepancy, normalised_discrepancy).
    normalised > 1.5 suggests a real periodic pattern at scale d.
    """
    # Extract subsequence along arithmetic progression of step d
    subseq = [returns_sign[i] for i in range(0, len(returns_sign), d) if i < len(returns_sign)]

    if len(subseq) < 2:
        return 0.0, 0.0

    # Max absolute partial sum (the discrepancy)
    partial = 0
    max_disc = 0
    for x in subseq:
        partial += x
        if abs(partial) > max_disc:
            max_disc = abs(partial)

    # Normalise by √N (random walk scaling)
    normalised = max_disc / math.sqrt(len(subseq))

    return float(max_disc), normalised


def erdos_gallai_valid(degree_sequence: List[int]) -> bool:
    """
    Erdős–Gallai theorem (1960, proved):

    A non-increasing sequence d₁ ≥ d₂ ≥ ... ≥ dₙ is a valid graph
    degree sequence if and only if:
      1. Σ dᵢ is even
      2. ∀ k ∈ {1,...,n}: Σ_{i=1}^k dᵢ ≤ k(k-1) + Σ_{i=k+1}^n min(dᵢ, k)

    For trading: validates that the k-NN graph constructed in Poincaré
    topology analysis has a geometrically realisable structure.
    If the degree sequence is invalid → the graph is degenerate →
    topology signal is unreliable.
    """
    if not degree_sequence:
        return True

    ds = sorted(degree_sequence, reverse=True)
    n = len(ds)
    total = sum(ds)

    # Condition 1: sum must be even
    if total % 2 != 0:
        return False

    # Condition 2: Erdős–Gallai inequalities
    prefix_sum = 0
    for k in range(1, n + 1):
        prefix_sum += ds[k - 1]
        rhs = k * (k - 1) + sum(min(ds[i], k) for i in range(k, n))
        if prefix_sum > rhs:
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  FelSemigroupSignal — trading signal
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FelReport:
    """Report from Fel semigroup analysis."""
    generators: List[int]          # detected periodicities (semigroup generators)
    genus: int                     # number of gaps = spectral "holes"
    frobenius: int                 # largest gap (largest unrepresented scale)
    gap_density: float             # genus / frobenius — how sparse the spectrum is
    K_invariants: List[float]      # K_0, K_1, K_2 — Fel's syzygy invariants
    fel_verified: bool             # True if Fel's formula matches direct computation

    # Erdős–Kac
    ek_expected: float             # expected # significant scales under null
    ek_z_score: float              # z-score: observed vs expected

    # Erdős discrepancy
    max_discrepancy: float         # max normalised discrepancy across generators
    discrepancy_scale: int         # scale with highest discrepancy

    # Erdős–Gallai
    graph_valid: bool              # True if k-NN graph has valid degree sequence

    # Signal
    signal_strength: str           # "strong", "moderate", "weak", "none"
    regime: str                    # "mean-reversion", "trending", "neutral"

    def __repr__(self) -> str:
        gens_str = ", ".join(str(g) for g in self.generators)
        lines = [
            f"FelReport: S = ⟨{gens_str}⟩  genus={self.genus}  F={self.frobenius}",
            f"  K₀={self.K_invariants[0]:.3f}  K₁={self.K_invariants[1]:.3f}  K₂={self.K_invariants[2]:.3f}",
            f"  Fel verified: {self.fel_verified}",
            f"  Erdős–Kac: z={self.ek_z_score:+.2f} (expected {self.ek_expected:.1f} significant scales)",
            f"  Erdős discrepancy: max={self.max_discrepancy:.3f} at scale {self.discrepancy_scale}",
            f"  Erdős–Gallai graph valid: {self.graph_valid}",
            f"  Signal: {self.signal_strength}  Regime: {self.regime}",
        ]
        return "\n".join(lines)


@dataclass
class FelSignalConfig:
    """Configuration for FelSemigroupSignal."""
    min_prices: int = 60
    significance_threshold: float = 0.75    # spectral |λ| threshold
    max_generators: int = 6                 # max periods to use as generators
    knn: int = 5                            # k-NN for Erdős–Gallai check
    ek_z_threshold: float = 1.5             # Erdős–Kac z-score for "real signal"
    discrepancy_threshold: float = 1.5      # normalised discrepancy for periodic pattern


class FelSemigroupSignal:
    """
    Trading signal via numerical semigroup gap analysis + Erdős theorems.

    Pipeline:
      1. Spectral analysis → detect significant periodicities at prime scales
      2. Build numerical semigroup from detected periods
      3. Compute gap set (missing scales) and Fel's K-invariants
      4. Erdős–Kac: is the number of significant scales unusual?
      5. Erdős discrepancy: do periodic patterns show real partial-sum growth?
      6. Erdős–Gallai: is the topology graph structurally valid?
      7. Combine into trading signal
    """

    def __init__(self, config: FelSignalConfig = None):
        self.config = config or FelSignalConfig()
        self.last_report: Optional[FelReport] = None

    def eval(self, prices: List[float]) -> Optional[FelReport]:
        """
        Run full Fel semigroup analysis on a price window.
        Returns FelReport with signal and regime.
        """
        n = len(prices)
        if n < self.config.min_prices:
            return None

        # Step 1: spectral analysis to find significant periods
        from spectral_window import spectral_report, log_returns
        from hecke_operators import sieve_primes

        primes = sieve_primes(min(n // 4, 30))
        if len(primes) < 2:
            return None

        report = spectral_report(
            prices, primes,
            significance_threshold=self.config.significance_threshold,
        )

        # Select scales that stand out: eigenvalue > median + 0.5 * MAD
        mags = sorted(s.eigenvalue_mag for s in report.scales)
        median_mag = mags[len(mags) // 2] if mags else 1.0
        mad = sorted(abs(m - median_mag) for m in mags)
        mad_val = mad[len(mad) // 2] if mad else 0.1
        adaptive_thr = median_mag + max(0.5 * mad_val, 0.05)

        significant_primes = [s.p for s in report.scales
                              if s.eigenvalue_mag >= adaptive_thr]
        n_significant = len(significant_primes)

        # Step 2: build numerical semigroup from significant periods
        generators = significant_primes[:self.config.max_generators]

        # If fewer than 2 generators, add top by eigenvalue magnitude
        if len(generators) < 2:
            by_mag = sorted(report.scales, key=lambda s: s.eigenvalue_mag, reverse=True)
            for s in by_mag:
                if s.p not in generators:
                    generators.append(s.p)
                if len(generators) >= 2:
                    break

        generators = sorted(set(generators))

        # Ensure gcd = 1 (required for finite gap set) — add smallest
        # coprime prime, not all small primes
        if len(generators) >= 1:
            g = _gcd_list(generators)
            if g != 1:
                for p in [2, 3, 5, 7, 11, 13]:
                    if p not in generators and g % p != 0:
                        generators.append(p)
                        generators.sort()
                        break

        try:
            semigroup = NumericalSemigroup(generators)
        except ValueError:
            return None

        # Step 3: Fel's K-invariants
        K = [semigroup.K_invariant(p) for p in range(3)]
        # Verify Fel's formula for K_0
        K_direct, K_fel = semigroup.verify_fel(0)
        fel_ok = abs(K_direct - K_fel) < 1e-6

        # Step 4: Erdős–Kac null test
        ek_exp, ek_std, ek_z = erdos_kac_null(n_significant, n)

        # Step 5: Erdős discrepancy on returns
        rets = log_returns(prices)
        returns_sign = [1 if r >= 0 else -1 for r in rets]

        max_disc = 0.0
        max_disc_scale = generators[0]
        for d in generators:
            _, norm_disc = erdos_discrepancy_bound(returns_sign, d)
            if norm_disc > max_disc:
                max_disc = norm_disc
                max_disc_scale = d

        # Step 6: Erdős–Gallai graph validation
        # Build a quick k-NN degree sequence from the embedded points
        from poincare_trading import (delay_embed, normalise_points,
                                       farthest_point_sampling,
                                       pairwise_distances, knn_graph)
        points = delay_embed(prices, dim=3)
        if len(points) > 80:
            points = normalise_points(points)
            points = farthest_point_sampling(points, 60)
        else:
            points = normalise_points(points)

        if len(points) > 5:
            D = pairwise_distances(points)
            graph = knn_graph(D, k=min(self.config.knn, len(points) - 1))
            degree_seq = [len(graph[i]) for i in range(len(graph))]
            graph_valid = erdos_gallai_valid(degree_seq)
        else:
            graph_valid = True

        # Step 7: combine into signal
        gap_density = semigroup.genus / max(semigroup.frobenius, 1)

        # Signal strength from Erdős–Kac + discrepancy
        if ek_z > self.config.ek_z_threshold and max_disc > self.config.discrepancy_threshold:
            signal = "strong"
        elif ek_z > 1.0 or max_disc > 1.0:
            signal = "moderate"
        elif ek_z > 0.5:
            signal = "weak"
        else:
            signal = "none"

        # Regime from gap structure + K-invariants + discrepancy
        # Small genus → dense spectral coverage at short scales → mean-reversion
        # Large genus → many missing scales → trending / directional
        # High discrepancy → periodic patterns → mean-reversion
        k_ratio = K[1] / K[0] if abs(K[0]) > 1e-10 else 0.0
        min_gen = min(generators) if generators else 2

        if semigroup.genus == 0:
            regime = "mean-reversion"  # all scales present
        elif semigroup.genus <= 5 and min_gen <= 5:
            # Few gaps + small generators → short-period structure → mean-reversion
            regime = "mean-reversion"
        elif semigroup.genus > 20 and k_ratio > 30:
            # Many gaps concentrated at large values → directional
            regime = "trending"
        elif max_disc > self.config.discrepancy_threshold:
            regime = "mean-reversion"  # strong periodic pattern
        elif semigroup.genus > 10:
            regime = "trending"
        else:
            regime = "neutral"

        self.last_report = FelReport(
            generators=generators,
            genus=semigroup.genus,
            frobenius=semigroup.frobenius,
            gap_density=gap_density,
            K_invariants=K,
            fel_verified=fel_ok,
            ek_expected=ek_exp,
            ek_z_score=ek_z,
            max_discrepancy=max_disc,
            discrepancy_scale=max_disc_scale,
            graph_valid=graph_valid,
            signal_strength=signal,
            regime=regime,
        )
        return self.last_report

    def describe(self) -> str:
        lines = [
            "FelSemigroupSignal (numerical semigroup gap analysis):",
            "  Methods: spectral → semigroup → Fel K-invariants + Erdős theorems",
            f"  Config: min_prices={self.config.min_prices}  "
            f"sig_thr={self.config.significance_threshold}  "
            f"max_gens={self.config.max_generators}",
        ]
        if self.last_report:
            lines.append("")
            lines.append(str(self.last_report))
        return "\n".join(lines)


def log_returns(prices: List[float]) -> List[float]:
    """Log returns from price series."""
    return [math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
            if prices[i - 1] > 0 and prices[i] > 0]


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    print("=" * 65)
    print("Fel Semigroup Signal + Erdős Theorems — Self-Test")
    print("=" * 65)

    # Test 1: numerical semigroup basics
    print("\n[1] Numerical semigroup S = ⟨3, 5⟩")
    S = NumericalSemigroup([3, 5])
    print(S.describe())
    K0_d, K0_f = S.verify_fel(0)
    K1_d, K1_f = S.verify_fel(1)
    print(f"  K_0: direct={K0_d:.4f}  Fel={K0_f:.4f}  match={abs(K0_d-K0_f)<1e-4}")
    print(f"  K_1: direct={K1_d:.4f}  Fel={K1_f:.4f}  match={abs(K1_d-K1_f)<1e-4}")

    # Test 2: Erdős–Kac
    print("\n[2] Erdős–Kac null test")
    exp, std, z = erdos_kac_null(5, 200)
    print(f"  5 significant scales in 200 prices: expected={exp:.2f}, z={z:+.2f}")

    # Test 3: Erdős discrepancy
    print("\n[3] Erdős discrepancy")
    signs = [1 if random.random() > 0.5 else -1 for _ in range(200)]
    raw, norm = erdos_discrepancy_bound(signs, 3)
    print(f"  Random ±1, step=3: raw={raw:.0f}, normalised={norm:.3f}")
    # Biased sequence (should have higher discrepancy)
    signs_biased = [1 if (i % 5 < 3) else -1 for i in range(200)]
    raw_b, norm_b = erdos_discrepancy_bound(signs_biased, 5)
    print(f"  Periodic ±1 (period=5), step=5: raw={raw_b:.0f}, normalised={norm_b:.3f}")

    # Test 4: Erdős–Gallai
    print("\n[4] Erdős–Gallai graph validation")
    print(f"  [3,3,3,3]: valid={erdos_gallai_valid([3,3,3,3])}")
    print(f"  [3,3,3,1]: valid={erdos_gallai_valid([3,3,3,1])}")
    print(f"  [5,5,5,5,5]: valid={erdos_gallai_valid([5,5,5,5,5])}")

    # Test 5: FelSemigroupSignal on synthetic data
    print("\n[5] FelSemigroupSignal on synthetic price data")
    signal = FelSemigroupSignal()

    # Random walk
    prices_rw = [100.0]
    for _ in range(200):
        prices_rw.append(prices_rw[-1] * math.exp(random.gauss(0, 0.001)))
    report = signal.eval(prices_rw)
    if report:
        print(f"  Random walk: {report.signal_strength} / {report.regime}")
        print(f"    generators={report.generators}, genus={report.genus}")

    # Trending
    prices_trend = [100.0]
    for t in range(200):
        prices_trend.append(prices_trend[-1] * math.exp(0.003 + random.gauss(0, 0.0005)))
    report = signal.eval(prices_trend)
    if report:
        print(f"  Trending: {report.signal_strength} / {report.regime}")
        print(f"    generators={report.generators}, genus={report.genus}")

    # Mean-reverting (OU)
    prices_ou = [100.0]
    for _ in range(200):
        last = prices_ou[-1]
        drift = -0.05 * (last - 100.0)
        prices_ou.append(last + drift + random.gauss(0, 0.08))
    report = signal.eval(prices_ou)
    if report:
        print(f"  OU mean-rev: {report.signal_strength} / {report.regime}")
        print(f"    generators={report.generators}, genus={report.genus}")

    print("\n" + "=" * 65)
    print("All tests completed.")
