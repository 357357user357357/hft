"""Hecke Operator Algebra via Gröbner Basis.

Hecke operators T_n satisfy two classical polynomial relations:

  1. Multiplicativity:  T_m * T_n - T_{mn} = 0   (when gcd(m,n) = 1)
  2. Eichler-Deligne:   T_p^2 - a_p * T_p + p^(k-1) = 0   (p prime)

Here a_p is the Hecke eigenvalue — estimated from the autocorrelation
of price returns at scale p. The Gröbner basis of these relations gives
a canonical form for any composed operator, and the eigenvalues a_p are
the spectral signal used for trading.

When SageMath is available, the Gröbner basis is computed via Singular's
optimised F4/F5 algorithms (orders of magnitude faster than pure-Python
Buchberger), and actual Hecke eigenvalues from modular forms can be used
to validate market-estimated eigenvalues.

Variable layout: variable i corresponds to T_{i+1}
  x0 = T_1, x1 = T_2, x2 = T_3, ...
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─── Backend selection ────────────────────────────────────────────────────────
# Try SageMath first (fast F4/Singular), fall back to pure-Python Buchberger.

_SAGE_AVAILABLE = False
_QQ = None
_PolynomialRing = None
_CuspForms = None
_Gamma0 = None
try:
    # Sage needs Singular on PATH
    _sage_bin = os.path.expanduser("~/miniforge3/envs/sage/bin")
    if os.path.isdir(_sage_bin) and _sage_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _sage_bin + ":" + os.environ.get("PATH", "")

    from sage.all import (QQ as _QQ, PolynomialRing as _PolynomialRing,
                          CuspForms as _CuspForms, Gamma0 as _Gamma0)
    _SAGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass

from groebner import (
    Polynomial, Monomial, Rational, MonomialOrder,
    GroebnerBasis, poly_reduce, Term
)


# ─── Utilities ────────────────────────────────────────────────────────────────

def sieve_primes(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


# ─── Hecke Algebra ────────────────────────────────────────────────────────────

class HeckeAlgebra:
    """
    Polynomial algebra Z[T_1, T_2, ..., T_N] modulo Hecke relations.

    The Gröbner basis of these relations:
      - reduces any operator product to a canonical form
      - lets us test membership (is this pattern in the ideal?)
      - eigenvalues a_p are extracted from market autocorrelation
    """

    def __init__(self, max_n: int = 20, weight: int = 2):
        """
        max_n:   consider operators T_1 ... T_{max_n}
        weight:  modular form weight k (affects p^{k-1} term in Eichler-Deligne)
        """
        self.max_n = max_n
        self.weight = weight
        self.num_vars = max_n          # variable i ↔ T_{i+1}
        self.order = MonomialOrder.GRevLex
        self.basis: Optional[GroebnerBasis] = None
        # Estimated Hecke eigenvalues a_p (set from market data)
        self.eigenvalues: Dict[int, float] = {}

    # ── index helpers ──────────────────────────────────────────────────────────

    def var_index(self, n: int) -> int:
        """Map T_n → variable index (0-based). T_1 → 0, T_2 → 1, ..."""
        assert 1 <= n <= self.max_n, f"T_{n} out of range [1, {self.max_n}]"
        return n - 1

    def _monomial_for(self, n: int, degree: int = 1) -> List[int]:
        """Exponent vector with T_n raised to `degree`."""
        exp = [0] * self.num_vars
        exp[self.var_index(n)] = degree
        return exp

    def _poly_from_terms(self, terms: List[Tuple[int, List[int]]]) -> Polynomial:
        return Polynomial.from_terms(self.num_vars, self.order, terms)

    # ── relation builders ──────────────────────────────────────────────────────

    def multiplicativity_relation(self, m: int, n: int) -> Optional[Polynomial]:
        """
        T_m * T_n - T_{mn} = 0   when gcd(m, n) = 1 and mn <= max_n.

        Returns None if mn > max_n (can't represent T_{mn}).
        """
        if gcd(m, n) != 1:
            return None
        mn = m * n
        if mn > self.max_n:
            return None

        # T_m * T_n: exponent vector with both m and n set to 1
        exp_mn_prod = [0] * self.num_vars
        exp_mn_prod[self.var_index(m)] = 1
        exp_mn_prod[self.var_index(n)] = 1

        # T_{mn}: exponent vector with mn set to 1
        exp_mn = self._monomial_for(mn)

        return self._poly_from_terms([
            (1,  exp_mn_prod),   # +T_m * T_n
            (-1, exp_mn),        # -T_{mn}
        ])

    def eichler_deligne_relation(self, p: int, a_p: float = 0.0) -> Polynomial:
        """
        T_p^2 - a_p * T_p + p^{k-1} = 0   (p prime, k = self.weight).

        a_p is the Hecke eigenvalue, estimated from market autocorrelation.
        When a_p = 0 (default), this is the "pure spectral" constraint.
        """
        ap_rat = Rational.from_f64(a_p)
        pk1 = p ** (self.weight - 1)

        exp_p2 = self._monomial_for(p, degree=2)
        exp_p1 = self._monomial_for(p, degree=1)
        exp_0  = [0] * self.num_vars   # constant term

        # Build using integer-scaled exact arithmetic.
        # Represent a_p as fraction p_num/p_den, multiply through by p_den
        # to clear denominator:
        #   p_den * T_p^2  -  p_num * T_p  +  p_den * p^{k-1}  = 0
        from fractions import Fraction
        ap_frac = Fraction(a_p).limit_denominator(10000)
        p_num = int(ap_frac.numerator)
        p_den = int(ap_frac.denominator)

        raw_terms = [
            (p_den,          exp_p2),
            (-p_num,         exp_p1),
            (p_den * pk1,    exp_0),
        ]
        raw_terms = [(c, e) for c, e in raw_terms if c != 0]
        return self._poly_from_terms(raw_terms)

    # ── build basis ────────────────────────────────────────────────────────────

    def build_relations(self, max_relations: int = 80) -> List[Polynomial]:
        """
        Generate Hecke relations for T_1 ... T_{max_n}.

        Prioritises Eichler-Deligne (primes) over multiplicativity (coprime pairs),
        and caps total relation count to prevent Buchberger's algorithm from
        hitting exponential blowup on large max_n.
        """
        polys: List[Polynomial] = []
        primes = sieve_primes(self.max_n)

        # Eichler-Deligne for each prime (always included — these are essential)
        for p in primes:
            a_p = self.eigenvalues.get(p, 0.0)
            poly = self.eichler_deligne_relation(p, a_p)
            if not poly.is_zero():
                polys.append(poly)

        # Multiplicativity for coprime pairs, prioritising small products
        # (small products are more constrained and reduce faster)
        mult_rels: List[Tuple[int, Polynomial]] = []
        indices = list(range(2, self.max_n + 1))
        for i, m in enumerate(indices):
            for n in indices[i + 1:]:
                rel = self.multiplicativity_relation(m, n)
                if rel is not None and not rel.is_zero():
                    mult_rels.append((m * n, rel))

        # Sort by product size (smallest first) and cap
        mult_rels.sort(key=lambda x: x[0])
        remaining = max_relations - len(polys)
        for _, rel in mult_rels[:max(remaining, 0)]:
            polys.append(rel)

        return polys

    def compute_basis(self) -> GroebnerBasis:
        """
        Compute the Gröbner basis of all Hecke relations.

        Uses SageMath's Singular backend (F4 algorithm) when available,
        which is orders of magnitude faster than pure-Python Buchberger.
        Falls back to the pure-Python implementation otherwise.
        """
        if _SAGE_AVAILABLE:
            return self._compute_basis_sage()
        polys = self.build_relations()
        if not polys:
            raise ValueError("No relations generated — increase max_n")
        self.basis = GroebnerBasis.compute(polys, self.order)
        return self.basis

    def _compute_basis_sage(self) -> GroebnerBasis:
        """Compute Gröbner basis using SageMath's Singular backend."""
        var_names = [f"x{i}" for i in range(self.num_vars)]
        R = _PolynomialRing(_QQ, var_names, order="degrevlex")
        gens = R.gens()

        sage_polys = []
        primes = sieve_primes(self.max_n)

        # Eichler-Deligne: T_p^2 - a_p*T_p + p^{k-1}
        for p in primes:
            a_p = self.eigenvalues.get(p, 0.0)
            from fractions import Fraction
            ap_frac = Fraction(a_p).limit_denominator(10000)
            idx = self.var_index(p)
            pk1 = p ** (self.weight - 1)
            poly = gens[idx]**2 - _QQ(ap_frac) * gens[idx] + _QQ(pk1)
            sage_polys.append(poly)

        # Multiplicativity: T_m*T_n - T_{mn} for coprime m,n with mn <= max_n
        indices = list(range(2, self.max_n + 1))
        for i, m in enumerate(indices):
            for n in indices[i + 1:]:
                if gcd(m, n) != 1:
                    continue
                mn = m * n
                if mn > self.max_n:
                    continue
                poly = gens[self.var_index(m)] * gens[self.var_index(n)] - gens[self.var_index(mn)]
                sage_polys.append(poly)

        if not sage_polys:
            raise ValueError("No relations generated — increase max_n")

        # Compute Gröbner basis via Singular (F4 — very fast)
        I = R.ideal(sage_polys)
        gb = I.groebner_basis()

        # Convert sage basis polynomials back to our GroebnerBasis format
        result_polys = []
        for sg in gb:
            terms_data = []
            for coeff, mono in sg:
                exp = [0] * self.num_vars
                for i, e in enumerate(mono.exponents()[0] if hasattr(mono, 'exponents') else [0]*self.num_vars):
                    if i < self.num_vars:
                        exp[i] = int(e)
                c_int = int(coeff.numerator())
                d_int = int(coeff.denominator())
                terms_data.append(Term(Rational(c_int, d_int), Monomial(exp)))
            if terms_data:
                poly = Polynomial(terms_data, self.num_vars, self.order)
                poly._normalize()
                result_polys.append(poly)

        self.basis = GroebnerBasis(result_polys, self.order, self.num_vars)
        return self.basis

    def set_eigenvalues_from_prices(self, prices: List[float]) -> None:
        """
        Estimate Hecke eigenvalues a_p from price return autocorrelation.

        For a price series r_t = log(p_t / p_{t-1}), the lag-p autocorrelation
        is a natural estimate of the Hecke eigenvalue at scale p:

            a_p ≈ 2 * p^{(k-1)/2} * autocorr(r, lag=p)

        The factor 2*p^{(k-1)/2} comes from the Ramanujan bound |a_p| ≤ 2p^{(k-1)/2}.
        This maps the [-1,1] autocorrelation into the admissible range for a_p.
        """
        import math
        n = len(prices)
        if n < 2:
            return

        # Log returns
        returns = [math.log(prices[i] / prices[i - 1])
                   for i in range(1, n) if prices[i - 1] > 0]

        if not returns:
            return

        mu = sum(returns) / len(returns)
        var = sum((r - mu) ** 2 for r in returns) / len(returns)
        if var < 1e-15:
            return

        primes = sieve_primes(self.max_n)
        for p in primes:
            n_valid = len(returns) - p
            if n_valid < max(p, 4):
                continue
            # Lag-p autocorrelation (unbiased: divide by number of valid pairs)
            cov = sum(
                (returns[i] - mu) * (returns[i - p] - mu)
                for i in range(p, len(returns))
            ) / n_valid
            autocorr = max(-1.0, min(1.0, cov / var))

            # Scale to Hecke eigenvalue range via Ramanujan bound
            ramanujan_bound = 2.0 * (p ** ((self.weight - 1) / 2.0))
            self.eigenvalues[p] = ramanujan_bound * autocorr

    def set_eigenvalues_from_modular_forms(self, level: int = 11) -> Dict[int, List[float]]:
        """
        Get actual Hecke eigenvalues a_p from modular newforms (via SageMath).

        For a cuspidal newform f of weight k and level N, the Hecke eigenvalues
        a_p are exact algebraic numbers satisfying the Ramanujan bound
        |a_p| <= 2*p^{(k-1)/2}.

        Returns dict mapping prime p -> list of a_p values (one per newform).
        These can be compared to market-estimated eigenvalues to check if
        the price autocorrelation structure matches a known modular form.

        Requires SageMath.
        """
        if not _SAGE_AVAILABLE:
            raise RuntimeError("SageMath required for modular form eigenvalues")

        S = _CuspForms(_Gamma0(level), self.weight)
        newforms = S.newforms('a')
        primes = sieve_primes(self.max_n)

        result: Dict[int, List[float]] = {}
        for p in primes:
            result[p] = []
            for nf in newforms:
                try:
                    a_p = float(nf[p])
                    result[p].append(a_p)
                except (TypeError, IndexError):
                    pass

        return result

    def match_modular_form(self, prices: List[float],
                           levels: List[int] = None) -> Optional[Tuple[int, float]]:
        """
        Find the modular form whose Hecke eigenvalues best match market data.

        Computes market eigenvalues from prices, then compares them to actual
        Hecke eigenvalues from newforms at each level. Returns (best_level, score)
        where score is the average |a_p^market - a_p^form| / ramanujan_bound.
        Lower is better; 0.0 = perfect match.

        Requires SageMath.
        """
        if not _SAGE_AVAILABLE:
            raise RuntimeError("SageMath required for modular form matching")

        if levels is None:
            levels = [11, 14, 15, 17, 19, 20, 21, 23]

        self.set_eigenvalues_from_prices(prices)
        if not self.eigenvalues:
            return None

        primes = sorted(self.eigenvalues.keys())

        best_level = None
        best_score = float('inf')

        for level in levels:
            try:
                form_eigs = self.set_eigenvalues_from_modular_forms(level)
            except Exception:
                continue

            for form_idx in range(max(len(v) for v in form_eigs.values()) if form_eigs else 0):
                total_err = 0.0
                count = 0
                for p in primes:
                    if p not in form_eigs or form_idx >= len(form_eigs[p]):
                        continue
                    market_ap = self.eigenvalues[p]
                    form_ap = form_eigs[p][form_idx]
                    ramanujan = 2.0 * (p ** ((self.weight - 1) / 2.0))
                    total_err += abs(market_ap - form_ap) / ramanujan
                    count += 1

                if count > 0:
                    score = total_err / count
                    if score < best_score:
                        best_score = score
                        best_level = level

        return (best_level, best_score) if best_level is not None else None

    def set_eigenvalues_from_spectral(self, prices: List[float]) -> None:
        """
        Estimate Hecke eigenvalues a_p from spectral analysis (QR on Hankel matrix).

        Instead of naive autocorrelation scaling, this uses the dominant eigenvalue
        of the p×p Hankel autocorrelation matrix — the same object whose
        characteristic polynomial relates to the Eichler-Deligne relation
        T_p^2 - a_p T_p + p^{k-1} = 0.

        The dominant eigenvalue λ_p of the Hankel matrix directly estimates |a_p|,
        with the sign recovered from the lag-p autocorrelation.
        """
        from spectral_window import analyze_scale, log_returns
        rets = log_returns(prices)
        if len(rets) < 4:
            return

        primes = sieve_primes(self.max_n)
        for p in primes:
            if p >= len(rets):
                continue
            spec = analyze_scale(rets, p, weight=self.weight)
            # Use the spectral eigenvalue magnitude, signed by autocorrelation
            sign = 1.0 if spec.autocorr >= 0 else -1.0
            ramanujan = 2.0 * (p ** ((self.weight - 1) / 2.0))
            # Clamp to Ramanujan bound
            a_p = sign * min(spec.eigenvalue_mag * ramanujan, ramanujan)
            self.eigenvalues[p] = a_p

    def reduce(self, poly: Polynomial) -> Polynomial:
        """Reduce a polynomial expression modulo the Hecke ideal."""
        if self.basis is None:
            raise RuntimeError("Call compute_basis() first")
        return self.basis.reduce(poly)

    def on_variety(self, values: List[float], tol: float = -1.0) -> bool:
        """
        Test whether the given operator values lie on the Hecke variety.
        values[i] = numerical value assigned to T_{i+1}.

        If tol < 0 (default), uses a weight-aware tolerance:
            tol = base_tol * max_p^{(k-1)/2}
        This accounts for the Ramanujan bound scaling — higher-weight forms
        and larger primes produce larger eigenvalues, so the absolute
        residual on the variety is naturally larger.
        """
        if self.basis is None:
            raise RuntimeError("Call compute_basis() first")
        if tol < 0:
            primes = sieve_primes(self.max_n)
            max_p = max(primes) if primes else 2
            tol = 1e-4 * (max_p ** ((self.weight - 1) / 2.0))
        return self.basis.on_variety(values, tol)

    # ── Zeta / L-function from Hecke eigenvalues ─────────────────────────────

    def l_function_value(self, prices: List[float], s: float = 0.5) -> complex:
        """
        Partial L-function of the market-estimated modular form at point s:

            L(s, f) = Σ_{n=1}^{max_n} a_n · n^{-s}

        where a_p are Hecke eigenvalues (from prices) and a_n for composite n
        is derived from multiplicativity.

        At s = 1/2 (critical line): L(1/2, f) is where the interesting
        zeros live (Riemann hypothesis for automorphic L-functions).
        Large |L(1/2)| → strong spectral signal → tradeable pattern.

        Returns the complex L-function value.
        """
        self.set_eigenvalues_from_prices(prices)
        if not self.eigenvalues:
            return complex(0, 0)

        result = complex(0, 0)
        for n in range(1, self.max_n + 1):
            a_n = self._multiplicative_extend(n)
            result += a_n * (n ** (-s))
        return result

    def _multiplicative_extend(self, n: int) -> float:
        """
        Extend Hecke eigenvalues from primes to all n via multiplicativity:
          a_1 = 1
          a_{mn} = a_m · a_n  if gcd(m,n)=1
          a_{p^{k+1}} = a_p · a_{p^k} - p^{k-1} · a_{p^{k-1}}
        """
        if n == 1:
            return 1.0

        # Factor n
        result = 1.0
        m = n
        for p in sieve_primes(n):
            if p * p > m:
                break
            if m % p == 0:
                k = 0
                while m % p == 0:
                    k += 1
                    m //= p
                result *= self._prime_power_coeff(p, k)
        if m > 1:  # remaining prime factor
            result *= self._prime_power_coeff(m, 1)
        return result

    def _prime_power_coeff(self, p: int, k: int) -> float:
        """a_{p^k} via the recurrence from Eichler-Deligne."""
        a_p = self.eigenvalues.get(p, 0.0)
        if k == 0:
            return 1.0
        if k == 1:
            return a_p
        # Recurrence: a_{p^{k}} = a_p * a_{p^{k-1}} - p^{w-1} * a_{p^{k-2}}
        prev2 = 1.0     # a_{p^0}
        prev1 = a_p     # a_{p^1}
        pw = p ** (self.weight - 1)
        for _ in range(2, k + 1):
            curr = a_p * prev1 - pw * prev2
            prev2 = prev1
            prev1 = curr
        return prev1

    def euler_product(self, prices: List[float], s: float = 0.5) -> complex:
        """
        Euler product form of the L-function:

            L(s, f) = ∏_p (1 - a_p p^{-s} + p^{k-1-2s})^{-1}

        This converges for Re(s) > (k+1)/2, but we evaluate at s=1/2
        as a formal signal. The product structure means each prime scale
        contributes independently — if one scale is noisy, it only affects
        its own factor.

        Returns the (truncated) Euler product value.
        """
        self.set_eigenvalues_from_prices(prices)
        if not self.eigenvalues:
            return complex(1, 0)

        result = complex(1, 0)
        for p in sieve_primes(self.max_n):
            a_p = self.eigenvalues.get(p, 0.0)
            p_s = p ** (-s)
            p_k = p ** (self.weight - 1 - 2*s)
            local_factor = 1.0 - a_p * p_s + p_k
            if abs(local_factor) > 1e-12:
                result /= local_factor
            else:
                result *= 1e6  # pole — signal is very strong at this scale
        return result

    def zeta_signal(self, prices: List[float]) -> Optional[str]:
        """
        Trading signal from the L-function evaluated on the critical line.

        Uses both the Dirichlet series L(1/2, f) and the Euler product form.
        The key insight: near a zero of L(s, f) on the critical line,
        the spectral structure is transitioning between regimes.

        Returns:
          "strong"   — |L(1/2)| > 2.0, strong spectral structure (tradeable)
          "weak"     — |L(1/2)| < 0.5, near a zero (regime transition, avoid)
          "moderate" — in between
          None       — insufficient data
        """
        L_dir = self.l_function_value(prices, s=0.5)
        L_eul = self.euler_product(prices, s=0.5)

        if L_dir == 0j:
            return None

        # Use geometric mean of both forms for robustness
        mag_dir = abs(L_dir)
        mag_eul = abs(L_eul)
        mag = math.sqrt(mag_dir * min(mag_eul, 100.0))

        if mag > 2.0:
            return "strong"
        elif mag < 0.5:
            return "weak"
        return "moderate"

    def describe(self) -> str:
        lines = [
            f"HeckeAlgebra(max_n={self.max_n}, weight={self.weight}, sage={_SAGE_AVAILABLE})",
            f"  Primes: {sieve_primes(self.max_n)}",
            f"  Eigenvalues: { {p: f'{v:.4f}' for p, v in self.eigenvalues.items()} }",
        ]
        if self.basis:
            lines.append(f"  Gröbner basis size: {self.basis.size()}")
        return "\n".join(lines)
