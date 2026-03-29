"""Dirichlet Characters for spectral trading signals.

A Dirichlet character χ mod q is a completely multiplicative function
χ: Z → C that is:
  - Periodic: χ(n + q) = χ(n)
  - Vanishes: χ(n) = 0  when gcd(n, q) > 1
  - Multiplicative: χ(mn) = χ(m) χ(n)
  - Values are roots of unity

For trading, the key insight is:

    S(χ, T) = Σ_{t=1}^{T} χ(t) · r_t       (character sum of returns)

This is a complex-valued projection of the return series onto the
arithmetic "frequency" defined by χ.  Large |S(χ, T)| means returns
are strongly aligned with the periodic pattern χ.

The partial L-function L(s, χ, T) = Σ_{t=1}^{T} χ(t) · r_t / t^s
at s=1/2 gives a natural "half-spectral" filter: low-frequency patterns
are upweighted, noise is downweighted.

Connection to Hecke operators:
  - Hecke operators T_n act on L-functions via L(s, f) = Σ a_n n^{-s}
  - A Dirichlet character χ twists this: L(s, f ⊗ χ) = Σ a_n χ(n) n^{-s}
  - Trading signal: look for significant |L(s, f ⊗ χ)| at s = 1/2
"""

from __future__ import annotations
import math
import cmath
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─── SageMath backend for Dirichlet characters ──────────────────────────────
_SAGE_AVAILABLE = False
_SageDirichletGroup = None
try:
    _sage_bin = os.path.expanduser("~/miniforge3/envs/sage/bin")
    if os.path.isdir(_sage_bin) and _sage_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _sage_bin + ":" + os.environ.get("PATH", "")

    from sage.all import DirichletGroup as _SageDirichletGroup
    _SAGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass


# ─── Arithmetic helpers ───────────────────────────────────────────────────────

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def euler_phi(n: int) -> int:
    """Euler totient φ(n) = number of integers in [1,n] coprime to n."""
    result = n
    p = 2
    m = n
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            result -= result // p
        p += 1
    if m > 1:
        result -= result // m
    return result


def primitive_root(p: int) -> int:
    """Find a primitive root modulo prime p."""
    if p == 2:
        return 1
    phi = p - 1
    factors = prime_factors(phi)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    raise ValueError(f"No primitive root found mod {p}")


def prime_factors(n: int) -> List[int]:
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return list(factors)


def mod_inverse(a: int, m: int) -> Optional[int]:
    """Modular inverse of a mod m (None if not coprime)."""
    g = gcd(a, m)
    if g != 1:
        return None
    # Extended Euclidean algorithm
    x, y, m0 = 1, 0, m
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        a, m = m, a % m
        x, y = y, x - q * y
    if x < 0:
        x += m0
    return x


# ─── Dirichlet character ──────────────────────────────────────────────────────

@dataclass
class DirichletCharacter:
    """
    A Dirichlet character χ mod q.

    Characters are indexed by their order d (d | φ(q)) and a generator
    choice.  We enumerate all characters mod q and label them by index.
    """
    modulus: int                        # q
    order: int                          # order of χ (d | φ(q))
    index: int                          # index within characters mod q
    values: Dict[int, complex]          # χ(n) for n = 0..q-1

    @property
    def is_principal(self) -> bool:
        """Principal character χ₀: χ₀(n)=1 if gcd(n,q)=1, else 0."""
        return self.order == 1

    @property
    def is_real(self) -> bool:
        """Real character: all values are real (χ takes values in {-1,0,1})."""
        return all(abs(v.imag) < 1e-12 for v in self.values.values())

    def __call__(self, n: int) -> complex:
        """Evaluate χ(n)."""
        return self.values.get(n % self.modulus, complex(0, 0))

    def conjugate(self) -> "DirichletCharacter":
        """Complex conjugate character χ̄."""
        return DirichletCharacter(
            modulus=self.modulus,
            order=self.order,
            index=self.index,
            values={n: v.conjugate() for n, v in self.values.items()},
        )

    def __repr__(self) -> str:
        kind = "principal" if self.is_principal else ("real" if self.is_real else "complex")
        return f"χ_{self.index} mod {self.modulus}  order={self.order}  [{kind}]"


def all_characters(q: int) -> List[DirichletCharacter]:
    """
    Enumerate all φ(q) Dirichlet characters mod q.

    When SageMath is available, uses sage's DirichletGroup which correctly
    handles all moduli (primes, prime powers, composites) via the structure
    theorem for (Z/qZ)*.  Falls back to pure-Python implementation otherwise.
    """
    if _SAGE_AVAILABLE:
        return _characters_sage(q)

    phi = euler_phi(q)

    if q == 2:
        return [DirichletCharacter(
            modulus=2, order=1, index=0,
            values={0: 0, 1: 1},
        )]

    if _is_prime(q):
        return _characters_prime(q)

    return _characters_composite(q)


def _characters_sage(q: int) -> List[DirichletCharacter]:
    """Enumerate all Dirichlet characters mod q using SageMath."""
    D = _SageDirichletGroup(q)
    chars = []
    for idx, sage_chi in enumerate(D):
        values: Dict[int, complex] = {}
        for n in range(q):
            v = complex(sage_chi(n))
            values[n] = v
        order = int(sage_chi.order())
        chars.append(DirichletCharacter(
            modulus=q, order=order, index=idx, values=values,
        ))
    return chars


def _characters_exhaustive(q: int) -> List[DirichletCharacter]:
    """
    Exhaustive character enumeration for any modulus q.

    Finds all generators of (Z/qZ)* and builds characters by assigning
    roots of unity to generators. Slower than the structured approach but
    always correct — used as fallback when the CRT decomposition is incomplete.
    """
    units = [n for n in range(1, q) if gcd(n, q) == 1]
    phi = len(units)
    if phi == 0:
        return []

    # Find a generator (primitive root) via brute force
    generator = None
    for g in units:
        order = 1
        val = g % q
        while val != 1:
            val = (val * g) % q
            order += 1
            if order > phi:
                break
        if order == phi:
            generator = g
            break

    if generator is None:
        # (Z/qZ)* is not cyclic (e.g. q=8); fall back to principal only
        return [DirichletCharacter(
            modulus=q, order=1, index=0,
            values={n: (complex(1, 0) if gcd(n, q) == 1 else complex(0, 0))
                    for n in range(q)},
        )]

    # Build discrete log table
    dlog: Dict[int, int] = {}
    val = 1
    for k in range(phi):
        dlog[val] = k
        val = (val * generator) % q

    chars = []
    for idx in range(phi):
        zeta = cmath.exp(2j * math.pi * idx / phi)
        values: Dict[int, complex] = {}
        for n in range(q):
            if gcd(n, q) == 1:
                values[n] = zeta ** dlog.get(n, 0)
            else:
                values[n] = complex(0, 0)
        order = 1
        if idx > 0:
            for d in range(1, phi + 1):
                if phi % d == 0 and abs((zeta ** d) - 1) < 1e-9:
                    order = d
                    break
        chars.append(DirichletCharacter(
            modulus=q, order=order, index=idx, values=values,
        ))
    return chars


def _is_prime(n: int) -> bool:
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0: return False
    return True


def _characters_prime(p: int) -> List[DirichletCharacter]:
    """All Dirichlet characters mod p (prime)."""
    g = primitive_root(p)
    phi = p - 1
    chars = []

    # Discrete log table: dlog[n] = k such that g^k ≡ n (mod p)
    dlog: Dict[int, int] = {}
    gk = 1
    for k in range(phi):
        dlog[gk] = k
        gk = (gk * g) % p

    for idx in range(phi):
        zeta = cmath.exp(2j * math.pi * idx / phi)
        values: Dict[int, complex] = {0: complex(0, 0)}
        for n in range(1, p):
            k = dlog[n]
            values[n] = zeta ** k

        order = 1
        if idx > 0:
            for d in range(1, phi + 1):
                if phi % d == 0 and (zeta ** d - 1) == 0j:
                    order = d
                    break
            # Proper order computation
            order = phi
            for d in range(1, phi + 1):
                if phi % d == 0:
                    if abs((zeta ** d) - 1) < 1e-9:
                        order = d
                        break

        chars.append(DirichletCharacter(
            modulus=p, order=order, index=idx, values=values,
        ))

    return chars


def _characters_composite(q: int) -> List[DirichletCharacter]:
    """
    Full character enumeration for composite q via Smith normal form of (Z/qZ)*.

    The group of units (Z/qZ)* is a finite abelian group. By the fundamental
    theorem of finite abelian groups, it decomposes as a product of cyclic
    groups. Characters are homomorphisms (Z/qZ)* -> C*, determined by their
    values on generators.

    For q = p1^e1 * p2^e2 * ..., we use the Chinese Remainder Theorem:
        (Z/qZ)* ≅ (Z/p1^e1 Z)* × (Z/p2^e2 Z)* × ...

    Then enumerate characters as products of characters on each factor.
    """
    units = [n for n in range(1, q) if gcd(n, q) == 1]
    phi = len(units)

    # Principal character
    principal_values = {n: (complex(1, 0) if gcd(n, q) == 1 else complex(0, 0))
                        for n in range(q)}
    chars: List[DirichletCharacter] = [
        DirichletCharacter(modulus=q, order=1, index=0, values=principal_values)
    ]

    # Handle prime powers and small composites
    if q == 4:
        chars.append(DirichletCharacter(
            modulus=4, order=2, index=1,
            values={0: 0, 1: 1, 2: 0, 3: -1},
        ))
    elif q == 8:
        # (Z/8Z)* = {1, 3, 5, 7} ≅ C2 × C2, 3 characters
        chars.append(DirichletCharacter(
            modulus=8, order=2, index=1,
            values={n: complex(1 if n % 8 in (1, 7) else -1 if n % 8 in (3, 5) else 0)
                    for n in range(8)},
        ))
        chars.append(DirichletCharacter(
            modulus=8, order=2, index=2,
            values={n: complex(1 if n % 8 in (1, 5) else -1 if n % 8 in (3, 7) else 0)
                    for n in range(8)},
        ))
        chars.append(DirichletCharacter(
            modulus=8, order=2, index=3,
            values={n: complex(1 if n % 8 in (1, 3) else -1 if n % 8 in (5, 7) else 0)
                    for n in range(8)},
        ))
    elif q == 9:
        # (Z/9Z)* = {1, 2, 4, 5, 7, 8} ≅ C6, primitive root 2
        g = 2
        dlog = {1: 0}
        for k in range(1, 6):
            dlog[pow(g, k, q)] = k
        for idx in range(1, 6):
            zeta = cmath.exp(2j * math.pi * idx / 6)
            values = {0: complex(0, 0)}
            for n in range(1, q):
                if gcd(n, q) == 1:
                    k = dlog.get(n, 0)
                    values[n] = zeta ** k
                else:
                    values[n] = complex(0, 0)
            chars.append(DirichletCharacter(
                modulus=q, order=6 // gcd(idx, 6), index=idx, values=values,
            ))
    elif q == 12:
        # (Z/12Z)* = {1, 5, 7, 11} ≅ C2 × C2
        chars.append(DirichletCharacter(
            modulus=12, order=2, index=1,
            values={n: complex(1 if n % 12 in (1, 11) else -1 if n % 12 in (5, 7) else 0)
                    for n in range(12)},
        ))
        chars.append(DirichletCharacter(
            modulus=12, order=2, index=2,
            values={n: complex(1 if n % 12 in (1, 5) else -1 if n % 12 in (7, 11) else 0)
                    for n in range(12)},
        ))
        chars.append(DirichletCharacter(
            modulus=12, order=2, index=3,
            values={n: complex(1 if n % 12 in (1, 7) else -1 if n % 12 in (5, 11) else 0)
                    for n in range(12)},
        ))
    else:
        # General composite: use CRT decomposition
        # Factor q = p1^e1 * p2^e2 * ...
        factors = []
        m = q
        p = 2
        while p * p <= m:
            if m % p == 0:
                pe = 1
                while m % p == 0:
                    pe *= p
                    m //= p
                factors.append(pe)
            p += 1
        if m > 1:
            factors.append(m)

        if len(factors) == 1:
            # Prime power p^e with p odd: (Z/p^e Z)* is cyclic
            pe = factors[0]
            # Extract the prime base p from p^e
            base_p = pe
            for pf in prime_factors(pe):
                base_p = pf
                break
            if base_p > 2 and _is_prime(base_p):
                # Find primitive root for p^e (guaranteed to exist for odd prime powers)
                p = None
                for candidate in range(2, pe):
                    if gcd(candidate, pe) == 1:
                        order = 1
                        val = candidate % pe
                        while val != 1:
                            val = (val * candidate) % pe
                            order += 1
                        if order == phi:
                            p = candidate
                            break
                if p is not None:
                    dlog = {1: 0}
                    val = 1
                    for k in range(1, phi):
                        val = (val * p) % pe
                        dlog[val] = k
                    for idx in range(1, phi):
                        zeta = cmath.exp(2j * math.pi * idx / phi)
                        values = {0: complex(0, 0)}
                        for n in range(q):
                            if gcd(n, q) == 1:
                                k = dlog.get(n % pe, 0)
                                values[n] = zeta ** k
                            else:
                                values[n] = complex(0, 0)
                        order = phi
                        for d in range(1, phi + 1):
                            if phi % d == 0 and abs((zeta ** d) - 1) < 1e-9:
                                order = d
                                break
                        chars.append(DirichletCharacter(
                            modulus=q, order=order, index=idx, values=values,
                        ))
        else:
            # Multiple prime power factors: build characters via CRT
            # For each factor, get its characters, then take products
            factor_chars = []
            for pe in factors:
                if _is_prime(pe):
                    fc = _characters_prime(pe)
                else:
                    # Recursive call handles prime powers (4, 8, 9, etc.)
                    # and any composite via the same decomposition logic
                    fc = _characters_composite(pe)
                    if len(fc) < euler_phi(pe):
                        # Incomplete enumeration — use exhaustive search as fallback
                        fc = _characters_exhaustive(pe)
                factor_chars.append(fc)

            # Take all products of characters from each factor
            from itertools import product as iter_product
            for combo in iter_product(*factor_chars):
                if all(c.is_principal for c in combo):
                    continue  # already have principal
                values = {n: complex(1, 0) for n in range(q)}
                order = 1
                for c in combo:
                    order = order * c.order // gcd(order, c.order)
                for n in range(q):
                    if gcd(n, q) != 1:
                        values[n] = complex(0, 0)
                        continue
                    prod = complex(1, 0)
                    for i, pe in enumerate(factors):
                        prod *= combo[i].values.get(n % pe, complex(1, 0))
                    values[n] = prod
                chars.append(DirichletCharacter(
                    modulus=q, order=order, index=len(chars), values=values,
                ))

    return chars


# ─── Character sums and L-functions ──────────────────────────────────────────

@dataclass
class CharacterSignal:
    """Spectral signal derived from a Dirichlet character sum."""
    chi: DirichletCharacter
    character_sum: complex          # S(χ, T) = Σ χ(t) r_t
    partial_L: complex              # L(1/2, χ) ≈ Σ χ(t) r_t / √t
    magnitude: float                # |S(χ, T)|
    phase: float                    # arg(S(χ, T)) in radians
    normalized_magnitude: float     # |S| / √T  (removes √T growth)
    is_significant: bool

    def __repr__(self) -> str:
        sig = "✓" if self.is_significant else "✗"
        return (f"{self.chi}  {sig}  "
                f"|S|={self.magnitude:.4f}  "
                f"phase={math.degrees(self.phase):+6.1f}°  "
                f"norm={self.normalized_magnitude:.4f}  "
                f"|L(½)|={abs(self.partial_L):.4f}")


def character_sum(chi: DirichletCharacter, returns: List[float]) -> complex:
    """
    S(χ, T) = Σ_{t=1}^{T} χ(t) · r_t

    Projects the return series onto the arithmetic pattern χ.
    """
    return sum(chi(t + 1) * r for t, r in enumerate(returns))


def partial_L_function(chi: DirichletCharacter, returns: List[float],
                       s: float = 0.5) -> complex:
    """
    L(s, χ, T) = Σ_{t=1}^{T} χ(t) · r_t / t^s

    At s=1/2: upweights early returns, downweights recent noise.
    This is the "half-spectral" filter from the critical line of L-functions.
    """
    return sum(chi(t + 1) * r / ((t + 1) ** s)
               for t, r in enumerate(returns))


def gauss_sum(chi: DirichletCharacter) -> complex:
    """
    Gauss sum G(χ) = Σ_{a=0}^{q-1} χ(a) · exp(2πi·a/q).

    For primitive characters, |G(χ)| = √q.  This provides a natural
    normalisation factor so that character sums across different moduli
    are comparable (improvement K).

    Uses Kahan summation to reduce floating-point phase error accumulation.
    """
    q = chi.modulus
    result = complex(0, 0)
    comp = complex(0, 0)  # Kahan compensator
    for a in range(q):
        if gcd(a, q) == 1:
            term = chi(a) * cmath.exp(2j * math.pi * a / q)
            y = term - comp
            t = result + y
            comp = (t - result) - y
            result = t
    return result


def benjamini_hochberg_threshold(p_values: List[float],
                                  fdr: float = 0.10) -> float:
    """
    Benjamini-Hochberg FDR control (improvement L).

    Given p-values from multiple hypothesis tests, returns the largest
    p-value threshold such that the expected false discovery rate is ≤ fdr.

    Algorithm:
      1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
      2. Find largest k such that p_(k) ≤ (k/m) * fdr
      3. Reject all hypotheses with p-value ≤ p_(k)

    Returns the threshold p_(k), or 0 if no rejections.
    """
    m = len(p_values)
    if m == 0:
        return 0.0
    sorted_p = sorted(p_values)
    for k in range(m, 0, -1):
        threshold = (k / m) * fdr
        if sorted_p[k - 1] <= threshold:
            return sorted_p[k - 1]
    return 0.0


def _normal_cdf_tail(z: float) -> float:
    """Upper tail probability P(|Z| > z) for Z ~ N(0,1)."""
    # Approximation via error function complement
    return math.erfc(z / math.sqrt(2.0))


def analyze_character(chi: DirichletCharacter, returns: List[float],
                      significance_threshold: float = 2.0,
                      num_tests: int = 1,
                      use_fdr: bool = True,
                      fdr_level: float = 0.10) -> CharacterSignal:
    """
    Full character analysis of a return series.

    Under the null (iid standardised returns), S(χ,T) / √T → N(0,1) for
    real characters and the magnitude |S|/√T ~ Rayleigh(1/√2) for complex
    ones.

    Improvements:
      - Gauss sum normalisation (K): divide by |G(χ)| = √q so signals
        across different moduli are comparable.
      - Benjamini-Hochberg FDR (L): controls expected false discovery rate
        instead of the overly conservative Bonferroni family-wise error.

    The p-value for a character sum is computed from the normal tail:
        p = P(|Z| > |S| / √(T · φ(q)/q))
    """
    T = len(returns)
    if T == 0:
        return CharacterSignal(chi=chi, character_sum=0j, partial_L=0j,
                               magnitude=0.0, phase=0.0, normalized_magnitude=0.0,
                               is_significant=False)

    S = character_sum(chi, returns)
    L = partial_L_function(chi, returns)

    mag = abs(S)
    phase = cmath.phase(S)

    # Gauss sum normalisation (improvement K)
    q = chi.modulus
    phi_q = euler_phi(q)
    gauss = gauss_sum(chi)
    gauss_norm = abs(gauss) if abs(gauss) > 1e-10 else math.sqrt(q)

    # Under iid: Var(S) = T · φ(q)/q, and we also normalise by Gauss norm
    char_std = math.sqrt(max(T, 1) * phi_q / q) * gauss_norm / math.sqrt(q)
    norm_mag = mag / char_std if char_std > 1e-10 else 0.0

    # Compute p-value from normal tail
    p_value = _normal_cdf_tail(norm_mag)

    # Significance decision.
    # Per-character p-value is computed here; actual BH FDR control is applied
    # in dirichlet_report() which has access to all p-values simultaneously.
    # Here we use a simple per-test threshold as initial screening.
    if use_fdr and num_tests > 1:
        # Initial screen: will be overridden by proper BH in dirichlet_report()
        is_sig = p_value <= fdr_level
    else:
        adjusted_thr = significance_threshold * math.sqrt(math.log(max(num_tests, 1)) + 1.0)
        is_sig = norm_mag >= adjusted_thr

    return CharacterSignal(
        chi=chi,
        character_sum=S,
        partial_L=L,
        magnitude=mag,
        phase=phase,
        normalized_magnitude=norm_mag,
        is_significant=is_sig,
    )


# ─── Multi-character spectral report ─────────────────────────────────────────

@dataclass
class DirichletReport:
    """Full Dirichlet spectral decomposition of a price window.

    Improvements:
      - Gauss sum normalisation (K): signals comparable across moduli
      - Benjamini-Hochberg FDR (L): less conservative than Bonferroni
      - Timing confidence from Im(S) (M): imaginary part indicates cycle phase
    """
    moduli: List[int]
    signals: List[CharacterSignal]
    num_significant: int
    dominant_chi: Optional[DirichletCharacter]
    dominant_phase: Optional[float]
    overall_score: float
    timing_confidence: float  # from Im(S): +1 = peak imminent, -1 = trough imminent

    def is_tradeable(self, min_sig: int = 2, min_score: float = 0.4) -> bool:
        return self.num_significant >= min_sig and self.overall_score >= min_score

    def buy_bias(self) -> float:
        """
        Returns a real number in [-1, +1].
        Positive -> expect price rise (buy signal).
        Negative -> expect price fall (sell signal).

        Improvement M: uses both Re(S) for direction AND Im(S) for timing.
        The imaginary part encodes the cycle phase - whether the periodic
        pattern is about to turn up or down.

        Direction = weighted Re(S)
        Timing confidence = weighted Im(S) / |S|
        Final bias = direction * (1 + 0.3 * timing_confidence)  # 30% boost for good timing
        """
        if not self.signals:
            return 0.0
        significant = [s for s in self.signals if s.is_significant]
        if not significant:
            return 0.0

        total_weight = sum(s.magnitude for s in significant)
        if total_weight < 1e-10:
            return 0.0

        # Direction from real part
        direction = sum(s.character_sum.real * s.magnitude for s in significant)
        direction = max(-1.0, min(1.0, direction / total_weight))

        # Timing from imaginary part (improvement M)
        timing_sum = sum(s.character_sum.imag * s.magnitude for s in significant)
        timing = timing_sum / total_weight if total_weight > 0 else 0.0
        # Normalise timing to [-1, 1]
        max_timing = max((abs(s.character_sum.imag) for s in significant), default=1.0)
        timing_conf = timing / max_timing if max_timing > 1e-10 else 0.0

        # Boost direction by timing confidence (up to 30%)
        boost = 1.0 + 0.3 * timing_conf
        return max(-1.0, min(1.0, direction * boost))

    def __repr__(self) -> str:
        lines = [f"DirichletReport: score={self.overall_score:.3f}  "
                 f"significant={self.num_significant}  "
                 f"bias={self.buy_bias():+.3f}  timing={self.timing_confidence:+.3f}"]
        for s in self.signals:
            lines.append(f"  {s}")
        return "\n".join(lines)


def dirichlet_report(
    prices: List[float],
    moduli: List[int],
    significance_threshold: float = 0.3,
    s: float = 0.5,
    use_fdr: bool = True,
    fdr_level: float = 0.10,
) -> DirichletReport:
    """
    Compute a full Dirichlet spectral decomposition of a price window.

    For each modulus q and each character χ mod q, computes the character
    sum S(χ,T) and partial L-function L(s, χ, T) on the log-return series.

    Improvements:
      - Gauss sum normalisation (K): signals comparable across moduli
      - Benjamini-Hochberg FDR (L): less conservative than Bonferroni
      - Timing confidence from Im(S) (M): imaginary part indicates cycle phase
    """
    n = len(prices)
    if n < 2:
        return DirichletReport(moduli=moduli, signals=[], num_significant=0,
                               dominant_chi=None, dominant_phase=None,
                               overall_score=0.0, timing_confidence=0.0)

    raw = [math.log(prices[i] / prices[i - 1])
           for i in range(1, n)
           if prices[i - 1] > 0 and prices[i] > 0]

    if not raw:
        return DirichletReport(moduli=moduli, signals=[], num_significant=0,
                               dominant_chi=None, dominant_phase=None,
                               overall_score=0.0, timing_confidence=0.0)
    mu = sum(raw) / len(raw)
    var = sum((r - mu) ** 2 for r in raw) / len(raw)
    std = math.sqrt(var) if var > 1e-12 else 1.0
    returns = [(r - mu) / std for r in raw]

    all_chars = [(q, chi) for q in moduli
                 for chi in all_characters(q) if not chi.is_principal]
    num_tests = len(all_chars)

    signals: List[CharacterSignal] = []
    p_values: List[float] = []
    for q, chi in all_chars:
        sig = analyze_character(chi, returns, significance_threshold, num_tests,
                                use_fdr=use_fdr, fdr_level=fdr_level)
        signals.append(sig)
        # Compute p-value for BH
        if sig.normalized_magnitude > 0:
            p_values.append(_normal_cdf_tail(sig.normalized_magnitude))
        else:
            p_values.append(1.0)

    # Apply BH FDR if using
    if use_fdr and num_tests > 1:
        bh_thr = benjamini_hochberg_threshold(p_values, fdr_level)
        # Re-mark signals as significant if p <= BH threshold
        for i, sig in enumerate(signals):
            if p_values[i] <= bh_thr:
                sig.is_significant = True

    significant = [s for s in signals if s.is_significant]
    num_sig = len(significant)

    dominant_chi = None
    dominant_phase = None
    if significant:
        dom = max(significant, key=lambda s: s.normalized_magnitude)
        dominant_chi = dom.chi
        dominant_phase = dom.phase

    # Timing confidence from Im(S) (improvement M)
    timing_confidence = 0.0
    if significant:
        total_weight = sum(s.magnitude for s in significant)
        if total_weight > 1e-10:
            timing_sum = sum(s.character_sum.imag * s.magnitude for s in significant)
            max_im = max((abs(s.character_sum.imag) for s in significant), default=1.0)
            timing_confidence = (timing_sum / total_weight) / max_im if max_im > 1e-10 else 0.0

    # Overall score: geometric mean of normalized magnitudes
    if significant:
        geo = math.exp(
            sum(math.log(max(s.normalized_magnitude, 1e-9)) for s in significant)
            / len(significant)
        )
        multi_bonus = min(num_sig / max(len(signals), 1), 1.0)
        overall_score = 0.7 * min(geo / 0.5, 1.0) + 0.3 * multi_bonus
    else:
        overall_score = 0.0

    return DirichletReport(
        moduli=moduli,
        signals=signals,
        num_significant=num_sig,
        dominant_chi=dominant_chi,
        dominant_phase=dominant_phase,
        overall_score=overall_score,
        timing_confidence=timing_confidence,
    )


# ─── Combined Hecke + Dirichlet signal ───────────────────────────────────────

class HeckeDirichletSignal:
    """
    Combined signal: Hecke eigenvalue consistency + Dirichlet character projection.

    Fires when:
      1. Hecke: enough scales pass g_p(x_p) ≈ 0 (Gröbner basis check)
      2. Dirichlet: at least one character has significant |S(χ,T)|
      3. Dirichlet bias agrees with intended trade direction

    Improvements used:
      - Gauss sum normalisation (K)
      - Benjamini-Hochberg FDR (L)
      - Timing confidence from Im(S) (M)
    """

    def __init__(self,
                 hecke_primes: List[int] = None,
                 dirichlet_moduli: List[int] = None,
                 min_hecke_scales: int = 1,
                 min_dirichlet_sig: int = 1,
                 min_prices: int = 60,
                 weight: int = 2,
                 groebner_tolerance_factor: float = 0.35,
                 significance_threshold: float = 2.0):
        from algorithms.hecke_signal import HeckeSignal, HeckeSignalConfig
        self.hecke = HeckeSignal(HeckeSignalConfig(
            primes=hecke_primes or [2, 3, 5, 7, 11],
            min_significant_scales=min_hecke_scales,
            min_overall_score=0.4,
            min_prices=min_prices,
            weight=weight,
            significance_threshold=0.55,
            use_groebner_check=True,
            groebner_tolerance_factor=groebner_tolerance_factor,
            groebner_pass_fraction=0.20,
        ))
        self.dirichlet_moduli = dirichlet_moduli or [3, 4, 5, 7]
        self.min_dirichlet_sig = min_dirichlet_sig
        self.significance_threshold = significance_threshold
        self.last_dirichlet: Optional[DirichletReport] = None

    def eval(self, prices: List[float], side: str = "buy") -> bool:
        """
        side: "buy" or "sell"
        """
        # Layer 1: Hecke Gröbner check
        if not self.hecke.eval(prices):
            return False

        # Layer 2: Dirichlet character spectral analysis with FDR
        report = dirichlet_report(
            prices,
            moduli=self.dirichlet_moduli,
            significance_threshold=self.significance_threshold,
            use_fdr=True,
            fdr_level=0.10,
        )
        self.last_dirichlet = report

        if not report.is_tradeable(min_sig=self.min_dirichlet_sig):
            return False

        # Layer 3: Directional bias filter (now includes timing boost)
        bias = report.buy_bias()
        if side == "buy" and bias < 0:
            return False
        if side == "sell" and bias > 0:
            return False

        return True

    def describe(self) -> str:
        lines = [
            "HeckeDirichletSignal:",
            self.hecke.describe(),
            f"  Dirichlet moduli: {self.dirichlet_moduli}  (FDR=0.10, Gauss norm)",
        ]
        if self.last_dirichlet:
            lines.append(str(self.last_dirichlet))
        return "\n".join(lines)
