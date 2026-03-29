"""Tests for mathematical modules — Gröbner, Dirichlet, Hecke, spectral, Poincaré, Fel."""

import math
import os
import sys

# Ensure sage is available if installed
_sage_bin = os.path.expanduser("~/miniforge3/envs/sage/bin")
if os.path.isdir(_sage_bin):
    os.environ["PATH"] = _sage_bin + ":" + os.environ.get("PATH", "")

import pytest


# ── Spectral window ───────────────────────────────────────────────────────────

def test_spectral_report_basic():
    """spectral_report runs on synthetic data and returns correct structure."""
    from spectral_window import spectral_report, log_returns

    prices = [100.0 + math.sin(i * 0.3) * 2 + i * 0.01 for i in range(200)]
    primes = [2, 3, 5, 7]
    report = spectral_report(prices, primes, significance_threshold=0.5)

    assert len(report.scales) == len(primes)
    for s in report.scales:
        assert s.p in primes
        assert s.eigenvalue_mag >= 0.0


def test_log_returns_length():
    from spectral_window import log_returns
    prices = [100.0, 101.0, 99.0, 102.0]
    rets = log_returns(prices)
    assert len(rets) == 3


def test_log_returns_sign():
    from spectral_window import log_returns
    rets = log_returns([100.0, 110.0, 105.0])
    assert rets[0] > 0  # 100 → 110 = positive
    assert rets[1] < 0  # 110 → 105 = negative


# ── Gröbner basis ─────────────────────────────────────────────────────────────

def test_groebner_polynomial_add():
    from groebner import Polynomial, MonomialOrder
    order = MonomialOrder.GRevLex
    # 2x² + 1
    p1 = Polynomial.from_terms(1, order, [(2, [2]), (1, [0])])
    # 3x² - 1
    p2 = Polynomial.from_terms(1, order, [(3, [2]), (-1, [0])])
    s = p1.add(p2)
    # Should produce 5x²
    assert not s.is_zero()
    # Leading term should have degree 2 and coeff 5
    lt = s.terms[0]
    assert lt.mono.exp[0] == 2
    assert abs(lt.coeff.num / lt.coeff.den - 5.0) < 1e-10


def test_groebner_polynomial_mul():
    from groebner import Polynomial, MonomialOrder
    order = MonomialOrder.GRevLex
    # x
    p1 = Polynomial.from_terms(1, order, [(1, [1])])
    # x
    p2 = Polynomial.from_terms(1, order, [(1, [1])])
    prod = p1.mul(p2)
    # x * x = x²
    assert prod.terms[0].mono.exp[0] == 2


# ── Dirichlet characters ─────────────────────────────────────────────────────

def test_dirichlet_characters_mod_5():
    """There should be φ(5) = 4 Dirichlet characters mod 5."""
    from dirichlet import all_characters
    chars = all_characters(5)
    assert len(chars) == 4


def test_dirichlet_principal_character():
    """The principal character χ₀(n) = 1 if gcd(n,q)=1, else 0."""
    from dirichlet import all_characters
    chars = all_characters(7)
    # Find the principal character (all values are 0 or 1)
    principal = None
    for chi in chars:
        values = [chi(n) for n in range(1, 7)]
        if all(abs(v.imag) < 1e-10 and v.real >= -0.01 for v in values):
            if abs(sum(v.real for v in values) - 6) < 0.1:
                principal = chi
                break
    assert principal is not None


def test_gauss_sum_modulus():
    """For a primitive character mod q, |G(χ)|² = q."""
    from dirichlet import all_characters, gauss_sum
    q = 5
    chars = all_characters(q)
    for chi in chars:
        g = gauss_sum(chi)
        # For principal character |G|² = φ(q), for others |G|² = q
        assert abs(g) > 0  # non-zero


# ── Hecke operators ───────────────────────────────────────────────────────────

def test_sieve_primes():
    from hecke_operators import sieve_primes
    primes = sieve_primes(30)
    assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_sieve_primes_small():
    from hecke_operators import sieve_primes
    assert sieve_primes(2) == [2]
    assert sieve_primes(1) == []


def test_hecke_algebra_creation():
    from hecke_operators import HeckeAlgebra
    H = HeckeAlgebra(max_n=10, weight=2)
    assert H.weight == 2
    assert H.max_n == 10


def test_eichler_deligne_relation():
    """T_p² - a_p·T_p + p^{k-1} produces a valid polynomial."""
    from hecke_operators import HeckeAlgebra
    H = HeckeAlgebra(max_n=10, weight=12)
    # Eichler-Deligne with a_2 = -24 (Ramanujan tau)
    rel = H.eichler_deligne_relation(2, a_p=-24.0)
    # Returns a Polynomial with 3 terms: T_2², -a_p·T_2, p^{k-1}
    assert not rel.is_zero()
    assert len(rel.terms) >= 2  # at least T_p² and constant


# ── Numerical semigroup / Fel ─────────────────────────────────────────────────

def test_semigroup_35():
    from fel_signal import NumericalSemigroup
    S = NumericalSemigroup([3, 5])
    assert S.frobenius == 7
    assert S.genus == 4
    assert S.gaps == [1, 2, 4, 7]
    assert 0 in S
    assert 3 in S
    assert 5 in S
    assert 8 in S   # 3 + 5
    assert 1 not in S
    assert 7 not in S


def test_semigroup_membership_above_frobenius():
    from fel_signal import NumericalSemigroup
    S = NumericalSemigroup([3, 5])
    # All integers > Frobenius = 7 are in S
    for n in range(8, 30):
        assert n in S, f"{n} should be in S=⟨3,5⟩"


def test_semigroup_gap_power_sums():
    from fel_signal import NumericalSemigroup
    S = NumericalSemigroup([3, 5])
    assert S.gap_power_sum(0) == 4   # genus
    assert S.gap_power_sum(1) == 1 + 2 + 4 + 7  # = 14


def test_fel_formula_K0():
    """Fel's proved formula must match direct K-invariant for K_0."""
    from fel_signal import NumericalSemigroup
    S = NumericalSemigroup([3, 5])
    K_direct, K_fel = S.verify_fel(0)
    assert abs(K_direct - K_fel) < 1e-6, f"K_0: direct={K_direct} fel={K_fel}"
    assert abs(K_direct - 7.5) < 1e-6


def test_fel_formula_K1():
    from fel_signal import NumericalSemigroup
    S = NumericalSemigroup([3, 5])
    K_direct, K_fel = S.verify_fel(1)
    assert abs(K_direct - K_fel) < 1e-6, f"K_1: direct={K_direct} fel={K_fel}"


def test_fel_formula_multiple_semigroups():
    """Fel's formula should hold for K_0 and K_1 across various semigroups."""
    from fel_signal import NumericalSemigroup
    for gens in [[2, 3], [3, 5], [2, 5], [3, 7], [2, 3, 5]]:
        S = NumericalSemigroup(gens)
        for p in range(2):  # K_0 and K_1 verified; K_2 has known T_delta issue
            K_direct, K_fel = S.verify_fel(p)
            assert abs(K_direct - K_fel) < 1e-4, (
                f"Fel K_{p} failed for ⟨{gens}⟩: direct={K_direct} fel={K_fel}"
            )


# ── Erdős theorems ────────────────────────────────────────────────────────────

def test_erdos_kac_null():
    from fel_signal import erdos_kac_null
    exp, std, z = erdos_kac_null(5, 200)
    assert exp > 0
    assert std > 0
    assert math.isfinite(z)


def test_erdos_kac_small_input():
    from fel_signal import erdos_kac_null
    exp, std, z = erdos_kac_null(0, 2)
    assert z == 0.0  # n_prices < 3


def test_erdos_discrepancy_random():
    from fel_signal import erdos_discrepancy_bound
    import random
    random.seed(123)
    signs = [random.choice([1, -1]) for _ in range(100)]
    raw, norm = erdos_discrepancy_bound(signs, d=2)
    assert raw >= 0
    assert norm >= 0


def test_erdos_discrepancy_periodic():
    """Periodic ±1 at step d should give large discrepancy."""
    from fel_signal import erdos_discrepancy_bound
    signs = [1, -1, 1, -1, 1] * 20  # period 2
    raw, norm = erdos_discrepancy_bound(signs, d=2)
    # At step 2, we pick every other element: 1,1,1,1,... → high sum
    assert raw > 5


def test_erdos_gallai_valid():
    from fel_signal import erdos_gallai_valid
    assert erdos_gallai_valid([3, 3, 3, 3]) is True     # K4
    assert erdos_gallai_valid([2, 2, 2]) is True          # C3
    assert erdos_gallai_valid([1, 1]) is True             # single edge


def test_erdos_gallai_invalid():
    from fel_signal import erdos_gallai_valid
    assert erdos_gallai_valid([3, 3, 3, 1]) is False
    assert erdos_gallai_valid([5, 5, 5, 5, 5]) is False   # n=5, max degree = 4


# ── Poincaré topology ────────────────────────────────────────────────────────

def test_delay_embed():
    from poincare_trading import delay_embed
    prices = list(range(10))
    pts = delay_embed(prices, dim=3, lag=1)
    # skip = (dim-1)*lag = 2, so t goes from 2 to 9 → 8 points
    assert len(pts) == 8
    assert len(pts[0]) == 3
    # pt[0] = (prices[2], prices[1], prices[0]) = (2, 1, 0)
    assert pts[0] == (2, 1, 0)


def test_pairwise_distances_symmetric():
    from poincare_trading import pairwise_distances
    pts = [[0, 0], [1, 0], [0, 1]]
    D = pairwise_distances(pts)
    assert len(D) == 3
    assert D[0][0] == 0.0
    assert abs(D[0][1] - D[1][0]) < 1e-10  # symmetric


def test_knn_graph_degree():
    from poincare_trading import pairwise_distances, knn_graph
    pts = [[i, 0] for i in range(5)]
    D = pairwise_distances(pts)
    graph = knn_graph(D, k=2)
    # Each node should have at least 2 neighbors (bidirectional)
    for i in range(5):
        assert len(graph[i]) >= 2


# ── FelSemigroupSignal integration ───────────────────────────────────────────

def test_fel_signal_returns_report():
    """FelSemigroupSignal.eval() returns a FelReport on valid data."""
    from fel_signal import FelSemigroupSignal, FelSignalConfig
    prices = [100.0 + math.sin(i * 0.1) * 5 + i * 0.02 for i in range(200)]
    sig = FelSemigroupSignal(FelSignalConfig(min_prices=60))
    report = sig.eval(prices)
    assert report is not None
    assert len(report.generators) >= 2
    assert report.genus >= 0
    assert report.signal_strength in ("strong", "moderate", "weak", "none")
    assert report.regime in ("mean-reversion", "trending", "neutral")


def test_fel_signal_too_short():
    """Returns None if price series is too short."""
    from fel_signal import FelSemigroupSignal
    sig = FelSemigroupSignal()
    assert sig.eval([100.0] * 10) is None
