"""Tests for p-adic number arithmetic."""

import pytest
from p_adic import PAdicNumber, padic_from_integer, LargeNumberArithmetic


# ── Basic p-adic Tests ───────────────────────────────────────────────────────

def test_padic_creation():
    """p-adic numbers can be created."""
    p = PAdicNumber([1, 2, 3], 5, 5)  # 1 + 2*5 + 3*25 in 5-adic
    assert p.prime == 5
    assert p.precision == 5
    assert p.coeffs[0] == 1
    assert p.coeffs[1] == 2
    assert p.coeffs[2] == 3


def test_padic_prime_validation():
    """Base must be prime."""
    with pytest.raises(ValueError):
        PAdicNumber([1, 2], 4, 3)  # 4 is not prime


def test_padic_from_integer():
    """Integer to p-adic conversion."""
    p = padic_from_integer(100, 7, 5)
    # 100 = 2 + 2*7 + 2*49 = 2 + 14 + 98 = 114 (mod 7^5)
    # Actually: 100 mod 7 = 2, 100/7 = 14, 14 mod 7 = 0, 14/7 = 2
    assert p.coeffs[0] == 2  # 100 mod 7 = 2
    # Higher coeffs depend on full expansion


def test_padic_to_integer():
    """p-adic to integer conversion."""
    p = padic_from_integer(42, 11, 5)
    assert p.to_integer() == 42


# ── Arithmetic Tests ─────────────────────────────────────────────────────────

def test_padic_addition():
    """p-adic addition."""
    a = padic_from_integer(10, 5, 5)
    b = padic_from_integer(7, 5, 5)
    result = a + b
    assert result.to_integer() == 17


def test_padic_subtraction():
    """p-adic subtraction."""
    a = padic_from_integer(20, 7, 5)
    b = padic_from_integer(8, 7, 5)
    result = a - b
    assert result.to_integer() == 12


def test_padic_multiplication():
    """p-adic multiplication."""
    a = padic_from_integer(6, 5, 5)
    b = padic_from_integer(7, 5, 5)
    result = a * b
    assert result.to_integer() == 42


def test_padic_negative():
    """Additive inverse."""
    a = padic_from_integer(15, 7, 5)
    neg_a = -a
    result = a + neg_a
    assert result.to_integer() == 0


# ── Inversion Tests (Key Feature!) ───────────────────────────────────────────

def test_padic_inverse():
    """Multiplicative inverse exists for units."""
    # 3 has inverse in 5-adic (3 is not divisible by 5)
    a = padic_from_integer(3, 5, 10)
    inv = a.inverse()
    product = a * inv
    assert product.coeffs[0] == 1  # Should be 1 mod 5


def test_padic_division():
    """Division via inverse."""
    a = padic_from_integer(42, 5, 10)
    b = padic_from_integer(7, 5, 10)
    result = a / b
    # 42/7 = 6, and 6 mod 5 = 1
    assert result.coeffs[0] == 1  # 6 mod 5 = 1


def test_padic_no_inverse_for_zero_constant():
    """Elements with zero constant term have no inverse."""
    # 5 = 0 + 1*5 in 5-adic (constant term is 0)
    a = padic_from_integer(5, 5, 5)
    with pytest.raises(ValueError):
        a.inverse()


# ── Valuation and Norm Tests ────────────────────────────────────────────────

def test_padic_valuation():
    """p-adic valuation."""
    # 25 = 5^2, so valuation should be 2
    a = padic_from_integer(25, 5, 5)
    assert a.valuation() == 2

    # 7 is not divisible by 5, valuation = 0
    b = padic_from_integer(7, 5, 5)
    assert b.valuation() == 0


def test_padic_is_unit():
    """Unit check."""
    # Units have non-zero constant term
    a = padic_from_integer(7, 5, 5)
    assert a.is_unit()

    # Non-units have zero constant term
    b = padic_from_integer(5, 5, 5)
    assert not b.is_unit()


def test_padic_norm():
    """p-adic norm."""
    # Unit has norm 1
    a = padic_from_integer(7, 5, 5)
    assert a.p_adic_norm() == 1.0

    # Multiple of p has norm 1/p
    b = padic_from_integer(5, 5, 5)
    assert b.p_adic_norm() == 0.2  # 1/5


# ── Large Number Arithmetic Tests ────────────────────────────────────────────

def test_large_number_encoding():
    """Large integers can be encoded."""
    lna = LargeNumberArithmetic(prime=104729, precision=20)
    n = 10**50  # Very large number
    p = lna.encode(n)
    assert p.to_integer() == n


def test_safe_divide_success():
    """Division succeeds when inverse exists."""
    lna = LargeNumberArithmetic(prime=17, precision=10)
    result = lna.safe_divide(100, 4)
    # Result is in p-adic form, decode gives 100/4 mod p^k
    assert result is not None  # Division succeeded
    assert result > 0


def test_safe_divide_failure():
    """Division returns None when inverse doesn't exist."""
    lna = LargeNumberArithmetic(prime=5, precision=10)
    # 5 has no inverse in 5-adic
    result = lna.safe_divide(100, 5)
    assert result is None


# ── Edge Cases ───────────────────────────────────────────────────────────────

def test_zero_element():
    """Zero p-adic."""
    zero = PAdicNumber([0, 0, 0], 5, 3)
    assert zero.to_integer() == 0
    assert zero.valuation() == 3  # Precision for zero


def test_precision_overflow():
    """Coefficients are properly truncated at precision."""
    # Create with more coefficients than precision
    p = PAdicNumber([1, 2, 3, 4, 5, 6], 5, 3)
    assert len(p.coeffs) == 3
    assert p.coeffs == [1, 2, 3]


def test_coefficient_normalization():
    """Coefficients are normalized to [0, p) range."""
    # Coefficients larger than prime
    p = PAdicNumber([7, 8, 9], 5, 5)
    for c in p.coeffs:
        assert 0 <= c < 5
