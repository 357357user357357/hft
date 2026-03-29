"""p-adic number arithmetic for HFT computations.

p-adic numbers provide:
- Efficient representation of very large integers
- Fast inversion (every non-zero p-adic has inverse)
- Natural handling of modular arithmetic
- Applications in cryptography and error correction

A p-adic number is represented as a series:
    x = a₀ + a₁p + a₂p² + a₃p³ + ...

where p is a prime and 0 ≤ aᵢ < p are the coefficients.

This implementation uses a finite truncation (precision n):
    x ≈ a₀ + a₁p + a₂p² + ... + aₙ₋₁pⁿ⁻¹
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from fractions import Fraction


@dataclass
class PAdicNumber:
    """
    p-adic number with prime base p and precision n.

    Represented as coefficients [a₀, a₁, ..., aₙ₋₁] where:
        x = Σ aᵢ * p^i  for i = 0 to n-1

    Attributes:
        coeffs: List of coefficients (least significant first)
        prime: The prime base p
        precision: Number of coefficients (precision n)
    """
    coeffs: List[int]
    prime: int
    precision: int

    def __post_init__(self):
        """Validate and normalize coefficients."""
        if not self._is_prime(self.prime):
            raise ValueError(f"Base must be prime, got {self.prime}")

        # Normalize coefficients
        self.coeffs = self._normalize_coeffs(self.coeffs, self.prime, self.precision)

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def _normalize_coeffs(coeffs: List[int], p: int, n: int) -> List[int]:
        """Normalize coefficients to [0, p) range with carry."""
        result = [0] * n

        for i, c in enumerate(coeffs):
            carry = c
            j = i
            while j < n and carry != 0:
                result[j] += carry
                carry = result[j] // p
                result[j] %= p
                j += 1

        return result

    # ── Basic Operations ──────────────────────────────────────────────────────

    def __add__(self, other: "PAdicNumber") -> "PAdicNumber":
        """Add two p-adic numbers."""
        if self.prime != other.prime:
            raise ValueError("Cannot add p-adics with different primes")

        max_len = max(len(self.coeffs), len(other.coeffs))
        result_coeffs = []

        carry = 0
        for i in range(self.precision):
            a = self.coeffs[i] if i < len(self.coeffs) else 0
            b = other.coeffs[i] if i < len(other.coeffs) else 0
            total = a + b + carry
            result_coeffs.append(total % self.prime)
            carry = total // self.prime

        return PAdicNumber(result_coeffs, self.prime, self.precision)

    def __neg__(self) -> "PAdicNumber":
        """Additive inverse using p's complement."""
        result = []
        borrow = 0

        for i in range(self.precision):
            a = self.coeffs[i] if i < len(self.coeffs) else 0
            neg = (self.prime - a - borrow) % self.prime
            result.append(neg)
            borrow = 1 if a + borrow > 0 else 0

        return PAdicNumber(result, self.prime, self.precision)

    def __sub__(self, other: "PAdicNumber") -> "PAdicNumber":
        """Subtract two p-adic numbers."""
        return self + (-other)

    def __mul__(self, other: "PAdicNumber") -> "PAdicNumber":
        """Multiply two p-adic numbers using convolution."""
        if self.prime != other.prime:
            raise ValueError("Cannot multiply p-adics with different primes")

        result_coeffs = [0] * self.precision

        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                if i + j < self.precision:
                    result_coeffs[i + j] += a * b

        # Normalize with carries
        result_coeffs = self._normalize_coeffs(result_coeffs, self.prime, self.precision)

        return PAdicNumber(result_coeffs, self.prime, self.precision)

    def __eq__(self, other: object) -> bool:
        """Check equality of p-adic numbers."""
        if not isinstance(other, PAdicNumber):
            return False
        if self.prime != other.prime or self.precision != other.precision:
            return False
        return self.coeffs == other.coeffs

    # ── Inversion (Key Feature!) ─────────────────────────────────────────────

    def inverse(self) -> "PAdicNumber":
        """
        Multiplicative inverse using Hensel lifting.

        For a p-adic x with x ≢ 0 (mod p), the inverse exists and can be
        computed efficiently using Newton-Raphson / Hensel lifting.

        Returns:
            y such that x * y ≡ 1 (mod p^n)
        """
        if self.coeffs[0] == 0:
            raise ValueError("p-adic with zero constant term has no inverse")

        # Step 1: Inverse mod p using Fermat's little theorem
        # a^(-1) ≡ a^(p-2) (mod p)
        a0 = self.coeffs[0]
        inv = pow(a0, self.prime - 2, self.prime)

        # Step 2: Hensel lifting to precision n
        # If y_k is inverse mod p^k, then:
        # y_{2k} = y_k * (2 - x * y_k) mod p^{2k}

        result = [inv]
        k = 1

        while k < self.precision:
            # Double precision each iteration
            new_precision = min(2 * k, self.precision)

            # Compute x * result mod p^{2k}
            product = self._truncate_mul(result, k)

            # Compute (2 - x * result) mod p^{2k}
            two_minus = [(2 if i == 0 else 0) for i in range(new_precision)]
            for i in range(len(product)):
                if i < new_precision:
                    two_minus[i] = (two_minus[i] - product[i]) % self.prime

            # Multiply: result * (2 - x * result)
            new_result = self._truncate_mul(result, k)
            new_result = self._truncate_mul(new_result, k)

            result = new_result[:new_precision]
            k = new_precision

        # Pad to full precision
        result = result + [0] * (self.precision - len(result))

        return PAdicNumber(result, self.prime, self.precision)

    def _truncate_mul(self, coeffs: List[int], k: int) -> List[int]:
        """Multiply truncated to k coefficients."""
        result = [0] * k
        for i, a in enumerate(self.coeffs[:k]):
            for j, b in enumerate(coeffs[:k]):
                if i + j < k:
                    result[i + j] += a * b
        return self._normalize_coeffs(result, self.prime, k)

    def __truediv__(self, other: "PAdicNumber") -> "PAdicNumber":
        """Division via multiplication by inverse."""
        return self * other.inverse()

    # ── Utility Methods ──────────────────────────────────────────────────────

    def to_integer(self) -> int:
        """Convert p-adic to integer (if all high coeffs are zero)."""
        result = 0
        for i, c in enumerate(self.coeffs):
            result += c * (self.prime ** i)
        return result

    def to_rational(self) -> Fraction:
        """
        Convert p-adic to rational number (if possible).

        Uses the fact that rationals have eventually periodic p-adic expansions.
        For finite expansions, just returns the integer value.
        """
        return Fraction(self.to_integer(), 1)

    def valuation(self) -> int:
        """
        p-adic valuation: highest power of p dividing this number.

        Returns:
            v such that x = p^v * u where u is a unit
        """
        for i, c in enumerate(self.coeffs):
            if c != 0:
                return i
        return self.precision  # Zero element

    def is_unit(self) -> bool:
        """Check if this p-adic is a unit (has multiplicative inverse)."""
        return self.coeffs[0] != 0

    def p_adic_norm(self) -> float:
        """
        Compute p-adic norm: |x|_p = p^(-v(x))

        Units have norm 1, multiples of p have norm 1/p, etc.
        """
        v = self.valuation()
        return self.prime ** (-v)

    def __repr__(self) -> str:
        coeff_str = " + ".join(
            f"{c}*{self.prime}^{i}" if i > 0 else str(c)
            for i, c in enumerate(self.coeffs) if c != 0
        )
        return f"PAdic({coeff_str or '0'}, p={self.prime})"


# ── p-adic Utilities ─────────────────────────────────────────────────────────

def padic_from_integer(n: int, p: int, precision: int) -> PAdicNumber:
    """Create p-adic from integer."""
    coeffs = []
    temp = n
    for _ in range(precision):
        coeffs.append(temp % p)
        temp //= p
    return PAdicNumber(coeffs, p, precision)


def padic_gcd(a: PAdicNumber, b: PAdicNumber) -> int:
    """
    Compute gcd using p-adic valuations.

    For p-adics, gcd is p^min(v_p(a), v_p(b))
    """
    if a.prime != b.prime:
        raise ValueError("Cannot compute gcd of p-adics with different primes")

    v_a = a.valuation()
    v_b = b.valuation()
    return a.prime ** min(v_a, v_b)


# ── Application: Large Number Arithmetic ─────────────────────────────────────

class LargeNumberArithmetic:
    """
    Efficient arithmetic for very large integers using p-adic representation.

    Benefits:
    - Inversion is O(log n) instead of potentially failing for integers
    - Division always works (for units)
    - Natural modular arithmetic
    """

    def __init__(self, prime: int = 104729, precision: int = 20):
        """
        Initialize with prime base and precision.

        Args:
            prime: Prime base (default: 104729, a large prime)
            precision: Number of coefficients (higher = more precision)
        """
        self.prime = prime
        self.precision = precision

    def encode(self, n: int) -> PAdicNumber:
        """Encode integer as p-adic."""
        return padic_from_integer(n, self.prime, self.precision)

    def decode(self, p: PAdicNumber) -> int:
        """Decode p-adic back to integer (if small enough)."""
        return p.to_integer()

    def safe_divide(self, a: int, b: int) -> Optional[int]:
        """
        Attempt division using p-adic arithmetic.

        Returns None if b has no inverse (b ≡ 0 mod p).
        """
        p_a = self.encode(a)
        p_b = self.encode(b)

        try:
            result = p_a / p_b
            return result.to_integer()
        except ValueError:
            return None  # b has no inverse
