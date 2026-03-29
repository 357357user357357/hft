"""Gröbner Basis — abstract multivariate polynomial algebra.

Implements Buchberger's algorithm with exact rational arithmetic and
three standard monomial orderings (Lex, GrLex, GRevLex).

When SageMath is available, delegates to Singular's optimised F4 algorithm
for the Gröbner basis computation (orders of magnitude faster).

Variables live in Q[x0, x1, ..., x_{n-1}].
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
from fractions import Fraction

# ─── SageMath backend ─────────────────────────────────────────────────────────
_SAGE_AVAILABLE = False
_QQ = None
_PolynomialRing = None
try:
    _sage_bin = os.path.expanduser("~/miniforge3/envs/sage/bin")
    if os.path.isdir(_sage_bin) and _sage_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _sage_bin + ":" + os.environ.get("PATH", "")

    from sage.all import QQ as _QQ, PolynomialRing as _PolynomialRing
    _SAGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass


# ═══════════════════════════════════════════════════════════════════════════
#  Rational — exact arithmetic over Q (using Python's fractions.Fraction)
# ═══════════════════════════════════════════════════════════════════════════

class Rational:
    """Exact rational number, always in lowest terms with positive denominator."""

    def __init__(self, num: int = 0, den: int = 1):
        if den == 0:
            raise ZeroDivisionError("zero denominator")
        self._f = Fraction(num, den)

    @property
    def num(self) -> int:
        return self._f.numerator

    @property
    def den(self) -> int:
        return self._f.denominator

    @staticmethod
    def zero() -> "Rational":
        return Rational(0, 1)

    @staticmethod
    def one() -> "Rational":
        return Rational(1, 1)

    def is_zero(self) -> bool:
        return self._f == 0

    def abs(self) -> "Rational":
        r = Rational.__new__(Rational)
        r._f = abs(self._f)
        return r

    def inv(self) -> "Rational":
        if self.is_zero():
            raise ZeroDivisionError("inverse of zero")
        r = Rational.__new__(Rational)
        r._f = Fraction(self._f.denominator, self._f.numerator)
        return r

    def to_f64(self) -> float:
        return float(self._f)

    @staticmethod
    def from_f64(v: float) -> "Rational":
        scale = 1_000_000
        r = Rational.__new__(Rational)
        r._f = Fraction(round(v * scale), scale)
        return r

    def __eq__(self, other) -> bool:
        if isinstance(other, Rational):
            return self._f == other._f
        return NotImplemented

    def __lt__(self, other) -> bool:
        return self._f < other._f

    def __le__(self, other) -> bool:
        return self._f <= other._f

    def __gt__(self, other) -> bool:
        return self._f > other._f

    def __ge__(self, other) -> bool:
        return self._f >= other._f

    def __neg__(self) -> "Rational":
        r = Rational.__new__(Rational)
        r._f = -self._f
        return r

    def __add__(self, other: "Rational") -> "Rational":
        r = Rational.__new__(Rational)
        r._f = self._f + other._f
        return r

    def __sub__(self, other: "Rational") -> "Rational":
        r = Rational.__new__(Rational)
        r._f = self._f - other._f
        return r

    def __mul__(self, other: "Rational") -> "Rational":
        r = Rational.__new__(Rational)
        r._f = self._f * other._f
        return r

    def __truediv__(self, other: "Rational") -> "Rational":
        return self * other.inv()

    def __repr__(self) -> str:
        if self.den == 1:
            return str(self.num)
        return f"{self.num}/{self.den}"

    def __hash__(self) -> int:
        return hash(self._f)


# ═══════════════════════════════════════════════════════════════════════════
#  Monomial
# ═══════════════════════════════════════════════════════════════════════════

class Monomial:
    """A monomial represented as an exponent vector."""

    def __init__(self, exp: List[int]):
        self.exp = list(exp)

    @staticmethod
    def one(nv: int) -> "Monomial":
        return Monomial([0] * nv)

    def num_vars(self) -> int:
        return len(self.exp)

    def total_deg(self) -> int:
        return sum(self.exp)

    def is_one(self) -> bool:
        return all(e == 0 for e in self.exp)

    def divides(self, other: "Monomial") -> bool:
        return all(a <= b for a, b in zip(self.exp, other.exp))

    def div(self, divisor: "Monomial") -> "Monomial":
        result = []
        for a, b in zip(self.exp, divisor.exp):
            assert a >= b, "monomial does not divide"
            result.append(a - b)
        return Monomial(result)

    def mul(self, other: "Monomial") -> "Monomial":
        return Monomial([a + b for a, b in zip(self.exp, other.exp)])

    def lcm(self, other: "Monomial") -> "Monomial":
        return Monomial([max(a, b) for a, b in zip(self.exp, other.exp)])

    def gcd(self, other: "Monomial") -> "Monomial":
        return Monomial([min(a, b) for a, b in zip(self.exp, other.exp)])

    def __eq__(self, other) -> bool:
        if isinstance(other, Monomial):
            return self.exp == other.exp
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self.exp))

    def __repr__(self) -> str:
        if self.is_one():
            return "1"
        parts = []
        for i, e in enumerate(self.exp):
            if e > 0:
                if e == 1:
                    parts.append(f"x{i}")
                else:
                    parts.append(f"x{i}^{e}")
        return "*".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  MonomialOrder
# ═══════════════════════════════════════════════════════════════════════════

class MonomialOrder(Enum):
    Lex = "Lex"
    GrLex = "GrLex"
    GRevLex = "GRevLex"

    def cmp(self, a: Monomial, b: Monomial) -> int:
        """Returns -1 if a < b, 0 if a == b, 1 if a > b"""
        if self == MonomialOrder.Lex:
            for ea, eb in zip(a.exp, b.exp):
                if ea < eb:
                    return -1
                if ea > eb:
                    return 1
            return 0
        elif self == MonomialOrder.GrLex:
            da, db = a.total_deg(), b.total_deg()
            if da != db:
                return -1 if da < db else 1
            for ea, eb in zip(a.exp, b.exp):
                if ea < eb:
                    return -1
                if ea > eb:
                    return 1
            return 0
        else:  # GRevLex
            da, db = a.total_deg(), b.total_deg()
            if da != db:
                return -1 if da < db else 1
            for ea, eb in reversed(list(zip(a.exp, b.exp))):
                if eb < ea:
                    return -1
                if eb > ea:
                    return 1
            return 0


# ═══════════════════════════════════════════════════════════════════════════
#  Polynomial
# ═══════════════════════════════════════════════════════════════════════════

class Term:
    def __init__(self, coeff: Rational, mono: Monomial):
        self.coeff = coeff
        self.mono = mono

    def __repr__(self) -> str:
        return f"Term({self.coeff}, {self.mono})"


class Polynomial:
    """Multivariate polynomial with exact rational coefficients.
    Terms are kept sorted in descending monomial order.
    """

    def __init__(self, terms: List[Term], num_vars: int, order: MonomialOrder):
        self.terms = terms
        self.num_vars = num_vars
        self.order = order

    @staticmethod
    def zero(nv: int, order: MonomialOrder) -> "Polynomial":
        return Polynomial([], nv, order)

    @staticmethod
    def from_terms(nv: int, order: MonomialOrder, data: List[Tuple]) -> "Polynomial":
        """Build from (integer_coeff, exponent_list) pairs."""
        terms = [Term(Rational(int(c), 1), Monomial(list(exp))) for c, exp in data]
        p = Polynomial(terms, nv, order)
        p._normalize()
        return p

    @staticmethod
    def var(i: int, nv: int, order: MonomialOrder) -> "Polynomial":
        exp = [0] * nv
        exp[i] = 1
        return Polynomial([Term(Rational.one(), Monomial(exp))], nv, order)

    @staticmethod
    def constant(c: Rational, nv: int, order: MonomialOrder) -> "Polynomial":
        if c.is_zero():
            return Polynomial.zero(nv, order)
        return Polynomial([Term(c, Monomial.one(nv))], nv, order)

    def is_zero(self) -> bool:
        return len(self.terms) == 0

    def lt(self) -> Optional[Term]:
        return self.terms[0] if self.terms else None

    def lm(self) -> Optional[Monomial]:
        return self.terms[0].mono if self.terms else None

    def lc(self) -> Optional[Rational]:
        return self.terms[0].coeff if self.terms else None

    def total_degree(self) -> int:
        return max((t.mono.total_deg() for t in self.terms), default=0)

    def neg(self) -> "Polynomial":
        terms = [Term(-t.coeff, t.mono) for t in self.terms]
        return Polynomial(terms, self.num_vars, self.order)

    def add(self, other: "Polynomial") -> "Polynomial":
        terms = list(self.terms) + list(other.terms)
        p = Polynomial(terms, self.num_vars, self.order)
        p._normalize()
        return p

    def sub(self, other: "Polynomial") -> "Polynomial":
        return self.add(other.neg())

    def scale(self, c: Rational, m: Monomial) -> "Polynomial":
        if c.is_zero():
            return Polynomial.zero(self.num_vars, self.order)
        terms = [Term(t.coeff * c, t.mono.mul(m)) for t in self.terms]
        return Polynomial(terms, self.num_vars, self.order)

    def mul(self, other: "Polynomial") -> "Polynomial":
        terms = []
        for a in self.terms:
            for b in other.terms:
                terms.append(Term(a.coeff * b.coeff, a.mono.mul(b.mono)))
        p = Polynomial(terms, self.num_vars, self.order)
        p._normalize()
        return p

    def monic(self) -> "Polynomial":
        if self.is_zero():
            return self._copy()
        lc_inv = self.lc().inv()
        terms = [Term(t.coeff * lc_inv, t.mono) for t in self.terms]
        return Polynomial(terms, self.num_vars, self.order)

    def eval(self, values: List[float]) -> float:
        assert len(values) == self.num_vars, "wrong number of variables"
        result = 0.0
        for t in self.terms:
            mono_val = 1.0
            for i, e in enumerate(t.mono.exp):
                mono_val *= values[i] ** e
            result += t.coeff.to_f64() * mono_val
        return result

    def _copy(self) -> "Polynomial":
        return Polynomial(list(self.terms), self.num_vars, self.order)

    def _normalize(self) -> None:
        order = self.order
        import functools
        def cmp_terms(a, b):
            return order.cmp(b.mono, a.mono)  # descending
        self.terms.sort(key=functools.cmp_to_key(cmp_terms))

        # Combine like terms
        merged: List[Term] = []
        for term in self.terms:
            if merged and merged[-1].mono == term.mono:
                merged[-1].coeff = merged[-1].coeff + term.coeff
            else:
                merged.append(Term(term.coeff, term.mono))
        # Drop zeros
        self.terms = [t for t in merged if not t.coeff.is_zero()]

    def __repr__(self) -> str:
        if self.is_zero():
            return "0"
        parts = []
        for i, t in enumerate(self.terms):
            neg = t.coeff < Rational.zero()
            ac = t.coeff.abs()
            if t.mono.is_one():
                s = repr(ac)
            elif ac == Rational.one():
                s = repr(t.mono)
            else:
                s = f"{ac}*{t.mono}"
            if i == 0:
                parts.append(f"-{s}" if neg else s)
            else:
                parts.append(f" - {s}" if neg else f" + {s}")
        return "".join(parts)


# Fix the _normalize method (avoid duplicate key= in sort)
def _poly_normalize(self) -> None:
    import functools
    order = self.order

    def cmp_terms(a: Term, b: Term) -> int:
        return order.cmp(b.mono, a.mono)  # descending

    self.terms.sort(key=functools.cmp_to_key(cmp_terms))

    merged: List[Term] = []
    for term in self.terms:
        if merged and merged[-1].mono == term.mono:
            merged[-1] = Term(merged[-1].coeff + term.coeff, merged[-1].mono)
        else:
            merged.append(Term(term.coeff, term.mono))
    self.terms = [t for t in merged if not t.coeff.is_zero()]


Polynomial._normalize = _poly_normalize


# ═══════════════════════════════════════════════════════════════════════════
#  Polynomial division and S-polynomial
# ═══════════════════════════════════════════════════════════════════════════

def poly_reduce(f: Polynomial, basis: List[Polynomial]) -> Polynomial:
    """Compute the remainder of f divided by basis."""
    nv = f.num_vars
    order = f.order
    r = Polynomial.zero(nv, order)
    p = Polynomial(list(f.terms), nv, order)

    while not p.is_zero():
        lm_p = p.lm()
        lc_p = p.lc()

        divided = False
        for g in basis:
            if g.is_zero():
                continue
            lm_g = g.lm()
            if lm_g.divides(lm_p):
                lc_g = g.lc()
                mono_q = lm_p.div(lm_g)
                coeff_q = lc_p / lc_g
                p = p.sub(g.scale(coeff_q, mono_q))
                divided = True
                break

        if not divided:
            lt = p.lt()
            r = r.add(Polynomial([Term(lt.coeff, lt.mono)], nv, order))
            p.terms.pop(0)

    return r


def s_polynomial(f: Polynomial, g: Polynomial) -> Polynomial:
    """Compute the S-polynomial of two polynomials."""
    lm_f = f.lm()
    lm_g = g.lm()
    lc_f = f.lc()
    lc_g = g.lc()

    gamma = lm_f.lcm(lm_g)
    mono_f = gamma.div(lm_f)
    mono_g = gamma.div(lm_g)

    term1 = f.scale(lc_f.inv(), mono_f)
    term2 = g.scale(lc_g.inv(), mono_g)
    return term1.sub(term2)


# ═══════════════════════════════════════════════════════════════════════════
#  Buchberger's Algorithm
# ═══════════════════════════════════════════════════════════════════════════

def _coprime_criterion(f: Polynomial, g: Polynomial) -> bool:
    lm_f = f.lm()
    lm_g = g.lm()
    if lm_f is None or lm_g is None:
        return False
    return lm_f.gcd(lm_g).is_one()


def buchberger(polys: List[Polynomial], order: MonomialOrder) -> List[Polynomial]:
    """Compute a Gröbner basis using Buchberger's algorithm."""
    import functools

    for p in polys:
        p.order = order
        def cmp_terms(a: Term, b: Term) -> int:
            return order.cmp(b.mono, a.mono)
        p.terms.sort(key=functools.cmp_to_key(cmp_terms))

    g = list(polys)
    pairs = [(i, j) for i in range(len(g)) for j in range(i + 1, len(g))]

    while pairs:
        i, j = pairs.pop()
        if _coprime_criterion(g[i], g[j]):
            continue
        s = s_polynomial(g[i], g[j])
        r = poly_reduce(s, g)
        if not r.is_zero():
            new_idx = len(g)
            for k in range(new_idx):
                pairs.append((k, new_idx))
            g.append(r)

    g = [p for p in g if not p.is_zero()]
    return g


def reduce_basis(basis: List[Polynomial]) -> List[Polynomial]:
    """Reduce a Gröbner basis to its unique minimal reduced form."""
    import functools
    order = basis[0].order

    g = [p.monic() for p in basis]

    # Remove redundant generators
    i = 0
    while i < len(g):
        lm_i = g[i].lm()
        redundant = any(
            j != i and not g[j].is_zero() and g[j].lm().divides(lm_i)
            for j in range(len(g))
        )
        if redundant:
            g.pop(i)
        else:
            i += 1

    # Reduce each element modulo the others
    for i in range(len(g)):
        others = [g[j] for j in range(len(g)) if j != i]
        g[i] = poly_reduce(g[i], others).monic()

    def cmp_polys(a: Polynomial, b: Polynomial) -> int:
        return order.cmp(a.lm(), b.lm())
    g.sort(key=functools.cmp_to_key(cmp_polys))
    g = [p for p in g if not p.is_zero()]
    return g


# ═══════════════════════════════════════════════════════════════════════════
#  Public API — GroebnerBasis
# ═══════════════════════════════════════════════════════════════════════════

class GroebnerBasis:
    """A reduced Gröbner basis for a polynomial ideal."""

    def __init__(self, polys: List[Polynomial], order: MonomialOrder, num_vars: int):
        self.polys = polys
        self.order = order
        self.num_vars = num_vars

    @staticmethod
    def compute(generators: List[Polynomial], order: MonomialOrder) -> "GroebnerBasis":
        assert generators, "need at least one generator"
        nv = generators[0].num_vars
        if _SAGE_AVAILABLE:
            return GroebnerBasis._compute_sage(generators, order, nv)
        raw = buchberger(generators, order)
        polys = reduce_basis(raw)
        return GroebnerBasis(polys, order, nv)

    @staticmethod
    def _compute_sage(generators: List[Polynomial], order: MonomialOrder,
                      nv: int) -> "GroebnerBasis":
        """Compute Gröbner basis via SageMath's Singular (F4 algorithm)."""
        order_map = {
            MonomialOrder.Lex: "lex",
            MonomialOrder.GrLex: "deglex",
            MonomialOrder.GRevLex: "degrevlex",
        }
        sage_order = order_map.get(order, "degrevlex")
        var_names = [f"x{i}" for i in range(nv)]
        R = _PolynomialRing(_QQ, var_names, order=sage_order)
        gens = R.gens()

        # Convert our Polynomial objects to sage polynomials
        sage_polys = []
        for poly in generators:
            sage_p = R.zero()
            for t in poly.terms:
                coeff = _QQ(t.coeff.num) / _QQ(t.coeff.den)
                mono = R.one()
                for i, e in enumerate(t.mono.exp):
                    if e > 0:
                        mono *= gens[i] ** e
                sage_p += coeff * mono
            if sage_p != R.zero():
                sage_polys.append(sage_p)

        if not sage_polys:
            return GroebnerBasis([], order, nv)

        I = R.ideal(sage_polys)
        gb = I.groebner_basis()

        # Convert back to our Polynomial format
        result_polys = []
        for sg in gb:
            terms_data = []
            for coeff, mono in sg:
                exp = list(mono.exponents()[0])
                # Pad or trim to nv
                while len(exp) < nv:
                    exp.append(0)
                exp = [int(e) for e in exp[:nv]]
                c_frac = Fraction(int(coeff.numerator()), int(coeff.denominator()))
                terms_data.append(Term(
                    Rational(c_frac.numerator, c_frac.denominator),
                    Monomial(exp),
                ))
            if terms_data:
                p = Polynomial(terms_data, nv, order)
                p._normalize()
                result_polys.append(p)

        return GroebnerBasis(result_polys, order, nv)

    def reduce(self, f: Polynomial) -> Polynomial:
        """Reduce f modulo this basis."""
        return poly_reduce(f, self.polys)

    def contains(self, f: Polynomial) -> bool:
        """Test ideal membership."""
        return self.reduce(f).is_zero()

    def eval_all(self, values: List[float]) -> List[float]:
        """Evaluate every basis polynomial at values."""
        return [p.eval(values) for p in self.polys]

    def on_variety(self, values: List[float], tol: float) -> bool:
        """Returns True if all basis polynomials evaluate within tol of zero."""
        return all(abs(v) <= tol for v in self.eval_all(values))

    def size(self) -> int:
        return len(self.polys)

    def __repr__(self) -> str:
        lines = [f"Gröbner basis ({len(self.polys)} polys, {self.order}):"]
        for i, p in enumerate(self.polys):
            lines.append(f"  g{i} = {p}")
        return "\n".join(lines)
