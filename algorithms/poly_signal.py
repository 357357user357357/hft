"""Polynomial Signal Model for the Averages algorithm.

Replaces the simple scalar delta condition
  delta = (long_avg - short_avg) / long_avg * 100
with a polynomial system whose Gröbner basis is computed once at
start-up and evaluated on every tick.
"""

from dataclasses import dataclass, field
from typing import List
from groebner import GroebnerBasis, MonomialOrder, Polynomial


# Variable index constants
class var:
    LONG_AVG = 0
    SHORT_AVG = 1
    PRICE = 2
    DELTA = 3


@dataclass
class Constraint:
    var_idx: int
    lo: float
    hi: float

    @staticmethod
    def between(var_idx: int, lo: float, hi: float) -> "Constraint":
        return Constraint(var_idx=var_idx, lo=lo, hi=hi)

    @staticmethod
    def at_least(var_idx: int, lo: float) -> "Constraint":
        return Constraint(var_idx=var_idx, lo=lo, hi=float('inf'))

    @staticmethod
    def at_most(var_idx: int, hi: float) -> "Constraint":
        return Constraint(var_idx=var_idx, lo=float('-inf'), hi=hi)

    def check(self, values: List[float]) -> bool:
        v = values[self.var_idx]
        return self.lo <= v <= self.hi


@dataclass
class PolySignalConfig:
    num_vars: int
    var_names: List[str]
    relations: List[Polynomial]
    constraints: List[Constraint]
    order: MonomialOrder
    tolerance: float


class PolySignal:
    """A compiled polynomial signal. Build once, call eval on every tick."""

    def __init__(self, basis: GroebnerBasis, constraints: List[Constraint],
                 tolerance: float, var_names: List[str]):
        self.basis = basis
        self.constraints = constraints
        self.tolerance = tolerance
        self.var_names = var_names

    @staticmethod
    def build(cfg: PolySignalConfig) -> "PolySignal":
        assert cfg.relations, "need at least one relation"
        basis = GroebnerBasis.compute(cfg.relations, cfg.order)
        return PolySignal(basis, cfg.constraints, cfg.tolerance, cfg.var_names)

    def eval(self, values: List[float]) -> bool:
        return (self.basis.on_variety(values, self.tolerance)
                and all(c.check(values) for c in self.constraints))

    def residuals(self, values: List[float]) -> List[float]:
        return self.basis.eval_all(values)

    def describe(self) -> str:
        names = [f"x{i} = {n}" for i, n in enumerate(self.var_names)]
        return (f"PolySignal [{self.basis.order}]  "
                f"vars: {', '.join(names)}  "
                f"basis size: {self.basis.size()}  "
                f"tol: {self.tolerance:.2e}")


def single_delta_signal(trigger_min_pct: float, trigger_max_pct: float,
                        tolerance: float) -> PolySignal:
    """Build the standard single-delta signal.

    delta = (long_avg - short_avg) / long_avg * 100
    expressed as: x0*x3 - 100*x0 + 100*x1 = 0

    Variables: x0=long_avg, x1=short_avg, x2=price, x3=delta
    """
    nv = 4
    order = MonomialOrder.GRevLex

    delta_rel = Polynomial.from_terms(nv, order, [
        (1,    [1, 0, 0, 1]),   # x0*x3
        (-100, [1, 0, 0, 0]),   # -100*x0
        (100,  [0, 1, 0, 0]),   # +100*x1
    ])

    cfg = PolySignalConfig(
        num_vars=nv,
        var_names=["long_avg", "short_avg", "price", "delta"],
        relations=[delta_rel],
        constraints=[Constraint.between(var.DELTA, trigger_min_pct, trigger_max_pct)],
        order=order,
        tolerance=tolerance,
    )

    return PolySignal.build(cfg)


def two_delta_signal(trigger_min_1: float, trigger_max_1: float,
                     trigger_min_2: float, trigger_max_2: float,
                     tolerance: float) -> PolySignal:
    """Build a two-timeframe polynomial signal.

    Variables (6 total):
      x0=long_avg_1, x1=short_avg_1, x2=delta_1
      x3=long_avg_2, x4=short_avg_2, x5=delta_2
    """
    nv = 6
    order = MonomialOrder.GRevLex

    delta1 = Polynomial.from_terms(nv, order, [
        (1,    [1, 0, 1, 0, 0, 0]),   # x0*x2
        (-100, [1, 0, 0, 0, 0, 0]),   # -100*x0
        (100,  [0, 1, 0, 0, 0, 0]),   # +100*x1
    ])
    delta2 = Polynomial.from_terms(nv, order, [
        (1,    [0, 0, 0, 1, 0, 1]),   # x3*x5
        (-100, [0, 0, 0, 1, 0, 0]),   # -100*x3
        (100,  [0, 0, 0, 0, 1, 0]),   # +100*x4
    ])

    cfg = PolySignalConfig(
        num_vars=nv,
        var_names=["long1", "short1", "delta1", "long2", "short2", "delta2"],
        relations=[delta1, delta2],
        constraints=[
            Constraint.between(2, trigger_min_1, trigger_max_1),
            Constraint.between(5, trigger_min_2, trigger_max_2),
        ],
        order=order,
        tolerance=tolerance,
    )

    return PolySignal.build(cfg)
