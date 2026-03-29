"""Spectral analysis of price windows using Hecke-like operators.

For each scale p (prime), we build a p×p autocorrelation matrix M_p
whose dominant eigenvalue λ_p estimates the Hecke eigenvalue a_p.

Key ideas:
  - |λ_p| close to 1  → persistent pattern at scale p (trade it)
  - |λ_p| < 0.4       → noise, skip
  - arg(λ_p)          → phase of the cycle (entry timing)
  - Multi-scale agreement across primes → higher-confidence signal
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from hecke_operators import HeckeAlgebra, sieve_primes


@dataclass
class ScaleSpectrum:
    """Spectral result at a single scale (prime p)."""
    p: int
    eigenvalue_mag: float      # |λ_p|  — strength of the pattern
    eigenvalue_phase: float    # arg(λ_p) in radians — cycle phase
    estimated_a_p: float       # Hecke eigenvalue estimate (Ramanujan-scaled)
    autocorr: float            # raw lag-p autocorrelation
    signal_power: float        # fraction of variance explained by dominant mode
    is_significant: bool       # True if |λ_p| >= threshold

    def __repr__(self) -> str:
        sig = "✓" if self.is_significant else "✗"
        return (f"Scale p={self.p:2d} {sig}  "
                f"|λ|={self.eigenvalue_mag:.4f}  "
                f"phase={math.degrees(self.eigenvalue_phase):+6.1f}°  "
                f"a_p={self.estimated_a_p:+.4f}  "
                f"autocorr={self.autocorr:+.4f}  "
                f"power={self.signal_power:.3f}")


@dataclass
class SpectralReport:
    """Full multi-scale spectral report for a price window."""
    scales: List[ScaleSpectrum]
    num_significant: int
    dominant_scale: Optional[int]   # p with highest |λ_p|
    consensus_phase: Optional[float]  # average phase (if scales agree)
    overall_score: float            # combined confidence score in [0, 1]

    def is_tradeable(self, min_significant: int = 2,
                     min_score: float = 0.5) -> bool:
        return self.num_significant >= min_significant and self.overall_score >= min_score

    def __repr__(self) -> str:
        lines = [f"SpectralReport: score={self.overall_score:.3f}  "
                 f"significant={self.num_significant}  "
                 f"dominant_p={self.dominant_scale}"]
        for s in self.scales:
            lines.append(f"  {s}")
        return "\n".join(lines)


# ─── Core Spectral Analysis ────────────────────────────────────────────────────

def log_returns(prices: List[float]) -> List[float]:
    """Compute log returns from price series."""
    return [
        math.log(prices[i] / prices[i - 1])
        for i in range(1, len(prices))
        if prices[i - 1] > 0 and prices[i] > 0
    ]


def autocorrelation(series: List[float], lag: int) -> float:
    """Lag-p autocorrelation of a series."""
    n = len(series)
    if n <= lag:
        return 0.0
    mu = sum(series) / n
    var = sum((x - mu) ** 2 for x in series) / n
    if var < 1e-12:
        return 0.0
    cov = sum(
        (series[i] - mu) * (series[i - lag] - mu)
        for i in range(lag, n)
    ) / n
    return cov / var


def hankel_matrix(series: List[float], size: int) -> List[List[float]]:
    """
    Build a size×size Hankel-like autocorrelation matrix from a series.

    M[i][j] = autocorrelation at lag |i-j|, weighted by exp(-|i-j|/size).
    This mimics the Hecke operator structure: M encodes how the pattern
    at one time step relates to another with a "modular" decay.
    """
    n = len(series)
    mu = sum(series) / n if n > 0 else 0.0
    var = sum((x - mu) ** 2 for x in series) / n if n > 0 else 1.0
    if var < 1e-12:
        var = 1.0

    # Pre-compute autocorrelations for lags 0..size-1
    ac = [autocorrelation(series, lag) for lag in range(size)]

    matrix: List[List[float]] = []
    for i in range(size):
        row = []
        for j in range(size):
            lag = abs(i - j)
            decay = math.exp(-lag / max(size / 4.0, 1.0))
            row.append(ac[lag] * decay)
        matrix.append(row)

    # Regularise diagonal for numerical stability
    for i in range(size):
        matrix[i][i] += 0.01

    return matrix


def power_iteration(matrix: List[List[float]], n_iter: int = 50) -> Tuple[float, List[float]]:
    """
    Dominant real eigenvalue via power iteration.

    Returns (eigenvalue, eigenvector).
    Uses deflation-free version: good enough for the dominant mode.
    """
    size = len(matrix)
    # Start with uniform vector
    v = [1.0 / math.sqrt(size)] * size

    eigenvalue = 0.0
    for _ in range(n_iter):
        # Mv
        mv = [sum(matrix[i][j] * v[j] for j in range(size)) for i in range(size)]
        # Rayleigh quotient
        eigenvalue = sum(mv[i] * v[i] for i in range(size))
        # Normalise
        norm = math.sqrt(sum(x * x for x in mv))
        if norm < 1e-12:
            break
        v = [x / norm for x in mv]

    return eigenvalue, v


def qr_eigenvalues_2x2(matrix: List[List[float]],
                        n_iter: int = 80) -> List[complex]:
    """
    Extract ALL eigenvalues of a real matrix via implicit QR iteration
    with Wilkinson shifts.  Returns complex eigenvalues.

    This captures the complex conjugate pair structure that power iteration
    misses.  The Eichler-Deligne polynomial x^2 - a_p*x + p^{k-1} has
    complex roots under the Ramanujan bound; QR iteration recovers these
    from the 2x2 trailing block of the Hessenberg form.

    For matrices up to 20x20 (our Hankel matrices), this is fast and exact.
    """
    size = len(matrix)
    if size == 0:
        return []
    if size == 1:
        return [complex(matrix[0][0])]

    # Work on a copy (will be destroyed)
    H = [row[:] for row in matrix]

    # Reduce to upper Hessenberg form via Householder reflections
    for k in range(size - 2):
        # Compute Householder vector for column k, rows k+1..n-1
        x = [H[i][k] for i in range(k + 1, size)]
        norm_x = math.sqrt(sum(v * v for v in x))
        if norm_x < 1e-15:
            continue
        sign = 1.0 if x[0] >= 0 else -1.0
        x[0] += sign * norm_x
        norm_v = math.sqrt(sum(v * v for v in x))
        if norm_v < 1e-15:
            continue
        v = [xi / norm_v for xi in x]
        m = len(v)

        # H = H - 2 * v * (v^T * H[k+1:, :])
        for j in range(size):
            dot = sum(v[i] * H[k + 1 + i][j] for i in range(m))
            for i in range(m):
                H[k + 1 + i][j] -= 2.0 * v[i] * dot

        # H = H - 2 * (H[:, k+1:] * v) * v^T
        for i in range(size):
            dot = sum(H[i][k + 1 + j] * v[j] for j in range(m))
            for j in range(m):
                H[i][k + 1 + j] -= 2.0 * dot * v[j]

    # QR iteration on Hessenberg matrix
    n = size
    eigenvalues: List[complex] = []

    for _ in range(n_iter * size):
        if n <= 0:
            break
        if n == 1:
            eigenvalues.append(complex(H[0][0]))
            n = 0
            break

        # Check for convergence of bottom subdiagonal
        if abs(H[n - 1][n - 2]) < 1e-12 * (abs(H[n - 1][n - 1]) + abs(H[n - 2][n - 2]) + 1e-30):
            eigenvalues.append(complex(H[n - 1][n - 1]))
            n -= 1
            continue

        # Check 2x2 block at bottom
        if n == 2 or (n >= 3 and abs(H[n - 2][n - 3]) < 1e-12 * (
                abs(H[n - 2][n - 2]) + abs(H[n - 3][n - 3]) + 1e-30)):
            # Extract eigenvalues of 2x2 block
            a11 = H[n - 2][n - 2]
            a12 = H[n - 2][n - 1]
            a21 = H[n - 1][n - 2]
            a22 = H[n - 1][n - 1]
            tr = a11 + a22
            det = a11 * a22 - a12 * a21
            disc = tr * tr - 4.0 * det
            if disc >= 0:
                sqrt_d = math.sqrt(disc)
                eigenvalues.append(complex((tr + sqrt_d) / 2.0))
                eigenvalues.append(complex((tr - sqrt_d) / 2.0))
            else:
                sqrt_d = math.sqrt(-disc)
                eigenvalues.append(complex(tr / 2.0, sqrt_d / 2.0))
                eigenvalues.append(complex(tr / 2.0, -sqrt_d / 2.0))
            n -= 2
            continue

        # Wilkinson shift: eigenvalue of trailing 2x2 closer to H[n-1][n-1]
        a11 = H[n - 2][n - 2]
        a12 = H[n - 2][n - 1]
        a21 = H[n - 1][n - 2]
        a22 = H[n - 1][n - 1]
        tr = a11 + a22
        det = a11 * a22 - a12 * a21
        disc = tr * tr - 4.0 * det
        if disc >= 0:
            sqrt_d = math.sqrt(disc)
            s1 = (tr + sqrt_d) / 2.0
            s2 = (tr - sqrt_d) / 2.0
            shift = s1 if abs(s1 - a22) < abs(s2 - a22) else s2
        else:
            shift = a22  # use real part

        # Shifted QR step
        for i in range(n):
            H[i][i] -= shift

        # QR factorization via Givens rotations
        cs = [0.0] * (n - 1)
        sn = [0.0] * (n - 1)
        for i in range(n - 1):
            a = H[i][i]
            b = H[i + 1][i]
            r = math.sqrt(a * a + b * b)
            if r < 1e-30:
                cs[i] = 1.0
                sn[i] = 0.0
                continue
            cs[i] = a / r
            sn[i] = b / r
            # Apply to rows i, i+1
            for j in range(n):
                t1 = cs[i] * H[i][j] + sn[i] * H[i + 1][j]
                t2 = -sn[i] * H[i][j] + cs[i] * H[i + 1][j]
                H[i][j] = t1
                H[i + 1][j] = t2

        # RQ: apply rotations from the right
        for i in range(n - 1):
            for j in range(n):
                t1 = cs[i] * H[j][i] + sn[i] * H[j][i + 1]
                t2 = -sn[i] * H[j][i] + cs[i] * H[j][i + 1]
                H[j][i] = t1
                H[j][i + 1] = t2

        # Undo shift
        for i in range(n):
            H[i][i] += shift

    # Remaining unconverged
    for i in range(n):
        eigenvalues.append(complex(H[i][i]))

    return eigenvalues


def phase_from_fft(series: List[float], lag: int) -> float:
    """
    Estimate the phase angle of the dominant frequency near 1/lag cycles/sample.
    Uses a simple DFT at frequency f = 1/lag.
    """
    n = len(series)
    if n == 0 or lag <= 0:
        return 0.0
    omega = 2.0 * math.pi / lag
    re = sum(series[t] * math.cos(omega * t) for t in range(n)) / n
    im = sum(series[t] * math.sin(omega * t) for t in range(n)) / n
    return math.atan2(im, re)


def analyze_scale(returns: List[float], p: int,
                  weight: int = 2,
                  significance_threshold: float = 0.55) -> ScaleSpectrum:
    """
    Full spectral analysis at scale p.

    Steps:
      1. Compute lag-p autocorrelation -> a_p estimate
      2. Build p*p Hankel matrix from the last p^2 returns
      3. QR iteration for ALL eigenvalues (including complex conjugate pairs)
      4. Dominant eigenvalue magnitude + phase from the complex eigenvalue
      5. Signal power = |lambda|^2 / trace(M) with minimum floor gate
    """
    # Use last p*p returns (or all if fewer)
    window = returns[-(p * p):] if len(returns) >= p * p else returns
    if len(window) < p:
        return ScaleSpectrum(p=p, eigenvalue_mag=0.0, eigenvalue_phase=0.0,
                             estimated_a_p=0.0, autocorr=0.0, signal_power=0.0,
                             is_significant=False)

    # Lag-p autocorrelation
    ac = autocorrelation(window, lag=p)

    # Hecke eigenvalue estimate (Ramanujan-bounded)
    ramanujan = 2.0 * (p ** ((weight - 1) / 2.0))
    a_p = ramanujan * ac

    # Hankel matrix (size = p, capped at 20 for performance)
    mat_size = min(p, 20)
    matrix = hankel_matrix(window[-mat_size * 4:], mat_size)

    # Full QR eigenvalue extraction (captures complex conjugate pairs)
    all_eigs = qr_eigenvalues_2x2(matrix)

    # Dominant eigenvalue by magnitude
    if all_eigs:
        dom = max(all_eigs, key=lambda z: abs(z))
        lam_mag = abs(dom)
        # Phase: from complex eigenvalue if it has imaginary part,
        # otherwise fall back to DFT
        if abs(dom.imag) > 1e-10:
            phase = math.atan2(dom.imag, dom.real)
        else:
            phase = phase_from_fft(window, p)
    else:
        # Fallback to power iteration
        lam, _ = power_iteration(matrix)
        lam_mag = abs(lam)
        phase = phase_from_fft(window, p)

    # Signal power: fraction of variance explained
    trace = sum(matrix[i][i] for i in range(mat_size))
    signal_power = (lam_mag ** 2) / max(trace, 1e-10)
    signal_power = min(signal_power, 1.0)

    # Significance requires BOTH eigenvalue magnitude AND signal power floor.
    # This prevents the Gröbner check from passing on pure noise where
    # both x_p and a_p are tiny (improvement F).
    MIN_SIGNAL_POWER = 0.05
    is_sig = (lam_mag >= significance_threshold and signal_power >= MIN_SIGNAL_POWER)

    return ScaleSpectrum(
        p=p,
        eigenvalue_mag=lam_mag,
        eigenvalue_phase=phase,
        estimated_a_p=a_p,
        autocorr=ac,
        signal_power=signal_power,
        is_significant=is_sig,
    )


# ─── Multi-Scale Report ────────────────────────────────────────────────────────

def spectral_report(
    prices: List[float],
    primes: List[int],
    weight: int = 2,
    significance_threshold: float = 0.55,
) -> SpectralReport:
    """
    Compute a full multi-scale spectral report.

    Requires at least max(primes)² prices for reliable estimation.
    """
    rets = log_returns(prices)

    scales: List[ScaleSpectrum] = []
    for p in primes:
        s = analyze_scale(rets, p,
                          weight=weight,
                          significance_threshold=significance_threshold)
        scales.append(s)

    significant = [s for s in scales if s.is_significant]
    num_sig = len(significant)

    # Dominant scale: highest signal power among significant ones
    dominant_scale = None
    if significant:
        best = max(significant, key=lambda s: s.signal_power)
        dominant_scale = best.p

    # Consensus phase: circular mean of phases of significant scales
    consensus_phase = None
    if significant:
        sin_sum = sum(math.sin(s.eigenvalue_phase) for s in significant)
        cos_sum = sum(math.cos(s.eigenvalue_phase) for s in significant)
        consensus_phase = math.atan2(sin_sum, cos_sum)

    # Overall score: geometric mean of |λ| across significant scales,
    # boosted by multi-scale agreement
    if significant:
        geo_mean = math.exp(
            sum(math.log(max(s.eigenvalue_mag, 1e-6)) for s in significant)
            / len(significant)
        )
        multi_scale_bonus = min(num_sig / max(len(primes), 1), 1.0)
        overall_score = 0.7 * geo_mean + 0.3 * multi_scale_bonus
    else:
        overall_score = 0.0

    return SpectralReport(
        scales=scales,
        num_significant=num_sig,
        dominant_scale=dominant_scale,
        consensus_phase=consensus_phase,
        overall_score=overall_score,
    )
