"""Frenet-Serret frames on GPU via Tilelang.

All geometry computation runs on GPU:
  - 3D curve embedding (price, delayed_price, volume)
  - Finite-difference derivatives (r', r'', r''')
  - Cross products, norms, normalization
  - Curvature κ and torsion τ
  - Trading signals (mean-reversion, regime detection)

Replaces the pure-Python Vector3D loop version in frenet_serret.py.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import tilelang
from tilelang import language as T
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Tilelang GPU kernels
# ─────────────────────────────────────────────────────────────────────────────

@tilelang.jit(target='cuda')
def _embed_curve_kernel(prices, volumes, delay):
    """Embed price series into 3D curve: r(t) = (P(t), P(t-delay), V(t))."""
    N = T.const("N")
    prices: T.Tensor[[N], T.float32]
    volumes: T.Tensor[[N], T.float32]
    D = T.const("D")

    # Output: 3 columns (x, y, z) for each point
    curve_x = T.empty([N], T.float32)
    curve_y = T.empty([N], T.float32)
    curve_z = T.empty([N], T.float32)

    for i in T.serial(N):
        curve_x[i] = prices[i]
        if i >= D:
            curve_y[i] = prices[i - D]
        else:
            curve_y[i] = prices[i]
        curve_z[i] = volumes[i]

    return curve_x, curve_y, curve_z


@tilelang.jit(target='cuda')
def _compute_derivatives(curve_x, curve_y, curve_z):
    """Compute r', r'', r''' via finite differences.

    r'(t)  ≈ (r(t+1) - r(t-1)) / 2
    r''(t) ≈ r(t+1) - 2*r(t) + r(t-1)
    r'''(t) ≈ (r(t+2) - r(t-2)) / 4
    """
    N = T.const("N")
    curve_x: T.Tensor[[N], T.float32]
    curve_y: T.Tensor[[N], T.float32]
    curve_z: T.Tensor[[N], T.float32]

    # First derivative
    rx1 = T.empty([N], T.float32)
    ry1 = T.empty([N], T.float32)
    rz1 = T.empty([N], T.float32)

    # Second derivative
    rx2 = T.empty([N], T.float32)
    ry2 = T.empty([N], T.float32)
    rz2 = T.empty([N], T.float32)

    # Third derivative
    rx3 = T.empty([N], T.float32)
    ry3 = T.empty([N], T.float32)
    rz3 = T.empty([N], T.float32)

    for i in T.serial(N):
        if i > 0 and i < N - 1:
            rx1[i] = (curve_x[i + 1] - curve_x[i - 1]) / 2.0
            ry1[i] = (curve_y[i + 1] - curve_y[i - 1]) / 2.0
            rz1[i] = (curve_z[i + 1] - curve_z[i - 1]) / 2.0

            rx2[i] = curve_x[i + 1] - 2.0 * curve_x[i] + curve_x[i - 1]
            ry2[i] = curve_y[i + 1] - 2.0 * curve_y[i] + curve_y[i - 1]
            rz2[i] = curve_z[i + 1] - 2.0 * curve_z[i] + curve_z[i - 1]
        else:
            rx1[i] = 0.0; ry1[i] = 0.0; rz1[i] = 0.0
            rx2[i] = 0.0; ry2[i] = 0.0; rz2[i] = 0.0

        if i > 1 and i < N - 2:
            rx3[i] = (curve_x[i + 2] - curve_x[i - 2]) / 4.0
            ry3[i] = (curve_y[i + 2] - curve_y[i - 2]) / 4.0
            rz3[i] = (curve_z[i + 2] - curve_z[i - 2]) / 4.0
        else:
            rx3[i] = 0.0; ry3[i] = 0.0; rz3[i] = 0.0

    return rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3


@tilelang.jit(target='cuda')
def _compute_frames(rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3):
    """Compute Frenet frames: T, N, B, curvature κ, torsion τ.

    T = r' / |r'|
    κ = |r' × r''| / |r'|³
    N = (r'' - (r''·T)T) / |...|
    B = T × N
    τ = (r' × r'') · r''' / |r' × r''|²
    """
    N = T.const("N")
    rx1: T.Tensor[[N], T.float32]
    ry1: T.Tensor[[N], T.float32]
    rz1: T.Tensor[[N], T.float32]
    rx2: T.Tensor[[N], T.float32)
    ry2: T.Tensor[[N], T.float32]
    rz2: T.Tensor[[N], T.float32]
    rx3: T.Tensor[[N], T.float32]
    ry3: T.Tensor[[N], T.float32]
    rz3: T.Tensor[[N], T.float32]

    # Output arrays
    Tx = T.empty([N], T.float32)
    Ty = T.empty([N], T.float32)
    Tz = T.empty([N], T.float32)
    Nx = T.empty([N], T.float32)
    Ny = T.empty([N], T.float32)
    Nz = T.empty([N], T.float32)
    Bx = T.empty([N], T.float32)
    By = T.empty([N], T.float32)
    Bz = T.empty([N], T.float32)
    curvature = T.empty([N], T.float32)
    torsion = T.empty([N], T.float32)

    for i in T.serial(N):
        # r' magnitude
        rp_norm = T.sqrt(rx1[i]*rx1[i] + ry1[i]*ry1[i] + rz1[i]*rz1[i])

        if rp_norm < 1e-10:
            Tx[i] = 0.0; Ty[i] = 0.0; Tz[i] = 0.0
            Nx[i] = 0.0; Ny[i] = 0.0; Nz[i] = 0.0
            Bx[i] = 0.0; By[i] = 0.0; Bz[i] = 0.0
            curvature[i] = 0.0
            torsion[i] = 0.0
        else:
            # Tangent: T = r' / |r'|
            Tx[i] = rx1[i] / rp_norm
            Ty[i] = ry1[i] / rp_norm
            Tz[i] = rz1[i] / rp_norm

            # Cross product: r' × r''
            cx = ry1[i] * rz2[i] - rz1[i] * ry2[i]
            cy = rz1[i] * rx2[i] - rx1[i] * rz2[i]
            cz = rx1[i] * ry2[i] - ry1[i] * rx2[i]
            cross_norm = T.sqrt(cx*cx + cy*cy + cz*cz)

            # Curvature: κ = |r' × r''| / |r'|³
            curvature[i] = cross_norm / (rp_norm * rp_norm * rp_norm)

            # Normal: N = (r'' - (r''·T)T) / |...|
            r2_dot_T = rx2[i]*Tx[i] + ry2[i]*Ty[i] + rz2[i]*Tz[i]
            nx = rx2[i] - r2_dot_T * Tx[i]
            ny = ry2[i] - r2_dot_T * Ty[i]
            nz = rz2[i] - r2_dot_T * Tz[i]
            n_norm = T.sqrt(nx*nx + ny*ny + nz*nz)

            if n_norm < 1e-10:
                Nx[i] = 0.0; Ny[i] = 0.0; Nz[i] = 0.0
            else:
                Nx[i] = nx / n_norm
                Ny[i] = ny / n_norm
                Nz[i] = nz / n_norm

            # Binormal: B = T × N
            Bx[i] = Ty[i] * Nz[i] - Tz[i] * Ny[i]
            By[i] = Tz[i] * Nx[i] - Tx[i] * Nz[i]
            Bz[i] = Tx[i] * Ny[i] - Ty[i] * Nx[i]

            # Torsion: τ = (r' × r'') · r''' / |r' × r''|²
            if cross_norm < 1e-10:
                torsion[i] = 0.0
            else:
                triple_prod = cx * rx3[i] + cy * ry3[i] + cz * rz3[i]
                torsion[i] = triple_prod / (cross_norm * cross_norm)

    return Tx, Ty, Tz, Nx, Ny, Nz, Bx, By, Bz, curvature, torsion


@tilelang.jit(target='cuda')
def _curvature_signal_kernel(curvature, window):
    """Generate curvature-based trading signal per window.

    High curvature → mean-reversion
    Low curvature → trending
    """
    N = T.const("N")
    curvature: T.Tensor[[N], T.float32]
    W = T.const("W")

    signal = T.empty([N], T.float32)
    for i in T.serial(N):
        if i < W - 1:
            signal[i] = 0.0
        else:
            avg_k = 0.0
            for j in T.serial(i - W + 1, i + 1):
                avg_k = avg_k + curvature[j]
            avg_k = avg_k / T.cast(W, T.float32)

            # Map to [-1, 1]: high curvature = mean-revert (positive)
            # Thresholds: 0.1 = neutral, 0.5 = strong
            if avg_k > 0.5:
                signal[i] = 1.0  # STRONG mean-revert
            elif avg_k > 0.2:
                signal[i] = 0.5  # WEAK mean-revert
            elif avg_k > 0.1:
                signal[i] = 0.0  # NEUTRAL
            elif avg_k > 0.05:
                signal[i] = -0.5  # WEAK trend
            else:
                signal[i] = -1.0  # STRONG trend

    return signal


@tilelang.jit(target='cuda')
def _torsion_signal_kernel(torsion, threshold):
    """Generate torsion-based regime signal.

    High |torsion| → regime change
    Low |torsion| → stable
    """
    N = T.const("N")
    torsion: T.Tensor[[N], T.float32]
    thr = T.const("THR")

    signal = T.empty([N], T.float32)
    for i in T.serial(N):
        t_val = T.abs(torsion[i])
        if t_val > thr:
            signal[i] = 1.0  # regime change
        else:
            signal[i] = 0.0  # stable

    return signal


@tilelang.jit(target='cuda')
def _frame_alignment_kernel(Tx, Ty, Tz, window):
    """Compute frame alignment score [0, 1].

    High alignment = consistent direction (trending)
    Low alignment = chaotic
    """
    N = T.const("N")
    Tx: T.Tensor[[N], T.float32]
    Ty: T.Tensor[[N], T.float32)
    Tz: T.Tensor[[N], T.float32]
    W = T.const("W")

    alignment = T.empty([N], T.float32)
    for i in T.serial(N):
        if i < W - 1:
            alignment[i] = 0.0
        else:
            # Average tangent
            avg_x = 0.0; avg_y = 0.0; avg_z = 0.0
            for j in T.serial(i - W + 1, i + 1):
                avg_x = avg_x + Tx[j]
                avg_y = avg_y + Ty[j]
                avg_z = avg_z + Tz[j]
            avg_x = avg_x / T.cast(W, T.float32)
            avg_y = avg_y / T.cast(W, T.float32)
            avg_z = avg_z / T.cast(W, T.float32)

            # Normalize
            norm = T.sqrt(avg_x*avg_x + avg_y*avg_y + avg_z*avg_z)
            if norm < 1e-10:
                alignment[i] = 0.0
            else:
                avg_x = avg_x / norm
                avg_y = avg_y / norm
                avg_z = avg_z / norm

                # Dot product with each recent tangent
            score = 0.0
            for j in T.serial(i - W + 1, i + 1):
                score = score + Tx[j]*avg_x + Ty[j]*avg_y + Tz[j]*avg_z
            alignment[i] = score / T.cast(W, T.float32)

    return alignment


@tilelang.jit(target='cuda')
def _mean_reversion_strength_kernel(Nx, Ny, Nz):
    """Compute mean-reversion strength from normal vector.

    Negative N.x = pointing back toward mean (mean-reversion signal).
    """
    N = T.const("N")
    Nx: T.Tensor[[N], T.float32]
    Ny: T.Tensor[[N], T.float32]
    Nz: T.Tensor[[N], T.float32]

    strength = T.empty([N], T.float32)
    for i in T.serial(N):
        strength[i] = -Nx[i]  # Positive = mean-reversion

    return strength


# ─────────────────────────────────────────────────────────────────────────────
# GPU Frenet-Serret Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class FrenetSerretGPU:
    """GPU-accelerated Frenet-Serret analysis via Tilelang.

    Usage:
        analyzer = FrenetSerretGPU(delay=5)
        result = analyzer.analyze(prices, volumes)
        print(result['curvature_signal'])
    """

    def __init__(self, delay: int = 5, curvature_window: int = 10,
                 torsion_threshold: float = 0.3):
        self.delay = delay
        self.curvature_window = curvature_window
        self.torsion_threshold = torsion_threshold

    def analyze(self, prices: np.ndarray,
                volumes: Optional[np.ndarray] = None) -> dict:
        """Run full Frenet-Serret analysis on GPU.

        Args:
            prices: Price series (1D array)
            volumes: Volume series (1D array, defaults to 1.0)

        Returns:
            dict with all computed signals and statistics
        """
        n = len(prices)
        if volumes is None:
            volumes = np.ones(n, dtype=np.float32)

        # Upload to GPU
        prices_dev = torch.from_numpy(prices.astype(np.float32)).cuda()
        volumes_dev = torch.from_numpy(volumes.astype(np.float32)).cuda()
        delay_dev = torch.tensor(self.delay, dtype=torch.int32, device="cuda")

        # 1. Embed curve
        cx, cy, cz = _embed_curve_kernel(
            prices_dev, volumes_dev, delay_dev, N=n, D=self.delay
        )

        # 2. Compute derivatives
        (rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3) = \
            _compute_derivatives(cx, cy, cz, N=n)

        # 3. Compute frames
        (Tx, Ty, Tz, Nx, Ny, Nz, Bx, By, Bz, curvature, torsion) = \
            _compute_frames(rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3, N=n)

        # 4. Trading signals
        curv_signal = _curvature_signal_kernel(
            curvature, torch.tensor(self.curvature_window, dtype=torch.int32, device="cuda"),
            N=n, W=self.curvature_window
        )
        torsion_signal = _torsion_signal_kernel(
            torsion, self.torsion_threshold,
            N=n, THR=self.torsion_threshold
        )
        alignment = _frame_alignment_kernel(
            Tx, Ty, Tz, torch.tensor(self.curvature_window, dtype=torch.int32, device="cuda"),
            N=n, W=self.curvature_window
        )
        mr_strength = _mean_reversion_strength_kernel(Nx, Ny, Nz, N=n)

        # Download results
        result = {
            'Tx': Tx.cpu().numpy(),
            'Ty': Ty.cpu().numpy(),
            'Tz': Tz.cpu().numpy(),
            'Nx': Nx.cpu().numpy(),
            'Ny': Ny.cpu().numpy(),
            'Nz': Nz.cpu().numpy(),
            'Bx': Bx.cpu().numpy(),
            'By': By.cpu().numpy(),
            'Bz': Bz.cpu().numpy(),
            'curvature': curvature.cpu().numpy(),
            'torsion': torsion.cpu().numpy(),
            'curvature_signal': curv_signal.cpu().numpy(),
            'torsion_signal': torsion_signal.cpu().numpy(),
            'frame_alignment': alignment.cpu().numpy(),
            'mean_reversion_strength': mr_strength.cpu().numpy(),
        }

        # Summary statistics
        result['avg_curvature'] = float(np.mean(result['curvature']))
        result['max_curvature'] = float(np.max(result['curvature']))
        result['avg_torsion'] = float(np.mean(result['torsion']))
        result['max_torsion'] = float(np.max(np.abs(result['torsion'])))
        result['current_regime'] = 'regime_change' if result['torsion_signal'][-1] > 0.5 else 'stable'
        result['curvature_signal'] = self._signal_label(result['curvature_signal'][-1])
        result['mean_reversion_strength'] = float(result['mean_reversion_strength'][-1])
        result['frame_alignment'] = float(result['frame_alignment'][-1])

        return result

    def _signal_label(self, val: float) -> str:
        if val > 0.75:
            return "STRONG_MEAN_REVERT"
        elif val > 0.25:
            return "WEAK_MEAN_REVERT"
        elif val > -0.25:
            return "NEUTRAL"
        elif val > -0.75:
            return "WEAK_TREND"
        else:
            return "STRONG_TREND"


# ── Convenience function ─────────────────────────────────────────────────────

def analyze_price_series_gpu(prices: np.ndarray,
                              volumes: Optional[np.ndarray] = None,
                              delay: int = 5) -> dict:
    """GPU Frenet-Serret analysis — one-liner.

    Args:
        prices: Price series
        volumes: Volume series (optional)
        delay: Embedding delay

    Returns:
        dict with all signals and statistics
    """
    analyzer = FrenetSerretGPU(delay=delay)
    return analyzer.analyze(prices, volumes)
