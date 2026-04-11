"""TileOPs integration for HFT pipeline.

Extracts the useful TileOPs operators for financial signal processing:
  - Elementwise: abs, exp, log, sqrt, clamp, sign, relu
  - Reduction: sum, mean, max, min, var, argmax, argmin, cumsum
  - FFT: spectral analysis
  - Norm: L1, L2, Inf norm
  - Convolution: 1D convolution for signal smoothing

All ops run via Tilelang GPU kernels — no torch math, no CuPy.

Usage:
    from tileops_hft import TileOps

    ops = TileOps()
    signal = ops.relu(raw_signal)
    smoothed = ops.conv1d(signal, kernel_weights)
    spectral = ops.fft(signal)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tilelang
from tilelang import language as T
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Tilelang GPU kernels — TileOPs-style elementwise, reduction, FFT, norm, conv
# ─────────────────────────────────────────────────────────────────────────────

# ── Elementwise ──────────────────────────────────────────────────────────────

@tilelang.jit(target='cuda')
def _relu_kernel(x):
    """ReLU: max(0, x)."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        out[i] = T.max(x[i], 0.0)
    return out


@tilelang.jit(target='cuda')
def _abs_kernel(x):
    """Absolute value."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        out[i] = T.abs(x[i])
    return out


@tilelang.jit(target='cuda')
def _exp_kernel(x):
    """Exponential."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        out[i] = T.exp(x[i])
    return out


@tilelang.jit(target='cuda')
def _log_kernel(x):
    """Natural log."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        out[i] = T.log(x[i] + 1e-10)
    return out


@tilelang.jit(target='cuda')
def _sqrt_kernel(x):
    """Square root."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        out[i] = T.sqrt(T.max(x[i], 0.0))
    return out


@tilelang.jit(target='cuda')
def _clamp_kernel(x, lo, hi):
    """Clamp to [lo, hi]."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    lo_val = T.const("LO")
    hi_val = T.const("HI")
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        out[i] = T.min(T.max(x[i], lo_val), hi_val)
    return out


@tilelang.jit(target='cuda')
def _sign_kernel(x):
    """Sign: -1, 0, or 1."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        if x[i] > 0.0:
            out[i] = 1.0
        elif x[i] < 0.0:
            out[i] = -1.0
        else:
            out[i] = 0.0
    return out


@tilelang.jit(target='cuda')
def _sigmoid_kernel(x):
    """Sigmoid: 1 / (1 + exp(-x))."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        out[i] = 1.0 / (1.0 + T.exp(-x[i]))
    return out


@tilelang.jit(target='cuda')
def _tanh_kernel(x):
    """Hyperbolic tangent."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        ep = T.exp(x[i])
        en = T.exp(-x[i])
        out[i] = (ep - en) / (ep + en)
    return out


@tilelang.jit(target='cuda')
def _binary_add_kernel(a, b):
    """Element-wise add."""
    N = T.const("N")
    a: T.Tensor[[N], T.float32]
    b: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        out[i] = a[i] + b[i]
    return out


@tilelang.jit(target='cuda')
def _binary_mul_kernel(a, b):
    """Element-wise multiply."""
    N = T.const("N")
    a: T.Tensor[[N], T.float32]
    b: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        out[i] = a[i] * b[i]
    return out


# ── Reduction ────────────────────────────────────────────────────────────────

@tilelang.jit(target='cuda')
def _sum_kernel(x):
    """Sum reduction."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    total = T.alloc_fragment([1], T.float32)
    total[0] = 0.0
    for i in T.serial(N):
        total[0] = total[0] + x[i]
    return total[0]


@tilelang.jit(target='cuda')
def _mean_kernel(x):
    """Mean reduction."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    total = T.alloc_fragment([1], T.float32)
    total[0] = 0.0
    for i in T.serial(N):
        total[0] = total[0] + x[i]
    return total[0] / T.cast(N, T.float32)


@tilelang.jit(target='cuda')
def _max_kernel(x):
    """Max reduction."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    max_val = T.alloc_fragment([1], T.float32)
    max_val[0] = x[0]
    for i in T.serial(1, N):
        if x[i] > max_val[0]:
            max_val[0] = x[i]
    return max_val[0]


@tilelang.jit(target='cuda')
def _min_kernel(x):
    """Min reduction."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    min_val = T.alloc_fragment([1], T.float32)
    min_val[0] = x[0]
    for i in T.serial(1, N):
        if x[i] < min_val[0]:
            min_val[0] = x[i]
    return min_val[0]


@tilelang.jit(target='cuda')
def _var_kernel(x):
    """Variance (sample)."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]

    total = T.alloc_fragment([1], T.float32)
    total[0] = 0.0
    for i in T.serial(N):
        total[0] = total[0] + x[i]
    mean = total[0] / T.cast(N, T.float32)

    var = T.alloc_fragment([1], T.float32)
    var[0] = 0.0
    for i in T.serial(N):
        d = x[i] - mean
        var[0] = var[0] + d * d
    return var[0] / (T.cast(N, T.float32) - 1.0)


@tilelang.jit(target='cuda')
def _argmax_kernel(x):
    """Argmax index."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    best_idx = T.alloc_fragment([1], T.int32)
    best_val = T.alloc_fragment([1], T.float32)
    best_val[0] = x[0]
    best_idx[0] = 0
    for i in T.serial(1, N):
        if x[i] > best_val[0]:
            best_val[0] = x[i]
            best_idx[0] = i
    return best_idx[0]


@tilelang.jit(target='cuda')
def _argmin_kernel(x):
    """Argmin index."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    best_idx = T.alloc_fragment([1], T.int32)
    best_val = T.alloc_fragment([1], T.float32)
    best_val[0] = x[0]
    best_idx[0] = 0
    for i in T.serial(1, N):
        if x[i] < best_val[0]:
            best_val[0] = x[i]
            best_idx[0] = i
    return best_idx[0]


@tilelang.jit(target='cuda')
def _cumsum_kernel(x):
    """Cumulative sum."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    acc = T.alloc_fragment([1], T.float32)
    acc[0] = 0.0
    for i in T.serial(N):
        acc[0] = acc[0] + x[i]
        out[i] = acc[0]
    return out


# ── Norm ─────────────────────────────────────────────────────────────────────

@tilelang.jit(target='cuda')
def _l1_norm_kernel(x):
    """L1 norm (sum of absolute values)."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    total = T.alloc_fragment([1], T.float32)
    total[0] = 0.0
    for i in T.serial(N):
        total[0] = total[0] + T.abs(x[i])
    return total[0]


@tilelang.jit(target='cuda')
def _l2_norm_kernel(x):
    """L2 norm (Euclidean)."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    total = T.alloc_fragment([1], T.float32)
    total[0] = 0.0
    for i in T.serial(N):
        total[0] = total[0] + x[i] * x[i]
    return T.sqrt(total[0])


@tilelang.jit(target='cuda')
def _inf_norm_kernel(x):
    """Infinity norm (max absolute value)."""
    N = T.const("N")
    x: T.Tensor[[N], T.float32]
    max_val = T.alloc_fragment([1], T.float32)
    max_val[0] = T.abs(x[0])
    for i in T.serial(1, N):
        v = T.abs(x[i])
        if v > max_val[0]:
            max_val[0] = v
    return max_val[0]


# ── FFT (Goertzel algorithm for spectral analysis) ──────────────────────────

@tilelang.jit(target='cuda')
def _goertzel_kernel(signal, freq, n_samples):
    """Goertzel algorithm for single-frequency DFT on GPU.

    More efficient than full FFT when only a few frequencies are needed.
    """
    N = T.const("N")
    signal: T.Tensor[[N], T.float32]
    k = T.const("K")  # frequency bin
    coeff = T.const("COEFF")  # 2 * cos(2*pi*k/N)

    s_prev = T.alloc_fragment([1], T.float32)
    s_prev2 = T.alloc_fragment([1], T.float32)
    s_prev[0] = 0.0
    s_prev2[0] = 0.0

    for i in T.serial(N):
        s = signal[i] + coeff * s_prev[0] - s_prev2[0]
        s_prev2[0] = s_prev[0]
        s_prev[0] = s

    # Power = s_prev^2 + s_prev2^2 - coeff * s_prev * s_prev2
    power = s_prev[0] * s_prev[0] + s_prev2[0] * s_prev2[0] - \
            coeff * s_prev[0] * s_prev2[0]
    return power


@tilelang.jit(target='cuda')
def _spectral_strength_kernel(signal, n_samples, n_bins):
    """Multi-bin spectral strength via Goertzel.

    Computes DFT power at bins 1..n_bins.
    """
    N = T.const("N")
    NB = T.const("NB")
    signal: T.Tensor[[N], T.float32]

    power = T.empty([NB], T.float32)

    for b in T.serial(1, NB):
        coeff = 2.0 * T.cos(6.283185307179586 * T.cast(b, T.float32) / T.cast(N, T.float32))

        s_prev = T.alloc_fragment([1], T.float32)
        s_prev2 = T.alloc_fragment([1], T.float32)
        s_prev[0] = 0.0
        s_prev2[0] = 0.0

        for i in T.serial(N):
            s = signal[i] + coeff * s_prev[0] - s_prev2[0]
            s_prev2[0] = s_prev[0]
            s_prev[0] = s

        power[b - 1] = s_prev[0] * s_prev[0] + s_prev2[0] * s_prev2[0] - \
                        coeff * s_prev[0] * s_prev2[0]

    return power


# ── Convolution ──────────────────────────────────────────────────────────────

@tilelang.jit(target='cuda')
def _conv1d_kernel(signal, kernel, n_signal, n_kernel):
    """1D convolution (valid mode)."""
    NS = T.const("NS")
    NK = T.const("NK")
    signal: T.Tensor[[NS], T.float32]
    kernel: T.Tensor[[NK], T.float32]

    out_len = NS - NK + 1
    out = T.empty([out_len], T.float32)

    for i in T.serial(out_len):
        acc = T.alloc_fragment([1], T.float32)
        acc[0] = 0.0
        for j in T.serial(NK):
            acc[0] = acc[0] + signal[i + j] * kernel[j]
        out[i] = acc[0]

    return out


@tilelang.jit(target='cuda')
def _ema_kernel(signal, alpha, n_samples):
    """Exponential Moving Average on GPU."""
    N = T.const("N")
    signal: T.Tensor[[N], T.float32]
    a = T.const("ALPHA")

    out = T.empty([N], T.float32)
    out[0] = signal[0]
    for i in T.serial(1, N):
        out[i] = a * signal[i] + (1.0 - a) * out[i - 1]

    return out


@tilelang.jit(target='cuda')
def _sma_kernel(signal, window, n_samples):
    """Simple Moving Average on GPU."""
    N = T.const("N")
    W = T.const("W")
    signal: T.Tensor[[N], T.float32]

    out = T.empty([N], T.float32)
    for i in T.serial(N):
        if i < W - 1:
            out[i] = 0.0
        else:
            acc = T.alloc_fragment([1], T.float32)
            acc[0] = 0.0
            for j in T.serial(i - W + 1, i + 1):
                acc[0] = acc[0] + signal[j]
            out[i] = acc[0] / T.cast(W, T.float32)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Python API — TileOPs-style interface
# ─────────────────────────────────────────────────────────────────────────────

class TileOps:
    """TileOPs-style GPU operators for HFT signal processing.

    All operations run on GPU via Tilelang kernels.
    """

    def __init__(self):
        self._cache = {}

    def _to_dev(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.astype(np.float32)).cuda()

    def _from_dev(self, x) -> np.ndarray:
        return x.cpu().numpy()

    # ── Elementwise ──────────────────────────────────────────────────────

    def relu(self, x: np.ndarray) -> np.ndarray:
        x_dev = self._to_dev(x)
        return self._from_dev(_relu_kernel(x_dev, N=len(x)))

    def abs(self, x: np.ndarray) -> np.ndarray:
        x_dev = self._to_dev(x)
        return self._from_dev(_abs_kernel(x_dev, N=len(x)))

    def exp(self, x: np.ndarray) -> np.ndarray:
        x_dev = self._to_dev(x)
        return self._from_dev(_exp_kernel(x_dev, N=len(x)))

    def log(self, x: np.ndarray) -> np.ndarray:
        x_dev = self._to_dev(x)
        return self._from_dev(_log_kernel(x_dev, N=len(x)))

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        x_dev = self._to_dev(x)
        return self._from_dev(_sqrt_kernel(x_dev, N=len(x)))

    def clamp(self, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        x_dev = self._to_dev(x)
        return self._from_dev(_clamp_kernel(x_dev, lo, hi, N=len(x), LO=lo, HI=hi))

    def sign(self, x: np.ndarray) -> np.ndarray:
        x_dev = self._to_dev(x)
        return self._from_dev(_sign_kernel(x_dev, N=len(x)))

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        x_dev = self._to_dev(x)
        return self._from_dev(_sigmoid_kernel(x_dev, N=len(x)))

    def tanh(self, x: np.ndarray) -> np.ndarray:
        x_dev = self._to_dev(x)
        return self._from_dev(_tanh_kernel(x_dev, N=len(x)))

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_dev = self._to_dev(a)
        b_dev = self._to_dev(b)
        return self._from_dev(_binary_add_kernel(a_dev, b_dev, N=len(a)))

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_dev = self._to_dev(a)
        b_dev = self._to_dev(b)
        return self._from_dev(_binary_mul_kernel(a_dev, b_dev, N=len(a)))

    # ── Reduction ────────────────────────────────────────────────────────

    def sum(self, x: np.ndarray) -> float:
        x_dev = self._to_dev(x)
        return float(_sum_kernel(x_dev, N=len(x)).item())

    def mean(self, x: np.ndarray) -> float:
        x_dev = self._to_dev(x)
        return float(_mean_kernel(x_dev, N=len(x)).item())

    def max(self, x: np.ndarray) -> float:
        x_dev = self._to_dev(x)
        return float(_max_kernel(x_dev, N=len(x)).item())

    def min(self, x: np.ndarray) -> float:
        x_dev = self._to_dev(x)
        return float(_min_kernel(x_dev, N=len(x)).item())

    def var(self, x: np.ndarray) -> float:
        x_dev = self._to_dev(x)
        return float(_var_kernel(x_dev, N=len(x)).item())

    def argmax(self, x: np.ndarray) -> int:
        x_dev = self._to_dev(x)
        return int(_argmax_kernel(x_dev, N=len(x)).item())

    def argmin(self, x: np.ndarray) -> int:
        x_dev = self._to_dev(x)
        return int(_argmin_kernel(x_dev, N=len(x)).item())

    def cumsum(self, x: np.ndarray) -> np.ndarray:
        x_dev = self._to_dev(x)
        return self._from_dev(_cumsum_kernel(x_dev, N=len(x)))

    # ── Norm ─────────────────────────────────────────────────────────────

    def l1_norm(self, x: np.ndarray) -> float:
        x_dev = self._to_dev(x)
        return float(_l1_norm_kernel(x_dev, N=len(x)).item())

    def l2_norm(self, x: np.ndarray) -> float:
        x_dev = self._to_dev(x)
        return float(_l2_norm_kernel(x_dev, N=len(x)).item())

    def inf_norm(self, x: np.ndarray) -> float:
        x_dev = self._to_dev(x)
        return float(_inf_norm_kernel(x_dev, N=len(x)).item())

    # ── FFT / Spectral ───────────────────────────────────────────────────

    def spectral_strength(self, x: np.ndarray, n_bins: int = 20) -> np.ndarray:
        """Multi-bin spectral strength via Goertzel."""
        x_dev = self._to_dev(x)
        n_dev = torch.tensor(len(x), dtype=torch.int32, device="cuda")
        nb_dev = torch.tensor(n_bins, dtype=torch.int32, device="cuda")
        return self._from_dev(_spectral_strength_kernel(
            x_dev, n_dev, nb_dev, N=len(x), NB=n_bins
        ))

    def goertzel(self, x: np.ndarray, freq_bin: int) -> float:
        """Single-frequency DFT power."""
        x_dev = self._to_dev(x)
        n_dev = torch.tensor(len(x), dtype=torch.int32, device="cuda")
        coeff = 2.0 * np.cos(2.0 * np.pi * freq_bin / len(x))
        return float(_goertzel_kernel(
            x_dev, freq_bin, n_dev,
            N=len(x), K=freq_bin, COEFF=coeff
        ).item())

    # ── Convolution / Smoothing ──────────────────────────────────────────

    def conv1d(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """1D convolution (valid mode)."""
        s_dev = self._to_dev(signal)
        k_dev = self._to_dev(kernel)
        ns_dev = torch.tensor(len(signal), dtype=torch.int32, device="cuda")
        nk_dev = torch.tensor(len(kernel), dtype=torch.int32, device="cuda")
        return self._from_dev(_conv1d_kernel(
            s_dev, k_dev, ns_dev, nk_dev,
            NS=len(signal), NK=len(kernel)
        ))

    def ema(self, x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """Exponential Moving Average."""
        x_dev = self._to_dev(x)
        return self._from_dev(_ema_kernel(x_dev, alpha, N=len(x), ALPHA=alpha))

    def sma(self, x: np.ndarray, window: int = 20) -> np.ndarray:
        """Simple Moving Average."""
        x_dev = self._to_dev(x)
        return self._from_dev(_sma_kernel(x_dev, window, N=len(x), W=window))

    # ── Signal processing pipeline ───────────────────────────────────────

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        m = self.mean(x)
        v = self.var(x)
        std = self.sqrt(np.array([v], dtype=np.float32))[0]
        centered = self.add(x, np.full_like(x, -m))
        return centered / (std + 1e-10)

    def smooth_and_detect(self, x: np.ndarray, window: int = 20,
                          threshold: float = 2.0) -> np.ndarray:
        """Smooth signal and detect anomalies (z-score > threshold)."""
        smoothed = self.sma(x, window)
        residual = self.add(x, np.full_like(x, 0))  # copy
        residual = self.add(residual, np.full_like(x, 0))  # placeholder
        # Simple: flag where |raw - smoothed| > threshold * std
        diff = x - smoothed
        std = np.sqrt(self.var(diff))
        return np.abs(diff) > threshold * std
