/// gpu_ops — CuPy replacement functions.
///
/// These provide the same interface as the original CuPy code in:
///   - walk_forward._resample_gpu
///   - gpu_walk_forward.{to_ret, cs, rolling, rolling_std}
///   - gpu_optimizer (MA cross via rolling_mean)
///
/// All functions take/return Vec/Python types — no numpy/CuPy dependency.

use pyo3::prelude::*;
use pyo3::types::PyList;

/// Scatter-add resampling: replaces cp.add.at + argsort pattern.
/// Returns (close_bars, vol_bars) as Python lists.
#[pyfunction]
pub fn resample_bars(
    py: Python<'_>,
    times: Vec<i64>,
    prices: Vec<f32>,
    qtys: Vec<f32>,
    bar_seconds: i32,
) -> PyResult<(PyObject, PyObject)> {
    if times.is_empty() {
        return Ok((py.None(), py.None()));
    }
    let bar_ms = bar_seconds as i64 * 1000;
    let t_start = (times[0] / bar_ms) * bar_ms;
    let t_last = (times[times.len() - 1] / bar_ms) * bar_ms;
    let n_bars = ((t_last - t_start) / bar_ms + 1) as usize;

    let mut vol_bars = vec![0.0f32; n_bars];
    let mut close_bars = vec![0.0f32; n_bars];

    for i in 0..times.len() {
        let idx = (((times[i] - t_start) / bar_ms) as usize).min(n_bars - 1);
        vol_bars[idx] += qtys[i];
        close_bars[idx] = prices[i];
    }

    let mut last = prices[0];
    for i in 0..n_bars {
        if close_bars[i] == 0.0 {
            close_bars[i] = last;
        } else {
            last = close_bars[i];
        }
    }

    let close_list = PyList::new(py, &close_bars)?;
    let vol_list = PyList::new(py, &vol_bars)?;
    Ok((close_list.into(), vol_list.into()))
}

/// Per-bar returns: ret[0]=0, ret[i]=(p[i]-p[i-1])/(|p[i-1]|+eps).
#[pyfunction]
pub fn compute_returns(prices: Vec<f32>) -> Vec<f32> {
    let n = prices.len();
    if n == 0 {
        return vec![];
    }
    let mut ret = vec![0.0f32; n];
    for i in 1..n {
        let denom = prices[i - 1].abs() + 1e-9;
        ret[i] = (prices[i] - prices[i - 1]) / denom;
    }
    ret
}

/// Rolling mean via cumsum (f64 precision), NaN-padded to same length.
#[pyfunction]
pub fn rolling_mean(prices: Vec<f32>, window: usize) -> Vec<f32> {
    let n = prices.len();
    if n == 0 || window == 0 {
        return vec![];
    }
    let mut out = vec![f32::NAN; n];
    if window > n {
        return out;
    }
    let mut cs = vec![0.0f64; n + 1];
    for i in 0..n {
        cs[i + 1] = cs[i] + prices[i] as f64;
    }
    for i in (window - 1)..n {
        let sum = cs[i + 1] - cs[i + 1 - window];
        out[i] = (sum / window as f64) as f32;
    }
    out
}

/// Rolling std via cumsum of x and x², NaN-padded to same length.
#[pyfunction]
pub fn rolling_std_rs(values: Vec<f32>, window: usize) -> Vec<f32> {
    let n = values.len();
    if n == 0 || window == 0 {
        return vec![];
    }
    let mut out = vec![f32::NAN; n];
    if window > n {
        return out;
    }
    let mut cs1 = vec![0.0f64; n + 1];
    let mut cs2 = vec![0.0f64; n + 1];
    for i in 0..n {
        cs1[i + 1] = cs1[i] + values[i] as f64;
        cs2[i + 1] = cs2[i] + (values[i] as f64).powi(2);
    }
    for i in (window - 1)..n {
        let sum_x = cs1[i + 1] - cs1[i + 1 - window];
        let sum_x2 = cs2[i + 1] - cs2[i + 1 - window];
        let mu = sum_x / window as f64;
        let mu2 = sum_x2 / window as f64;
        let var = (mu2 - mu * mu).max(0.0);
        out[i] = var.sqrt() as f32;
    }
    out
}

/// Threshold signal: +1 if >thr, -1 if <-thr, 0 otherwise.
#[pyfunction]
pub fn threshold_signal(signal: Vec<f32>, threshold: f32) -> Vec<f32> {
    signal
        .iter()
        .map(|s| {
            if *s > threshold {
                1.0
            } else if *s < -threshold {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Forward returns: fwd[i] = sum(ret[i..i+hold]) via cumsum.
#[pyfunction]
pub fn forward_returns(ret: Vec<f32>, hold: usize) -> Vec<f32> {
    let n = ret.len();
    if n == 0 || hold == 0 {
        return vec![0.0; n];
    }
    if hold >= n {
        return vec![0.0; n];
    }
    let mut cs = vec![0.0f64; n + 1];
    for i in 0..n {
        cs[i + 1] = cs[i] + ret[i] as f64;
    }
    let limit = n - hold;
    let mut fwd = vec![0.0f32; n];
    for i in 0..limit {
        fwd[i] = (cs[i + hold] - cs[i]) as f32;
    }
    fwd
}

/// Backtest sweep: find best threshold sharpe for one (signal, hold) pair.
/// Returns (sharpe, threshold, n_trades).
#[pyfunction]
pub fn backtest_sweep(
    signal: Vec<f32>,
    fwd_ret: Vec<f32>,
    thresholds: Vec<f32>,
    hold: i32,
) -> (f32, f32, usize) {
    let n = signal.len();
    let valid_len = n.min(fwd_ret.len());
    if n == 0 || valid_len == 0 {
        return (-999.0, 0.0, 0);
    }
    let mut best_sh = -999.0f32;
    let mut best_thr = 0.0f32;
    let mut best_n_trades = 0usize;

    for &thr in &thresholds {
        let mut n_trades = 0usize;
        let mut pnl_sum = 0.0f64;
        let mut pnl_sq_sum = 0.0f64;

        for i in 0..valid_len {
            let entry = if signal[i] > thr {
                1.0
            } else if signal[i] < -thr {
                -1.0
            } else {
                0.0
            };
            if entry != 0.0 {
                let p = entry as f64 * fwd_ret[i] as f64;
                n_trades += 1;
                pnl_sum += p;
                pnl_sq_sum += p * p;
            }
        }

        if n_trades >= 3 {
            let mean_p = pnl_sum / n_trades as f64;
            let var_p = (pnl_sq_sum / n_trades as f64 - mean_p * mean_p).max(0.0);
            let std_p = var_p.sqrt() + 1e-9;
            let sh = (mean_p / std_p) * (252.0 / hold as f64).sqrt();
            if sh as f32 > best_sh {
                best_sh = sh as f32;
                best_thr = thr;
                best_n_trades = n_trades;
            }
        }
    }
    (best_sh, best_thr, best_n_trades)
}

/// MA gap: (fma - sma) / (|sma| + eps).
#[pyfunction]
pub fn ma_gap(fma: Vec<f32>, sma: Vec<f32>) -> Vec<f32> {
    let n = fma.len().min(sma.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let denom = sma[i].abs() + 1e-9;
        out.push((fma[i] - sma[i]) / denom);
    }
    out
}
