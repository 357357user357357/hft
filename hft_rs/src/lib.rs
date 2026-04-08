/// hft_rs — Unified Rust-accelerated crate for HFT.
///
/// Modules:
///   ricci  — Ollivier-Ricci curvature, HMM, Kalman pair arb
///   analytics — Hurst, order flow, autocorr, rolling Sharpe, ADF, vol regime
///   gpu_ops — CuPy replacement: resample, returns, rolling mean/std,
///             threshold, forward returns, backtest sweep, MA gap

mod ricci;
mod analytics;
mod gpu_ops;

use pyo3::prelude::*;

#[pymodule]
pub fn hft_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Analytics ──────────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(analytics::analyze_symbol, m)?)?;
    // ── GPU / CuPy replacements ────────────────────────────────────────
    m.add_function(wrap_pyfunction!(gpu_ops::resample_bars, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_ops::compute_returns, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_ops::rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_ops::rolling_std_rs, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_ops::threshold_signal, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_ops::forward_returns, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_ops::backtest_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_ops::ma_gap, m)?)?;
    // ── Ricci curvature ────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(ricci::wasserstein_1, m)?)?;
    m.add_function(wrap_pyfunction!(ricci::ollivier_ricci, m)?)?;
    m.add_function(wrap_pyfunction!(ricci::all_edge_curvatures, m)?)?;
    m.add_function(wrap_pyfunction!(ricci::ricci_flow, m)?)?;
    m.add_function(wrap_pyfunction!(ricci::hmm_regime_fit, m)?)?;
    m.add_function(wrap_pyfunction!(ricci::hmm_vol_fit, m)?)?;
    // ── Pair arbitrage ─────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(ricci::kalman_pair_update, m)?)?;
    m.add_function(wrap_pyfunction!(ricci::kalman_pair_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ricci::pair_zscore, m)?)?;
    m.add_function(wrap_pyfunction!(ricci::engle_granger_coint, m)?)?;
    m.add_function(wrap_pyfunction!(ricci::pair_backtest_run, m)?)?;
    Ok(())
}
