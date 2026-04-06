/// analytics_rs — Rust-accelerated CPU analytics for HFT.
///
/// Replaces the pure-Python `_cpu_analyze_symbol` hot path.
/// Computes over ~10K ticks:
///   - Hurst exponent (R/S, 12 lag scales)
///   - Order flow imbalance
///   - Multi-lag autocorrelation (lags 1-50)
///   - Rolling Sharpe (100 windows)
///   - ADF-lite stationarity
///   - Volatility regime ratio

use pyo3::prelude::*;
use std::collections::HashMap;

const TARGET: usize = 10_000;

/// Extend prices to TARGET length by bootstrapping.
fn extend_prices(prices: &[f64]) -> Vec<f64> {
    if prices.len() >= TARGET {
        return prices[prices.len() - TARGET..].to_vec();
    }
    let mut ext = prices.to_vec();
    let n = prices.len();
    let mut idx = 1usize;
    while ext.len() < TARGET {
        let i = idx % n.max(2);
        let i = if i == 0 { 1 } else { i };
        let ret = (prices[i] - prices[i - 1]) / (prices[i - 1] + 1e-10);
        let last = *ext.last().unwrap();
        ext.push(last * (1.0 + ret));
        idx += 1;
    }
    ext
}

/// Compute returns from prices.
fn returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / (w[0] + 1e-10))
        .collect()
}

/// Hurst exponent via R/S analysis over 12 lag scales.
fn hurst(rets: &[f64]) -> f64 {
    let lags: &[usize] = &[16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048];
    let n = rets.len();
    let mut ns: Vec<f64> = Vec::new();
    let mut rs: Vec<f64> = Vec::new();

    for &lag in lags {
        let lag = lag.min(n);
        if lag > n || lag < 4 {
            continue;
        }
        let sub = &rets[n - lag..];
        let m: f64 = sub.iter().sum::<f64>() / lag as f64;
        let dev: Vec<f64> = sub.iter().map(|x| x - m).collect();
        let mut cs = Vec::with_capacity(lag);
        let mut c = 0.0f64;
        for &d in &dev {
            c += d;
            cs.push(c);
        }
        let r = cs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - cs.iter().cloned().fold(f64::INFINITY, f64::min);
        let s = (dev.iter().map(|d| d * d).sum::<f64>() / lag as f64).sqrt();
        if s > 1e-10 && r > 0.0 {
            ns.push(lag as f64);
            rs.push(r / s);
        }
    }
    if ns.len() < 3 {
        return 0.5;
    }
    // Linear regression on log-log
    let lx: Vec<f64> = ns.iter().map(|x| x.ln()).collect();
    let ly: Vec<f64> = rs.iter().map(|x| x.ln()).collect();
    let k = lx.len() as f64;
    let mx: f64 = lx.iter().sum::<f64>() / k;
    let my: f64 = ly.iter().sum::<f64>() / k;
    let num: f64 = lx.iter().zip(ly.iter()).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = lx.iter().map(|x| (x - mx).powi(2)).sum();
    (num / (den + 1e-10)).clamp(0.0, 1.0)
}

/// Order flow imbalance (fraction of positive returns in last 100).
fn ofi(rets: &[f64]) -> f64 {
    let n = rets.len().min(100);
    let tail = &rets[rets.len() - n..];
    let pos = tail.iter().filter(|&&r| r > 0.0).count();
    pos as f64 / n as f64
}

/// Best absolute autocorrelation across lags 1-50.
fn autocorr(rets: &[f64]) -> f64 {
    let n = rets.len().min(5000);
    let ac_rets = &rets[rets.len() - n..];
    let mut best = 0.0f64;
    for lag in 1..=50 {
        if lag >= ac_rets.len() {
            break;
        }
        let r1 = &ac_rets[lag..];
        let r2 = &ac_rets[..ac_rets.len() - lag];
        let k = r1.len() as f64;
        let m1: f64 = r1.iter().sum::<f64>() / k;
        let m2: f64 = r2.iter().sum::<f64>() / k;
        let num: f64 = r1.iter().zip(r2.iter()).map(|(a, b)| (a - m1) * (b - m2)).sum();
        let d1: f64 = r1.iter().map(|x| (x - m1).powi(2)).sum::<f64>().sqrt();
        let d2: f64 = r2.iter().map(|x| (x - m2).powi(2)).sum::<f64>().sqrt();
        let ac = num / (d1 * d2 + 1e-10);
        if ac.abs() > best.abs() {
            best = ac;
        }
    }
    best
}

/// Best Sharpe ratio across 100 rolling windows (200-4000 ticks).
fn rolling_sharpe(rets: &[f64]) -> f64 {
    let mut best = 0.0f64;
    let mut w = 200;
    while w <= 4000 {
        if w > rets.len() {
            w += 36;
            continue;
        }
        let rw = &rets[rets.len() - w..];
        let m: f64 = rw.iter().sum::<f64>() / w as f64;
        let s: f64 = (rw.iter().map(|x| (x - m).powi(2)).sum::<f64>() / w as f64).sqrt();
        let sh = (m / (s + 1e-10)) * (252.0_f64).sqrt();
        if sh.abs() > best.abs() {
            best = sh;
        }
        w += 36;
    }
    best
}

/// ADF-lite: regression coefficient of delta(price) on lagged price.
fn adf_lite(prices: &[f64]) -> f64 {
    let n = prices.len().min(500);
    let pw = &prices[prices.len() - n..];
    let delta: Vec<f64> = pw.windows(2).map(|w| w[1] - w[0]).collect();
    let lag_p = &pw[..pw.len() - 1];
    let md: f64 = delta.iter().sum::<f64>() / delta.len() as f64;
    let ml: f64 = lag_p.iter().sum::<f64>() / lag_p.len() as f64;
    let cov: f64 = delta
        .iter()
        .zip(lag_p.iter())
        .map(|(d, l)| (d - md) * (l - ml))
        .sum();
    let var: f64 = lag_p.iter().map(|x| (x - ml).powi(2)).sum();
    cov / (var + 1e-10)
}

/// Volatility regime: ratio of short-term to long-term volatility.
fn vol_regime(rets: &[f64]) -> f64 {
    let short_n = rets.len().min(50);
    let long_n = rets.len().min(500);
    let short_vol = (rets[rets.len() - short_n..]
        .iter()
        .map(|r| r * r)
        .sum::<f64>()
        / short_n as f64)
        .sqrt();
    let long_vol = (rets[rets.len() - long_n..]
        .iter()
        .map(|r| r * r)
        .sum::<f64>()
        / long_n as f64)
        .sqrt();
    short_vol / (long_vol + 1e-10)
}

/// Full analytics pipeline — drop-in replacement for _cpu_analyze_symbol.
///
/// Args:
///   prices: list of floats (raw tick prices, minimum 20)
///
/// Returns:
///   dict with keys: hurst, ofi, autocorr, sharpe, adf, vol_regime
#[pyfunction]
fn analyze_symbol(prices: Vec<f64>) -> PyResult<HashMap<String, f64>> {
    let mut result = HashMap::new();
    result.insert("hurst".into(), 0.5);
    result.insert("ofi".into(), 0.5);
    result.insert("autocorr".into(), 0.0);
    result.insert("sharpe".into(), 0.0);
    result.insert("adf".into(), 0.0);
    result.insert("vol_regime".into(), 0.0);

    if prices.len() < 20 {
        return Ok(result);
    }

    let ext = extend_prices(&prices);
    let rets = returns(&ext);
    if rets.is_empty() {
        return Ok(result);
    }

    result.insert("hurst".into(), hurst(&rets));
    result.insert("ofi".into(), ofi(&rets));
    result.insert("autocorr".into(), autocorr(&rets));
    result.insert("sharpe".into(), rolling_sharpe(&rets));
    result.insert("adf".into(), adf_lite(&ext));
    result.insert("vol_regime".into(), vol_regime(&rets));

    Ok(result)
}

#[pymodule]
fn analytics_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_symbol, m)?)?;
    Ok(())
}
