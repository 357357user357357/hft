/// ricci_rs — Rust-accelerated Ollivier–Ricci curvature for HFT regime detection.
///
/// Exposes to Python:
///   wasserstein_1(mu, nu, D) -> f64
///   ollivier_ricci(i, j, graph, D, alpha) -> f64
///   all_edge_curvatures(graph, D, alpha) -> (list[float], dict)
///   ricci_flow(graph, D, steps, dt, alpha) -> (list[list[float]], list[float])

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;

const MAX_SUPPORT: usize = 16;
const INF: f64 = 1e30;
const EPS: f64 = 1e-14;
const SOLVE_ITERS: usize = 30;

// ── Core LP solver (pure Rust, zero allocation on hot path) ──────────────────

fn wasserstein_1_inner(
    nodes_i: &[usize],
    weights_i: &[f64],
    nodes_j: &[usize],
    weights_j: &[f64],
    dist: &[Vec<f64>],
) -> f64 {
    let m = nodes_i.len().min(MAX_SUPPORT);
    let n = nodes_j.len().min(MAX_SUPPORT);
    if m == 0 || n == 0 { return 0.0; }

    let ts: f64 = weights_i[..m].iter().sum();
    let td: f64 = weights_j[..n].iter().sum();

    let mut supply = [0f64; MAX_SUPPORT];
    let mut demand = [0f64; MAX_SUPPORT];
    for a in 0..m { supply[a] = if ts > 1e-15 { weights_i[a] / ts } else { weights_i[a] }; }
    for b in 0..n { demand[b] = if td > 1e-15 { weights_j[b] / td } else { weights_j[b] }; }

    let mut cost = [[0f64; MAX_SUPPORT]; MAX_SUPPORT];
    for a in 0..m {
        for b in 0..n {
            cost[a][b] = dist[nodes_i[a]][nodes_j[b]];
        }
    }

    // North-West Corner initial feasible solution
    let mut flow = [[0f64; MAX_SUPPORT]; MAX_SUPPORT];
    let (mut ai, mut bj) = (0usize, 0usize);
    while ai < m && bj < n {
        let amount = supply[ai].min(demand[bj]);
        flow[ai][bj] = amount;
        supply[ai] -= amount;
        demand[bj] -= amount;
        if supply[ai] < EPS { ai += 1; }
        if demand[bj] < EPS { bj += 1; }
    }

    // Stepping-stone optimality iterations
    let mut u = [INF; MAX_SUPPORT];
    let mut v = [INF; MAX_SUPPORT];

    for _ in 0..SOLVE_ITERS {
        // Reset duals
        u[0] = 0.0;
        for x in u[1..m].iter_mut() { *x = INF; }
        for x in v[..n].iter_mut() { *x = INF; }

        // Propagate duals from basic cells
        let mut changed = true;
        while changed {
            changed = false;
            for a in 0..m {
                for b in 0..n {
                    if flow[a][b] > EPS {
                        if u[a] < INF && v[b] >= INF {
                            v[b] = cost[a][b] - u[a];
                            changed = true;
                        } else if v[b] < INF && u[a] >= INF {
                            u[a] = cost[a][b] - v[b];
                            changed = true;
                        }
                    }
                }
            }
        }
        for a in 0..m { if u[a] >= INF { u[a] = 0.0; } }
        for b in 0..n { if v[b] >= INF { v[b] = 0.0; } }

        // Find most-negative reduced cost (entering variable)
        let mut best_rc = -1e-10f64;
        let mut best_a = usize::MAX;
        let mut best_b = usize::MAX;
        for a in 0..m {
            for b in 0..n {
                let rc = cost[a][b] - u[a] - v[b];
                if rc < best_rc { best_rc = rc; best_a = a; best_b = b; }
            }
        }
        if best_a == usize::MAX { break; }

        // Pivot: find stepping-stone loop
        let (ea, eb) = (best_a, best_b);
        let mut min_shift = f64::INFINITY;
        let mut pivot_b = usize::MAX;
        let mut pivot_a = usize::MAX;
        for b in 0..n {
            if b != eb && flow[ea][b] > EPS {
                for a in 0..m {
                    if a != ea && flow[a][eb] > EPS {
                        let shift = flow[ea][b].min(flow[a][eb]);
                        if shift < min_shift { min_shift = shift; pivot_b = b; pivot_a = a; }
                    }
                }
            }
        }
        if min_shift <= EPS || min_shift.is_infinite() || pivot_b == usize::MAX { break; }

        flow[ea][eb] += min_shift;
        flow[ea][pivot_b] -= min_shift;
        flow[pivot_a][eb] -= min_shift;
        flow[pivot_a][pivot_b] += min_shift;
    }

    let mut total = 0.0f64;
    for a in 0..m { for b in 0..n { total += flow[a][b] * cost[a][b]; } }
    total
}

// ── Lazy random-walk measure ─────────────────────────────────────────────────

fn lazy_measure(node: usize, neighbours: &[usize], alpha: f64) -> (Vec<usize>, Vec<f64>) {
    let deg = neighbours.len();
    if deg == 0 { return (vec![node], vec![1.0]); }
    let w_nbr = (1.0 - alpha) / deg as f64;
    let mut nodes = Vec::with_capacity(deg + 1);
    let mut weights = Vec::with_capacity(deg + 1);
    nodes.push(node);
    weights.push(alpha);
    for &nb in neighbours { nodes.push(nb); weights.push(w_nbr); }
    (nodes, weights)
}

// ── Parse helpers ────────────────────────────────────────────────────────────

fn parse_dist(d: &Bound<'_, PyList>) -> PyResult<Vec<Vec<f64>>> {
    d.iter().map(|row| {
        row.downcast::<PyList>()?.iter()
            .map(|x| x.extract::<f64>())
            .collect::<PyResult<Vec<f64>>>()
    }).collect()
}

fn parse_graph(graph: &Bound<'_, PyDict>) -> PyResult<HashMap<usize, Vec<usize>>> {
    let mut adj = HashMap::new();
    for (k, v) in graph.iter() {
        let node = k.extract::<usize>()?;
        let nbrs: Vec<usize> = v.downcast::<PyList>()?.iter()
            .map(|x| x.extract::<usize>())
            .collect::<PyResult<_>>()?;
        adj.insert(node, nbrs);
    }
    Ok(adj)
}

// ── Python-facing functions ──────────────────────────────────────────────────

/// wasserstein_1(mu, nu, D) -> float
#[pyfunction]
fn wasserstein_1(
    _py: Python<'_>,
    mu: &Bound<'_, PyList>,
    nu: &Bound<'_, PyList>,
    d: &Bound<'_, PyList>,
) -> PyResult<f64> {
    let dist = parse_dist(d)?;
    let mut ni = Vec::new(); let mut wi = Vec::new();
    for item in mu.iter() {
        let t = item.downcast::<PyTuple>()?;
        ni.push(t.get_item(0)?.extract::<usize>()?);
        wi.push(t.get_item(1)?.extract::<f64>()?);
    }
    let mut nj = Vec::new(); let mut wj = Vec::new();
    for item in nu.iter() {
        let t = item.downcast::<PyTuple>()?;
        nj.push(t.get_item(0)?.extract::<usize>()?);
        wj.push(t.get_item(1)?.extract::<f64>()?);
    }
    Ok(wasserstein_1_inner(&ni, &wi, &nj, &wj, &dist))
}

/// ollivier_ricci(i, j, graph, D, alpha=0.5) -> float
#[pyfunction]
#[pyo3(signature = (i, j, graph, d, alpha=0.5))]
fn ollivier_ricci(
    _py: Python<'_>,
    i: usize,
    j: usize,
    graph: &Bound<'_, PyDict>,
    d: &Bound<'_, PyList>,
    alpha: f64,
) -> PyResult<f64> {
    let dist = parse_dist(d)?;
    let dij = dist[i][j];
    if dij < 1e-12 { return Ok(0.0); }
    let adj = parse_graph(graph)?;
    let empty: Vec<usize> = vec![];
    let nbrs_i = adj.get(&i).unwrap_or(&empty);
    let nbrs_j = adj.get(&j).unwrap_or(&empty);
    let (ni, wi) = lazy_measure(i, nbrs_i, alpha);
    let (nj, wj) = lazy_measure(j, nbrs_j, alpha);
    let w1 = wasserstein_1_inner(&ni, &wi, &nj, &wj, &dist);
    Ok(1.0 - w1 / dij)
}

/// all_edge_curvatures(graph, D, alpha=0.5) -> (list[float], dict[(int,int)->float])
///
/// Parses graph and D once, then computes all edge curvatures in Rust.
/// This is the main speedup: avoids re-parsing D for every edge call from Python.
#[pyfunction]
#[pyo3(signature = (graph, d, alpha=0.5))]
fn all_edge_curvatures(
    py: Python<'_>,
    graph: &Bound<'_, PyDict>,
    d: &Bound<'_, PyList>,
    alpha: f64,
) -> PyResult<(Vec<f64>, PyObject)> {
    let dist = parse_dist(d)?;
    let adj = parse_graph(graph)?;

    let mut nodes: Vec<usize> = adj.keys().cloned().collect();
    nodes.sort();

    let mut kappas: Vec<f64> = Vec::new();
    let edge_kappa_py = PyDict::new(py);
    let mut visited: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let empty: Vec<usize> = vec![];

    for &i in &nodes {
        let nbrs_i = adj.get(&i).unwrap_or(&empty);
        for &j in nbrs_i {
            if visited.contains(&(j, i)) { continue; }
            visited.insert((i, j));

            let dij = dist[i][j];
            let kappa = if dij < 1e-12 {
                0.0
            } else {
                let nbrs_j = adj.get(&j).unwrap_or(&empty);
                let (ni, wi) = lazy_measure(i, nbrs_i, alpha);
                let (nj, wj) = lazy_measure(j, nbrs_j, alpha);
                let w1 = wasserstein_1_inner(&ni, &wi, &nj, &wj, &dist);
                1.0 - w1 / dij
            };

            kappas.push(kappa);
            edge_kappa_py.set_item((i, j), kappa)?;
            edge_kappa_py.set_item((j, i), kappa)?;
        }
    }

    Ok((kappas, edge_kappa_py.into()))
}

/// ricci_flow(graph, D, steps=3, dt=0.05, alpha=0.5)
///   -> (D_new: list[list[float]], kappa_history: list[float])
///
/// Full Ricci flow with volume normalisation — all steps in Rust.
#[pyfunction]
#[pyo3(signature = (graph, d, steps=3, dt=0.05, alpha=0.5))]
fn ricci_flow(
    py: Python<'_>,
    graph: &Bound<'_, PyDict>,
    d: &Bound<'_, PyList>,
    steps: usize,
    dt: f64,
    alpha: f64,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut dist = parse_dist(d)?;
    let adj = parse_graph(graph)?;
    let mut nodes: Vec<usize> = adj.keys().cloned().collect();
    nodes.sort();
    let empty: Vec<usize> = vec![];
    let mut history: Vec<f64> = Vec::with_capacity(steps);

    for _ in 0..steps {
        let mut edge_kappa: HashMap<(usize, usize), f64> = HashMap::new();
        let mut visited: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        let mut kappas_step: Vec<f64> = Vec::new();
        let mut edge_pairs: Vec<(usize, usize)> = Vec::new();

        for &i in &nodes {
            let nbrs_i = adj.get(&i).unwrap_or(&empty);
            for &j in nbrs_i {
                if visited.contains(&(j, i)) { continue; }
                visited.insert((i, j));
                edge_pairs.push((i, j));
                let dij = dist[i][j];
                let kappa = if dij < 1e-12 {
                    0.0
                } else {
                    let nbrs_j = adj.get(&j).unwrap_or(&empty);
                    let (ni, wi) = lazy_measure(i, nbrs_i, alpha);
                    let (nj, wj) = lazy_measure(j, nbrs_j, alpha);
                    let w1 = wasserstein_1_inner(&ni, &wi, &nj, &wj, &dist);
                    1.0 - w1 / dij
                };
                edge_kappa.insert((i, j), kappa);
                edge_kappa.insert((j, i), kappa);
                kappas_step.push(kappa);
            }
        }

        let mean_k = if kappas_step.is_empty() { 0.0 }
            else { kappas_step.iter().sum::<f64>() / kappas_step.len() as f64 };
        history.push(mean_k);

        let total_old: f64 = edge_pairs.iter().map(|&(i, j)| dist[i][j]).sum();
        let mut d_new = dist.clone();
        for &(i, j) in &edge_pairs {
            let kappa = edge_kappa.get(&(i, j)).cloned().unwrap_or(0.0);
            let new_d = (dist[i][j] - kappa * dist[i][j] * dt).max(1e-6);
            d_new[i][j] = new_d;
            d_new[j][i] = new_d;
        }
        let total_new: f64 = edge_pairs.iter().map(|&(i, j)| d_new[i][j]).sum();
        if total_new > 1e-12 && total_old > 1e-12 {
            let scale = total_old / total_new;
            for &(i, j) in &edge_pairs {
                d_new[i][j] *= scale;
                d_new[j][i] *= scale;
            }
        }
        dist = d_new;
    }

    // Convert dist → nested Python list via return type (pyo3 auto-converts Vec<Vec<f64>>)
    let _ = py; // py used implicitly by pyo3 for return conversion
    Ok((dist, history))
}

// ── Simons SDE: Baum-Welch HMM forward pass ─────────────────────────────────
//
// Both RegimeSwitchResult.fit() and HMMVolResult.fit() spend ~70% of their
// time in the forward pass: calling g()/gq() (Gaussian PDF) inside a Python
// loop over n*n_iter steps.  This Rust version runs the entire EM loop
// (forward + M-step) natively.
//
// hmm_regime_fit(rets, n_iter) -> (mu0, mu1, s0, s1, p01, p10, alpha_last_0)
//   Returns fitted 2-state HMM on log-returns for RegimeSwitchResult.
//
// hmm_vol_fit(obs, n_iter) -> (var0, var1, p01, p10, alpha_last_0)
//   Returns fitted 2-state HMM on squared returns for HMMVolResult.

#[inline(always)]
fn gauss_pdf(x: f64, mu: f64, s: f64) -> f64 {
    let z = (x - mu) / s;
    (-0.5 * z * z).exp() / (s * std::f64::consts::TAU.sqrt())
}

#[inline(always)]
fn gauss_pdf_zero_mean_sq(x_sq: f64, var: f64) -> f64 {
    // PDF of N(0, sqrt(var)) evaluated at a point where obs = x^2 (chi-sq proxy)
    let v = var.max(1e-12);
    (-0.5 * x_sq / v).exp() / (v.sqrt() * std::f64::consts::TAU.sqrt())
}

/// hmm_regime_fit(rets: list[float], n_iter: int) -> (mu0, mu1, s0, s1, p01, p10, alpha_last_0)
#[pyfunction]
#[pyo3(signature = (rets, n_iter=10))]
fn hmm_regime_fit(
    _py: Python<'_>,
    rets: Vec<f64>,
    n_iter: usize,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64)> {
    let n = rets.len();
    if n < 4 {
        return Ok((0.0, 0.0, 0.01, 0.01, 0.1, 0.1, 0.5));
    }

    // Initial params: split by median
    let mut sorted = rets.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[n / 2];

    let hi: Vec<f64> = rets.iter().cloned().filter(|&r| r >= med).collect();
    let lo: Vec<f64> = rets.iter().cloned().filter(|&r| r <  med).collect();

    let mean_std = |xs: &[f64]| -> (f64, f64) {
        if xs.is_empty() { return (0.0, 0.01); }
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        let v = xs.iter().map(|x| (x-m)*(x-m)).sum::<f64>() / xs.len() as f64;
        (m, v.sqrt().max(1e-6))
    };

    let (mut mu0, mut s0) = mean_std(&hi);
    let (mut mu1, mut s1) = mean_std(&lo);
    let (mut p01, mut p10) = (0.05f64, 0.05f64);

    let mut alpha = vec![[0f64; 2]; n];

    for _ in 0..n_iter {
        // Forward pass
        let b0 = gauss_pdf(rets[0], mu0, s0);
        let b1 = gauss_pdf(rets[0], mu1, s1);
        alpha[0] = [0.5 * b0, 0.5 * b1];
        let sc = alpha[0][0] + alpha[0][1];
        if sc > 0.0 { alpha[0][0] /= sc; alpha[0][1] /= sc; }

        for t in 1..n {
            let b0 = gauss_pdf(rets[t], mu0, s0);
            let b1 = gauss_pdf(rets[t], mu1, s1);
            alpha[t][0] = (alpha[t-1][0]*(1.0-p01) + alpha[t-1][1]*p10) * b0;
            alpha[t][1] = (alpha[t-1][0]*p01         + alpha[t-1][1]*(1.0-p10)) * b1;
            let sc = alpha[t][0] + alpha[t][1];
            if sc > 0.0 { alpha[t][0] /= sc; alpha[t][1] /= sc; }
        }

        // M-step
        let sw0 = alpha.iter().map(|a| a[0]).sum::<f64>().max(1.0);
        let sw1 = alpha.iter().map(|a| a[1]).sum::<f64>().max(1.0);
        mu0 = alpha.iter().zip(rets.iter()).map(|(a, &r)| a[0]*r).sum::<f64>() / sw0;
        mu1 = alpha.iter().zip(rets.iter()).map(|(a, &r)| a[1]*r).sum::<f64>() / sw1;
        s0 = (alpha.iter().zip(rets.iter()).map(|(a, &r)| a[0]*(r-mu0)*(r-mu0)).sum::<f64>() / sw0).sqrt().max(1e-6);
        s1 = (alpha.iter().zip(rets.iter()).map(|(a, &r)| a[1]*(r-mu1)*(r-mu1)).sum::<f64>() / sw1).sqrt().max(1e-6);

        let t01: f64 = (1..n).map(|t| alpha[t][0]*p01*gauss_pdf(rets[t],mu1,s1)).sum();
        let s00: f64 = (1..n).map(|t| alpha[t][0]*(1.0-p01)*gauss_pdf(rets[t],mu0,s0)).sum();
        p01 = t01 / (t01 + s00 + 1e-9);
        let t10: f64 = (1..n).map(|t| alpha[t][1]*p10*gauss_pdf(rets[t],mu0,s0)).sum();
        let s11: f64 = (1..n).map(|t| alpha[t][1]*(1.0-p10)*gauss_pdf(rets[t],mu1,s1)).sum();
        p10 = t10 / (t10 + s11 + 1e-9);
    }

    Ok((mu0, mu1, s0, s1, p01, p10, alpha[n-1][0]))
}

/// hmm_vol_fit(obs: list[float], n_iter: int) -> (var0, var1, p01, p10, alpha_last_0)
#[pyfunction]
#[pyo3(signature = (obs, n_iter=8))]
fn hmm_vol_fit(
    _py: Python<'_>,
    obs: Vec<f64>,
    n_iter: usize,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let n = obs.len();
    if n < 4 {
        return Ok((1e-4, 1e-3, 0.1, 0.3, 0.8));
    }

    let mut sorted = obs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[n / 2];

    let hi: Vec<f64> = obs.iter().cloned().filter(|&o| o >  med).collect();
    let lo: Vec<f64> = obs.iter().cloned().filter(|&o| o <= med).collect();
    let mean = |xs: &[f64]| if xs.is_empty() { 1e-4 } else { xs.iter().sum::<f64>() / xs.len() as f64 };

    let (mut var0, mut var1) = (mean(&lo).max(1e-12), mean(&hi).max(1e-12));
    let (mut p01, mut p10) = (0.1f64, 0.3f64);

    let mut alpha = vec![[0f64; 2]; n];

    for _ in 0..n_iter {
        let b0 = gauss_pdf_zero_mean_sq(obs[0], var0);
        let b1 = gauss_pdf_zero_mean_sq(obs[0], var1);
        alpha[0] = [0.5*b0, 0.5*b1];
        let sc = alpha[0][0]+alpha[0][1];
        if sc > 0.0 { alpha[0][0]/=sc; alpha[0][1]/=sc; }

        for t in 1..n {
            let b0 = gauss_pdf_zero_mean_sq(obs[t], var0);
            let b1 = gauss_pdf_zero_mean_sq(obs[t], var1);
            alpha[t][0] = (alpha[t-1][0]*(1.0-p01)+alpha[t-1][1]*p10)*b0;
            alpha[t][1] = (alpha[t-1][0]*p01        +alpha[t-1][1]*(1.0-p10))*b1;
            let sc = alpha[t][0]+alpha[t][1];
            if sc > 0.0 { alpha[t][0]/=sc; alpha[t][1]/=sc; }
        }

        let sw0 = alpha.iter().map(|a| a[0]).sum::<f64>().max(1.0);
        let sw1 = alpha.iter().map(|a| a[1]).sum::<f64>().max(1.0);
        var0 = (alpha.iter().zip(obs.iter()).map(|(a,&o)| a[0]*o).sum::<f64>()/sw0).max(1e-12);
        var1 = (alpha.iter().zip(obs.iter()).map(|(a,&o)| a[1]*o).sum::<f64>()/sw1).max(1e-12);

        let t01: f64 = (1..n).map(|t| alpha[t][0]*p01*gauss_pdf_zero_mean_sq(obs[t],var1)).sum();
        let s00: f64 = (1..n).map(|t| alpha[t][0]*(1.0-p01)*gauss_pdf_zero_mean_sq(obs[t],var0)).sum();
        p01 = t01/(t01+s00+1e-9);
        let t10: f64 = (1..n).map(|t| alpha[t][1]*p10*gauss_pdf_zero_mean_sq(obs[t],var0)).sum();
        let s11: f64 = (1..n).map(|t| alpha[t][1]*(1.0-p10)*gauss_pdf_zero_mean_sq(obs[t],var1)).sum();
        p10 = t10/(t10+s11+1e-9);
    }

    Ok((var0, var1, p01, p10, alpha[n-1][0]))
}

// ── Pair Arbitrage: Kalman filter spread tracker ─────────────────────────────
//
// Statistical pair arbitrage requires tracking a time-varying hedge ratio β
// and the residual spread  s_t = y_t − β·x_t  in real time.
//
// We use a 1-D Kalman filter (local-level model) to estimate β online:
//   State:      β_t  (hedge ratio, scalar)
//   Transition: β_t = β_{t-1} + w_t,   w_t ~ N(0, Q)   (random walk)
//   Observation:y_t = β_t·x_t + v_t,   v_t ~ N(0, R)   (measurement)
//
// After each update we return the spread z-score so Python can decide
// whether to enter/exit a mean-reversion trade.
//
// kalman_pair_update(beta, P, y, x, Q, R)
//   -> (beta_new, P_new, spread, spread_zscore, kalman_gain)
//
// kalman_pair_batch(ys, xs, Q, R)
//   -> (betas, Ps, spreads)   — full history for backtest
//
// pair_zscore(spreads, window)
//   -> (zscores, mean, std)   — rolling z-score of spread series
//
// engle_granger_coint(ys, xs)
//   -> (beta_ols, resid_adf_stat, coint_pvalue_approx)
//   Approx ADF on the OLS residuals; fast, no scipy needed.

/// Single Kalman filter step for pair hedge-ratio tracking.
///
/// Arguments:
///   beta  – current hedge ratio estimate
///   p     – current error covariance
///   y     – price of leg-A at this bar
///   x     – price of leg-B at this bar
///   q     – process noise variance (controls how fast β can drift)
///   r     – observation noise variance (measurement error)
///
/// Returns: (beta_new, p_new, spread, gain)
#[pyfunction]
#[pyo3(signature = (beta, p, y, x, q=1e-5, r=1e-3))]
fn kalman_pair_update(
    _py: Python<'_>,
    beta: f64,
    p: f64,
    y: f64,
    x: f64,
    q: f64,
    r: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    // Predict
    let p_pred = p + q;
    // Innovation (spread before update)
    let spread = y - beta * x;
    // Innovation variance
    let s = p_pred * x * x + r;
    // Kalman gain
    let gain = if s.abs() < 1e-30 { 0.0 } else { p_pred * x / s };
    // Update
    let beta_new = beta + gain * spread;
    let p_new    = (1.0 - gain * x) * p_pred;
    // Spread after update (residual with new β)
    let spread_new = y - beta_new * x;
    Ok((beta_new, p_new.max(1e-12), spread_new, gain))
}

/// Run Kalman filter over full price history (batch, for backtest).
///
/// Arguments:
///   ys  – price series for leg-A  (length N)
///   xs  – price series for leg-B  (length N)
///   q   – process noise
///   r   – observation noise
///
/// Returns: (betas, p_vars, spreads)  — each length N
#[pyfunction]
#[pyo3(signature = (ys, xs, q=1e-5, r=1e-3))]
fn kalman_pair_batch(
    _py: Python<'_>,
    ys: Vec<f64>,
    xs: Vec<f64>,
    q: f64,
    r: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = ys.len().min(xs.len());
    if n == 0 {
        return Ok((vec![], vec![], vec![]));
    }

    let mut betas   = Vec::with_capacity(n);
    let mut p_vars  = Vec::with_capacity(n);
    let mut spreads = Vec::with_capacity(n);

    // Initialise: OLS estimate on first 20 points (or n if shorter)
    let init = n.min(20);
    let sx: f64  = xs[..init].iter().sum();
    let sy: f64  = ys[..init].iter().sum();
    let sxx: f64 = xs[..init].iter().map(|x| x*x).sum();
    let sxy: f64 = xs[..init].iter().zip(ys[..init].iter()).map(|(x,y)| x*y).sum();
    let denom = sxx - sx*sx / init as f64;
    let mut beta = if denom.abs() > 1e-12 {
        (sxy - sx*sy / init as f64) / denom
    } else { 1.0 };
    let mut p = 1.0_f64;  // start with high uncertainty

    for i in 0..n {
        let x = xs[i]; let y = ys[i];
        // Predict
        let p_pred = p + q;
        // Innovation
        let spread = y - beta * x;
        // Innovation variance
        let innov_var = p_pred * x * x + r;
        // Kalman gain
        let gain = if innov_var.abs() < 1e-30 { 0.0 } else { p_pred * x / innov_var };
        // Update
        beta = beta + gain * spread;
        p    = ((1.0 - gain * x) * p_pred).max(1e-12);
        spreads.push(y - beta * x);
        betas.push(beta);
        p_vars.push(p);
    }

    Ok((betas, p_vars, spreads))
}

/// Rolling z-score of a spread series — O(N) using Welford's online algorithm.
///
/// Welford's method maintains a running mean and sum-of-squared-deviations
/// in a sliding window with O(1) work per step (add new, remove oldest).
/// This is ~window× faster than the naive O(N×window) approach.
///
/// Arguments:
///   spreads – raw spread values (length N)
///   window  – lookback window for mean/std (default 60)
///
/// Returns: (zscores, rolling_mean, rolling_std)  — each length N
/// Points before window is full return z=0.
#[pyfunction]
#[pyo3(signature = (spreads, window=60))]
fn pair_zscore(
    _py: Python<'_>,
    spreads: Vec<f64>,
    window: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = spreads.len();
    let w = window.max(2);
    let mut zscores = vec![0.0f64; n];
    let mut means   = vec![0.0f64; n];
    let mut stds    = vec![0.0f64; n];

    // Welford's online mean + M2 (sum of squared deviations from mean).
    // For a sliding window of size w we add the new element and remove the
    // oldest using the standard incremental update formulae.
    let mut win_mean = 0.0f64;
    let mut win_m2   = 0.0f64;   // Σ(x - mean)²
    let mut count    = 0usize;

    for i in 0..n {
        let x_new = spreads[i];

        if count < w {
            // Window not yet full — standard Welford add
            count += 1;
            let delta  = x_new - win_mean;
            win_mean  += delta / count as f64;
            let delta2 = x_new - win_mean;
            win_m2    += delta * delta2;
        } else {
            // Window full — sliding update: add x_new, remove x_old
            let x_old  = spreads[i - w];
            let old_mean = win_mean;
            win_mean  += (x_new - x_old) / w as f64;
            // M2 update: Chan's parallel algorithm for sliding window
            win_m2    += (x_new - x_old) * (x_new - win_mean + x_old - old_mean);
            // Numerical stability floor
            if win_m2 < 0.0 { win_m2 = 0.0; }
        }

        let var = win_m2 / count as f64;
        let std = var.sqrt();
        means[i] = win_mean;
        stds[i]  = std;

        if count >= w && std > 1e-12 {
            zscores[i] = (x_new - win_mean) / std;
        }
    }

    Ok((zscores, means, stds))
}

/// Engle-Granger cointegration test (approximate).
///
/// Fits OLS regression y ~ β·x, then runs approximate ADF on residuals.
/// Returns (beta_ols, adf_stat, p_value_approx).
///
/// p_value_approx uses MacKinnon (1994) response surface for 1% / 5% / 10%:
///   τ_1% ≈ -3.43,  τ_5% ≈ -2.86,  τ_10% ≈ -2.57   (for n→∞)
/// We interpolate linearly.  Not exact but good enough for screening.
#[pyfunction]
fn engle_granger_coint(
    _py: Python<'_>,
    ys: Vec<f64>,
    xs: Vec<f64>,
) -> PyResult<(f64, f64, f64)> {
    let n = ys.len().min(xs.len());
    if n < 10 {
        return Ok((1.0, 0.0, 1.0));
    }

    // OLS: β = Σ(x·y) / Σ(x²)  (no intercept — log-price ratio)
    let sx:  f64 = xs[..n].iter().sum();
    let sy:  f64 = ys[..n].iter().sum();
    let sxx: f64 = xs[..n].iter().map(|v| v*v).sum();
    let sxy: f64 = xs[..n].iter().zip(ys[..n].iter()).map(|(x,y)| x*y).sum();
    let nf = n as f64;
    let denom = sxx - sx*sx/nf;
    let beta  = if denom.abs() > 1e-12 { (sxy - sx*sy/nf) / denom } else { 1.0 };
    let alpha_intercept = (sy - beta*sx) / nf;

    // Residuals
    let resid: Vec<f64> = ys[..n].iter().zip(xs[..n].iter())
        .map(|(&y, &x)| y - beta*x - alpha_intercept)
        .collect();

    // ADF(0): stat = (Σ e_t · Δe_t) / (σ · sqrt(Σ e_t²))
    // where Δe_t = e_t - e_{t-1}
    let mut sum_e2: f64 = 0.0;
    let mut sum_e_de: f64 = 0.0;
    for t in 1..n {
        let e    = resid[t-1];
        let de   = resid[t] - resid[t-1];
        sum_e2  += e * e;
        sum_e_de += e * de;
    }

    // Variance of Δe
    let mean_de = resid.windows(2).map(|w| w[1]-w[0]).sum::<f64>() / (n-1) as f64;
    let var_de  = resid.windows(2)
        .map(|w| { let d = (w[1]-w[0]) - mean_de; d*d })
        .sum::<f64>() / (n-2).max(1) as f64;

    let denom2 = var_de.sqrt() * sum_e2.sqrt();
    let adf_stat = if denom2 > 1e-30 { sum_e_de / denom2 } else { 0.0 };

    // Approximate p-value via MacKinnon response surface (2-tail, no trend)
    // Critical values at n→∞: 1%=-3.43, 5%=-2.86, 10%=-2.57
    // Finite-sample correction: add c1/n + c2/n²
    let tau1 = -3.43 + 3.39 / nf;
    let tau5 = -2.86 + 2.74 / nf;
    let tau10= -2.57 + 1.99 / nf;

    let p_approx = if adf_stat < tau1 { 0.01 }
        else if adf_stat < tau5  { 0.05 }
        else if adf_stat < tau10 { 0.10 }
        else                     { 0.50 };

    Ok((beta, adf_stat, p_approx))
}

// ── Pair backtest simulation loop ────────────────────────────────────────────
//
// Runs all four algorithm strategies (shot, depth_shot, averages, vector)
// over the pre-computed z-score series in a single Rust pass.
//
// This replaces the innermost Python for-loop in pair_backtest.py which is
// the remaining hot path (~18ms per run × 28 pairs = 500ms total).
//
// Each algorithm is a small state machine:
//   shot        — enter when |z| > entry; exit when |z| < exit or timeout
//   depth_shot  — same but gate with volume-ratio confirmation
//   averages    — enter on short-MA vs long-MA spread divergence
//   vector      — enter on z-score velocity burst
//
// Trade P&L = spread_change / |entry_price_a| * 100  minus total_cost_pct
//
// Returns per-algorithm stats as a flat tuple so Python can unpack cheaply
// without building a Dict on the Rust side:
//   (shot_trades, shot_wins, shot_pnl,
//    depth_trades, depth_wins, depth_pnl,
//    avg_trades,   avg_wins,   avg_pnl,
//    vec_trades,   vec_wins,   vec_pnl)

#[allow(clippy::too_many_arguments)]
fn run_algo_sim(
    algo: u8,                   // 0=shot 1=depth_shot 2=averages 3=vector
    zscores:      &[f64],
    spreads:      &[f64],
    prices_a:     &[f64],
    vol_ratio:    &[f64],       // volumes_a[i] / volumes_b[i], pre-computed
    lookback:     usize,
    max_hold:     usize,
    zscore_entry: f64,
    zscore_exit:  f64,
    zscore_stop:  f64,
    total_cost:   f64,          // fee+slippage as fraction (not %)
    avg_short:    usize,
    avg_long:     usize,
    avg_trig_pct: f64,
    vec_vel_bars: usize,
    vec_min_vel:  f64,
) -> (i32, i32, f64, f64, f64) {
    // Returns (total_trades, winning_trades, total_pnl_pct, sharpe_ratio, max_drawdown_pct)
    let n = zscores.len();
    let mut total_trades = 0i32;
    let mut winning      = 0i32;
    let mut total_pnl    = 0.0f64;

    // For Sharpe: online mean + M2 over per-trade P&Ls (Welford)
    let mut pnl_mean = 0.0f64;
    let mut pnl_m2   = 0.0f64;

    // For max drawdown: track cumulative equity curve
    let mut cum_equity = 0.0f64;
    let mut peak       = 0.0f64;
    let mut max_dd     = 0.0f64;

    let mut in_pos      = false;
    let mut pos_side    = 0i8;   // +1 = long spread, -1 = short spread
    let mut entry_bar   = 0usize;
    let mut entry_sp    = 0.0f64;
    let mut entry_pa    = 0.0f64;

    for i in lookback..n {
        let z  = zscores[i];
        let sp = spreads[i];
        let pa = prices_a[i];

        // ── Exit ─────────────────────────────────────────────────────────────
        if in_pos {
            let held = i - entry_bar;
            let mut exit_now = false;

            match algo {
                0 => { // shot
                    if pos_side > 0 && z >= -zscore_exit { exit_now = true; }
                    if pos_side < 0 && z <=  zscore_exit { exit_now = true; }
                    if z.abs() > zscore_stop              { exit_now = true; }
                    if held >= max_hold                   { exit_now = true; }
                }
                1 => { // depth_shot — volume-ratio confirmation on exit
                    let vr = vol_ratio[i];
                    if pos_side > 0 && z >= -zscore_exit && vr < 1.5 { exit_now = true; }
                    if pos_side < 0 && z <=  zscore_exit && vr > 0.7 { exit_now = true; }
                    if z.abs() > zscore_stop { exit_now = true; }
                    if held >= max_hold      { exit_now = true; }
                }
                2 => { // averages — exit when short MA reverts to long MA
                    if i >= avg_long {
                        let s_start = i + 1 - avg_short;
                        let l_start = i + 1 - avg_long;
                        let short_avg: f64 = spreads[s_start..=i].iter().sum::<f64>() / avg_short as f64;
                        let long_avg:  f64 = spreads[l_start..=i].iter().sum::<f64>() / avg_long  as f64;
                        let la_abs = long_avg.abs().max(1e-9);
                        let delta_pct = (short_avg - long_avg) / la_abs * 100.0;
                        let thr = avg_trig_pct * 0.3;
                        if pos_side > 0 && delta_pct >= -thr { exit_now = true; }
                        if pos_side < 0 && delta_pct <=  thr { exit_now = true; }
                    }
                    if held >= max_hold { exit_now = true; }
                }
                _ => { // vector — exit when velocity reverses
                    if i >= vec_vel_bars {
                        let vel = z - zscores[i - vec_vel_bars];
                        if pos_side > 0 && vel > 0.0 { exit_now = true; }
                        if pos_side < 0 && vel < 0.0 { exit_now = true; }
                    }
                    if held >= max_hold { exit_now = true; }
                }
            }

            if exit_now {
                let normaliser = entry_pa.abs().max(1e-6);
                let raw_pnl = if pos_side > 0 {
                    (sp - entry_sp) / normaliser
                } else {
                    (entry_sp - sp) / normaliser
                };
                let net_pct = raw_pnl * 100.0 - total_cost;
                total_pnl    += net_pct;
                total_trades += 1;
                if net_pct > 0.0 { winning += 1; }

                // Welford online Sharpe update
                let k = total_trades as f64;
                let delta  = net_pct - pnl_mean;
                pnl_mean  += delta / k;
                let delta2 = net_pct - pnl_mean;
                pnl_m2    += delta * delta2;

                // Max drawdown update
                cum_equity += net_pct;
                if cum_equity > peak { peak = cum_equity; }
                let dd = if peak > 0.0 { (peak - cum_equity) / peak * 100.0 } else { 0.0 };
                if dd > max_dd { max_dd = dd; }

                in_pos   = false;
                pos_side = 0;
            }
        }

        // ── Entry ─────────────────────────────────────────────────────────────
        if !in_pos {
            let mut entered  = false;
            let mut new_side = 0i8;

            match algo {
                0 => { // shot
                    if z >  zscore_entry { entered = true; new_side = -1; }
                    else if z < -zscore_entry { entered = true; new_side = 1; }
                }
                1 => { // depth_shot — require volume confirmation
                    let vr = vol_ratio[i];
                    if z >  zscore_entry && vr > 1.2 { entered = true; new_side = -1; }
                    else if z < -zscore_entry && vr < 0.8 { entered = true; new_side = 1; }
                }
                2 => { // averages
                    if i >= avg_long {
                        let s_start = i + 1 - avg_short;
                        let l_start = i + 1 - avg_long;
                        let short_avg: f64 = spreads[s_start..=i].iter().sum::<f64>() / avg_short as f64;
                        let long_avg:  f64 = spreads[l_start..=i].iter().sum::<f64>() / avg_long  as f64;
                        let la_abs = long_avg.abs().max(1e-9);
                        let delta_pct = (short_avg - long_avg) / la_abs * 100.0;
                        if delta_pct >  avg_trig_pct && z > 1.0 { entered = true; new_side = -1; }
                        else if delta_pct < -avg_trig_pct && z < -1.0 { entered = true; new_side = 1; }
                    }
                }
                _ => { // vector
                    if i >= vec_vel_bars {
                        let vel = z - zscores[i - vec_vel_bars];
                        if vel >  vec_min_vel && z > 1.0 { entered = true; new_side = -1; }
                        else if vel < -vec_min_vel && z < -1.0 { entered = true; new_side = 1; }
                    }
                }
            }

            if entered {
                in_pos    = true;
                pos_side  = new_side;
                entry_bar = i;
                entry_sp  = sp;
                entry_pa  = pa;
            }
        }
    }

    // Annualised Sharpe from per-trade P&Ls (assuming ~252 trades/year proxy)
    let sharpe = if total_trades > 1 {
        let variance = pnl_m2 / (total_trades as f64 - 1.0);
        let std = variance.sqrt();
        if std > 1e-12 { pnl_mean / std * (252.0f64).sqrt() } else { 0.0 }
    } else { 0.0 };

    (total_trades, winning, total_pnl, sharpe, max_dd)
}

/// Full pair backtest simulation — all 4 algorithms in one Rust pass.
///
/// Pre-requisite: caller has already computed zscores and spreads via
/// kalman_pair_batch() + pair_zscore().  This function only runs the
/// trading simulation loop, which is the hot path (~18ms in Python).
///
/// Arguments:
///   zscores     – z-score series (length N)
///   spreads     – raw spread series (length N)
///   prices_a    – leg-A price series (length N)
///   volumes_a   – leg-A volume series (or empty → uniform 1.0)
///   volumes_b   – leg-B volume series (or empty → uniform 1.0)
///   lookback    – bars before trading starts
///   max_hold    – max bars to hold a position
///   zscore_entry – entry threshold   (default 2.0)
///   zscore_exit  – exit threshold    (default 0.5)
///   zscore_stop  – stop-loss z-score (default 4.0)
///   fee_pct      – one-way fee in %  (default 0.04)
///   slippage_pct – one-way slippage  (default 0.02)
///   avg_short    – averages short window in bars (default 10)
///   avg_long     – averages long window in bars  (default 60)
///   avg_trig_pct – averages trigger deviation %  (default 0.3)
///   vec_vel_bars – vector velocity lookback bars (default 5)
///   vec_min_vel  – vector minimum velocity       (default 0.5)
///
/// Returns 12-tuple:
///   (shot_trades, shot_wins, shot_pnl,
///    depth_trades, depth_wins, depth_pnl,
///    avg_trades,   avg_wins,   avg_pnl,
///    vec_trades,   vec_wins,   vec_pnl)
#[pyfunction]
#[pyo3(signature = (
    zscores, spreads, prices_a,
    volumes_a=None, volumes_b=None,
    lookback=200, max_hold=50,
    zscore_entry=2.0, zscore_exit=0.5, zscore_stop=4.0,
    fee_pct=0.04, slippage_pct=0.02,
    avg_short=10, avg_long=60, avg_trig_pct=0.3,
    vec_vel_bars=5, vec_min_vel=0.5
))]
#[allow(clippy::too_many_arguments)]
fn pair_backtest_run(
    _py: Python<'_>,
    zscores:      Vec<f64>,
    spreads:      Vec<f64>,
    prices_a:     Vec<f64>,
    volumes_a:    Option<Vec<f64>>,
    volumes_b:    Option<Vec<f64>>,
    lookback:     usize,
    max_hold:     usize,
    zscore_entry: f64,
    zscore_exit:  f64,
    zscore_stop:  f64,
    fee_pct:      f64,
    slippage_pct: f64,
    avg_short:    usize,
    avg_long:     usize,
    avg_trig_pct: f64,
    vec_vel_bars: usize,
    vec_min_vel:  f64,
) -> PyResult<Vec<f64>> {
    // Returns flat Vec<f64> of length 20: (trades,wins,pnl,sharpe,mdd) × 4 algos.
    // PyO3 only auto-converts tuples ≤ 12 elements; Vec<f64> has no limit.
    let n = zscores.len().min(spreads.len()).min(prices_a.len());
    // Total round-trip cost in %.
    // Python passes fee_pct=0.04 meaning 4 bps = 0.04%.
    // Python formula: net_pnl = raw_pnl * 100 - (4*(fee+slip)/100)
    //   where the /100 converts the already-percent inputs to fraction.
    // So total_cost here = 4*(fee_pct + slippage_pct) / 100.0
    let total_cost = 4.0 * (fee_pct + slippage_pct) / 100.0;

    // Pre-compute vol_ratio: vol_a / vol_b  (uniform 1.0 if not provided)
    let vol_ratio: Vec<f64> = {
        let va = volumes_a.as_deref().unwrap_or(&[]);
        let vb = volumes_b.as_deref().unwrap_or(&[]);
        (0..n).map(|i| {
            let a = if i < va.len() { va[i] } else { 1.0 };
            let b = if i < vb.len() { vb[i] } else { 1.0 };
            a / b.max(1e-9)
        }).collect()
    };

    let common_args = (
        &zscores[..n], &spreads[..n], &prices_a[..n], &vol_ratio[..],
        lookback, max_hold,
        zscore_entry, zscore_exit, zscore_stop, total_cost,
        avg_short, avg_long, avg_trig_pct,
        vec_vel_bars, vec_min_vel,
    );

    let (st, sw, sp, s_sharpe, s_mdd) = run_algo_sim(0, common_args.0, common_args.1, common_args.2,
        common_args.3, common_args.4, common_args.5, common_args.6, common_args.7,
        common_args.8, common_args.9, common_args.10, common_args.11, common_args.12,
        common_args.13, common_args.14);
    let (dt, dw, dp, d_sharpe, d_mdd) = run_algo_sim(1, common_args.0, common_args.1, common_args.2,
        common_args.3, common_args.4, common_args.5, common_args.6, common_args.7,
        common_args.8, common_args.9, common_args.10, common_args.11, common_args.12,
        common_args.13, common_args.14);
    let (at, aw, ap, a_sharpe, a_mdd) = run_algo_sim(2, common_args.0, common_args.1, common_args.2,
        common_args.3, common_args.4, common_args.5, common_args.6, common_args.7,
        common_args.8, common_args.9, common_args.10, common_args.11, common_args.12,
        common_args.13, common_args.14);
    let (vt, vw, vp, v_sharpe, v_mdd) = run_algo_sim(3, common_args.0, common_args.1, common_args.2,
        common_args.3, common_args.4, common_args.5, common_args.6, common_args.7,
        common_args.8, common_args.9, common_args.10, common_args.11, common_args.12,
        common_args.13, common_args.14);

    // Returns flat Vec<f64> of 20 values: (trades,wins,pnl,sharpe,mdd) × 4
    Ok(vec![
        st as f64, sw as f64, sp, s_sharpe, s_mdd,
        dt as f64, dw as f64, dp, d_sharpe, d_mdd,
        at as f64, aw as f64, ap, a_sharpe, a_mdd,
        vt as f64, vw as f64, vp, v_sharpe, v_mdd,
    ])
}

#[pymodule]
fn ricci_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wasserstein_1, m)?)?;
    m.add_function(wrap_pyfunction!(ollivier_ricci, m)?)?;
    m.add_function(wrap_pyfunction!(all_edge_curvatures, m)?)?;
    m.add_function(wrap_pyfunction!(ricci_flow, m)?)?;
    m.add_function(wrap_pyfunction!(hmm_regime_fit, m)?)?;
    m.add_function(wrap_pyfunction!(hmm_vol_fit, m)?)?;
    // Pair arbitrage
    m.add_function(wrap_pyfunction!(kalman_pair_update, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_pair_batch, m)?)?;
    m.add_function(wrap_pyfunction!(pair_zscore, m)?)?;
    m.add_function(wrap_pyfunction!(engle_granger_coint, m)?)?;
    m.add_function(wrap_pyfunction!(pair_backtest_run, m)?)?;
    Ok(())
}
