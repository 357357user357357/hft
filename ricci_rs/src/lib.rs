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

/// Rolling z-score of a spread series.
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

    for i in 0..n {
        let start = if i + 1 >= w { i + 1 - w } else { 0 };
        let slice = &spreads[start..=i];
        let len = slice.len() as f64;
        let mean = slice.iter().sum::<f64>() / len;
        let var  = slice.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / len;
        let std  = var.sqrt();
        means[i] = mean;
        stds[i]  = std;
        if i + 1 >= w && std > 1e-12 {
            zscores[i] = (spreads[i] - mean) / std;
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
    Ok(())
}
