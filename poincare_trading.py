"""Poincaré geometry for trading — Perelman's tools, Poincaré's criterion.

The Poincaré conjecture (Perelman 2003): every simply-connected closed
3-manifold is homeomorphic to S³.

Conceptual distinction (important):
  - The **conjecture** is the *criterion* (the destination): a manifold that
    satisfies (simply-connected + closed + 3-dimensional) must be S³.
  - **Perelman's Ricci flow** is the *tool* (the compass): it evolves the
    metric toward round-sphere geometry, revealing whether the manifold
    converges to S³ or diverges toward a hyperbolic space.
  - **Persistent homology** (Whitehead / algebraic topology) tests
    simply-connectedness via Betti numbers β₁, β₂.

Trading interpretation:
  The Poincaré condition (S³) ↔ mean-reversion manifold:
    - All price trajectories eventually return (compactness).
    - No persistent directional loops (β₁ = 0, simply connected).
    - Ricci flow converges → positive curvature everywhere → sphere.

  Hyperbolic manifold (non-S³) ↔ trending regime:
    - Non-contractible loops exist (β₁ > 0).
    - Negative Ricci curvature → trajectories diverge.

Implementation layers:
1. **Delay-embedding** (Takens): scalar prices → ℝ^d point cloud.
   Adaptive lag via autocorrelation zero-crossing.
2. **Ollivier–Ricci curvature + Ricci flow** (Perelman's tool):
   Wasserstein-1 via Kantorovich LP on k-NN graph.
   Flow convergence → approaching S³ fixed point.
3. **Persistent homology** via gudhi (fast C++ Vietoris–Rips):
   Betti numbers β₀/β₁/β₂. Falls back to pure Python if gudhi absent.
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

# Fast C++ persistent homology via gudhi (MIT licence).
# Falls back to the pure-Python VR implementation when absent.
try:
    import gudhi as _gudhi
    _GUDHI_AVAILABLE = True
except ImportError:
    _GUDHI_AVAILABLE = False

# ── Rust-accelerated LP solver (ricci_rs) ────────────────────────────────────
# ricci_rs.all_edge_curvatures(graph, D) computes the full Ollivier-Ricci
# curvature for every edge in a single Rust call (~8-12x faster than Python).
# ricci_rs.ricci_flow(graph, D, steps, dt) runs the full flow in Rust.
# Falls back to pure Python if the extension is not installed.
try:
    import ricci_rs as _ricci_rs
    # maturin may install as a package with __init__.py re-exporting the .so,
    # or the .so may be the top-level module — handle both layouts.
    if not hasattr(_ricci_rs, 'all_edge_curvatures'):
        _ricci_rs = _ricci_rs.ricci_rs  # unwrap nested module
    _RUST_AVAILABLE = True
except Exception:
    _RUST_AVAILABLE = False

# ── Vectorised distance backend (GPU → NumPy fallback) ───────────────────────
# When HFT_USE_GPU=1, try CuPy for O(n²) matrix ops.  Both paths use the
# same vectorised formula:  D[i,j] = ||P[i] - P[j]||₂
#   X = (n,d) matrix;  ||X[i]-X[j]||² = ||X[i]||² + ||X[j]||² - 2·X[i]·X[j]
# This is ~100× faster than the nested Python loop for n=80 landmarks.
import numpy as _np

_xp = _np  # default: NumPy
if os.environ.get("HFT_USE_GPU"):
    try:
        import cupy as _cupy
        _probe = _cupy.array([1.0, 2.0])
        _ = (_probe + _probe).get()
        _xp = _cupy
    except Exception:
        pass  # silently fall back to NumPy


# ─── 1. Delay Embedding (Takens) ─────────────────────────────────────────────

def _autocorrelation_lag(series: List[float], lag: int) -> float:
    """Lag-k autocorrelation of a series."""
    n = len(series)
    if n <= lag:
        return 0.0
    mu = sum(series) / n
    var = sum((x - mu) ** 2 for x in series) / n
    if var < 1e-12:
        return 0.0
    cov = sum((series[i] - mu) * (series[i - lag] - mu)
              for i in range(lag, n)) / n
    return cov / var


def optimal_lag(prices: List[float], max_lag: int = 50) -> int:
    """
    Estimate optimal Takens embedding lag via first zero-crossing of
    autocorrelation.  If no zero-crossing found, use first minimum.

    This ensures delay vectors span independent directions of the attractor,
    giving the best manifold reconstruction.
    """
    n = len(prices)
    if n < 4:
        return 1
    # Log returns for stationarity
    rets = [math.log(prices[i] / prices[i - 1])
            for i in range(1, n)
            if prices[i - 1] > 0 and prices[i] > 0]
    if len(rets) < 4:
        return 1

    max_lag = min(max_lag, len(rets) // 3)
    prev_ac = _autocorrelation_lag(rets, 1)
    best_lag = 1
    min_abs_ac = abs(prev_ac)

    for lag in range(2, max_lag + 1):
        ac = _autocorrelation_lag(rets, lag)
        # Zero-crossing: sign change
        if prev_ac > 0 and ac <= 0:
            return lag
        if prev_ac < 0 and ac >= 0:
            return lag
        # Track minimum absolute autocorrelation as fallback
        if abs(ac) < min_abs_ac:
            min_abs_ac = abs(ac)
            best_lag = lag
        prev_ac = ac

    return max(best_lag, 1)


def delay_embed(prices: List[float], dim: int = 3,
                lag: int = -1) -> List[Tuple[float, ...]]:
    """
    Embed a scalar price series into R^dim via Takens delay embedding.

        x_t = (p_t, p_{t-lag}, p_{t-2*lag}, ..., p_{t-(dim-1)*lag})

    If lag=-1 (default), automatically estimate optimal lag from
    autocorrelation zero-crossing.

    Returns list of dim-dimensional points.
    """
    if lag < 0:
        lag = optimal_lag(prices)
    n = len(prices)
    skip = (dim - 1) * lag
    points = []
    for t in range(skip, n):
        pt = tuple(prices[t - k * lag] for k in range(dim))
        points.append(pt)
    return points


def normalise_points(points: List[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
    """Centre and scale to unit bounding box."""
    if not points:
        return points
    d = len(points[0])
    mins = [min(p[i] for p in points) for i in range(d)]
    maxs = [max(p[i] for p in points) for i in range(d)]
    ranges = [max(maxs[i] - mins[i], 1e-10) for i in range(d)]
    return [tuple((p[i] - mins[i]) / ranges[i] for i in range(d)) for p in points]


def farthest_point_sampling(points: List[Tuple[float, ...]],
                            n_landmarks: int) -> List[Tuple[float, ...]]:
    """
    Farthest-point (greedy) landmark sampling — vectorised via NumPy/CuPy.

    Selects n_landmarks points that maximise minimum inter-point distance,
    preserving geometric coverage far better than uniform stride subsampling.
    """
    if len(points) <= n_landmarks:
        return list(points)
    n = len(points)
    try:
        # Vectorised: compute distances from one point to all others as a vector
        X = _np.array(points, dtype=_np.float64)  # (n, d) — keep on CPU for indexing
        selected: List[int] = [0]
        selected_set: Set[int] = {0}
        # ||X[0] - X[i]||₂ for all i
        diff = X - X[0]
        min_dist = _np.sqrt((diff * diff).sum(axis=1))

        for _ in range(n_landmarks - 1):
            # mask selected points
            min_dist[list(selected_set)] = -1.0
            farthest = int(_np.argmax(min_dist))
            selected.append(farthest)
            selected_set.add(farthest)
            diff2 = X - X[farthest]
            d_new = _np.sqrt((diff2 * diff2).sum(axis=1))
            _np.minimum(min_dist, d_new, out=min_dist)

        return [points[i] for i in selected]
    except Exception:
        # pure-Python fallback
        selected_py: List[int] = [0]
        selected_set_py: Set[int] = {0}
        min_dist_py = [euclidean(points[0], points[i]) for i in range(n)]
        for _ in range(n_landmarks - 1):
            farthest = max((i for i in range(n) if i not in selected_set_py),
                           key=lambda i: min_dist_py[i])
            selected_py.append(farthest)
            selected_set_py.add(farthest)
            for i in range(n):
                d = euclidean(points[farthest], points[i])
                if d < min_dist_py[i]:
                    min_dist_py[i] = d
        return [points[i] for i in selected_py]


# ─── 2. Distance utilities ────────────────────────────────────────────────────

def euclidean(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def pairwise_distances(points: List[Tuple[float, ...]]) -> List[List[float]]:
    """Vectorised pairwise Euclidean distances. Uses CuPy if GPU available."""
    n = len(points)
    try:
        X = _xp.array(points, dtype=_xp.float64)      # (n, d)
        sq = (_xp.einsum("ij,ij->i", X, X))           # (n,) row squared norms
        # ||X[i]-X[j]||² = sq[i] + sq[j] - 2·X[i]·X[j]
        cross = X @ X.T                                # (n, n)
        D2 = sq[:, None] + sq[None, :] - 2.0 * cross
        # clamp small negatives from float arithmetic
        D2 = _xp.maximum(D2, 0.0)
        D_mat = _xp.sqrt(D2)
        if _xp is not _np:
            D_mat = D_mat.get()                        # GPU → CPU
        else:
            D_mat = _np.asarray(D_mat)
        return D_mat.tolist()
    except Exception:
        # pure-Python fallback
        D = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = euclidean(points[i], points[j])
                D[i][j] = d
                D[j][i] = d
        return D


# ─── 3. Ollivier–Ricci Curvature ─────────────────────────────────────────────

def knn_graph(D: List[List[float]], k: int) -> Dict[int, List[int]]:
    """Build k-nearest-neighbour graph from distance matrix."""
    n = len(D)
    graph: Dict[int, List[int]] = {}
    for i in range(n):
        neighbours = sorted(range(n), key=lambda j: D[i][j])
        graph[i] = [j for j in neighbours[1:k+1]]  # exclude self
    return graph


def wasserstein_1_lp(mu: List[Tuple[int, float]],
                     nu: List[Tuple[int, float]],
                     D: List[List[float]]) -> float:
    """
    Exact Wasserstein-1 (earth-mover) distance via Kantorovich LP.

    Solves the optimal transport problem:
        min  Σ_{i,j} c_{ij} · f_{ij}
        s.t. Σ_j f_{ij} = mu_i     (supply)
             Σ_i f_{ij} = nu_j     (demand)
             f_{ij} >= 0

    Uses the *network simplex* method for small supports (k+1 nodes each),
    which is exact and fast for the ~6×6 problems we encounter.

    For supports of size m, n, this is O(m·n·(m+n)) — perfectly fine for
    the Ollivier-Ricci setup where m = n = k+1 ≈ 6.
    """
    nodes_i = [nd for nd, _ in mu]
    nodes_j = [nd for nd, _ in nu]
    weights_i = [w for _, w in mu]
    weights_j = [w for _, w in nu]
    m = len(nodes_i)
    n = len(nodes_j)

    # Balance supply and demand (required for feasibility)
    total_supply = sum(weights_i)
    total_demand = sum(weights_j)
    if abs(total_supply - total_demand) > 1e-10:
        # Normalise to equal mass
        if total_supply > 1e-15:
            weights_i = [w / total_supply for w in weights_i]
        if total_demand > 1e-15:
            weights_j = [w / total_demand for w in weights_j]

    # Cost matrix
    C = [[D[nodes_i[a]][nodes_j[b]] for b in range(n)] for a in range(m)]

    # Solve via north-west corner + MODI (modified distribution).
    # --- North-West Corner initial BFS ---
    flow = [[0.0] * n for _ in range(m)]
    supply = list(weights_i)
    demand = list(weights_j)
    ai, bj = 0, 0
    while ai < m and bj < n:
        amount = min(supply[ai], demand[bj])
        flow[ai][bj] = amount
        supply[ai] -= amount
        demand[bj] -= amount
        if supply[ai] < 1e-14:
            ai += 1
        if demand[bj] < 1e-14:
            bj += 1

    # --- Stepping stone improvement (up to 20 iterations) ---
    INF_DUAL = 1e30
    for _ in range(20):
        # Compute dual variables u, v via: u_i + v_j = c_ij for basic (flow>0) cells
        u: List[float] = [INF_DUAL] * m
        v: List[float] = [INF_DUAL] * n
        u[0] = 0.0
        changed = True
        while changed:
            changed = False
            for a in range(m):
                for b in range(n):
                    if flow[a][b] > 1e-14:
                        if u[a] < INF_DUAL and v[b] >= INF_DUAL:
                            v[b] = C[a][b] - u[a]
                            changed = True
                        elif v[b] < INF_DUAL and u[a] >= INF_DUAL:
                            u[a] = C[a][b] - v[b]
                            changed = True
        # Fill any unset duals with 0
        for a in range(m):
            if u[a] >= INF_DUAL:
                u[a] = 0.0
        for b in range(n):
            if v[b] >= INF_DUAL:
                v[b] = 0.0

        # Find most negative reduced cost
        best_rc = -1e-10
        best_cell: Optional[Tuple[int, int]] = None
        for a in range(m):
            for b in range(n):
                rc = C[a][b] - u[a] - v[b]
                if rc < best_rc:
                    best_rc = rc
                    best_cell = (a, b)

        if best_cell is None:
            break  # Optimal

        # Find stepping stone loop and pivot
        ea, eb = best_cell
        min_shift = float('inf')
        pivot_row_b = -1
        pivot_col_a = -1
        for b in range(n):
            if b != eb and flow[ea][b] > 1e-14:
                for a in range(m):
                    if a != ea and flow[a][eb] > 1e-14:
                        shift = min(flow[ea][b], flow[a][eb])
                        if shift < min_shift:
                            min_shift = shift
                            pivot_row_b = b
                            pivot_col_a = a

        if min_shift <= 1e-14 or min_shift == float('inf') or pivot_row_b < 0:
            break

        # Apply pivot
        flow[ea][eb] += min_shift
        flow[ea][pivot_row_b] -= min_shift
        flow[pivot_col_a][eb] -= min_shift
        flow[pivot_col_a][pivot_row_b] += min_shift

    # Compute total cost
    cost = sum(flow[a][b] * C[a][b] for a in range(m) for b in range(n))
    return cost


def ollivier_ricci_curvature(i: int, j: int,
                              graph: Dict[int, List[int]],
                              D: List[List[float]],
                              alpha: float = 0.5) -> float:
    """
    Ollivier–Ricci curvature κ(i,j) for edge (i,j):

        κ(i,j) = 1 − W₁(mᵢ, mⱼ) / d(i,j)

    where mₓ is the α-lazy random walk measure at x:
        mₓ(x) = α
        mₓ(y) = (1-α)/deg(x)  for y ~ x

    Positive κ → positive curvature (sphere-like, mean-reverting).
    Negative κ → negative curvature (hyperbolic-like, trending).
    """
    dij = D[i][j]
    if dij < 1e-12:
        return 0.0

    def lazy_measure(node: int) -> List[Tuple[int, float]]:
        nbrs = graph[node]
        deg = len(nbrs)
        if deg == 0:
            return [(node, 1.0)]
        w_nbr = (1.0 - alpha) / deg
        m = [(node, alpha)] + [(nb, w_nbr) for nb in nbrs]
        return m

    mu = lazy_measure(i)
    nu = lazy_measure(j)

    w1 = wasserstein_1_lp(mu, nu, D)
    return 1.0 - w1 / dij


def compute_all_edge_curvatures(graph: Dict[int, List[int]],
                                D: List[List[float]]
                                ) -> Tuple[List[float], Dict[Tuple[int, int], float]]:
    """
    Compute Ollivier-Ricci curvature for every undirected edge.
    Uses Rust extension when available (~8-12x faster than Python).
    Returns (flat list of kappas, dict mapping (i,j)->kappa).
    """
    if _RUST_AVAILABLE:
        kappas, edge_kappa = _ricci_rs.all_edge_curvatures(graph, D)
        return kappas, edge_kappa

    # Pure-Python fallback
    kappas: List[float] = []
    edge_kappa: Dict[Tuple[int, int], float] = {}
    n = len(graph)
    visited: set = set()
    for i in range(n):
        for j in graph[i]:
            if (j, i) not in visited:
                visited.add((i, j))
                k = ollivier_ricci_curvature(i, j, graph, D)
                kappas.append(k)
                edge_kappa[(i, j)] = k
                edge_kappa[(j, i)] = k
    return kappas, edge_kappa


def mean_ricci_curvature(graph: Dict[int, List[int]],
                         D: List[List[float]]) -> Tuple[float, float, float]:
    """
    Compute mean, std and negative-edge-fraction of Ollivier–Ricci curvature.
    Returns (mean_kappa, std_kappa, neg_fraction).
    """
    kappas, _ = compute_all_edge_curvatures(graph, D)
    if not kappas:
        return 0.0, 0.0, 0.0
    mean = sum(kappas) / len(kappas)
    var = sum((k - mean) ** 2 for k in kappas) / len(kappas)
    neg_frac = sum(1 for k in kappas if k < 0) / len(kappas)
    return mean, math.sqrt(var), neg_frac


def scalar_curvature(graph: Dict[int, List[int]],
                     edge_kappa: Dict[Tuple[int, int], float]
                     ) -> Tuple[List[float], float, float]:
    """
    Vertex-level scalar curvature: S(v) = Σ_{u~v} κ(v,u).

    On a Riemannian manifold, scalar curvature is the trace of the Ricci
    tensor.  On a graph, the natural analogue sums Ollivier-Ricci curvature
    over all edges incident to each vertex.

    Hyperbolic manifolds have S(v) < 0 at almost every vertex.
    Spheres have S(v) > 0 everywhere.

    Returns (list of per-vertex S(v), mean_scalar, neg_vertex_fraction).
    """
    n = len(graph)
    scalars: List[float] = []
    for i in range(n):
        sv = sum(edge_kappa.get((i, j), 0.0) for j in graph[i])
        scalars.append(sv)
    if not scalars:
        return [], 0.0, 0.0
    mean_s = sum(scalars) / len(scalars)
    neg_vert_frac = sum(1 for s in scalars if s < 0) / len(scalars)
    return scalars, mean_s, neg_vert_frac


# ─── 4. Discrete Ricci Flow ───────────────────────────────────────────────────

def ricci_flow_step(graph: Dict[int, List[int]],
                    D: List[List[float]],
                    dt: float = 0.1,
                    normalise: bool = True) -> List[List[float]]:
    """
    One step of discrete Ricci flow with optional volume normalisation:

        d/dt d(i,j) = −κ(i,j) · d(i,j)

    After the update, if normalise=True, rescale all distances so that the
    total edge weight (sum of all graph-edge distances) is preserved.
    This is analogous to Perelman's normalised Ricci flow, which prevents
    the metric from collapsing to a point or expanding to infinity and
    ensures convergence toward the round sphere S³.

    Returns updated distance matrix.
    """
    n = len(D)
    D_new = [row[:] for row in D]
    edge_pairs: List[Tuple[int, int]] = []
    visited = set()
    for i in range(n):
        for j in graph[i]:
            if (j, i) not in visited:
                visited.add((i, j))
                edge_pairs.append((i, j))
                kappa = ollivier_ricci_curvature(i, j, graph, D)
                delta = -kappa * D[i][j] * dt
                new_d = max(D[i][j] + delta, 1e-6)
                D_new[i][j] = new_d
                D_new[j][i] = new_d

    # Volume normalisation: preserve total edge weight
    if normalise and edge_pairs:
        total_old = sum(D[i][j] for i, j in edge_pairs)
        total_new = sum(D_new[i][j] for i, j in edge_pairs)
        if total_new > 1e-12 and total_old > 1e-12:
            scale = total_old / total_new
            for i, j in edge_pairs:
                D_new[i][j] *= scale
                D_new[j][i] *= scale

    return D_new


def run_ricci_flow(graph: Dict[int, List[int]],
                   D: List[List[float]],
                   steps: int = 5,
                   dt: float = 0.05,
                   _alpha: float = 0.5) -> Tuple[List[List[float]], List[float]]:
    """
    Run `steps` steps of discrete Ricci flow.
    Uses Rust extension when available (all steps in one call, no Python overhead).
    Returns (final distance matrix, history of mean curvature).
    """
    if _RUST_AVAILABLE:
        return _ricci_rs.ricci_flow(graph, D, steps, dt)

    # Pure-Python fallback
    history = []
    for _ in range(steps):
        mean_k, _, _ = mean_ricci_curvature(graph, D)
        history.append(mean_k)
        D = ricci_flow_step(graph, D, dt)
    return D, history


# ─── 5. Persistent Homology (Vietoris–Rips, β₀ and β₁) ──────────────────────

class UnionFind:
    """Union-Find for β₀ (connected components)."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.count -= 1
        return True


@dataclass
class PersistencePair:
    """A persistence pair (birth, death) for a topological feature."""
    dim: int          # 0 = component, 1 = loop, 2 = void
    birth: float      # filtration value when feature appears
    death: float      # filtration value when feature disappears (inf = still alive)

    @property
    def lifetime(self) -> float:
        if math.isinf(self.death):
            return math.inf
        return self.death - self.birth

    @property
    def is_essential(self) -> bool:
        """Feature with infinite lifetime (never dies)."""
        return math.isinf(self.death)


def vietoris_rips_persistence(points: List[Tuple[float, ...]],
                               max_edge: float = 0.5,
                               num_steps: int = 50
                               ) -> List[PersistencePair]:
    """
    Compute persistent homology of the Vietoris-Rips filtration up to dim 2
    (components beta_0, loops beta_1, voids beta_2).

    Algorithm:
      - Sort all pairwise edges by length.
      - Sweep from 0 to max_edge; add edges and track simplices.
      - beta_0 via Union-Find.
      - beta_1 via cycle detection + triangle filling.
      - beta_2 via tetrahedron filling: when a tetrahedron (4 vertices, all
        6 edges, all 4 triangles present) completes, it may kill a void
        (beta_2 feature) or birth one when its boundary encloses empty space.
    """
    n = len(points)
    if n < 2:
        return []

    D = pairwise_distances(points)

    # All edges sorted by length
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if D[i][j] <= max_edge:
                edges.append((D[i][j], i, j))
    edges.sort()

    pairs: List[PersistencePair] = []
    uf = UnionFind(n)

    component_birth = [0.0] * n

    for d, i, j in edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri != rj:
            bi = component_birth[ri]
            bj = component_birth[rj]
            if bi <= bj:
                pairs.append(PersistencePair(dim=0, birth=bj, death=d))
                uf.union(i, j)
                new_r = uf.find(i)
                component_birth[new_r] = bi
            else:
                pairs.append(PersistencePair(dim=0, birth=bi, death=d))
                uf.union(i, j)
                new_r = uf.find(i)
                component_birth[new_r] = bj
        else:
            pairs.append(PersistencePair(dim=1, birth=d, death=math.inf))

    # Essential components
    surviving = set(uf.find(i) for i in range(n))
    for r in surviving:
        pairs.append(PersistencePair(dim=0, birth=component_birth[r], death=math.inf))

    # --- Triangles: close beta_1 loops ---
    edge_set = {(min(i, j), max(i, j)): d for d, i, j in edges}
    triangles: List[Tuple[float, int, int, int]] = []
    triangle_set: set = set()  # track which triangles exist for beta_2

    for idx_i in range(n):
        for idx_j in range(idx_i + 1, n):
            eij = edge_set.get((idx_i, idx_j))
            if eij is None:
                continue
            for idx_k in range(idx_j + 1, n):
                eik = edge_set.get((min(idx_i, idx_k), max(idx_i, idx_k)))
                ejk = edge_set.get((min(idx_j, idx_k), max(idx_j, idx_k)))
                if eik is not None and ejk is not None:
                    tri_scale = max(eij, eik, ejk)
                    triangles.append((tri_scale, idx_i, idx_j, idx_k))
                    triangle_set.add((idx_i, idx_j, idx_k))
    triangles.sort()

    open_loops = [p for p in pairs if p.dim == 1 and math.isinf(p.death)]
    # Standard TDA: kill the youngest loop (most recently born) first.
    # This is the "eldest rule" — the oldest feature survives longest,
    # which gives the canonical persistence pairing.
    open_loops.sort(key=lambda p: p.birth, reverse=True)  # youngest first

    for tri_scale, i, j, k in triangles:
        for lp in open_loops:
            if math.isinf(lp.death) and lp.birth <= tri_scale:
                lp.death = tri_scale
                break

    # --- Tetrahedra: detect beta_2 voids ---
    # A tetrahedron on vertices {a,b,c,d} exists when all 6 edges and all 4
    # triangles are present.  A void (beta_2) is born when the 4 boundary
    # triangles are all present but the tetrahedron interior is "empty";
    # it dies when the tetrahedron fills it in.
    #
    # Approximation: a beta_2 feature is born at the scale of the 5th edge
    # (when the boundary sphere is nearly complete) and dies at the 6th edge
    # (when the tetrahedron closes).

    if n >= 4:
        for a in range(n):
            for b in range(a + 1, n):
                eab = edge_set.get((a, b))
                if eab is None:
                    continue
                for c in range(b + 1, n):
                    eac = edge_set.get((a, c))
                    ebc = edge_set.get((b, c))
                    if eac is None or ebc is None:
                        continue
                    for d_idx in range(c + 1, n):
                        ead = edge_set.get((a, d_idx))
                        ebd = edge_set.get((b, d_idx))
                        ecd = edge_set.get((c, d_idx))
                        if ead is None or ebd is None or ecd is None:
                            continue
                        # All 6 edges present — check all 4 triangles
                        faces = [
                            (a, b, c), (a, b, d_idx),
                            (a, c, d_idx), (b, c, d_idx),
                        ]
                        if all(f in triangle_set for f in faces):
                            edge_lengths = sorted([eab, eac, ebc, ead, ebd, ecd])
                            birth_2 = edge_lengths[4]  # 5th edge: boundary nearly done
                            death_2 = edge_lengths[5]  # 6th edge: tetrahedron fills void
                            if death_2 > birth_2 + 1e-12:
                                pairs.append(PersistencePair(
                                    dim=2, birth=birth_2, death=death_2))

    return pairs


def betti_numbers(pairs: List[PersistencePair],
                  scale: float = math.inf) -> Tuple[int, int, int]:
    """
    Betti numbers β₀, β₁, β₂ at a given filtration scale.

    A feature is alive at `scale` if birth ≤ scale < death.
    """
    b0 = sum(1 for p in pairs if p.dim == 0 and p.birth <= scale
             and (math.isinf(p.death) or p.death > scale))
    b1 = sum(1 for p in pairs if p.dim == 1 and p.birth <= scale
             and (math.isinf(p.death) or p.death > scale))
    b2 = sum(1 for p in pairs if p.dim == 2 and p.birth <= scale
             and (math.isinf(p.death) or p.death > scale))
    return b0, b1, b2


def total_persistence(pairs: List[PersistencePair], dim: int = 1) -> float:
    """
    Total persistence of dim-k features:  Σ (death - birth)  (finite pairs only).
    Measures the total "topological energy" of loops/holes.
    """
    return sum(p.lifetime for p in pairs
               if p.dim == dim and not math.isinf(p.lifetime))


# ─── 6. Topology Report ──────────────────────────────────────────────────────

@dataclass
class TopologyReport:
    """
    Full Poincaré-geometry analysis of a price window.

    Interpretation:
      regime = "mean-reversion"  ->  S3-like manifold (positive Ricci, simply connected)
      regime = "trending"        ->  hyperbolic-like (negative curvature, non-trivial topology)
      regime = "neutral"         ->  indeterminate

    Key discriminators (from Poincaré/Perelman):
      - mean_ricci > 0: positive curvature -> sphere-like -> mean-reversion
      - neg_ricci_frac > 0.4: >40% edges with negative curvature -> hyperbolic -> trending
      - mean_scalar < 0: negative scalar curvature everywhere -> hyperbolic
      - neg_scalar_frac > 0.5: most vertices have S(v) < 0 -> trending
      - ricci_flow_convergence: small = already at S3 fixed point
    """
    # Embedding
    num_points: int
    embed_dim: int
    embed_lag: int             # adaptive lag used

    # Ricci curvature (edge-level)
    mean_ricci: float          # positive = sphere-like, negative = hyperbolic
    std_ricci: float           # high = mixed curvature (oscillation/cycles)
    neg_ricci_frac: float      # fraction of edges with kappa < 0; > 0.4 -> trending
    ricci_flow_convergence: float  # |kappa_final - kappa_initial| / |kappa_initial|

    # Scalar curvature (vertex-level, trace of Ricci)
    mean_scalar: float         # positive = sphere, negative = hyperbolic
    neg_scalar_frac: float     # fraction of vertices with S(v) < 0

    # Betti numbers (at max filtration scale)
    beta0: int                 # connected components (1 = single manifold)
    beta1: int                 # independent loops (0 = simply connected)
    beta2: int                 # enclosed voids / bubbles

    # Persistence
    loop_persistence: float    # total persistence of beta_1 features
    void_persistence: float    # total persistence of beta_2 features

    # Derived geometry signal
    simply_connected: bool     # pi_1 ~ 0 (Poincaré: if True AND Ricci > 0 -> S3)
    poincare_score: float      # in [-1, +1]; +1 = S3 (mean reversion), -1 = hyperbolic (trend)
    regime: str                # "mean-reversion" | "trending" | "neutral"

    def __repr__(self) -> str:
        poincare_verdict = (
            "S3-like (Poincaré)" if self.simply_connected and self.mean_ricci > 0
            else "hyperbolic" if self.neg_ricci_frac > 0.4
            else "non-trivial topology"
        )
        lines = [
            f"TopologyReport: regime={self.regime}  score={self.poincare_score:+.3f}  lag={self.embed_lag}",
            f"  Ricci: kappa={self.mean_ricci:+.4f} +/- {self.std_ricci:.4f}  "
            f"neg_frac={self.neg_ricci_frac:.2f}  convergence={self.ricci_flow_convergence:.4f}",
            f"  Scalar: S={self.mean_scalar:+.4f}  neg_vert_frac={self.neg_scalar_frac:.2f}",
            f"  Betti: b0={self.beta0}  b1={self.beta1}  b2={self.beta2}  "
            f"loop_pers={self.loop_persistence:.4f}  void_pers={self.void_persistence:.4f}",
            f"  Simply connected: {self.simply_connected}  -> {poincare_verdict}",
        ]
        return "\n".join(lines)


def poincare_analysis(
    prices: List[float],
    embed_dim: int = 3,
    embed_lag: int = -1,
    knn: int = 5,
    ricci_flow_steps: int = 3,
    max_edge_fraction: float = 0.4,
    subsample: int = 80,
) -> TopologyReport:
    """
    Full Poincaré-conjecture geometric analysis of a price window.

    Steps:
      1. Delay-embed prices into R^embed_dim (Takens) with adaptive lag.
      2. Normalise and subsample via farthest-point landmarks.
      3. Build k-NN graph.
      4. Compute Ollivier-Ricci curvature (proper Kantorovich W1).
      5. Compute scalar curvature (vertex-level trace of Ricci).
      6. Run discrete Ricci flow.
      7. Compute persistent homology (Vietoris-Rips, beta_0/1/2).
      8. Score and classify regime.

    Returns TopologyReport.
    """
    _empty = lambda lag_val: TopologyReport(
        num_points=0, embed_dim=embed_dim, embed_lag=lag_val,
        mean_ricci=0.0, std_ricci=0.0, neg_ricci_frac=0.5,
        ricci_flow_convergence=0.0,
        mean_scalar=0.0, neg_scalar_frac=0.5,
        beta0=1, beta1=0, beta2=0,
        loop_persistence=0.0, void_persistence=0.0,
        simply_connected=True, poincare_score=0.0, regime="neutral",
    )

    # Step 1: embed with adaptive lag
    used_lag = embed_lag if embed_lag > 0 else optimal_lag(prices)
    points = delay_embed(prices, dim=embed_dim, lag=used_lag)
    if len(points) < knn + 2:
        return _empty(used_lag)

    points = normalise_points(points)

    # Step 2: farthest-point landmark subsampling (preserves geometry)
    if len(points) > subsample:
        points = farthest_point_sampling(points, subsample)

    n = len(points)

    # Step 3: pairwise distances and k-NN graph
    D = pairwise_distances(points)
    graph = knn_graph(D, k=min(knn, n - 1))

    # Step 4: Ricci curvature (proper Kantorovich LP for W1)
    kappas_init, edge_kappa_init = compute_all_edge_curvatures(graph, D)
    if not kappas_init:
        return _empty(used_lag)
    kappa_init = sum(kappas_init) / len(kappas_init)
    neg_frac_init = sum(1 for k in kappas_init if k < 0) / len(kappas_init)

    # Step 5: scalar curvature (vertex-level)
    _, mean_scalar_init, neg_scalar_frac_init = scalar_curvature(graph, edge_kappa_init)

    # Step 6: Ricci flow
    D_flowed, kappa_history = run_ricci_flow(graph, D, steps=ricci_flow_steps, dt=0.05)
    kappas_final, edge_kappa_final = compute_all_edge_curvatures(graph, D_flowed)
    kappa_final = sum(kappas_final) / len(kappas_final) if kappas_final else 0.0
    kappa_std_final = math.sqrt(
        sum((k - kappa_final) ** 2 for k in kappas_final) / len(kappas_final)
    ) if kappas_final else 0.0
    neg_frac_final = (sum(1 for k in kappas_final if k < 0) / len(kappas_final)
                      if kappas_final else 0.5)
    _, mean_scalar_final, neg_scalar_frac_final = scalar_curvature(graph, edge_kappa_final)

    # Relative convergence with absolute floor to avoid div-by-near-zero
    denom = max(abs(kappa_init), abs(kappa_final), 0.01)
    convergence = abs(kappa_final - kappa_init) / denom

    # Step 7: persistent homology (beta_0, beta_1, beta_2)
    max_d = max(D[i][j] for i in range(n) for j in range(i + 1, n))
    max_edge = max_edge_fraction * max_d

    if _GUDHI_AVAILABLE:
        # Fast C++ Vietoris-Rips via gudhi (orders of magnitude faster than
        # the pure-Python implementation for n > 30).
        rc = _gudhi.RipsComplex(points=[list(p) for p in points],
                                max_edge_length=max_edge)
        st = rc.create_simplex_tree(max_dimension=3)
        st.compute_persistence()
        persistence = st.persistence()          # [(dim, (birth, death)), ...]
        betti = st.betti_numbers()              # [β₀, β₁, β₂, ...]
        b0 = betti[0] if len(betti) > 0 else 1
        b1 = betti[1] if len(betti) > 1 else 0
        b2 = betti[2] if len(betti) > 2 else 0
        loop_pers = sum(d - b for dim, (b, d) in persistence
                        if dim == 1 and not math.isinf(d))
        void_pers = sum(d - b for dim, (b, d) in persistence
                        if dim == 2 and not math.isinf(d))
        significant_loops = sum(1 for dim, (b, d) in persistence
                                if dim == 1 and not math.isinf(d) and d - b > 0.02)
        inf_loops = sum(1 for dim, (b, d) in persistence
                        if dim == 1 and math.isinf(d))
    else:
        # Pure-Python fallback (slow for large n, kept for zero-dependency use).
        pers_pairs = vietoris_rips_persistence(points, max_edge=max_edge)
        b0, b1, b2 = betti_numbers(pers_pairs, scale=max_edge)
        loop_pers = total_persistence(pers_pairs, dim=1)
        void_pers = total_persistence(pers_pairs, dim=2)
        significant_loops = sum(
            1 for p in pers_pairs
            if p.dim == 1 and not math.isinf(p.death) and p.lifetime > 0.02
        )
        inf_loops = sum(1 for p in pers_pairs if p.dim == 1 and math.isinf(p.death))
    simply_connected = (significant_loops == 0) and (inf_loops == 0)

    # Poincaré score: +1 = S3 (mean reversion), -1 = hyperbolic (trending)
    #
    # Four components from Perelman's Ricci flow geometry:
    #
    # 1. ricci_component: mean Ollivier-Ricci curvature after flow
    #    Scaled adaptively by curvature std to produce z-score-like values.
    # 2. neg_frac_component: edge-level negative curvature fraction
    # 3. scalar_component: vertex-level scalar curvature (trace of Ricci)
    # 4. topology_component: loop persistence from Vietoris-Rips

    # Adaptive scaling: use curvature std instead of magic constants.
    # If std is tiny (flat curvature), any nonzero mean is significant.
    ricci_scale = max(kappa_std_final, 0.05)
    ricci_component = math.tanh(kappa_final / ricci_scale)        # [-1, +1]
    neg_frac_component = 1.0 - 2.0 * neg_frac_final              # +1 if all pos, -1 if all neg

    scalar_scale = max(abs(mean_scalar_final), 0.1) if abs(mean_scalar_final) > 1e-8 else 0.1
    scalar_component = math.tanh(mean_scalar_final / scalar_scale)  # [-1, +1]

    loop_score = -min(loop_pers * 2.0, 1.0)
    if inf_loops > 0:
        loop_score = -1.0
    topology_component = 1.0 if simply_connected else loop_score

    # Weighted: Ricci(30%) + neg_frac(25%) + scalar(30%) + topology(15%)
    poincare_score = (0.30 * ricci_component
                      + 0.25 * neg_frac_component
                      + 0.30 * scalar_component
                      + 0.15 * topology_component)

    if poincare_score > 0.25:
        regime = "mean-reversion"
    elif poincare_score < 0.10:
        regime = "trending"
    else:
        regime = "neutral"

    return TopologyReport(
        num_points=n,
        embed_dim=embed_dim,
        embed_lag=used_lag,
        mean_ricci=kappa_final,
        std_ricci=kappa_std_final,
        neg_ricci_frac=neg_frac_final,
        ricci_flow_convergence=convergence,
        mean_scalar=mean_scalar_final,
        neg_scalar_frac=neg_scalar_frac_final,
        beta0=b0,
        beta1=b1,
        beta2=b2,
        loop_persistence=loop_pers,
        void_persistence=void_pers,
        simply_connected=simply_connected,
        poincare_score=poincare_score,
        regime=regime,
    )


# ─── 7. Trading Signal ────────────────────────────────────────────────────────

@dataclass
class PoincareSignalConfig:
    """Configuration for the Poincaré topology trading signal."""
    embed_dim: int = 3              # Takens embedding dimension
    embed_lag: int = -1             # lag (-1 = auto via autocorrelation zero-crossing)
    knn: int = 5                    # k-NN graph neighbours
    ricci_flow_steps: int = 3       # Ricci flow iterations
    max_edge_fraction: float = 0.4  # Vietoris-Rips max edge (fraction of diameter)
    subsample: int = 60             # max points (farthest-point sampling)
    min_prices: int = 40            # minimum price window
    # Signal thresholds
    mean_reversion_threshold: float = 0.25  # min poincare_score for MR signal
    trending_threshold: float = 0.10        # max poincare_score for trend signal


class PoincareSignal:
    """
    Trading signal based on Poincaré conjecture topology analysis.

    Fires "mean_reversion" when the price manifold is S³-like (positive Ricci,
    simply connected → Poincaré says it IS S³ topologically).

    Fires "trending" when topology has non-trivial loops or negative curvature
    (non-simply-connected manifold → cannot be S³).
    """

    def __init__(self, config: PoincareSignalConfig = None):
        self.config = config or PoincareSignalConfig()
        self.last_report: Optional[TopologyReport] = None

    def eval(self, prices: List[float]) -> Optional[str]:
        """
        Analyse price window.
        Returns: "mean_reversion", "trending", or None.
        """
        if len(prices) < self.config.min_prices:
            return None

        report = poincare_analysis(
            prices,
            embed_dim=self.config.embed_dim,
            embed_lag=self.config.embed_lag,
            knn=self.config.knn,
            ricci_flow_steps=self.config.ricci_flow_steps,
            max_edge_fraction=self.config.max_edge_fraction,
            subsample=self.config.subsample,
        )
        self.last_report = report

        if report.poincare_score >= self.config.mean_reversion_threshold:
            return "mean_reversion"
        elif report.poincare_score <= self.config.trending_threshold:
            return "trending"
        return None

    def describe(self) -> str:
        lines = [
            "PoincareSignal (topology-based regime detector):",
            "  Method: Takens delay embedding + Ollivier-Ricci flow + Vietoris-Rips homology",
            f"  Embed: R^{self.config.embed_dim}  lag={self.config.embed_lag}  "
            f"k-NN={self.config.knn}  Ricci steps={self.config.ricci_flow_steps}",
        ]
        if self.last_report:
            lines.append("")
            lines.append(str(self.last_report))
        return "\n".join(lines)


# ─── 8. Combined Hecke + Poincaré signal ─────────────────────────────────────

class HeckePoincareSignal:
    """
    Three-layer signal with regime-adaptive thresholds:
      Layer 1: Poincaré topology (regime detection)
      Layer 2: Hecke spectral consistency (Gröbner basis check)
      Layer 3: Dirichlet character projection (directional bias)

    Cross-layer feedback (improvement N):
      - In mean-reversion regime: tighten Gröbner tolerance (structure should
        be strong for a sphere-like manifold).
      - In trending regime: loosen Dirichlet significance (weaker periodic
        signal expected in hyperbolic geometry).

    Single-pass evaluation (improvement O):
      - Computes Hecke + Dirichlet once, then branches on buy_bias() sign
        instead of calling eval() twice.
    """

    def __init__(self,
                 hecke_primes: List[int] = None,
                 dirichlet_moduli: List[int] = None,
                 min_prices: int = 80,
                 poincare_config: PoincareSignalConfig = None):
        self.poincare = PoincareSignal(poincare_config or PoincareSignalConfig(
            min_prices=min_prices,
        ))
        self.hecke_primes = hecke_primes or [2, 3, 5, 7, 11]
        self.dirichlet_moduli = dirichlet_moduli or [3, 5, 7]
        self.min_prices = min_prices
        # Import once, store class ref for regime-adaptive construction
        from dirichlet import HeckeDirichletSignal, dirichlet_report
        self._HDS = HeckeDirichletSignal
        self._dirichlet_report = dirichlet_report
        # Default signal (will be rebuilt per-eval with regime-adaptive params)
        self.hecke_dirichlet = self._HDS(
            hecke_primes=self.hecke_primes,
            dirichlet_moduli=self.dirichlet_moduli,
            min_prices=min_prices,
        )

    def eval(self, prices: List[float]) -> Optional[str]:
        """
        Returns: "buy", "sell", or None.

        Single-pass: runs Hecke check once, then Dirichlet once, and reads
        buy_bias() to determine direction. Regime from Poincaré modulates
        the Hecke/Dirichlet thresholds.
        """
        # Layer 1: topology regime
        regime = self.poincare.eval(prices)
        if regime is None:
            return None

        # Regime-adaptive thresholds
        if regime == "mean_reversion":
            # Tighter Gröbner — sphere-like manifold should have strong spectral structure
            groebner_tol = 0.25
            dirichlet_thr = 2.0
        else:
            # Looser Gröbner, lower Dirichlet threshold for trending regime
            groebner_tol = 0.45
            dirichlet_thr = 1.5

        # Rebuild Hecke signal with regime-adaptive tolerance
        self.hecke_dirichlet = self._HDS(
            hecke_primes=self.hecke_primes,
            dirichlet_moduli=self.dirichlet_moduli,
            min_prices=self.min_prices,
            groebner_tolerance_factor=groebner_tol,
            significance_threshold=dirichlet_thr,
        )

        # Layer 2: Hecke Gröbner check (single pass)
        if not self.hecke_dirichlet.hecke.eval(prices):
            return None

        # Layer 3: Dirichlet spectral analysis (single pass)
        report = self._dirichlet_report(
            prices,
            moduli=self.dirichlet_moduli,
            significance_threshold=dirichlet_thr,
        )
        self.hecke_dirichlet.last_dirichlet = report

        if not report.is_tradeable(min_sig=1):
            return None

        # Direction from bias (single read, no double eval)
        bias = report.buy_bias()
        if bias > 0:
            return "buy"
        elif bias < 0:
            return "sell"
        return None

    def describe(self) -> str:
        regime_str = "unknown"
        if self.poincare.last_report:
            regime_str = self.poincare.last_report.regime
        return "\n".join([
            f"HeckePoincareSignal (regime={regime_str}):",
            self.poincare.describe(),
            "",
            self.hecke_dirichlet.describe(),
        ])


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    print("=" * 60)
    print("Poincaré Conjecture Trading Geometry — Self Test")
    print("=" * 60)

    # Test 1: Random walk (should be neutral/mean-reverting)
    print("\n[1] Random walk (expect: mean-reversion or neutral)")
    prices_rw = [100.0]
    for _ in range(200):
        prices_rw.append(prices_rw[-1] * math.exp(random.gauss(0, 0.001)))
    report_rw = poincare_analysis(prices_rw)
    print(report_rw)

    # Test 2: Trending series (expect: trending)
    print("\n[2] Trending series (expect: trending)")
    prices_trend = [100.0]
    for t in range(200):
        prices_trend.append(prices_trend[-1] * math.exp(0.003 + random.gauss(0, 0.0005)))
    report_trend = poincare_analysis(prices_trend)
    print(report_trend)

    # Test 3: Mean-reverting (OU process, expect: mean-reversion)
    print("\n[3] Ornstein–Uhlenbeck mean-reverting (expect: mean-reversion)")
    prices_ou = [100.0]
    for _ in range(200):
        last = prices_ou[-1]
        drift = -0.05 * (last - 100.0)
        prices_ou.append(last + drift + random.gauss(0, 0.08))
    report_ou = poincare_analysis(prices_ou)
    print(report_ou)

    # Test 4: Oscillating (periodic, should have loops → trending or neutral)
    print("\n[4] Periodic oscillation (expect: loops / trending)")
    prices_osc = [100.0]
    for t in range(200):
        prices_osc.append(100.0 + 2.0 * math.sin(2 * math.pi * t / 20)
                          + random.gauss(0, 0.05))
    report_osc = poincare_analysis(prices_osc)
    print(report_osc)

    # Test 5: PoincareSignal
    print("\n[5] PoincareSignal on all 4 series:")
    sig = PoincareSignal()
    for name, prices in [("random walk", prices_rw), ("trend", prices_trend),
                          ("OU", prices_ou), ("oscillation", prices_osc)]:
        result = sig.eval(prices)
        score = sig.last_report.poincare_score if sig.last_report else 0
        print(f"  {name:<14}: signal={result}  score={score:+.3f}")
