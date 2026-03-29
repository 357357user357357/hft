"""Whitehead topology signal for HFT regime-change detection.

J.H.C. Whitehead's contributions used here:

1. **CW complexes** (Whitehead's invention):
   Price trajectory is modelled as a CW complex built by attaching cells
   of increasing dimension to the Vietoris-Rips filtration:
     0-cells = price levels (vertices)
     1-cells = short-term transitions (edges, β₀/β₁)
     2-cells = mean-reversion cycles (filled triangles, closes β₁ loops)
     3-cells = volatility bubbles (tetrahedra, β₂)

2. **Whitehead torsion** (simple homotopy theory):
   A homotopy equivalence between two CW complexes is *simple* iff its
   Whitehead torsion τ ∈ Wh(π₁) vanishes.
   Trading interpretation:
     - Short-lived persistence bars (lifetime ≪ diameter) → simple
       homotopy equivalence → same regime, just parameter noise.
     - Long-lived bars → non-trivial torsion → genuine topological change
       → regime transition signal.
   We approximate τ via the ratio of long-lived to short-lived bar counts.

3. **Whitehead theorem** (homotopy groups):
   A map inducing isomorphisms on ALL homotopy groups is a homotopy
   equivalence.  We track changes in β₁ (π₁) and β₂ (π₂) simultaneously;
   a sudden joint change flags a full regime transition (both loops and
   voids restructure at once).

4. **Reeb graph / Mapper** (topological skeleton):
   The Mapper algorithm (Singh–Mémoli–Carlsson) compresses the price
   manifold into a graph whose nodes are regime clusters and whose edges
   are transitions. Uses gudhi's SimplexTree + a 1D cover on the
   normalised price axis.

Requires: gudhi (MIT, pip install gudhi)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import gudhi as _gudhi
    _GUDHI_AVAILABLE = True
except ImportError:
    _GUDHI_AVAILABLE = False

from poincare_trading import (
    delay_embed, normalise_points, farthest_point_sampling,
    optimal_lag, pairwise_distances,
)


# ─── Persistence utilities ────────────────────────────────────────────────────

@dataclass
class BarSummary:
    """Summary of a single persistence bar."""
    dim: int
    birth: float
    death: float        # math.inf for essential features

    @property
    def lifetime(self) -> float:
        return math.inf if math.isinf(self.death) else self.death - self.birth

    @property
    def is_essential(self) -> bool:
        return math.isinf(self.death)


def _compute_persistence(points: List[Tuple[float, ...]],
                         max_edge: float,
                         max_dim: int = 3) -> List[BarSummary]:
    """
    Compute Vietoris-Rips persistence up to max_dim via gudhi.
    Returns list of BarSummary sorted by (dim, birth).
    """
    if not _GUDHI_AVAILABLE or not points:
        return []
    rc = _gudhi.RipsComplex(points=[list(p) for p in points],
                             max_edge_length=max_edge)
    st = rc.create_simplex_tree(max_dimension=max_dim)
    st.compute_persistence()
    bars = []
    for dim, (birth, death) in st.persistence():
        bars.append(BarSummary(dim=dim, birth=birth, death=death))
    bars.sort(key=lambda b: (b.dim, b.birth))
    return bars


# ─── Whitehead Torsion Proxy ──────────────────────────────────────────────────

@dataclass
class WhiteheadTorsion:
    """
    Discrete approximation of Whitehead torsion from a persistence diagram.

    torsion_ratio > threshold → non-trivial torsion → regime change.

    Fields:
        long_lived_count:  bars with lifetime > long_threshold * diameter
        short_lived_count: bars with lifetime < short_threshold * diameter
        essential_count:   bars with infinite lifetime (never die)
        torsion_ratio:     long_lived / (short_lived + 1) — high = complex topology
        is_simple:         True if torsion is below threshold (same regime)
        diameter:          diameter of the point cloud (max pairwise distance)
    """
    long_lived_count: int
    short_lived_count: int
    essential_count: int
    torsion_ratio: float
    is_simple: bool
    diameter: float
    # Per-dimension counts for Whitehead theorem check
    beta1_changes: int   # significant β₁ bars
    beta2_changes: int   # significant β₂ bars


def compute_whitehead_torsion(
    bars: List[BarSummary],
    diameter: float,
    long_threshold: float = 0.15,   # bar lifetime > 15% of diameter = long-lived
    short_threshold: float = 0.03,  # bar lifetime < 3% of diameter = noise
    torsion_threshold: float = 0.5, # torsion_ratio above this = non-simple
) -> WhiteheadTorsion:
    """
    Estimate Whitehead torsion from persistence bars.

    Long-lived bars represent non-trivial topological structure that persists
    across the filtration — analogous to non-trivial Whitehead torsion.
    Short-lived bars are topological noise (simple homotopy equivalences).
    """
    if not bars or diameter < 1e-12:
        return WhiteheadTorsion(0, 0, 0, 0.0, True, diameter, 0, 0)

    long_lived = 0
    short_lived = 0
    essential = 0
    beta1 = 0
    beta2 = 0

    for b in bars:
        if b.dim == 0:          # β₀ = components, skip (always 1 at end)
            continue
        if b.is_essential:
            essential += 1
            if b.dim == 1:
                beta1 += 1
            elif b.dim == 2:
                beta2 += 1
            continue
        lt = b.lifetime / diameter
        if lt > long_threshold:
            long_lived += 1
            if b.dim == 1:
                beta1 += 1
            elif b.dim == 2:
                beta2 += 1
        elif lt < short_threshold:
            short_lived += 1

    torsion_ratio = long_lived / (short_lived + 1)
    is_simple = torsion_ratio < torsion_threshold and essential == 0

    return WhiteheadTorsion(
        long_lived_count=long_lived,
        short_lived_count=short_lived,
        essential_count=essential,
        torsion_ratio=torsion_ratio,
        is_simple=is_simple,
        diameter=diameter,
        beta1_changes=beta1,
        beta2_changes=beta2,
    )


# ─── Reeb Graph / Mapper skeleton ────────────────────────────────────────────

@dataclass
class ReebNode:
    """A node in the Reeb graph (a regime cluster)."""
    node_id: int
    mean_price: float           # representative price level
    point_count: int            # how many price points fall here
    cover_interval: Tuple[float, float]  # [low, high] on the filter axis


@dataclass
class ReebEdge:
    """An edge in the Reeb graph (a regime transition)."""
    src: int
    dst: int
    shared_points: int          # points in the overlap / transition zone


@dataclass
class ReebGraph:
    """
    Topological skeleton of the price manifold via the Mapper algorithm.

    Nodes = regime clusters.
    Edges = transitions between regimes.
    Isolated nodes = distinct regime islands (no transition between them).
    High-degree nodes = hub regimes (price keeps returning here).
    """
    nodes: List[ReebNode]
    edges: List[ReebEdge]

    @property
    def num_regimes(self) -> int:
        return len(self.nodes)

    @property
    def num_transitions(self) -> int:
        return len(self.edges)

    @property
    def hub_regime(self) -> Optional[ReebNode]:
        """Node with the highest degree (most connected regime)."""
        if not self.nodes:
            return None
        degree: Dict[int, int] = {n.node_id: 0 for n in self.nodes}
        for e in self.edges:
            degree[e.src] = degree.get(e.src, 0) + 1
            degree[e.dst] = degree.get(e.dst, 0) + 1
        best_id = max(degree, key=lambda k: degree[k])
        return next(n for n in self.nodes if n.node_id == best_id)


def build_reeb_graph(
    points: List[Tuple[float, ...]],
    prices: List[float],
    n_intervals: int = 5,
    overlap_pct: float = 0.3,
) -> ReebGraph:
    """
    Build a Reeb/Mapper graph from delay-embedded price points.

    Algorithm (1D Mapper):
    1. Filter function f = normalised price (0→1 along price axis).
    2. Cover = n_intervals overlapping intervals on [0, 1].
    3. For each interval, collect points whose filter value falls inside.
    4. Cluster each interval's points via single-linkage (connected components
       of the sub-cloud within max_edge distance).
    5. Nodes = clusters; edges = non-empty intersections between adjacent
       interval clusters.

    This is a discrete approximation of the Reeb graph of the price manifold
    with respect to the price filter function.
    """
    if len(points) < 4:
        return ReebGraph(nodes=[], edges=[])

    n = len(points)
    # Filter values: use first coordinate of each embedded point (= price_t)
    f_vals = [p[0] for p in points]
    f_min = min(f_vals)
    f_max = max(f_vals)
    f_range = max(f_max - f_min, 1e-10)
    f_norm = [(v - f_min) / f_range for v in f_vals]

    step = 1.0 / n_intervals
    half_overlap = step * overlap_pct / 2.0

    # Build intervals
    intervals: List[Tuple[float, float]] = []
    for k in range(n_intervals):
        lo = k * step - half_overlap
        hi = (k + 1) * step + half_overlap
        intervals.append((max(0.0, lo), min(1.0, hi)))

    # For each interval, collect point indices inside it
    interval_points: List[List[int]] = []
    for lo, hi in intervals:
        members = [i for i, fv in enumerate(f_norm) if lo <= fv <= hi]
        interval_points.append(members)

    # Cluster each interval's points via connected components
    # (edges exist if distance < median pairwise distance of the full cloud)
    all_dists = []
    sample = points[:min(30, n)]
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(sample[i], sample[j])))
            all_dists.append(d)
    median_dist = sorted(all_dists)[len(all_dists) // 2] if all_dists else 0.1

    node_counter = 0
    reeb_nodes: List[ReebNode] = []
    # interval_clusters[k] = list of frozenset(point_indices) = clusters in interval k
    interval_clusters: List[List[frozenset]] = []

    for k, members in enumerate(interval_points):
        if not members:
            interval_clusters.append([])
            continue
        # Union-find clustering within this interval
        parent = {i: i for i in members}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for a_idx, a in enumerate(members):
            for b in members[a_idx + 1:]:
                d = math.sqrt(sum((points[a][dim] - points[b][dim]) ** 2
                                  for dim in range(len(points[a]))))
                if d <= median_dist * 1.5:
                    union(a, b)

        # Collect clusters
        clusters_map: Dict[int, List[int]] = {}
        for i in members:
            root = find(i)
            clusters_map.setdefault(root, []).append(i)

        clusters = []
        for cluster_pts in clusters_map.values():
            mean_p = sum(prices[min(i, len(prices) - 1)] for i in cluster_pts) / len(cluster_pts)
            node = ReebNode(
                node_id=node_counter,
                mean_price=mean_p,
                point_count=len(cluster_pts),
                cover_interval=intervals[k],
            )
            reeb_nodes.append(node)
            clusters.append(frozenset(cluster_pts))
            node_counter += 1

        interval_clusters.append(clusters)

    # Build edges: two clusters in adjacent intervals share an edge if their
    # point sets intersect (a point appears in both → transition zone).
    reeb_edges: List[ReebEdge] = []
    node_offset = 0
    offsets: List[int] = []
    for clusters in interval_clusters:
        offsets.append(node_offset)
        node_offset += len(clusters)

    for k in range(len(interval_clusters) - 1):
        src_clusters = interval_clusters[k]
        dst_clusters = interval_clusters[k + 1]
        src_base = offsets[k]
        dst_base = offsets[k + 1]
        for si, sc in enumerate(src_clusters):
            for di, dc in enumerate(dst_clusters):
                shared = len(sc & dc)
                if shared > 0:
                    reeb_edges.append(ReebEdge(
                        src=reeb_nodes[src_base + si].node_id,
                        dst=reeb_nodes[dst_base + di].node_id,
                        shared_points=shared,
                    ))

    return ReebGraph(nodes=reeb_nodes, edges=reeb_edges)


# ─── Whitehead Report ─────────────────────────────────────────────────────────

@dataclass
class WhiteheadReport:
    """
    Full Whitehead topology analysis of a price window.

    regime_change:   True if non-trivial Whitehead torsion detected
    torsion:         detailed torsion proxy breakdown
    reeb:            Mapper/Reeb graph (regime skeleton)
    poincare_check:  True if Whitehead theorem is satisfied (β₁ and β₂ both zero)
    signal:          "regime_change" | "same_regime" | "insufficient_data"
    """
    num_points: int
    embed_lag: int
    diameter: float
    torsion: WhiteheadTorsion
    reeb: ReebGraph
    poincare_check: bool    # β₁ = β₂ = 0 → Whitehead theorem satisfied → Poincaré condition
    regime_change: bool
    signal: str             # "regime_change" | "same_regime" | "insufficient_data"

    def __repr__(self) -> str:
        lines = [
            f"WhiteheadReport: signal={self.signal}  regimes={self.reeb.num_regimes}"
            f"  transitions={self.reeb.num_transitions}",
            f"  Torsion: ratio={self.torsion.torsion_ratio:.3f}"
            f"  long={self.torsion.long_lived_count}"
            f"  short={self.torsion.short_lived_count}"
            f"  essential={self.torsion.essential_count}"
            f"  simple={self.torsion.is_simple}",
            f"  β₁_changes={self.torsion.beta1_changes}"
            f"  β₂_changes={self.torsion.beta2_changes}"
            f"  Poincaré-check={self.poincare_check}",
        ]
        if self.reeb.hub_regime:
            h = self.reeb.hub_regime
            lines.append(f"  Hub regime: price≈{h.mean_price:.4f}  pts={h.point_count}")
        return "\n".join(lines)


def whitehead_analysis(
    prices: List[float],
    embed_dim: int = 3,
    embed_lag: int = -1,
    subsample: int = 60,
    max_edge_fraction: float = 0.5,
    reeb_intervals: int = 5,
    torsion_threshold: float = 0.5,
    min_prices: int = 30,
) -> WhiteheadReport:
    """
    Whitehead topology analysis of a price window.

    Steps:
    1. Delay-embed prices (Takens) with adaptive lag.
    2. Normalise and subsample via farthest-point landmarks.
    3. Compute Vietoris-Rips persistent homology (gudhi).
    4. Estimate Whitehead torsion from persistence bar lifetimes.
    5. Build Reeb/Mapper graph (topological regime skeleton).
    6. Check Whitehead theorem (β₁ = β₂ = 0 → Poincaré condition).
    7. Emit regime-change signal if torsion is non-trivial.

    Returns WhiteheadReport.
    """
    _empty = lambda lag: WhiteheadReport(
        num_points=0,
        embed_lag=lag,
        diameter=0.0,
        torsion=WhiteheadTorsion(0, 0, 0, 0.0, True, 0.0, 0, 0),
        reeb=ReebGraph(nodes=[], edges=[]),
        poincare_check=True,
        regime_change=False,
        signal="insufficient_data",
    )

    if not _GUDHI_AVAILABLE or len(prices) < min_prices:
        lag = embed_lag if embed_lag > 0 else optimal_lag(prices)
        return _empty(lag)

    used_lag = embed_lag if embed_lag > 0 else optimal_lag(prices)
    points = delay_embed(prices, dim=embed_dim, lag=used_lag)
    if len(points) < 6:
        return _empty(used_lag)

    points = normalise_points(points)
    if len(points) > subsample:
        points = farthest_point_sampling(points, subsample)

    n = len(points)
    # Diameter of point cloud
    flat_dists = [
        math.sqrt(sum((points[i][d] - points[j][d]) ** 2 for d in range(len(points[0]))))
        for i in range(n) for j in range(i + 1, n)
    ]
    diameter = max(flat_dists) if flat_dists else 1.0
    max_edge = max_edge_fraction * diameter

    # Persistent homology via gudhi
    bars = _compute_persistence(points, max_edge=max_edge, max_dim=3)

    # Whitehead torsion proxy
    torsion = compute_whitehead_torsion(
        bars, diameter=diameter, torsion_threshold=torsion_threshold
    )

    # Reeb / Mapper graph
    reeb = build_reeb_graph(points, prices, n_intervals=reeb_intervals)

    # Whitehead theorem check: β₁ = β₂ = 0 → all homotopy groups below dim 3
    # are trivial → Poincaré condition is satisfied (manifold could be S³)
    poincare_check = (torsion.beta1_changes == 0 and torsion.beta2_changes == 0
                      and torsion.essential_count == 0)

    regime_change = not torsion.is_simple

    if n < 6:
        signal = "insufficient_data"
    elif regime_change:
        signal = "regime_change"
    else:
        signal = "same_regime"

    return WhiteheadReport(
        num_points=n,
        embed_lag=used_lag,
        diameter=diameter,
        torsion=torsion,
        reeb=reeb,
        poincare_check=poincare_check,
        regime_change=regime_change,
        signal=signal,
    )
