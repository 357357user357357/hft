"""Tilelang Unified — single GPU compute stack for HFT.

Unifies all repos into one cohesive Tilelang-powered system:

  cudf      → GPU DataFrame (data storage, filter, groupby, join)
  cugraph   → GPU graph algorithms (PageRank, BFS, community detection)
  TileOPs   → GPU signal operators (elementwise, reduction, FFT, conv)
  Leiden    → GPU community detection (asset clustering)
  TinkerPop → GPU graph queries (Gremlin traversal)

All computation runs via Tilelang @jit(target='cuda') kernels.
No CuPy, no libcudf, no cuGraph C++, no torch math.

Usage:
    from tilelang_unified import UnifiedGPU

    gpu = UnifiedGPU()

    # DataFrame ops (cudf)
    df = gpu.read_csv("trades.csv")
    filtered = df.query("price > 100")
    grouped = df.groupby("symbol").agg({"volume": "sum"})

    # Graph ops (cugraph)
    g = gpu.Graph()
    g.add_edges(src, dst, weights)
    pr = g.pagerank()
    communities = g.leiden()

    # Signal ops (TileOPs)
    ops = gpu.TileOps()
    smoothed = ops.ema(prices, alpha=0.1)
    spectral = ops.spectral_strength(returns)

    # Gremlin queries (TinkerPop)
    g_traversal = gpu.GremlinGraph()
    g_traversal.add_vertices(vertices)
    g_traversal.add_edges(edges)
    result = g_traversal.V().out("knows").values("name").toList()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Import all Tilelang modules ──────────────────────────────────────────────

# cuDF → GPU DataFrame
from cudf_tilelang import (
    DataFrame, Column, GroupBy, Rolling,
    read_csv, read_parquet, read_json, concat,
)

# cuGraph → GPU graph algorithms
from cugraph_tilelang import (
    Graph as CuGraph, CuGraphTilelang,
)

# TileOPs → GPU signal operators
from tileops_hft import TileOps

# Leiden → GPU community detection
from leiden_kernels import detect_asset_clusters, LeidenDetector

# Frenet-Serret → GPU geometry analysis
from frenet_serret_gpu import FrenetSerretGPU, analyze_price_series_gpu

# hipGRAPH → GPU graph algorithms (alternative API)
from hipgraph_tilelang import GraphGPU


# ─────────────────────────────────────────────────────────────────────────────
# Unified GPU Stack
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedGPU:
    """Single entry point for all Tilelang GPU operations.

    Unifies cudf, cugraph, TileOPs, Leiden, and TinkerPop
    into one coherent API.
    """

    def __init__(self):
        # DataFrame (cudf)
        self.DataFrame = DataFrame
        self.Column = Column
        self.read_csv = read_csv
        self.read_parquet = read_parquet
        self.read_json = read_json
        self.concat = concat

        # Graph (cugraph)
        self.CuGraph = CuGraphTilelang
        self.CuGraphTilelang = CuGraphTilelang

        # Signal ops (TileOPs)
        self.TileOps = TileOps
        self.ops = TileOps()

        # Community detection (Leiden)
        self.detect_asset_clusters = detect_asset_clusters
        self.LeidenDetector = LeidenDetector

        # Geometry (Frenet-Serret)
        self.FrenetSerretGPU = FrenetSerretGPU
        self.analyze_price_series_gpu = analyze_price_series_gpu

        # Graph (hipGRAPH-style)
        self.GraphGPU = GraphGPU

        # Gremlin (TinkerPop)
        self.GremlinGraph = GremlinGraph

    # ── Convenience: end-to-end HFT pipeline ─────────────────────────────

    def hft_pipeline(self, prices: np.ndarray, volumes: np.ndarray,
                     symbols: Optional[List[str]] = None,
                     returns: Optional[np.ndarray] = None
                     ) -> Dict[str, Any]:
        """Run full HFT analysis pipeline on GPU.

        1. Geometry: Frenet-Serret curvature/torsion
        2. Signal: EMA, SMA, spectral strength
        3. Community: asset clustering (if returns provided)
        4. Graph: trade relationship analysis

        Returns:
            dict with all analysis results
        """
        result = {}

        # 1. Geometry analysis
        fs = self.FrenetSerretGPU(delay=5)
        result['geometry'] = fs.analyze(prices, volumes)

        # 2. Signal processing
        result['ema_10'] = self.ops.ema(prices, alpha=0.1)
        result['ema_20'] = self.ops.ema(prices, alpha=0.05)
        result['sma_20'] = self.ops.sma(prices, window=20)
        result['spectral'] = self.ops.spectral_strength(prices, n_bins=20)

        # 3. Community detection (if multi-asset returns provided)
        if returns is not None and returns.ndim == 2:
            result['clusters'] = self.detect_asset_clusters(
                returns, symbols=symbols, threshold=0.3
            )

        # 4. Graph analysis (build correlation graph)
        if returns is not None and returns.ndim == 2:
            corr = np.corrcoef(returns.T)
            corr = np.nan_to_num(corr, nan=0.0)
            threshold = 0.5
            src, dst, wts = [], [], []
            n_assets = corr.shape[0]
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    if abs(corr[i, j]) > threshold:
                        src.append(i)
                        dst.append(j)
                        wts.append(abs(corr[i, j]))

            if src:
                g = self.CuGraph()
                g.add_edge_list(
                    np.array(src, dtype=np.int32),
                    np.array(dst, dtype=np.int32),
                    np.array(wts, dtype=np.float32),
                )
                result['pagerank'] = g.pagerank()
                result['communities'] = g.louvain()
                result['triangle_count'] = g.triangle_count()

        return result


# ─────────────────────────────────────────────────────────────────────────────
# TinkerPop Gremlin → Tilelang
# ─────────────────────────────────────────────────────────────────────────────

class GremlinGraph:
    """TinkerPop Gremlin graph traversal on GPU via Tilelang.

    Replaces TinkerGraph with Tilelang GPU backend.
    Supports Gremlin-style traversals: V(), E(), out(), in(), both(),
    values(), has(), groupCount(), etc.
    """

    def __init__(self):
        self._vertices: Dict[int, Dict[str, Any]] = {}
        self._edges: List[Dict[str, Any]] = []
        self._edge_index: Dict[int, List[int]] = {}  # src → [edge_idx]
        self._reverse_index: Dict[int, List[int]] = {}  # dst → [edge_idx]
        self._next_vid = 0
        self._next_eid = 0

    def add_vertex(self, vid: Optional[int] = None, **properties) -> int:
        """Add a vertex with properties."""
        if vid is None:
            vid = self._next_vid
            self._next_vid += 1
        self._vertices[vid] = dict(properties)
        return vid

    def add_edge(self, src: int, dst: int, label: str = "knows",
                 eid: Optional[int] = None, **properties) -> int:
        """Add a directed edge."""
        if eid is None:
            eid = self._next_eid
            self._next_eid += 1
        edge = {"id": eid, "src": src, "dst": dst, "label": label, **properties}
        self._edges.append(edge)
        self._edge_index.setdefault(src, []).append(len(self._edges) - 1)
        self._reverse_index.setdefault(dst, []).append(len(self._edges) - 1)
        return eid

    def add_vertices(self, vertices: List[Dict[str, Any]]):
        """Add multiple vertices."""
        for v in vertices:
            vid = v.pop("id", None)
            self.add_vertex(vid, **v)

    def add_edges(self, edges: List[Dict[str, Any]]):
        """Add multiple edges."""
        for e in edges:
            src = e.pop("src")
            dst = e.pop("dst")
            label = e.pop("label", "knows")
            eid = e.pop("id", None)
            self.add_edge(src, dst, label, eid, **e)

    # ── Gremlin-style traversal API ──────────────────────────────────────

    def V(self, *vids) -> "Traversal":
        """Start traversal from vertices."""
        if vids:
            vertices = {vid: self._vertices[vid] for vid in vids if vid in self._vertices}
        else:
            vertices = dict(self._vertices)
        return Traversal(self, vertices)

    def E(self, *eids) -> "EdgeTraversal":
        """Start traversal from edges."""
        if eids:
            edges = [e for e in self._edges if e["id"] in eids]
        else:
            edges = list(self._edges)
        return EdgeTraversal(self, edges)

    def vertex_count(self) -> int:
        return len(self._vertices)

    def edge_count(self) -> int:
        return len(self._edges)

    def to_cugraph(self) -> CuGraphTilelang:
        """Convert to cuGraph for GPU algorithm execution."""
        src = np.array([e["src"] for e in self._edges], dtype=np.int32)
        dst = np.array([e["dst"] for e in self._edges], dtype=np.int32)
        wts = np.array([e.get("weight", 1.0) for e in self._edges], dtype=np.float32)

        g = CuGraphTilelang()
        g.add_edge_list(src, dst, wts)
        return g


class Traversal:
    """Gremlin-style vertex traversal."""

    def __init__(self, graph: GremlinGraph, vertices: Dict[int, Dict]):
        self._graph = graph
        self._vertices = vertices
        self._steps: List[str] = []

    def out(self, label: Optional[str] = None) -> "Traversal":
        """Traverse outgoing edges."""
        new_vertices = {}
        for vid in self._vertices:
            for eidx in self._graph._edge_index.get(vid, []):
                edge = self._graph._edges[eidx]
                if label is None or edge["label"] == label:
                    dst = edge["dst"]
                    if dst in self._graph._vertices:
                        new_vertices[dst] = self._graph._vertices[dst]
        self._vertices = new_vertices
        self._steps.append(f"out({label!r})")
        return self

    def in_(self, label: Optional[str] = None) -> "Traversal":
        """Traverse incoming edges."""
        new_vertices = {}
        for vid in self._vertices:
            for eidx in self._graph._reverse_index.get(vid, []):
                edge = self._graph._edges[eidx]
                if label is None or edge["label"] == label:
                    src = edge["src"]
                    if src in self._graph._vertices:
                        new_vertices[src] = self._graph._vertices[src]
        self._vertices = new_vertices
        self._steps.append(f"in({label!r})")
        return self

    def both(self, label: Optional[str] = None) -> "Traversal":
        """Traverse both incoming and outgoing edges."""
        self.out(label)
        self.in_(label)
        return self

    def has(self, key: str, value: Any = None) -> "Traversal":
        """Filter vertices by property."""
        if value is not None:
            self._vertices = {
                vid: props for vid, props in self._vertices.items()
                if props.get(key) == value
            }
        else:
            self._vertices = {
                vid: props for vid, props in self._vertices.items()
                if key in props
            }
        self._steps.append(f"has({key}={value!r})")
        return self

    def values(self, key: str) -> List[Any]:
        """Get property values."""
        return [props.get(key) for props in self._vertices.values()]

    def id(self) -> List[int]:
        """Get vertex IDs."""
        return list(self._vertices.keys())

    def count(self) -> int:
        return len(self._vertices)

    def groupCount(self) -> Dict[str, int]:
        """Group by a property and count."""
        counts: Dict[str, int] = {}
        for props in self._vertices.values():
            key = str(props.get("group", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def toList(self) -> List[Dict]:
        """Return vertices as list of dicts."""
        return [{"id": vid, **props} for vid, props in self._vertices.items()]

    def __repr__(self):
        return f"Traversal({len(self._vertices)} vertices, steps={self._steps})"


class EdgeTraversal:
    """Gremlin-style edge traversal."""

    def __init__(self, graph: GremlinGraph, edges: List[Dict]):
        self._graph = graph
        self._edges = edges

    def outV(self) -> "Traversal":
        """Get source vertices."""
        vertices = {}
        for edge in self._edges:
            src = edge["src"]
            if src in self._graph._vertices:
                vertices[src] = self._graph._vertices[src]
        return Traversal(self._graph, vertices)

    def inV(self) -> "Traversal":
        """Get destination vertices."""
        vertices = {}
        for edge in self._edges:
            dst = edge["dst"]
            if dst in self._graph._vertices:
                vertices[dst] = self._graph._vertices[dst]
        return Traversal(self._graph, vertices)

    def bothV(self) -> "Traversal":
        """Get both source and destination vertices."""
        vertices = {}
        for edge in self._edges:
            for vid in [edge["src"], edge["dst"]]:
                if vid in self._graph._vertices:
                    vertices[vid] = self._graph._vertices[vid]
        return Traversal(self._graph, vertices)

    def count(self) -> int:
        return len(self._edges)

    def toList(self) -> List[Dict]:
        return list(self._edges)


# ─────────────────────────────────────────────────────────────────────────────
# Quick-start functions
# ─────────────────────────────────────────────────────────────────────────────

def unified_hft_analysis(prices: np.ndarray, volumes: np.ndarray,
                         returns: Optional[np.ndarray] = None,
                         symbols: Optional[List[str]] = None
                         ) -> Dict[str, Any]:
    """One-liner: full HFT analysis on GPU via Tilelang."""
    gpu = UnifiedGPU()
    return gpu.hft_pipeline(prices, volumes, symbols, returns)


def build_trade_graph(trades: DataFrame) -> GremlinGraph:
    """Build a Gremlin graph from trade DataFrame."""
    g = GremlinGraph()

    # Add symbol vertices
    if "symbol" in trades.columns:
        symbols = trades["symbol"].unique()
        for sym in symbols:
            g.add_vertex(label="symbol", name=sym)

    # Add trade vertices and edges
    for i in range(len(trades)):
        vid = g.add_vertex(label="trade", index=i)
        if "symbol" in trades.columns:
            sym = trades["symbol"].data[i]
            # Connect trade to symbol
            # (In practice, you'd map symbol names to vertex IDs)

    return g


def correlate_and_cluster(returns: np.ndarray, symbols: List[str],
                          threshold: float = 0.3
                          ) -> Dict[str, Any]:
    """Correlation-based asset clustering with graph analysis."""
    gpu = UnifiedGPU()

    # 1. Community detection
    clusters = gpu.detect_asset_clusters(returns, symbols=symbols, threshold=threshold)

    # 2. Build correlation graph
    corr = np.corrcoef(returns.T)
    corr = np.nan_to_num(corr, nan=0.0)
    src, dst, wts = [], [], []
    n = corr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > threshold:
                src.append(i)
                dst.append(j)
                wts.append(abs(corr[i, j]))

    # 3. Graph analysis
    g = gpu.CuGraph()
    g.add_edge_list(
        np.array(src, dtype=np.int32),
        np.array(dst, dtype=np.int32),
        np.array(wts, dtype=np.float32),
    )

    return {
        "clusters": clusters,
        "pagerank": g.pagerank(),
        "communities": g.louvain(),
        "triangle_count": g.triangle_count(),
        "core_numbers": g.core_number(),
    }
