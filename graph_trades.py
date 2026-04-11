"""Graph-based trade analysis via Tilelang GPU.

Replaces rustworkx/networkx with pure Tilelang GPU graph algorithms.
All traversal, PageRank, community detection, and pattern matching run
on GPU via @tilelang.jit(target='cuda') kernels.

Usage:
    from graph_trades import TradeGraphGPU

    g = TradeGraphGPU()
    g.add_price_node(timestamp=..., price=...)
    g.add_signal_node(timestamp=..., signal_type="momentum", strength=0.8)
    g.add_trade_node(timestamp=..., side="long", entry=..., exit=..., pnl=...)

    # GPU graph analysis
    pr = g.pagerank()           # most influential nodes
    communities = g.communities()  # trade clusters
    bfs = g.bfs_from_latest()   # traversal from latest node
    similar = g.find_similar_setups(r=..., theta=...)  # pattern match
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import statistics
import numpy as np

from cugraph_tilelang import Graph as CuGraph, pagerank, louvain, bfs, triangle_count, core_number
from tilelang_unified import GremlinGraph


@dataclass
class GraphStats:
    total_nodes: int
    total_edges: int
    node_types: Dict[str, int]
    density: float
    pagerank_top: List[Dict]
    triangle_count: int
    core_numbers: Dict[str, int]


class TradeGraphGPU:
    """GPU-accelerated trade graph via Tilelang.

    Replaces rustworkx/networkx with Tilelang GPU kernels for:
    - PageRank (most influential trades/signals)
    - Community detection (trade clusters)
    - BFS traversal (signal → trade → regime paths)
    - Triangle counting (arbitrage cycle detection)
    - Core decomposition (trade network resilience)
    - Gremlin-style queries (V().out().has().values())
    """

    def __init__(self):
        # Node storage
        self._nodes: Dict[int, Dict[str, Any]] = {}
        self._node_counter = 0
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}

        # Edge storage (for Gremlin traversal)
        self._gremlin = GremlinGraph()

        # GPU graph (for algorithms)
        self._gpu_graph: Optional[CuGraph] = None
        self._needs_rebuild = False

    # ── Node Creation ──────────────────────────────────────────────────────

    def _next_idx(self, node_id: str) -> int:
        idx = self._node_counter
        self._node_counter += 1
        self._id_to_idx[node_id] = idx
        self._idx_to_id[idx] = node_id
        return idx

    def add_price_node(self, timestamp: int, price: float,
                       polar_r: float = None, polar_theta: float = None,
                       **kwargs) -> str:
        node_id = f"price_{self._node_counter}"
        idx = self._next_idx(node_id)
        self._nodes[idx] = {
            "node_type": "price",
            "timestamp": timestamp,
            "price": price,
            "polar_r": polar_r,
            "polar_theta": polar_theta,
            **kwargs,
        }
        self._gremlin.add_vertex(idx, **self._nodes[idx])
        self._needs_rebuild = True
        return node_id

    def add_signal_node(self, timestamp: int, signal_type: str,
                        strength: float, regime: str = None,
                        **kwargs) -> str:
        node_id = f"signal_{self._node_counter}"
        idx = self._next_idx(node_id)
        self._nodes[idx] = {
            "node_type": "signal",
            "timestamp": timestamp,
            "signal_type": signal_type,
            "strength": strength,
            "regime": regime,
            **kwargs,
        }
        self._gremlin.add_vertex(idx, **self._nodes[idx])
        self._needs_rebuild = True
        return node_id

    def add_trade_node(self, timestamp: int, side: str, entry: float,
                       exit_price: float, pnl_pct: float, size: float,
                       exit_reason: str = None, **kwargs) -> str:
        node_id = f"trade_{self._node_counter}"
        idx = self._next_idx(node_id)
        self._nodes[idx] = {
            "node_type": "trade",
            "timestamp": timestamp,
            "side": side,
            "entry_price": entry,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "size_usdt": size,
            "exit_reason": exit_reason,
            **kwargs,
        }
        self._gremlin.add_vertex(idx, **self._nodes[idx])
        self._needs_rebuild = True
        return node_id

    def add_regime_node(self, start_ts: int, regime_type: str,
                        end_ts: int = None, **kwargs) -> str:
        node_id = f"regime_{self._node_counter}"
        idx = self._next_idx(node_id)
        self._nodes[idx] = {
            "node_type": "regime",
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "regime_type": regime_type,
            **kwargs,
        }
        self._gremlin.add_vertex(idx, **self._nodes[idx])
        self._needs_rebuild = True
        return node_id

    # ── Edge Creation ──────────────────────────────────────────────────────

    def add_price_signal_edge(self, price_node: str, signal_node: str,
                              confidence: float = 1.0) -> None:
        src = self._id_to_idx[price_node]
        dst = self._id_to_idx[signal_node]
        self._gremlin.add_edge(src, dst, "generates", weight=confidence)
        self._needs_rebuild = True

    def add_signal_trade_edge(self, signal_node: str, trade_node: str,
                              delay_ms: float = None) -> None:
        src = self._id_to_idx[signal_node]
        dst = self._id_to_idx[trade_node]
        self._gremlin.add_edge(src, dst, "executed_as", delay_ms=delay_ms)
        self._needs_rebuild = True

    def add_trade_regime_edge(self, trade_node: str, regime_node: str) -> None:
        src = self._id_to_idx[trade_node]
        dst = self._id_to_idx[regime_node]
        self._gremlin.add_edge(src, dst, "occurred_in")
        self._needs_rebuild = True

    def add_correlation_edge(self, node1: str, node2: str,
                             correlation: float) -> None:
        src = self._id_to_idx[node1]
        dst = self._id_to_idx[node2]
        self._gremlin.add_edge(src, dst, "correlated_with", weight=correlation)
        self._gremlin.add_edge(dst, src, "correlated_with", weight=correlation)
        self._needs_rebuild = True

    def _build_gpu_graph(self):
        """Build CSR graph for GPU algorithms."""
        if not self._needs_rebuild:
            return

        src_list, dst_list, wts = [], [], []
        for edge in self._gremlin._edges:
            src_list.append(edge["src"])
            dst_list.append(edge["dst"])
            wts.append(edge.get("weight", 1.0))

        if src_list:
            self._gpu_graph = CuGraph()
            self._gpu_graph.add_edge_list(
                np.array(src_list, dtype=np.int32),
                np.array(dst_list, dtype=np.int32),
                np.array(wts, dtype=np.float32),
            )
        self._needs_rebuild = False

    # ── GPU Graph Algorithms ───────────────────────────────────────────────

    def pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """GPU PageRank — most influential nodes."""
        self._build_gpu_graph()
        if self._gpu_graph is None:
            return {}
        pr = pagerank(self._gpu_graph, alpha=alpha)
        return {
            self._idx_to_id.get(i, str(i)): float(pr[i])
            for i in range(len(pr))
        }

    def communities(self) -> Dict[str, List[str]]:
        """GPU community detection — trade clusters."""
        self._build_gpu_graph()
        if self._gpu_graph is None:
            return {}
        labels = louvain(self._gpu_graph)
        clusters: Dict[str, List[str]] = {}
        for i, label in enumerate(labels):
            name = f"cluster_{int(label)}"
            node_id = self._idx_to_id.get(i, str(i))
            clusters.setdefault(name, []).append(node_id)
        return clusters

    def bfs_from(self, source_idx: int) -> Dict[str, int]:
        """GPU BFS from a source node."""
        self._build_gpu_graph()
        if self._gpu_graph is None:
            return {}
        dist = bfs(self._gpu_graph, source=source_idx)
        return {
            self._idx_to_id.get(i, str(i)): int(dist[i])
            for i in range(len(dist))
        }

    def bfs_from_latest(self) -> Dict[str, int]:
        """BFS from the most recently added node."""
        if not self._nodes:
            return {}
        latest_idx = max(self._nodes.keys())
        return self.bfs_from(latest_idx)

    def count_triangles(self) -> int:
        """GPU triangle counting — arbitrage cycle detection."""
        self._build_gpu_graph()
        if self._gpu_graph is None:
            return 0
        return triangle_count(self._gpu_graph)

    def core_numbers(self) -> Dict[str, int]:
        """GPU core decomposition — network resilience."""
        self._build_gpu_graph()
        if self._gpu_graph is None:
            return {}
        cores = core_number(self._gpu_graph)
        return {
            self._idx_to_id.get(i, str(i)): int(cores[i])
            for i in range(len(cores))
        }

    # ── Gremlin Traversal ──────────────────────────────────────────────────

    def V(self, *vids):
        """Start Gremlin traversal from vertices."""
        return self._gremlin.V(*vids)

    def E(self, *eids):
        """Start Gremlin traversal from edges."""
        return self._gremlin.E(*eids)

    # ── Query Methods ──────────────────────────────────────────────────────

    def find_trades_by_signal(self, signal_type: str) -> List[str]:
        """Find all trades triggered by a signal type."""
        return [
            self._idx_to_id[succ]
            for vid, data in self._nodes.items()
            if data.get("signal_type") == signal_type
            for edge in self._gremlin._edges
            if edge["src"] == vid and edge["label"] == "executed_as"
            for succ in [edge["dst"]]
            if self._nodes.get(succ, {}).get("node_type") == "trade"
        ]

    def find_trades_in_regime(self, regime_type: str) -> List[str]:
        """Find all trades that occurred in a regime."""
        trades = []
        for vid, data in self._nodes.items():
            if data.get("regime_type") == regime_type:
                for edge in self._gremlin._edges:
                    if edge["dst"] == vid and edge["label"] == "occurred_in":
                        src = edge["src"]
                        if self._nodes.get(src, {}).get("node_type") == "trade":
                            trades.append(self._idx_to_id[src])
        return trades

    def get_trade_pnl_stats(self, regime_type: str = None) -> Dict:
        """PnL statistics, optionally filtered by regime."""
        pnl_values = []
        for vid, data in self._nodes.items():
            if data.get("node_type") != "trade":
                continue
            if regime_type:
                in_regime = any(
                    self._nodes.get(edge["dst"], {}).get("regime_type") == regime_type
                    for edge in self._gremlin._edges
                    if edge["src"] == vid and edge["label"] == "occurred_in"
                )
                if not in_regime:
                    continue
            pnl_values.append(data.get("pnl_pct", 0))

        if not pnl_values:
            return {"count": 0}
        return {
            "count": len(pnl_values),
            "mean_pnl": statistics.mean(pnl_values),
            "median_pnl": statistics.median(pnl_values),
            "std_pnl": statistics.stdev(pnl_values) if len(pnl_values) > 1 else 0,
            "total_pnl": sum(pnl_values),
            "win_rate": sum(1 for p in pnl_values if p > 0) / len(pnl_values) * 100,
        }

    def find_similar_setups(self, current_r: float, current_theta: float,
                            tolerance: float = 0.1) -> List[Dict]:
        """Find similar historical price setups via polar coordinates."""
        similar = []
        for vid, data in self._nodes.items():
            if data.get("node_type") != "price":
                continue
            r = data.get("polar_r")
            theta = data.get("polar_theta")
            if r is None or theta is None:
                continue
            r_diff = abs(r - current_r) / current_r if current_r > 0 else abs(r - current_r)
            theta_diff = abs(theta - current_theta)
            if r_diff < tolerance and theta_diff < tolerance:
                similar.append({
                    "node_id": self._idx_to_id[vid],
                    "timestamp": data.get("timestamp"),
                    "r": r,
                    "theta": theta,
                    "r_diff": r_diff,
                    "theta_diff": theta_diff,
                })
        similar.sort(key=lambda x: x["r_diff"] + x["theta_diff"])
        return similar

    def get_graph_stats(self) -> Dict:
        """Full graph statistics including GPU analysis."""
        node_counts: Dict[str, int] = {}
        for data in self._nodes.values():
            t = data.get("node_type", "unknown")
            node_counts[t] = node_counts.get(t, 0) + 1

        n_nodes = len(self._nodes)
        n_edges = len(self._gremlin._edges)
        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

        # GPU analysis
        pr_top = {}
        triangles = 0
        cores = {}
        try:
            pr_top = self.pagerank()
            triangles = self.count_triangles()
            cores = self.core_numbers()
        except Exception:
            pass

        # Top 5 by PageRank
        top_5 = sorted(pr_top.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_nodes": n_nodes,
            "total_edges": n_edges,
            "node_types": node_counts,
            "density": density,
            "pagerank_top": [{"node": k, "score": v} for k, v in top_5],
            "triangle_count": triangles,
            "core_numbers_summary": cores,
            "backend": "tilelang_gpu",
        }

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._gremlin._edges)

    def get_node_data(self, node_id: str) -> Dict[str, Any]:
        idx = self._id_to_idx.get(node_id)
        return self._nodes.get(idx, {})

    def has_edge(self, src: str, dst: str) -> bool:
        si = self._id_to_idx.get(src)
        di = self._id_to_idx.get(dst)
        if si is None or di is None:
            return False
        return any(e["src"] == si and e["dst"] == di for e in self._gremlin._edges)

    def export_to_dict(self) -> Dict:
        nodes = [{"id": self._idx_to_id[vid], **data} for vid, data in self._nodes.items()]
        edges = [dict(e) for e in self._gremlin._edges]
        return {"nodes": nodes, "edges": edges}

    def import_from_dict(self, data: Dict) -> None:
        self.__init__()
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            node_type = node.get("node_type", "")
            ts = node.get("timestamp", node.get("start_timestamp", 0))
            idx = self._next_idx(node_id)
            self._nodes[idx] = {"node_type": node_type, "timestamp": ts, **node}
        for edge in data.get("edges", []):
            self._gremlin._edges.append(edge)
        self._needs_rebuild = True
