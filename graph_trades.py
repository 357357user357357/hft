"""Graph-based trade analysis using rustworkx (Apache-2.0, Rust+PyO3).

Represents trades, signals, and price data as a directed graph for:
- Pattern matching (find similar historical setups)
- Relationship tracking (which signals led to which trades)
- Query by traversal (find all trades in a regime)

Uses rustworkx (IBM/Qiskit, Apache-2.0) backed by petgraph in Rust.
Falls back to networkx if rustworkx is not installed.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import statistics

try:
    import rustworkx as rx
    _RX_AVAILABLE = True
except ImportError:
    import networkx as nx
    _RX_AVAILABLE = False


class TradeGraph:
    """Directed graph for trade analysis.

    Internally uses rustworkx.PyDiGraph (Rust-backed, ~3x faster than
    networkx for traversal-heavy queries).  A string-ID ↔ integer-index
    mapping preserves the original API surface.
    """

    def __init__(self):
        if _RX_AVAILABLE:
            self._g = rx.PyDiGraph()
            self._id_to_idx: Dict[str, int] = {}
            self._idx_to_id: Dict[int, str] = {}
        else:
            self._g = nx.DiGraph()
        self._node_counter = 0

    # ── internal helpers ───────────────────────────────────────────────────

    def _next_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def _add_node(self, node_id: str, data: Dict[str, Any]) -> str:
        if _RX_AVAILABLE:
            idx = self._g.add_node(data)
            self._id_to_idx[node_id] = idx
            self._idx_to_id[idx] = node_id
        else:
            self._g.add_node(node_id, **data)
        return node_id

    def _node_data(self, node_id: str) -> Dict[str, Any]:
        if _RX_AVAILABLE:
            return self._g[self._id_to_idx[node_id]]
        return dict(self._g.nodes[node_id])

    def _iter_nodes(self):
        """Yield (node_id, data_dict) for every node."""
        if _RX_AVAILABLE:
            for idx in self._g.node_indices():
                yield self._idx_to_id[idx], self._g[idx]
        else:
            yield from self._g.nodes(data=True)

    def _successor_ids(self, node_id: str) -> List[str]:
        if _RX_AVAILABLE:
            idx = self._id_to_idx[node_id]
            return [self._idx_to_id[i] for i in self._g.successor_indices(idx)]
        return list(self._g.successors(node_id))

    def _predecessor_ids(self, node_id: str) -> List[str]:
        if _RX_AVAILABLE:
            idx = self._id_to_idx[node_id]
            return [self._idx_to_id[i] for i in self._g.predecessor_indices(idx)]
        return list(self._g.predecessors(node_id))

    # ── Node Creation ──────────────────────────────────────────────────────

    def add_price_node(self, timestamp: int, price: float,
                       polar_r: float = None, polar_theta: float = None,
                       **kwargs) -> str:
        node_id = self._next_id("price")
        return self._add_node(node_id, {
            "node_type": "price",
            "timestamp": timestamp,
            "price": price,
            "polar_r": polar_r,
            "polar_theta": polar_theta,
            **kwargs,
        })

    def add_signal_node(self, timestamp: int, signal_type: str,
                        strength: float, regime: str = None,
                        **kwargs) -> str:
        node_id = self._next_id("signal")
        return self._add_node(node_id, {
            "node_type": "signal",
            "timestamp": timestamp,
            "signal_type": signal_type,
            "strength": strength,
            "regime": regime,
            **kwargs,
        })

    def add_trade_node(self, timestamp: int, side: str, entry: float,
                       exit: float, pnl_pct: float, size: float,
                       exit_reason: str = None, **kwargs) -> str:
        node_id = self._next_id("trade")
        return self._add_node(node_id, {
            "node_type": "trade",
            "timestamp": timestamp,
            "side": side,
            "entry_price": entry,
            "exit_price": exit,
            "pnl_pct": pnl_pct,
            "size_usdt": size,
            "exit_reason": exit_reason,
            **kwargs,
        })

    def add_regime_node(self, start_ts: int, regime_type: str,
                        end_ts: int = None, **kwargs) -> str:
        node_id = self._next_id("regime")
        return self._add_node(node_id, {
            "node_type": "regime",
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "regime_type": regime_type,
            **kwargs,
        })

    # ── Edge Creation ──────────────────────────────────────────────────────

    def _add_edge(self, src: str, dst: str, data: Dict[str, Any]) -> None:
        if _RX_AVAILABLE:
            self._g.add_edge(self._id_to_idx[src], self._id_to_idx[dst], data)
        else:
            self._g.add_edge(src, dst, **data)

    def add_price_signal_edge(self, price_node: str, signal_node: str,
                              confidence: float = 1.0) -> None:
        self._add_edge(price_node, signal_node,
                       {"relation": "generated", "confidence": confidence})

    def add_signal_trade_edge(self, signal_node: str, trade_node: str,
                              delay_ms: float = None) -> None:
        self._add_edge(signal_node, trade_node,
                       {"relation": "executed_as", "delay_ms": delay_ms})

    def add_trade_regime_edge(self, trade_node: str, regime_node: str) -> None:
        self._add_edge(trade_node, regime_node, {"relation": "occurred_in"})

    def add_correlation_edge(self, node1: str, node2: str,
                             correlation: float) -> None:
        self._add_edge(node1, node2,
                       {"relation": "correlated_with", "weight": correlation})
        self._add_edge(node2, node1,
                       {"relation": "correlated_with", "weight": correlation})

    # ── Query Methods ──────────────────────────────────────────────────────

    def find_trades_by_signal(self, signal_type: str) -> List[str]:
        trades = []
        for node_id, data in self._iter_nodes():
            if data.get("signal_type") == signal_type:
                for succ_id in self._successor_ids(node_id):
                    if self._node_data(succ_id).get("node_type") == "trade":
                        trades.append(succ_id)
        return trades

    def find_trades_in_regime(self, regime_type: str) -> List[str]:
        trades = []
        for node_id, data in self._iter_nodes():
            if data.get("regime_type") == regime_type:
                for pred_id in self._predecessor_ids(node_id):
                    if self._node_data(pred_id).get("node_type") == "trade":
                        trades.append(pred_id)
        return trades

    def get_trade_pnl_stats(self, regime_type: str = None) -> Dict:
        pnl_values = []
        for node_id, data in self._iter_nodes():
            if data.get("node_type") != "trade":
                continue
            if regime_type:
                in_regime = any(
                    self._node_data(s).get("regime_type") == regime_type
                    for s in self._successor_ids(node_id)
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
        similar = []
        for node_id, data in self._iter_nodes():
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
                    "node_id": node_id,
                    "timestamp": data.get("timestamp"),
                    "r": r,
                    "theta": theta,
                    "r_diff": r_diff,
                    "theta_diff": theta_diff,
                })
        similar.sort(key=lambda x: x["r_diff"] + x["theta_diff"])
        return similar

    def get_graph_stats(self) -> Dict:
        node_counts: Dict[str, int] = {}
        for _, data in self._iter_nodes():
            t = data.get("node_type", "unknown")
            node_counts[t] = node_counts.get(t, 0) + 1

        if _RX_AVAILABLE:
            n_nodes = len(self._g)
            n_edges = self._g.num_edges()
            # Directed density = edges / (n*(n-1))
            density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
        else:
            n_nodes = self._g.number_of_nodes()
            n_edges = self._g.number_of_edges()
            import networkx as _nx
            density = _nx.density(self._g)

        return {
            "total_nodes": n_nodes,
            "total_edges": n_edges,
            "node_types": node_counts,
            "density": density,
            "backend": "rustworkx" if _RX_AVAILABLE else "networkx",
        }

    def node_count(self) -> int:
        return len(self._g) if _RX_AVAILABLE else self._g.number_of_nodes()

    def edge_count(self) -> int:
        return self._g.num_edges() if _RX_AVAILABLE else self._g.number_of_edges()

    def get_node_data(self, node_id: str) -> Dict[str, Any]:
        return self._node_data(node_id)

    def has_edge(self, src: str, dst: str) -> bool:
        if _RX_AVAILABLE:
            si, di = self._id_to_idx.get(src), self._id_to_idx.get(dst)
            if si is None or di is None:
                return False
            return self._g.has_edge(si, di)
        return self._g.has_edge(src, dst)

    def export_to_dict(self) -> Dict:
        """Export graph to a serialisable dictionary."""
        nodes = []
        for node_id, data in self._iter_nodes():
            nodes.append({"id": node_id, **data})

        edges = []
        if _RX_AVAILABLE:
            for src_idx, dst_idx, edge_data in self._g.weighted_edge_list():
                edges.append({
                    "source": self._idx_to_id[src_idx],
                    "target": self._idx_to_id[dst_idx],
                    **(edge_data or {}),
                })
        else:
            for u, v, d in self._g.edges(data=True):
                edges.append({"source": u, "target": v, **d})

        return {"nodes": nodes, "edges": edges}

    def import_from_dict(self, data: Dict) -> None:
        """Import graph from a dictionary produced by export_to_dict."""
        self.__init__()
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            node_type = node.get("node_type", "")
            ts = node.get("timestamp", node.get("start_timestamp", 0))
            self._add_node(node_id, {"node_type": node_type, "timestamp": ts, **node})
        for edge in data.get("edges", []):
            src = edge.pop("source")
            dst = edge.pop("target")
            self._add_edge(src, dst, edge)
