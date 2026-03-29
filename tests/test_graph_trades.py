"""Tests for rustworkx-backed trade graph."""

from graph_trades import TradeGraph


def test_graph_basic():
    g = TradeGraph()
    assert g.node_count() == 0


def test_add_price_node():
    g = TradeGraph()
    node_id = g.add_price_node(
        timestamp=1700000000000,
        price=50000.0,
        polar_r=1.05,
        polar_theta=0.3,
    )
    assert node_id.startswith("price_")
    data = g.get_node_data(node_id)
    assert data["price"] == 50000.0
    assert data["polar_r"] == 1.05
    assert data["polar_theta"] == 0.3


def test_add_signal_node():
    g = TradeGraph()
    node_id = g.add_signal_node(
        timestamp=1700000000000,
        signal_type="MEAN_REVERT_LONG",
        strength=0.8,
        regime="mean-reversion",
    )
    assert node_id.startswith("signal_")
    data = g.get_node_data(node_id)
    assert data["signal_type"] == "MEAN_REVERT_LONG"
    assert data["strength"] == 0.8


def test_add_trade_node():
    g = TradeGraph()
    node_id = g.add_trade_node(
        timestamp=1700000000000,
        side="Buy",
        entry=50000.0,
        exit=50100.0,
        pnl_pct=0.2,
        size=100.0,
        exit_reason="TakeProfit",
    )
    assert node_id.startswith("trade_")
    data = g.get_node_data(node_id)
    assert data["entry_price"] == 50000.0
    assert data["pnl_pct"] == 0.2


def test_add_edges():
    g = TradeGraph()
    price_id = g.add_price_node(1000, 50000.0, polar_r=1.0, polar_theta=0.1)
    signal_id = g.add_signal_node(1001, "MEAN_REVERT_LONG", 0.8)
    trade_id = g.add_trade_node(1002, "Buy", 50000.0, 50100.0, 0.2, 100.0)

    g.add_price_signal_edge(price_id, signal_id, confidence=0.9)
    g.add_signal_trade_edge(signal_id, trade_id, delay_ms=50)

    assert g.edge_count() == 2
    assert g.has_edge(price_id, signal_id)
    assert g.has_edge(signal_id, trade_id)


def test_find_trades_by_signal():
    g = TradeGraph()
    signal1 = g.add_signal_node(1000, "MEAN_REVERT_LONG", 0.8)
    signal2 = g.add_signal_node(2000, "TREND_LONG", 0.9)
    trade1 = g.add_trade_node(1001, "Buy", 50000.0, 50100.0, 0.2, 100.0)
    trade2 = g.add_trade_node(2001, "Buy", 50000.0, 50200.0, 0.4, 100.0)
    g.add_signal_trade_edge(signal1, trade1)
    g.add_signal_trade_edge(signal2, trade2)

    mr_trades = g.find_trades_by_signal("MEAN_REVERT_LONG")
    assert len(mr_trades) == 1
    assert mr_trades[0] == trade1


def test_get_trade_pnl_stats():
    g = TradeGraph()
    g.add_trade_node(1000, "Buy", 50000.0, 50100.0, 0.2, 100.0)
    g.add_trade_node(2000, "Buy", 50000.0, 49900.0, -0.2, 100.0)
    g.add_trade_node(3000, "Buy", 50000.0, 50300.0, 0.6, 100.0)

    stats = g.get_trade_pnl_stats()
    assert stats["count"] == 3
    assert abs(stats["mean_pnl"] - 0.2) < 0.001
    assert abs(stats["total_pnl"] - 0.6) < 0.001
    assert abs(stats["win_rate"] - 66.67) < 0.1


def test_find_similar_setups():
    g = TradeGraph()
    g.add_price_node(1000, 50000.0, polar_r=1.0, polar_theta=0.1)
    g.add_price_node(2000, 50000.0, polar_r=1.05, polar_theta=0.15)
    g.add_price_node(3000, 50000.0, polar_r=2.0, polar_theta=3.0)

    similar = g.find_similar_setups(current_r=1.02, current_theta=0.12, tolerance=0.1)
    assert len(similar) >= 2
    assert similar[0]["r"] in [1.0, 1.05]


def test_graph_stats():
    g = TradeGraph()
    g.add_price_node(1000, 50000.0)
    g.add_signal_node(1001, "MEAN_REVERT_LONG", 0.8)
    g.add_trade_node(1002, "Buy", 50000.0, 50100.0, 0.2, 100.0)

    stats = g.get_graph_stats()
    assert stats["total_nodes"] == 3
    assert stats["node_types"]["price"] == 1
    assert stats["node_types"]["signal"] == 1
    assert stats["node_types"]["trade"] == 1


def test_export_import():
    g1 = TradeGraph()
    g1.add_price_node(1000, 50000.0, polar_r=1.0, polar_theta=0.1)
    g1.add_signal_node(1001, "MEAN_REVERT_LONG", 0.8)

    data = g1.export_to_dict()

    g2 = TradeGraph()
    g2.import_from_dict(data)
    assert g2.node_count() == 2
    assert g2.edge_count() == 0


def test_regime_filtering():
    g = TradeGraph()
    regime_mr = g.add_regime_node(1000, "MEAN_REVERSION", end_ts=5000)
    regime_trend = g.add_regime_node(5001, "TRENDING", end_ts=10000)
    trade1 = g.add_trade_node(2000, "Buy", 50000.0, 50100.0, 0.2, 100.0)
    trade2 = g.add_trade_node(6000, "Buy", 50000.0, 50300.0, 0.6, 100.0)
    g.add_trade_regime_edge(trade1, regime_mr)
    g.add_trade_regime_edge(trade2, regime_trend)

    mr_trades = g.find_trades_in_regime("MEAN_REVERSION")
    assert len(mr_trades) == 1
    assert mr_trades[0] == trade1

    trend_trades = g.find_trades_in_regime("TRENDING")
    assert len(trend_trades) == 1
    assert trend_trades[0] == trade2
