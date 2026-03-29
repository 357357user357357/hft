"""Tests for data parsing and order book construction."""

import csv
import io
import tempfile
import zipfile
from pathlib import Path

from data import AggTrade, _parse_row, _parse_csv, load_agg_trades_csv, build_synthetic_book


# ── AggTrade parsing ──────────────────────────────────────────────────────────

def _sample_row(**overrides):
    row = {
        "agg_trade_id": "100",
        "price": "50000.50",
        "quantity": "0.001",
        "first_trade_id": "200",
        "last_trade_id": "201",
        "transact_time": "1700000000000",
        "is_buyer_maker": "true",
    }
    row.update(overrides)
    return row


def test_parse_row_basic():
    t = _parse_row(_sample_row())
    assert t.agg_trade_id == 100
    assert t.price == 50000.50
    assert t.quantity == 0.001
    assert t.first_trade_id == 200
    assert t.last_trade_id == 201
    assert t.transact_time == 1700000000000
    assert t.is_buyer_maker is True


def test_parse_row_qty_alias():
    """Binance uses 'qty' in some CSVs instead of 'quantity'."""
    row = _sample_row()
    row["qty"] = row.pop("quantity")
    t = _parse_row(row)
    assert t.quantity == 0.001


def test_parse_row_buyer_maker_variants():
    assert _parse_row(_sample_row(is_buyer_maker="true")).is_buyer_maker is True
    assert _parse_row(_sample_row(is_buyer_maker="True")).is_buyer_maker is True
    assert _parse_row(_sample_row(is_buyer_maker="1")).is_buyer_maker is True
    assert _parse_row(_sample_row(is_buyer_maker="false")).is_buyer_maker is False
    assert _parse_row(_sample_row(is_buyer_maker="0")).is_buyer_maker is False


def test_timestamp_ns():
    t = _parse_row(_sample_row(transact_time="1700000000000"))
    assert t.timestamp_ns() == 1700000000000 * 1_000_000


def test_quote_volume():
    t = _parse_row(_sample_row(price="100.0", quantity="2.5"))
    assert t.quote_volume() == 250.0


# ── CSV parsing ───────────────────────────────────────────────────────────────

def _csv_string(rows):
    """Build a CSV string from a list of dicts."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def test_parse_csv_multiple_rows():
    rows = [
        _sample_row(agg_trade_id="1", price="100.0"),
        _sample_row(agg_trade_id="2", price="101.0"),
        _sample_row(agg_trade_id="3", price="99.5"),
    ]
    trades = _parse_csv(io.StringIO(_csv_string(rows)))
    assert len(trades) == 3
    assert trades[0].price == 100.0
    assert trades[2].price == 99.5


def test_load_csv_plain(tmp_path):
    rows = [_sample_row(agg_trade_id=str(i)) for i in range(5)]
    csv_file = tmp_path / "trades.csv"
    csv_file.write_text(_csv_string(rows))
    trades = load_agg_trades_csv(csv_file)
    assert len(trades) == 5


def test_load_csv_zip(tmp_path):
    rows = [_sample_row(agg_trade_id=str(i)) for i in range(3)]
    csv_data = _csv_string(rows)
    zip_path = tmp_path / "trades.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("inner.csv", csv_data)
    trades = load_agg_trades_csv(zip_path)
    assert len(trades) == 3


# ── Synthetic order book ──────────────────────────────────────────────────────

def _make_trades(prices, base_time=1700000000000):
    return [
        AggTrade(
            agg_trade_id=i,
            price=p,
            quantity=1.0,
            first_trade_id=i,
            last_trade_id=i,
            transact_time=base_time + i * 100,
            is_buyer_maker=False,
        )
        for i, p in enumerate(prices)
    ]


def test_build_synthetic_book_structure():
    trades = _make_trades([100.0, 100.5, 101.0])
    book = build_synthetic_book(trades, levels=5, tick_size=0.1)
    assert len(book.bids) == 5
    assert len(book.asks) == 5
    # Bids below last price, asks above
    assert all(p < 101.0 for p, _ in book.bids)
    assert all(p > 101.0 for p, _ in book.asks)
    # Bids descending, asks ascending
    assert book.bids[0][0] > book.bids[-1][0]
    assert book.asks[0][0] < book.asks[-1][0]


def test_build_synthetic_book_mid_price():
    trades = _make_trades([100.0])
    book = build_synthetic_book(trades, levels=3, tick_size=1.0)
    mid = book.mid_price()
    assert mid is not None
    assert abs(mid - 100.0) < 2.0  # mid should be near last trade price


def test_build_synthetic_book_empty():
    book = build_synthetic_book([], levels=5, tick_size=0.1)
    assert book.bids == []
    assert book.asks == []
    assert book.mid_price() is None
