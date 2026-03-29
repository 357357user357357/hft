"""Data loading utilities for Binance aggTrades CSV files
Base URL: https://data.binance.vision/?prefix=data/futures/um/monthly/aggTrades
"""

import csv
import gzip
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import math


@dataclass
class AggTrade:
    """Binance aggTrades CSV row
    Columns: agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker
    """
    agg_trade_id: int
    price: float
    quantity: float
    first_trade_id: int
    last_trade_id: int
    transact_time: int   # Unix timestamp in milliseconds
    is_buyer_maker: bool  # true = seller is maker (price went down)

    def timestamp_ns(self) -> int:
        """Timestamp in nanoseconds"""
        return self.transact_time * 1_000_000

    def quote_volume(self) -> float:
        """Quote asset volume (price × quantity)"""
        return self.price * self.quantity


def _parse_row(row: dict) -> AggTrade:
    # Handle both 'quantity' and 'qty' column names
    qty_key = 'qty' if 'qty' in row else 'quantity'
    return AggTrade(
        agg_trade_id=int(row['agg_trade_id']),
        price=float(row['price']),
        quantity=float(row[qty_key]),
        first_trade_id=int(row['first_trade_id']),
        last_trade_id=int(row['last_trade_id']),
        transact_time=int(row['transact_time']),
        is_buyer_maker=row['is_buyer_maker'].strip().lower() in ('true', '1'),
    )


def _parse_csv(fileobj) -> List[AggTrade]:
    reader = csv.DictReader(fileobj)
    trades = []
    for i, row in enumerate(reader):
        try:
            trades.append(_parse_row(row))
        except (KeyError, ValueError) as e:
            raise ValueError(f"Malformed CSV row {i + 1}: {e}. Columns found: {list(row.keys())}") from e
    return trades


def load_agg_trades_csv(path: Path) -> List[AggTrade]:
    """Load aggTrades from a CSV file (possibly gzip-compressed or zip)"""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == '.zip':
        with zipfile.ZipFile(path, 'r') as archive:
            names = archive.namelist()
            if not names:
                raise ValueError(f"Zip archive is empty: {path}")
            with archive.open(names[0]) as inner:
                text = io.TextIOWrapper(inner, encoding='utf-8')
                return _parse_csv(text)
    elif ext == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return _parse_csv(f)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return _parse_csv(f)


@dataclass
class SimpleOrderBook:
    """Minimal order book snapshot for DepthShot simulation"""
    bids: List  # (price, qty) sorted descending
    asks: List  # (price, qty) sorted ascending

    def __init__(self):
        self.bids = []
        self.asks = []

    def cumulative_bid_volume(self, depth: int) -> float:
        return sum(p * q for p, q in self.bids[:depth])

    def cumulative_ask_volume(self, depth: int) -> float:
        return sum(p * q for p, q in self.asks[:depth])

    def bid_price_at_volume(self, target_volume: float) -> Optional[float]:
        cum = 0.0
        for price, qty in self.bids:
            cum += price * qty
            if cum >= target_volume:
                return price
        return None

    def ask_price_at_volume(self, target_volume: float) -> Optional[float]:
        cum = 0.0
        for price, qty in self.asks:
            cum += price * qty
            if cum >= target_volume:
                return price
        return None

    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    def mid_price(self) -> Optional[float]:
        b = self.best_bid()
        a = self.best_ask()
        if b is not None and a is not None:
            return (b + a) / 2.0
        return None


def build_synthetic_book(recent_trades: List[AggTrade], levels: int, tick_size: float) -> SimpleOrderBook:
    """Build a synthetic order book from recent trades"""
    book = SimpleOrderBook()
    if not recent_trades:
        return book

    last_price = recent_trades[-1].price
    total_vol = sum(t.quantity for t in recent_trades)
    vol_per_level = total_vol / levels

    for i in range(1, levels + 1):
        bid_price = last_price - tick_size * i
        ask_price = last_price + tick_size * i
        qty = vol_per_level * math.exp(-0.3 * i)
        book.bids.append((bid_price, qty))
        book.asks.append((ask_price, qty))

    return book
