"""Signal cache for faster backtests.

Caches computed scorecards by timestamp window to avoid recomputing
signals on overlapping windows.

Usage:
    from signal_cache import SignalCache

    cache = SignalCache(window_size=60)
    card = cache.get_or_compute(indexer, symbol, prices, volumes, timestamp)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from instrument_index import InstrumentIndexer, InstrumentScorecard


@dataclass
class CacheEntry:
    """Cached scorecard with metadata."""
    scorecard: InstrumentScorecard
    timestamp: float
    price_hash: str
    access_count: int = 0
    last_access: float = 0.0


class SignalCache:
    """
    LRU cache for instrument scorecards.

    The cache key is a hash of the price/volume window, so identical
    windows (common in overlapping backtest windows) hit the cache.
    """

    def __init__(self, max_size: int = 1000, ttl_secs: float = 300.0):
        self.max_size = max_size
        self.ttl_secs = ttl_secs
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._hits = 0
        self._misses = 0

    def get_or_compute(
        self,
        indexer: InstrumentIndexer,
        symbol: str,
        prices: List[float],
        volumes: Optional[List[float]],
        timestamp: Optional[float] = None,
    ) -> InstrumentScorecard:
        """Get scorecard from cache or compute and cache it."""
        key = self._make_key(prices, volumes)
        now = timestamp or time.time()

        # Check cache hit
        entry = self._cache.get(key)
        if entry is not None:
            # Check TTL
            if now - entry.timestamp < self.ttl_secs:
                entry.access_count += 1
                entry.last_access = now
                self._touch_key(key)
                self._hits += 1
                return entry.scorecard

        # Cache miss - compute
        self._misses += 1
        scorecard = indexer.update(symbol, prices, volumes)

        # Store in cache
        self._store(key, scorecard, now)
        return scorecard

    def _make_key(self, prices: List[float], volumes: Optional[List[float]]) -> str:
        """Make cache key from price/volume window."""
        # Quantize prices to reduce hash collisions from floating point
        quantize = lambda x: round(x, 6)
        price_str = ",".join(f"{quantize(p):.8f}" for p in prices[-60:])  # Last 60 bars
        if volumes:
            vol_str = ",".join(f"{quantize(v):.4f}" for v in volumes[-60:])
        else:
            vol_str = "novol"
        combined = f"{price_str}|{vol_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _store(self, key: str, scorecard: InstrumentScorecard, timestamp: float) -> None:
        """Store scorecard in cache."""
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            self._evict()

        entry = CacheEntry(
            scorecard=scorecard,
            timestamp=timestamp,
            price_hash=key,
            access_count=1,
            last_access=timestamp,
        )
        self._cache[key] = entry
        self._access_order.append(key)

    def _touch_key(self, key: str) -> None:
        """Update access order for LRU."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict(self) -> None:
        """Evict oldest entry."""
        if not self._access_order:
            return
        oldest_key = self._access_order.pop(0)
        if oldest_key in self._cache:
            del self._cache[oldest_key]

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    def print_stats(self) -> None:
        """Print cache statistics."""
        s = self.stats()
        print(f"Cache: {s['size']}/{s['max_size']} entries, "
              f"hits={s['hits']}, misses={s['misses']}, "
              f"hit_rate={s['hit_rate']*100:.1f}%")


class CachedSignalBacktest:
    """
    Signal backtest with caching for faster execution.

    Drop-in replacement for SignalBacktest that uses SignalCache
    to avoid recomputing signals on overlapping windows.
    """

    def __init__(self, config, cache_size: int = 500):
        self.config = config
        self._indexer = InstrumentIndexer()
        self._cache = SignalCache(max_size=cache_size)
        self._stats = None

    def run(self, prices: List[float], volumes: Optional[List[float]] = None):
        """Run cached signal backtest."""
        # Import here to avoid circular import
        from signal_backtest import SignalBacktest, SignalStats

        # Use regular backtest but with cached indexer
        backtest = SignalBacktest(self.config)
        backtest._indexer = self._indexer

        # Wrap indexer.update to use cache
        original_update = self._indexer.update

        def cached_update(symbol, p, v):
            return self._cache.get_or_compute(self._indexer, symbol, p, v, time.time())

        self._indexer.update = cached_update
        stats = backtest.run(prices, volumes)
        self._indexer.update = original_update
        self._stats = stats

        return stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()


if __name__ == "__main__":
    # Demo cache effectiveness
    from instrument_index import InstrumentIndexer
    import random

    random.seed(42)
    prices = [100 + random.gauss(0, 1) for _ in range(100)]
    volumes = [random.uniform(100, 1000) for _ in range(len(prices))]

    indexer = InstrumentIndexer()
    cache = SignalCache(max_size=100)

    print("Testing cache with overlapping windows...")
    window_size = 50

    # Without cache
    t0 = time.time()
    for i in range(window_size, len(prices)):
        window = prices[i-window_size:i+1]
        indexer.update("TEST", window, volumes[i-window_size:i+1])
    t1 = time.time()
    print(f"Without cache: {t1-t0:.2f}s")

    # With cache (overlapping windows should hit)
    t0 = time.time()
    for i in range(window_size, len(prices)):
        window = prices[i-window_size:i+1]
        cache.get_or_compute(indexer, "TEST", window, volumes[i-window_size:i+1])
    t1 = time.time()
    print(f"With cache: {t1-t0:.2f}s")
    cache.print_stats()
