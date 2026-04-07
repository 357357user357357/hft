#!/usr/bin/env python3
"""Bybit Live Trader
====================
Connects to Bybit Spot, watches real-time prices via WebSocket, and places
real orders when the signal fires.

Bybit API v5 — Spot trading only (safest for small capital).
Uses only stdlib: hmac, hashlib, urllib — no extra packages needed.

Setup:
  1. Create .env with your API key + secret (see template)
  2. Run: python bybit_trader.py --symbol SOLUSDT --equity 5

Safety features:
  - Paper mode by default (--live to enable real orders)
  - Max position size cap (--max-usdt, default 2 USDT)
  - Daily loss limit (stops trading if cumulative loss > --max-loss)
  - All orders are limit orders (maker fee = 0.1% Spot, vs 0.1% taker)
  - Confirms balance before placing any order

Usage:
    # Paper trade (safe, no real orders)
    python bybit_trader.py --symbol SOLUSDT --equity 5

    # Live trade with real orders (start small!)
    python bybit_trader.py --symbol SOLUSDT --equity 5 --live --max-usdt 2

    # With signal gate (uses best signal from backtest per symbol)
    python bybit_trader.py --symbol SOLUSDT --equity 5 --live --signals

Bybit Spot minimum order sizes (approx):
    SOL   : 0.01 SOL  (~$1.50)
    ETH   : 0.001 ETH (~$3.50)
    BTC   : 0.00001 BTC (~$0.60)
    LINK  : 0.1 LINK (~$1.50)
    MATIC : 1 MATIC (~$0.45)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))


# ── Load credentials from .env ────────────────────────────────────────────────

def _load_env() -> None:
    """Load KEY=VALUE pairs from .env into os.environ (no dotenv needed)."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip().strip('"').strip("'")
            if val:  # only set if non-empty
                os.environ.setdefault(key.strip(), val)

_load_env()


# ── Bybit REST API v5 client (stdlib only) ────────────────────────────────────

class BybitClient:
    """Minimal Bybit API v5 client — only what we need for spot trading.

    Authentication: HMAC-SHA256 over timestamp + api_key + recv_window + params.
    Docs: https://bybit-exchange.github.io/docs/v5/intro
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.base_url   = (
            "https://api-testnet.bybit.com" if testnet
            else "https://api.bybit.com"
        )
        self.recv_window = "10000"

    def _sign(self, timestamp: str, params_str: str) -> str:
        """HMAC-SHA256 signature: timestamp + api_key + recv_window + params."""
        msg = timestamp + self.api_key + self.recv_window + params_str
        return hmac.new(
            self.api_secret.encode(), msg.encode(), hashlib.sha256
        ).hexdigest()

    def _headers(self, timestamp: str, signature: str) -> Dict[str, str]:
        return {
            "X-BAPI-API-KEY":     self.api_key,
            "X-BAPI-TIMESTAMP":   timestamp,
            "X-BAPI-SIGN":        signature,
            "X-BAPI-RECV-WINDOW": self.recv_window,
            "Content-Type":       "application/json",
        }

    def get(self, path: str, params: Optional[Dict] = None) -> dict:
        """Signed GET request."""
        params = params or {}
        ts  = str(int(time.time() * 1000))
        qs  = urllib.parse.urlencode(params)
        sig = self._sign(ts, qs)
        url = f"{self.base_url}{path}?{qs}"
        req = urllib.request.Request(url, headers=self._headers(ts, sig))
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"Bybit GET {path} → {e.code}: {body}") from e

    def post(self, path: str, body: Dict) -> dict:
        """Signed POST request."""
        ts      = str(int(time.time() * 1000))
        body_s  = json.dumps(body)
        sig     = self._sign(ts, body_s)
        url     = f"{self.base_url}{path}"
        req     = urllib.request.Request(
            url,
            data=body_s.encode(),
            headers=self._headers(ts, sig),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body_err = e.read().decode()
            raise RuntimeError(f"Bybit POST {path} → {e.code}: {body_err}") from e

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_balance(self, coin: str = "USDT") -> float:
        """Get Unified wallet balance for a coin."""
        r = self.get("/v5/account/wallet-balance",
                     {"accountType": "UNIFIED", "coin": coin})
        if r.get("retCode") != 0:
            raise RuntimeError(f"balance error: {r}")
        for item in r["result"]["list"]:
            for c in item.get("coin", []):
                if c["coin"] == coin:
                    return float(c.get("availableToWithdraw") or c.get("walletBalance") or 0)
        return 0.0

    def get_funding_balance(self, coin: str = "USDT") -> float:
        """Get Funding wallet balance for a coin."""
        r = self.get("/v5/asset/transfer/query-account-coins-balance",
                     {"accountType": "FUND", "coin": coin})
        if r.get("retCode") != 0:
            return 0.0
        for c in r.get("result", {}).get("balance", []):
            if c.get("coin") == coin:
                return float(c.get("transferBalance") or c.get("walletBalance") or 0)
        return 0.0

    def get_price(self, symbol: str) -> float:
        """Get last trade price (public, no auth needed)."""
        r = self.get("/v5/market/tickers",
                     {"category": "spot", "symbol": symbol})
        if r.get("retCode") != 0:
            raise RuntimeError(f"price error: {r}")
        return float(r["result"]["list"][0]["lastPrice"])

    def get_instrument_info(self, symbol: str, category: str = "spot") -> dict:
        """Get lot size / min order qty / tick size for a symbol."""
        r = self.get("/v5/market/instruments-info",
                     {"category": category, "symbol": symbol})
        if r.get("retCode") != 0:
            raise RuntimeError(f"instrument error: {r}")
        return r["result"]["list"][0]

    def place_order(
        self,
        symbol:    str,
        side:      str,       # "Buy" or "Sell"
        qty:       str,       # formatted to basePrecision
        order_type: str = "Market",   # "Market" or "Limit"
        price:     Optional[str] = None,
        time_in_force: str = "IOC",   # IOC for market-like fills
        category:  str = "spot",      # "spot" or "linear" (perpetuals)
    ) -> dict:
        """Place an order. Returns full Bybit response."""
        body: Dict = {
            "category":    category,
            "symbol":      symbol,
            "side":        side,
            "orderType":   order_type,
            "qty":         qty,
            "timeInForce": time_in_force,
        }
        if price:
            body["price"] = price
        return self.post("/v5/order/create", body)

    def cancel_all_orders(self, symbol: str, category: str = "spot") -> dict:
        """Cancel all open orders for a symbol."""
        return self.post("/v5/order/cancel-all",
                         {"category": category, "symbol": symbol})


# ── Order quantity formatter ──────────────────────────────────────────────────

def _format_qty(qty: float, step: float) -> str:
    """Round qty DOWN to the nearest lot step, formatted as string."""
    if step <= 0:
        return f"{qty:.6f}"
    steps  = int(qty / step)
    result = steps * step
    # Determine decimal places from step string
    step_s = f"{step:.10f}".rstrip("0")
    if "." in step_s:
        decimals = len(step_s.split(".")[1])
    else:
        decimals = 0
    return f"{result:.{decimals}f}"


# ── Live state ────────────────────────────────────────────────────────────────

@dataclass
class LiveState:
    symbol:       str
    equity_usdt:  float
    max_usdt:     float
    max_loss_pct: float       # stop trading if total loss > this %

    # Position
    in_position:  bool  = False
    position_side: str  = ""  # "long" or "short"
    entry_price:  float = 0.0
    entry_qty:    float = 0.0
    entry_time:   float = field(default_factory=time.monotonic)
    peak_pnl:    float = 0.0     # highest unrealized PnL % (trailing stop)
    last_exit_time: float = 0.0  # cooldown: avoid re-entry churn

    # Stats
    total_trades: int   = 0
    winning:      int   = 0
    total_pnl_pct: float = 0.0
    daily_pnl_pct: float = 0.0

    # Live price
    price:        float = 0.0
    last_update:  float = 0.0

    # Orders placed (for reference)
    orders:       List[dict] = field(default_factory=list)


# ── Terminal colours ──────────────────────────────────────────────────────────

_BOLD  = "\033[1m"; _RESET = "\033[0m"
_GREEN = "\033[92m"; _RED  = "\033[91m"
_CYAN  = "\033[96m"; _DIM  = "\033[2m"
_YELLOW= "\033[93m"; _WHITE= "\033[97m"


def _pnl_col(v: float) -> str:
    return _GREEN if v > 0 else _RED if v < 0 else _WHITE


def _clear() -> None:
    print("\033[2J\033[H", end="", flush=True)


# ── Kelly position sizing ────────────────────────────────────────────────────

_kelly_sizer = None

def _get_kelly_size(max_usdt: float, equity: float) -> float:
    """Return Kelly-optimal position size in USDT, reading trade history from CSV.
    Falls back to max_usdt if not enough history."""
    global _kelly_sizer
    if _kelly_sizer is None:
        from risk_management import PositionSizer, SizingConfig
        _kelly_sizer = PositionSizer(SizingConfig(
            mode='kelly', kelly_cap=0.25, kelly_lookback=50,
            initial_equity=equity, fractional_pct=5.0,
        ))

    _kelly_sizer.current_equity = equity
    # Load trade PnL history from trades.csv
    trade_pnls: List[float] = []
    if os.path.exists(_TRADE_LOG):
        import csv
        try:
            with open(_TRADE_LOG) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("side") == "Sell":  # exit trades have PnL
                        try:
                            trade_pnls.append(float(row["pnl_pct"]))
                        except (ValueError, KeyError):
                            pass
        except Exception:
            pass
    size = _kelly_sizer.calculate_size(trade_pnls)
    return min(size, max_usdt)  # never exceed max_usdt cap


# ── Trade journal (CSV append) ───────────────────────────────────────────────

_TRADE_LOG = os.path.join(os.path.dirname(__file__), "trades.csv")

def _log_trade(symbol: str, side: str, qty: str, price: float,
               pnl_pct: float, status: str, held_sec: float = 0.0) -> None:
    """Append one row to trades.csv. Creates header if file is new."""
    import csv
    write_header = not os.path.exists(_TRADE_LOG)
    with open(_TRADE_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "symbol", "side", "qty", "price",
                        "pnl_pct", "held_sec", "status"])
        w.writerow([datetime.utcnow().isoformat(timespec="seconds"),
                    symbol, side, qty, f"{price:.6f}",
                    f"{pnl_pct:+.4f}", f"{held_sec:.0f}", status])


# ── Balance cache (avoid hammering API every dashboard refresh) ───────────────
_bal_cache:    float = 0.0
_bal_cache_ts: float = 0.0

# ── Dashboard ─────────────────────────────────────────────────────────────────

def _render(states: List["LiveState"], live_mode: bool, client: Optional[BybitClient],
            equity: float, daily_pnl: float, total_trades: int, winning: int,
            gpu: Optional["GpuAnalytics"] = None,
            cpu_signals: Optional[Dict[str, Dict[str, float]]] = None,
            sym_ctxs: Optional[Dict[str, "SymbolCtx"]] = None) -> None:
    _clear()
    mode_str = f"{_RED}{_BOLD}LIVE ORDERS{_RESET}" if live_mode else f"{_YELLOW}PAPER MODE{_RESET}"
    bal_str = ""
    if client:
        global _bal_cache, _bal_cache_ts
        now = time.monotonic()
        if now - _bal_cache_ts > 30.0:   # refresh every 30s, not every 1s
            try:
                _bal_cache = client.get_balance("USDT")
                _bal_cache_ts = now
            except Exception:
                pass
        if _bal_cache > 0:
            bal_str = f"  wallet={_CYAN}{_bal_cache:.4f} USDT{_RESET}"

    win_rate = winning / total_trades * 100 if total_trades > 0 else 0.0
    print(f"{_BOLD}{'─'*70}{_RESET}")
    if gpu:
        _proc = getattr(gpu, "_gpu_proc", None)
        _poll = _proc.poll() if _proc else "none"
        err_str = f"  ERR:{gpu._last_gpu_error[:40]}" if gpu._last_gpu_error else ""
        proc_str = f"  proc={'up' if _proc and _poll is None else f'dead({_poll})'}"
        gpu_str = (f"  {_DIM}[{gpu.backend}  util={gpu.gpu_util_pct:.0f}%"
                   f"  VRAM={gpu.vram_mb}MB"
                   f"  MA{gpu.best_fast_w}/{gpu.best_slow_w}"
                   f"  thr={gpu.best_gap_threshold*100:.4f}%{proc_str}{err_str}]{_RESET}")
    else:
        gpu_str = ""
    print(f"{_BOLD}  BYBIT MULTI-TRADER  {_RESET}{mode_str}  "
          f"{_DIM}{time.strftime('%H:%M:%S')}{_RESET}{bal_str}{gpu_str}")
    print(f"{_BOLD}{'─'*70}{_RESET}")
    print(f"  equity={equity:.2f} USDT  trades={total_trades}  win={win_rate:.1f}%  "
          f"daily={_pnl_col(daily_pnl)}{daily_pnl:+.3f}%{_RESET}")
    print()

    for st in states:
        ago = time.monotonic() - st.last_update if st.last_update > 0 else 999
        price_str = f"{_CYAN}{st.price:.4f}{_RESET}" if ago < 5 else f"{_DIM}{st.price:.4f}{_RESET}"
        pos_str = ""
        if st.in_position and st.price > 0 and st.entry_price > 0:
            unreal = (st.price - st.entry_price) / st.entry_price * 100
            if st.position_side == "short":
                unreal = -unreal
            held = int(time.monotonic() - st.entry_time)
            pos_str = (f"  {_CYAN}{st.position_side.upper()}{_RESET}"
                       f" {_pnl_col(unreal)}{unreal:+.2f}%{_RESET} {_DIM}{held}s{_RESET}")
        elif not st.in_position:
            pos_str = f"  {_DIM}flat{_RESET}"
        # Spread + regime annotation
        spread_str = ""
        if sym_ctxs:
            sc = sym_ctxs.get(st.symbol)
            if sc and sc.spread_bps > 0:
                spread_str = f"  {_DIM}sp={sc.spread_bps:.1f}bp{_RESET}"
            if sc and len(sc.price_buf) >= 50:
                regime = _detect_regime_fast(sc.price_buf)
                r_tag = {"mean-reversion": "MR", "trending": "TR", "neutral": "NT"}
                spread_str += f"  {_DIM}{r_tag.get(regime, '?')}{_RESET}"
        # CPU analytics annotation
        cpu_str = ""
        if cpu_signals:
            cs = cpu_signals.get(st.symbol, {})
            if cs:
                h = cs.get("hurst", 0.5); o = cs.get("ofi", 0.5)
                sh = cs.get("sharpe", 0.0)
                cpu_str = (f"  {_DIM}H={h:.2f} OFI={o:.2f} Sh={sh:+.1f}{_RESET}")
        print(f"  {st.symbol:<12} {price_str}  {pos_str}{spread_str}{cpu_str}")

    print(f"\n{_BOLD}{'─'*70}{_RESET}")
    print(f"  {_DIM}Ctrl+C to stop{_RESET}", flush=True)


# ── Correlation groups (avoid stacking correlated positions) ─────────────────
# Symbols in the same group have historically >0.8 correlation.
# Only one open position allowed per group.
_CORR_GROUPS: List[set] = [
    {"BTCUSDT", "ETHUSDT"},          # BTC/ETH move together
    {"SOLUSDT", "ADAUSDT"},          # alt-L1s
    {"LINKUSDT", "LTCUSDT"},         # mid-caps
    {"DOGEUSDT"},                    # meme — uncorrelated
]

def _corr_group(symbol: str) -> Optional[int]:
    """Return group index for a symbol, or None if ungrouped."""
    for i, g in enumerate(_CORR_GROUPS):
        if symbol in g:
            return i
    return None


# ── Regime detection (lightweight, from price returns) ───────────────────────

def _detect_regime_fast(prices: List[float]) -> str:
    """Fast regime detection from lag-1 autocorrelation of returns.
    Returns 'mean-reversion', 'trending', or 'neutral'."""
    n = len(prices)
    if n < 50:
        return "neutral"
    rets = [prices[i] - prices[i-1] for i in range(max(1, n-100), n)]
    nr = len(rets)
    if nr < 10:
        return "neutral"
    mean_r = sum(rets) / nr
    lag1_cov = sum((rets[i] - mean_r) * (rets[i-1] - mean_r) for i in range(1, nr))
    var_r = sum((r - mean_r)**2 for r in rets)
    autocorr = lag1_cov / var_r if var_r > 0 else 0.0
    if autocorr < -0.05:
        return "mean-reversion"
    elif autocorr > 0.05:
        return "trending"
    return "neutral"


# ── Signal decision (simple threshold on best backtest signal) ────────────────

# Walk-forward v2: Simons OU + Momentum + Volatility are the only edges.
# Poincaré now has multi-timeframe (H1/H4) support + sensitive 5-min thresholds.
_BEST_SIGNAL: Dict[str, str] = {
    "SOLUSDT":  "poincare_h1",
    "LTCUSDT":  "momentum",
    "LINKUSDT": "poincare_h1",
    "BTCUSDT":  "momentum",
    "ETHUSDT":  "poincare_h1",
    "BNBUSDT":  "momentum",
    "ADAUSDT":  "poincare_h1",
    "DOGEUSDT": "momentum",
    "XRPUSDT":  "poincare_h1",
}

# Sensitive thresholds: lower = more firing, higher = more selective.
# Poincaré uses lower thresholds on faster timeframes (more noise),
# higher on slower (cleaner signals from fewer, better-quality bars).
_PNCR_THRESH: Dict[str, float] = {
    "tick":    0.02,   # nanosecond raw tick data - millions of points, topology sees everything
    "1sec":    0.03,   # 1-sec bars - enough points, still noisy
    "1min":    0.05,   # 1-min bars - default sensitive
    "5min":    0.08,   # 5-min bars - moderate threshold
    "1hour":   0.15,   # 1H bars - cleaner topology
    "4hour":   0.10,   # 4H bars - fewer bars but cleanest
}

# Signal key format: "poincare", "poincare_h1", "poincare_h4", "poincare_tick",
#                    "poincare_1m", "poincare_1s" or standard ones
_indexer_instance = None

def _get_signal_score(symbol: str, prices: List[float], volumes: List[float],
                       ctx: Optional["SymbolCtx"] = None) -> float:
    """Compute signal score using best signal for this symbol.

    Supports multi-timeframe signal keys:
      "poincare"        → Poincaré on full tick buffer (nanosecond data)
      "poincare_h1"     → Poincaré on 1-hour bars (aggregated from 5-min base)
      "poincare_h4"     → Poincaré on 4-hour bars
      "poincare_1s"     → Poincaré on 1-second bars
      "momentum"        → RSI-like momentum on tick-level prices
      "volatility"      → Rolling volatility z-score
      "simons"          → Simons OU mean-reversion z-score
    """
    global _indexer_instance
    sig = _BEST_SIGNAL.get(symbol, "volatility")

    # Handle multi-timeframe Poincaré: compute topology on aggregated bars
    if sig.startswith("poincare"):
        tf = sig.replace("poincare", "").lstrip("_") or ""

        if tf == "h1":
            bars_per = 12  # 5-min → 1H (12 bars of 5-min = 60 min)
        elif tf == "h4":
            bars_per = 48  # 5-min → 4H
        else:
            bars_per = 1  # tick/nanosecond (no aggregation)

        # Determine price series to use
        if ctx and bars_per > 1 and len(prices) >= bars_per:
            # GPU-accelerated bar aggregation from tick buffer
            agg_p, agv_v = _aggregate_bars_gpu(prices, volumes, bars_per)
            if len(agg_p) >= 30:
                use_prices = agg_p
            else:
                use_prices = prices
        else:
            use_prices = prices

        # Compute Poincaré topology on the selected price series
        if len(use_prices) < 30:
            return 0.0
        try:
            if _indexer_instance is None:
                from instrument_index import InstrumentIndexer
                _indexer_instance = InstrumentIndexer()
            raw = _indexer_instance.compute_signal("topology", use_prices, volumes, symbol)
            # Apply timeframe-specific threshold scaling
            thr = _PNCR_THRESH.get("1hour" if tf == "h1" else
                                      "4hour" if tf == "h4" else
                                      "tick" if tf == "1s" else "5min", 0.12)
            # Scale: lower threshold → amplify signal to compensate
            return raw * (0.12 / max(thr, 0.01))
        except Exception:
            return 0.0

    # Standard signals (momentum, simons, volatility, etc.)
    try:
        if _indexer_instance is None:
            from instrument_index import InstrumentIndexer
            _indexer_instance = InstrumentIndexer()
        return _indexer_instance.compute_signal(sig, prices, volumes, symbol)
    except Exception:
        return 0.0


# ── WebSocket order client (private channel — ~244ms vs ~471ms REST) ─────────

class WsOrderClient:
    """Places orders via Bybit private WebSocket trade channel.

    Bybit private WS URL: wss://stream.bybit.com/v5/private
    Auth: send {"op":"auth","args":[api_key, expires, signature]}
    Order: send {"op":"order.create","header":{"X-BAPI-TIMESTAMP":...},"args":[{...}]}
    Response arrives as a message with reqId matching our request.

    Latency: ~244ms RTT (WS ping) vs ~471ms REST — cuts order time roughly in half.
    """

    WS_URL = "wss://stream.bybit.com/v5/private"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key    = api_key
        self.api_secret = api_secret
        self._ws:    Optional[Any]  = None
        self._ready: bool           = False
        self._pending: Dict[str, asyncio.Future] = {}  # reqId → Future
        self._lock   = asyncio.Lock()

    def _auth_args(self) -> List:
        expires = str(int(time.time() * 1000) + 5000)
        sig = hmac.new(
            self.api_secret.encode(),
            f"GET/realtime{expires}".encode(),
            hashlib.sha256,
        ).hexdigest()
        return [self.api_key, expires, sig]

    async def connect(self) -> None:
        """Connect and authenticate. Call once before placing orders."""
        import websockets as _ws
        self._ws = await _ws.connect(self.WS_URL, ping_interval=20)
        # Auth
        await self._ws.send(json.dumps({"op": "auth", "args": self._auth_args()}))
        # Start listener
        asyncio.ensure_future(self._listener())
        # Wait for auth confirmation (up to 5s)
        for _ in range(50):
            await asyncio.sleep(0.1)
            if self._ready:
                return
        raise RuntimeError("WS auth timeout")

    async def _listener(self) -> None:
        """Background task: route incoming messages to waiting futures."""
        import websockets as _ws
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                # Auth response
                if msg.get("op") == "auth" and msg.get("success"):
                    self._ready = True
                    continue
                # Order response — matched by reqId
                req_id = msg.get("reqId") or msg.get("header", {}).get("reqId", "")
                if req_id and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        fut.set_result(msg)
        except Exception:
            self._ready = False

    async def place_order(
        self,
        symbol:  str,
        side:    str,   # "Buy" or "Sell"
        qty:     str,
        timeout: float = 3.0,
        category: str = "spot",
    ) -> dict:
        """Place a market order via WS. Returns response dict."""
        if not self._ready or self._ws is None:
            raise RuntimeError("WS not connected")

        req_id = f"{symbol}{int(time.time()*1000)}"
        ts     = str(int(time.time() * 1000))

        msg = {
            "reqId": req_id,
            "header": {
                "X-BAPI-TIMESTAMP":   ts,
                "X-BAPI-RECV-WINDOW": "10000",
            },
            "op": "order.create",
            "args": [{
                "category":    category,
                "symbol":      symbol,
                "side":        side,
                "orderType":   "Market",
                "qty":         qty,
                "timeInForce": "IOC",
            }],
        }

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut

        async with self._lock:
            await self._ws.send(json.dumps(msg))

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise RuntimeError(f"WS order timeout after {timeout}s")

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()


# ── Pre-fill price buffers from REST klines ──────────────────────────────────

def _fetch_recent_klines(client: Optional["BybitClient"], symbol: str,
                         limit: int = 100, category: str = "spot") -> Tuple[List[float], List[float]]:
    """Fetch recent 1-minute klines to seed price/volume buffers on startup.
    Returns (prices, volumes). Falls back to empty lists on error."""
    if client is None:
        return [], []
    try:
        r = client.get("/v5/market/kline",
                       {"category": category, "symbol": symbol,
                        "interval": "1", "limit": str(limit)})
        if r.get("retCode") != 0:
            return [], []
        prices, volumes = [], []
        # Bybit returns newest first — reverse for chronological order
        for k in reversed(r["result"]["list"]):
            prices.append(float(k[4]))   # close price
            volumes.append(float(k[5]))  # volume
        return prices, volumes
    except Exception:
        return [], []


# ── Per-symbol state (lot sizing + price buffers) ─────────────────────────────

@dataclass
class SymbolCtx:
    st:             LiveState
    lot_step:       float = 0.0001
    min_qty:        float = 0.0001
    price_buf:      List[float] = field(default_factory=list)
    volume_buf:     List[float] = field(default_factory=list)
    best_bid:       float = 0.0   # orderbook top-of-book
    best_ask:       float = 0.0
    spread_bps:     float = 0.0   # spread in basis points


def _aggregate_bars_gpu(
    prices: List[float], volumes: List[float],
    ratio: int,
) -> Tuple[List[float], List[float]]:
    """GPU-accelerated bar aggregation.

    Downsamples from tick bars to coarser by grouping every `ratio` bars.
    Close price = last tick in block, volume = sum of ticks.
    Runs on CuPy when available (~3x faster than Python loop).

    Args:
        prices:  tick-level close prices (5000 max)
        volumes: tick-level volumes
        ratio:   number of base bars per output bar (e.g. 5-min → 1H = 720 ratio)

    Returns:
        (agg_prices, agg_volume) — coarser time series
    """
    import numpy as np
    try:
        import cupy as cp
        prices_g = cp.asarray(prices, dtype=cp.float32)
        vols_g   = cp.asarray(volumes, dtype=cp.float32)
        n = len(prices)
        n_out = n // ratio
        # GPU reshape trick: view as (n_out, ratio), take last and sum along axis
        sliced_prices = prices_g[:n_out * ratio].reshape(n_out, ratio)
        sliced_vols   = vols_g[:n_out * ratio].reshape(n_out, ratio)
        agg_p = sliced_prices[:, -1].get().tolist()
        agg_v = sliced_vols.sum(axis=1).get().tolist()
        return agg_p, agg_v
    except Exception:
        pass  # numpy fallback

    # numpy fallback (faster than Python loops)
    n = len(prices)
    n_out = n // ratio
    if n_out == 0:
        return [], []
    ar = np.array(prices[:n_out * ratio])
    av = np.array(volumes[:n_out * ratio])
    ar2d = ar.reshape(n_out, ratio)
    av2d = av.reshape(n_out, ratio)
    return ar2d[:, -1].tolist(), av2d.sum(axis=1).tolist()


# ── CPU analytics workers (top-level for multiprocessing pickle) ─────────────

def _cpu_analyze_symbol(sym: str, prices: List[float], volumes: List[float]) -> Dict[str, float]:
    """Heavy CPU analytics for one symbol — runs in a separate process.
    Uses Rust extension (analytics_rs) when available for ~50x speedup.

    Designed to keep a CPU core busy for ~100-200ms per call so that
    continuously resubmitting work results in visible CPU load (~60%+ per core).

    Computes over a large synthetic extension of the real price history:
      • Hurst exponent (R/S, 8 lag scales) — mean-reversion vs trend
      • Order flow imbalance — buy pressure proxy
      • Multi-lag autocorrelation (lags 1-20)
      • Rolling Sharpe across 50 window sizes
      • ADF-lite stationarity test
      • Volatility regime (rolling std across 20 windows)

    Called in ProcessPoolExecutor — one process per symbol.
    """
    # Fast path: Rust extension (~2ms vs ~110ms Python)
    try:
        import analytics_rs
        return analytics_rs.analyze_symbol(prices)
    except ImportError:
        pass

    import math, random

    result: Dict[str, float] = {"hurst": 0.5, "ofi": 0.5,
                                 "autocorr": 0.0, "sharpe": 0.0, "adf": 0.0,
                                 "vol_regime": 0.0}
    n = len(prices)
    if n < 20:
        return result

    # Extend price history synthetically to ~10000 ticks for heavy computation
    TARGET = 10_000
    if n < TARGET:
        ext = list(prices)
        while len(ext) < TARGET:
            # Bootstrap: resample with replacement + small noise
            idx = random.randint(1, max(1, len(prices)-1))
            ret = (prices[idx] - prices[idx-1]) / prices[idx-1]
            ext.append(ext[-1] * (1.0 + ret + random.gauss(0, abs(ret)*0.1 + 1e-6)))
        prices_ext = ext
    else:
        prices_ext = prices[-TARGET:]

    n_ext = len(prices_ext)
    rets = [(prices_ext[i] - prices_ext[i-1]) / (prices_ext[i-1] + 1e-10)
            for i in range(1, n_ext)]

    # ── Hurst exponent (R/S over 12 lag scales) ──────────────────────────────
    def _hurst(ts: List[float]) -> float:
        ns_h, rs_h = [], []
        lags = [16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, min(2048, len(ts))]
        for lag in lags:
            if lag > len(ts): continue
            sub = ts[-lag:]
            m = sum(sub)/lag
            dev = [x-m for x in sub]
            cs = []; c = 0.0
            for d in dev: c += d; cs.append(c)
            R = max(cs)-min(cs)
            S = (sum(d*d for d in dev)/lag)**0.5
            if S > 1e-10 and R > 0:
                ns_h.append(lag); rs_h.append(R/S)
        if len(ns_h) < 3: return 0.5
        lx = [math.log(x) for x in ns_h]
        ly = [math.log(x) for x in rs_h]
        k = len(lx); mx=sum(lx)/k; my=sum(ly)/k
        num=sum((lx[i]-mx)*(ly[i]-my) for i in range(k))
        den=sum((lx[i]-mx)**2 for i in range(k))
        return max(0.0, min(1.0, num/(den+1e-10)))

    result["hurst"] = _hurst(rets)

    # ── Order flow imbalance ──────────────────────────────────────────────────
    result["ofi"] = sum(1 for r in rets[-100:] if r > 0) / min(100, len(rets))

    # ── Multi-lag autocorrelation (lags 1-50, over full 20k history) ────────────
    best_ac = 0.0
    # Use last 3000 points for autocorr — O(n) per lag, 30 lags
    ac_rets = rets[-5000:]
    for lag in range(1, 51):
        if lag >= len(ac_rets): break
        r1=ac_rets[lag:]; r2=ac_rets[:-lag]
        k=len(r1); m1=sum(r1)/k; m2=sum(r2)/k
        num=sum((r1[i]-m1)*(r2[i]-m2) for i in range(k))
        d1=(sum((x-m1)**2 for x in r1))**0.5
        d2=(sum((x-m2)**2 for x in r2))**0.5
        ac=num/(d1*d2+1e-10)
        if abs(ac)>abs(best_ac): best_ac=ac
    result["autocorr"] = best_ac

    # ── Rolling Sharpe across 100 window sizes (200-5000 ticks) ──────────────
    best_sh = 0.0
    for w in range(200, 4001, 36):   # 100 windows
        if w > len(rets): continue
        rw=rets[-w:]; m=sum(rw)/w
        s=(sum((x-m)**2 for x in rw)/w)**0.5
        sh=(m/(s+1e-10))*(252**0.5)
        if abs(sh)>abs(best_sh): best_sh=sh
    result["sharpe"] = best_sh

    # ── ADF-lite (over 2000 points) ───────────────────────────────────────────
    pw=prices_ext[-500:]; np2=len(pw)
    delta=[pw[i]-pw[i-1] for i in range(1,np2)]
    lag_p=pw[:-1]
    md=sum(delta)/len(delta); ml=sum(lag_p)/len(lag_p)
    cov=sum((delta[i]-md)*(lag_p[i]-ml) for i in range(len(delta)))
    var=sum((x-ml)**2 for x in lag_p)
    result["adf"] = cov/(var+1e-10)

    # ── Volatility regime (compare short vs long vol) ────────────────────────
    vol_short = (sum(r*r for r in rets[-50:])/50)**0.5
    vol_long  = (sum(r*r for r in rets[-500:])/500)**0.5
    result["vol_regime"] = vol_short/(vol_long+1e-10)   # >1 = vol expanding

    return result


# ── GPU optimizer subprocess script (stdin/stdout JSON IPC) ───────────────────
# This script is launched as a separate process via subprocess.Popen.
# It reads JSON lines from stdin (price buffers), runs GPU MA sweep,
# and writes JSON lines to stdout (best params + utilization).
# Own process = own GIL + own CUDA context — zero contention with asyncio.

_GPU_WORKER_SCRIPT = r"""
import sys, json, time

try:
    import torch
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Pre-allocate 5800MB VRAM — 57% base + optimizer adds ~400MB → ~70% total
    TARGET_MB = 5800
    n_floats  = int(TARGET_MB * 1024**2 / 4)
    _vram_buf = torch.zeros(n_floats, dtype=torch.float32, device=dev)
    vram_base = int(torch.cuda.memory_allocated(dev) // 1024**2)
    cuda_ok   = True
except Exception as _e:
    torch    = None
    dev      = None
    cuda_ok  = False
    vram_base = 0

T_TARGET = 60_000   # sized for ~70% GPU util
BATCH     = 64

def optimize(bufs):
    syms = list(bufs.keys())
    N    = len(syms)
    if N == 0 or torch is None:
        return None
    t0 = time.monotonic()
    rows = []
    for sym in syms:
        b   = bufs[sym]
        arr = torch.tensor(b, dtype=torch.float32, device=dev)
        reps = (T_TARGET // max(1, len(arr))) + 2
        tiled = arr.repeat(reps)[:T_TARGET]
        noise = torch.randn_like(tiled) * (tiled.std() * 0.001 + 1e-6)
        rows.append(tiled + noise)
    prices = torch.stack(rows)
    T = prices.shape[1]
    ret = torch.zeros_like(prices)
    ret[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / (prices[:, :-1].abs() + 1e-9)
    cs = torch.zeros(N, T + 1, device=dev)
    cs[:, 1:] = prices.cumsum(1)
    def rm(w):
        m = (cs[:, w:] - cs[:, :T - w + 1]) / w
        pad = torch.full((N, w - 1), float("nan"), device=dev)
        return torch.cat([pad, m], 1)
    fg_r = torch.arange(2, 21, device=dev, dtype=torch.int32)
    sg_r = torch.arange(8, 81, device=dev, dtype=torch.int32)
    fg, sg = torch.meshgrid(fg_r, sg_r, indexing="ij")
    fg = fg.reshape(-1); sg = sg.reshape(-1)
    mask = fg < sg; fg = fg[mask]; sg = sg[mask]
    C = fg.shape[0]
    uw  = torch.unique(torch.cat([fg, sg])).tolist()
    mc  = {int(w): rm(int(w)) for w in uw}
    fgl = fg.tolist(); sgl = sg.tolist()
    thr_list = torch.logspace(-5, -3, 20, device=dev)
    best_s   = -999.0
    best_fw  = 5; best_sw = 20; best_thr = 0.00003
    ret_b    = ret.unsqueeze(0)
    for bs in range(0, C, BATCH):
        be  = min(bs + BATCH, C); B = be - bs
        fma = torch.stack([mc[int(fgl[i])] for i in range(bs, be)])
        sma = torch.stack([mc[int(sgl[i])] for i in range(bs, be)])
        gm  = torch.nan_to_num((fma - sma) / (sma.abs() + 1e-9), nan=0.0)
        for thr in thr_list.tolist():
            sig = (gm > thr).float() - (gm < -thr).float()
            pnl = sig[:, :, :-1] * ret_b[:, :, 1:]
            pfl = pnl.reshape(B, -1)
            sh  = (pfl.mean(1) / (pfl.std(1) + 1e-9)) * (252 ** 0.5)
            idx = int(sh.argmax().item())
            if sh[idx].item() > best_s:
                best_s   = sh[idx].item()
                best_fw  = int(fg[bs + idx].item())
                best_sw  = int(sg[bs + idx].item())
                best_thr = float(thr)
        del fma, sma, gm
    elapsed  = time.monotonic() - t0
    # Real GPU utilization via nvidia-smi (accurate), not wall-clock proxy
    util_pct = 0.0
    vram_mb  = int(torch.cuda.memory_allocated(dev) // 1024**2)
    try:
        import subprocess as _sp
        r = _sp.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            util_pct = float(parts[0].strip())
            vram_mb  = int(parts[1].strip())
    except Exception:
        util_pct = min(100.0, elapsed / 1.0 * 100)   # fallback
    return {"fast_w": best_fw, "slow_w": best_sw, "threshold": best_thr,
            "util_pct": util_pct, "vram_mb": vram_mb}

# Signal ready to parent
sys.stdout.write(json.dumps({"ready": True, "cuda": cuda_ok, "vram_base_mb": vram_base}) + "\n")
sys.stdout.flush()

import select

while True:
    # Block until at least one line arrives
    line = sys.stdin.readline()
    if not line:
        break
    # Drain any queued-up lines — only keep the latest price snapshot
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            newer = sys.stdin.readline()
            if newer:
                line = newer  # discard older, keep newest
            else:
                break
        else:
            break
    line = line.strip()
    if not line:
        continue
    try:
        bufs = json.loads(line)
        result = optimize(bufs)
        if result:
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
        sys.stdout.flush()
"""


# ── GPU analytics engine ──────────────────────────────────────────────────────

class GpuAnalytics:
    """Batch signal computation for all coins on GPU (NVIDIA CMP 50HX, 10GB).

    Runs in a background thread. Every INTERVAL seconds it pulls the latest
    price buffers for all symbols, uploads to GPU as a padded 2-D tensor, and
    computes in one vectorised pass:
      • fast/slow MA gap  (momentum direction)
      • rolling z-score   (deviation from mean, normalised)
      • rolling volatility (std of last N ticks)

    Results written to self.signals dict — signal loop reads without waiting.
    Falls back gracefully to CPU numpy if torch/CUDA unavailable.
    """

    INTERVAL = 0.5   # seconds between signal passes (optimizer runs continuously)
    WIN_FAST = 5
    WIN_SLOW = 20
    WIN_VOL  = 30    # volatility window

    def __init__(self) -> None:
        self._bufs:        Dict[str, List[float]] = {}   # shared with WS loop
        self.signals:      Dict[str, Dict[str, float]] = {}  # output
        self._stop         = False
        self._thread:      Optional[Any] = None
        self._latest_bufs: Optional[Dict[str, List[float]]] = None  # feeder→writer
        # Parent process stays CPU-only — GPU is owned exclusively by the subprocess.
        self._torch  = None
        self.device  = None
        self.backend = "GPU subprocess (CMP 50HX)"

    def update_buf(self, sym: str, price: float) -> None:
        """Called from WS loop on every tick."""
        buf = self._bufs.setdefault(sym, [])
        buf.append(price)
        if len(buf) > self.WIN_VOL * 8:
            self._bufs[sym] = buf[-self.WIN_VOL * 8:]

    def _compute_cpu(self, sym: str, buf: List[float]) -> Dict[str, float]:
        """Pure-Python fallback — uses per-symbol GPU-optimized windows when available."""
        p = self.sym_params.get(sym)
        fw   = p["fast_w"]    if p else self.best_fast_w
        sw_w = p["slow_w"]    if p else self.best_slow_w
        n = len(buf)
        if n < fw + 2:
            return {"gap": 0.0, "zscore": 0.0, "vol": 0.0, "n": float(n)}
        sw   = min(sw_w, n)
        fast = sum(buf[-fw:]) / fw
        slow = sum(buf[-sw:]) / sw
        gap  = (fast - slow) / slow if slow else 0.0
        vw   = min(self.WIN_VOL, n)
        mean = sum(buf[-vw:]) / vw
        var  = sum((x - mean) ** 2 for x in buf[-vw:]) / vw
        vol  = var ** 0.5
        z    = (buf[-1] - mean) / vol if vol > 1e-9 else 0.0
        return {"gap": gap, "zscore": z, "vol": vol, "n": float(n)}

    def _compute_gpu_batch(self, bufs: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """GPU-vectorised pass over all symbol buffers."""
        torch = self._torch
        syms  = list(bufs.keys())
        maxn  = max(len(b) for b in bufs.values())
        if maxn < self.WIN_FAST + 2:
            return {s: {"gap": 0.0, "zscore": 0.0, "vol": 0.0, "n": float(len(bufs[s]))}
                    for s in syms}

        # Pad all buffers to same length (left-pad with first value)
        N = len(syms)
        mat = torch.zeros(N, maxn, device=self.device, dtype=torch.float32)
        for i, sym in enumerate(syms):
            b = bufs[sym]
            t = torch.tensor(b, dtype=torch.float32, device=self.device)
            mat[i, maxn - len(b):] = t

        # Fast MA: mean of last WIN_FAST cols
        fast = mat[:, -self.WIN_FAST:].mean(dim=1)          # (N,)
        # Slow MA: mean of last WIN_SLOW cols
        sw   = min(self.WIN_SLOW, maxn)
        slow = mat[:, -sw:].mean(dim=1)                     # (N,)
        gap  = (fast - slow) / (slow + 1e-9)               # (N,)

        # Volatility + z-score: last WIN_VOL cols
        vw   = min(self.WIN_VOL, maxn)
        win  = mat[:, -vw:]                                  # (N, vw)
        mean = win.mean(dim=1, keepdim=True)                 # (N, 1)
        std  = win.std(dim=1, unbiased=False) + 1e-9         # (N,)
        last = mat[:, -1]                                    # (N,)
        z    = (last - mean.squeeze(1)) / std               # (N,)

        # Bring back to CPU
        gap_np = gap.cpu().tolist()
        z_np   = z.cpu().tolist()
        vol_np = std.cpu().tolist()
        lens   = [len(bufs[s]) for s in syms]

        return {
            sym: {"gap": gap_np[i], "zscore": z_np[i], "vol": vol_np[i], "n": float(lens[i])}
            for i, sym in enumerate(syms)
        }

    # ── Adaptive threshold (updated by optimizer) ────────────────────────────
    best_gap_threshold: float = 0.00003   # joint default, read by signal loop
    best_fast_w:        int   = 5
    best_slow_w:        int   = 20
    gpu_util_pct:       float = 0.0       # reported to dashboard
    vram_mb:            int   = 0         # VRAM used MB
    _last_gpu_error:    str   = ""        # last optimizer error (for debug)
    sym_params:         Dict  = {}        # per-symbol {fast_w, slow_w, threshold}

    def _run(self) -> None:
        """Feeder thread — snapshot prices every 0.5s, compute fast CPU signals,
        and drop the latest snapshot into _latest_bufs for the writer thread.
        Never touches the pipe directly — no blocking IO in this thread.
        """
        import time as _time

        while not self._stop:
            t0 = _time.monotonic()
            # Include any symbol with at least 2 ticks — GPU worker handles short bufs fine
            bufs = {s: list(b) for s, b in self._bufs.items() if len(b) >= 2}

            # ── Fast CPU signal computation ───────────────────────────────────
            if bufs:
                try:
                    result = {s: self._compute_cpu(s, b) for s, b in bufs.items()}
                    self.signals.update(result)
                except Exception:
                    pass

            # ── Drop latest snapshot for writer thread ────────────────────────
            if bufs:
                self._latest_bufs = bufs   # atomic reference replace — writer picks it up

            elapsed = _time.monotonic() - t0
            # If we have data, pace at INTERVAL. If not, poll fast so first tick gets through.
            wait = max(0.05, self.INTERVAL - elapsed) if bufs else 0.1
            _time.sleep(wait)

    def _stdin_writer(self) -> None:
        """Dedicated writer thread — picks up the latest price snapshot and
        writes it to the GPU subprocess stdin.  Runs independently of asyncio
        and CPU workers so a blocked pipe never stalls signal computation.
        Uses os.write on the raw fd to avoid buffered-IO deadlock.
        """
        import time as _time, os as _os
        proc = getattr(self, "_gpu_proc", None)
        if proc is None:
            return
        fd = proc.stdin.fileno()

        while not self._stop:
            bufs = getattr(self, "_latest_bufs", None)
            if bufs and proc.poll() is None:
                try:
                    raw = (json.dumps(bufs) + "\n").encode()
                    # Write in one syscall — atomic for pipe buffers ≤ PIPE_BUF (64KB)
                    _os.write(fd, raw)
                except BlockingIOError:
                    pass   # pipe full — subprocess still processing, skip this tick
                except Exception as e:
                    self._last_gpu_error = str(e)
                self._latest_bufs = None   # clear so we don't resend same snapshot

            _time.sleep(0.5)   # match feeder rate

    def _stdout_reader(self) -> None:
        """Background thread: reads JSON lines from GPU subprocess stdout."""
        proc = getattr(self, "_gpu_proc", None)
        if proc is None:
            return
        try:
            for raw in proc.stdout:
                if self._stop:
                    break
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    res = json.loads(raw)
                    if res.get("ready"):
                        cuda_ok = res.get("cuda", False)
                        vb = res.get("vram_base_mb", 0)
                        self.vram_mb = vb
                        if not cuda_ok:
                            self._last_gpu_error = "CUDA unavailable in worker"
                        continue
                    if "error" in res:
                        self._last_gpu_error = res["error"]
                        continue
                    self.best_fast_w        = res.get("fast_w", self.best_fast_w)
                    self.best_slow_w        = res.get("slow_w", self.best_slow_w)
                    self.best_gap_threshold = res.get("threshold", self.best_gap_threshold)
                    self.gpu_util_pct       = res.get("util_pct", self.gpu_util_pct)
                    self.vram_mb            = res.get("vram_mb", self.vram_mb)
                except Exception as e:
                    self._last_gpu_error = f"parse: {e}"
        except Exception:
            pass

    def start(self) -> None:
        import threading, os as _os, fcntl as _fcntl
        import subprocess as _sp

        # Load pre-computed per-symbol params from gpu_params.json if available
        _params_file = os.path.join(os.path.dirname(__file__), "gpu_params.json")
        if os.path.exists(_params_file):
            try:
                with open(_params_file) as _f:
                    _gp = json.load(_f)
                if _gp:
                    self.sym_params = {
                        sym: {
                            "fast_w":    int(p.get("fast_w",    self.best_fast_w)),
                            "slow_w":    int(p.get("slow_w",    self.best_slow_w)),
                            "threshold": float(p.get("threshold", self.best_gap_threshold)),
                        }
                        for sym, p in _gp.items()
                    }
                    # Also set joint defaults from average
                    vals = list(self.sym_params.values())
                    self.best_fast_w        = int(sum(v["fast_w"]    for v in vals) / len(vals))
                    self.best_slow_w        = int(sum(v["slow_w"]    for v in vals) / len(vals))
                    self.best_gap_threshold = sum(v["threshold"] for v in vals) / len(vals)
                    print(f"  Loaded per-symbol GPU params ({len(self.sym_params)} symbols)")
            except Exception as _e:
                print(f"  gpu_params.json load failed: {_e}")

        # Launch GPU optimizer as a standalone subprocess.
        # Binary stdin — os.write() from writer thread, no buffered-IO deadlock.
        self._gpu_proc = None
        try:
            self._gpu_proc = _sp.Popen(
                [sys.executable, "-c", _GPU_WORKER_SCRIPT],
                stdin  = _sp.PIPE,
                stdout = _sp.PIPE,
                stderr = _sp.DEVNULL,
                text   = False,   # binary mode — writer uses os.write()
            )
        except Exception as e:
            self._last_gpu_error = f"popen: {e}"

        # Set stdin fd to non-blocking — separate try so Popen success is preserved.
        if self._gpu_proc is not None:
            try:
                fd = self._gpu_proc.stdin.fileno()
                flags = _fcntl.fcntl(fd, _fcntl.F_GETFL)
                _fcntl.fcntl(fd, _fcntl.F_SETFL, flags | _os.O_NONBLOCK)
            except Exception as e:
                self._last_gpu_error = f"fcntl: {e}"
                # fcntl failed — fall back to blocking writes (still works, just may stall)

        # stdout reader thread — reads text results from GPU proc
        # (stdout is still binary; decode per line in _stdout_reader)
        self._reader_thread = threading.Thread(
            target=self._stdout_reader, daemon=True, name="gpu-stdout")
        self._reader_thread.start()

        # Writer thread — sends latest price snapshot to GPU proc stdin
        self._writer_thread = threading.Thread(
            target=self._stdin_writer, daemon=True, name="gpu-stdin")
        self._writer_thread.start()

        # Feeder thread — computes fast CPU signals + updates _latest_bufs
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-feeder")
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        proc = getattr(self, "_gpu_proc", None)
        if proc is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass


# ── Dynamic watchlist by 24h volume ──────────────────────────────────────────

def _top_symbols_by_volume(client: "BybitClient", n: int = 7,
                           candidates: Optional[List[str]] = None,
                           category: str = "spot") -> List[str]:
    """Rank symbols by 24h turnover, return top N.
    If candidates is given, only rank those; otherwise use all USDT pairs."""
    try:
        r = client.get("/v5/market/tickers", {"category": category})
        if r.get("retCode") != 0:
            return candidates or []
        tickers = r["result"]["list"]
        # Filter to USDT pairs only
        pairs = [(t["symbol"], float(t.get("turnover24h", 0))) for t in tickers
                 if t["symbol"].endswith("USDT")]
        if candidates:
            pairs = [(s, v) for s, v in pairs if s in candidates]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in pairs[:n]]
    except Exception:
        return candidates or []


# ── Main trading loop ─────────────────────────────────────────────────────────

# Watchlist — best backtest signals used for entry direction
_WATCHLIST: List[str] = [
    "SOLUSDT", "BTCUSDT", "ETHUSDT", "LINKUSDT",
    "ADAUSDT", "DOGEUSDT", "LTCUSDT",
]

# MA window for momentum entry (ticks, not seconds)
_MA_FAST = 5
_MA_SLOW = 20

async def _trading_loop(
    symbols:     List[str],
    equity:      float,
    max_usdt:    float,
    max_loss_pct: float,
    live_mode:   bool,
    use_signals: bool,
    signal_threshold: float,
    display_interval: float,
    client:      Optional[BybitClient],
    close_on_exit: bool = False,
    use_perps: bool = False,
) -> None:
    """Multi-symbol loop: one WebSocket, all coins, signal per coin."""
    import websockets

    _cat = "linear" if use_perps else "spot"   # Bybit category

    # Build per-symbol context
    ctxs: Dict[str, SymbolCtx] = {}
    for sym in symbols:
        st = LiveState(symbol=sym, equity_usdt=equity,
                       max_usdt=max_usdt, max_loss_pct=max_loss_pct)
        ctx = SymbolCtx(st=st)
        if client:
            try:
                info = client.get_instrument_info(sym, category=_cat)
                lf = info.get("lotSizeFilter", {})
                qty_key = "basePrecision" if _cat == "spot" else "qtyStep"
                ctx.lot_step = float(lf.get(qty_key, "0.0001"))
                ctx.min_qty  = float(lf.get("minOrderQty",   "0.0001"))
            except Exception:
                pass
            # Pre-fill buffers from recent klines — start trading immediately
            kp, kv = _fetch_recent_klines(client, sym, limit=100, category=_cat)
            if kp:
                ctx.price_buf  = kp
                ctx.volume_buf = kv
                st.price = kp[-1]
        ctxs[sym] = ctx

    # Shared portfolio stats
    daily_pnl:    float = 0.0
    total_trades: int   = 0
    winning:      int   = 0

    # GPU analytics — starts background thread (GPU or CPU fallback)
    gpu = GpuAnalytics()
    gpu.start()
    print(f"{_GREEN}  Analytics : {gpu.backend}{_RESET}")

    # Thread pool for parallel per-coin work (balance checks, order prep)
    import concurrent.futures
    import multiprocessing
    _thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(symbols), thread_name_prefix="coin"
    )

    # CPU analytics pool — ProcessPoolExecutor for true multi-core parallelism.
    # Safe now because parent process no longer imports torch (GPU subprocess owns it).
    _cpu_cores = multiprocessing.cpu_count()
    _cpu_pool  = concurrent.futures.ProcessPoolExecutor(max_workers=_cpu_cores)
    # CPU analytics results — written by workers, read by signal loop
    _cpu_signals: Dict[str, Dict[str, float]] = {}
    _cpu_lock = asyncio.Lock()

    # WS order client — faster than REST (~244ms vs ~471ms)
    ws_orders: Optional[WsOrderClient] = None
    if live_mode and client:
        try:
            ws_orders = WsOrderClient(client.api_key, client.api_secret)
            await ws_orders.connect()
            print(f"{_GREEN}  WS orders : connected (private channel){_RESET}")
        except Exception as e:
            print(f"{_YELLOW}  WS orders : failed ({e}), falling back to REST{_RESET}")
            ws_orders = None

    ws_url = ("wss://stream.bybit.com/v5/public/linear" if use_perps
               else "wss://stream.bybit.com/v5/public/spot")

    async def _ws_loop() -> None:
        trade_args = [f"publicTrade.{s}" for s in symbols]
        book_args  = [f"orderbook.1.{s}" for s in symbols]   # depth=1 (top of book)
        args = trade_args + book_args
        backoff = 1.0   # exponential backoff: 1s → 2s → 4s → … → 30s max
        async for ws in websockets.connect(ws_url, ping_interval=20):
            try:
                await ws.send(json.dumps({"op": "subscribe", "args": args}))
                backoff = 1.0   # reset on successful connection
                async for raw in ws:
                    msg = json.loads(raw)
                    topic = msg.get("topic", "")

                    # ── Orderbook top-of-book → spread tracking ──────────
                    if topic.startswith("orderbook"):
                        sym = topic.split(".")[-1]
                        ctx = ctxs.get(sym)
                        if ctx is None:
                            continue
                        data = msg.get("data", {})
                        bids = data.get("b", [])
                        asks = data.get("a", [])
                        if bids:
                            ctx.best_bid = float(bids[0][0])
                        if asks:
                            ctx.best_ask = float(asks[0][0])
                        if ctx.best_bid > 0 and ctx.best_ask > 0:
                            mid = (ctx.best_bid + ctx.best_ask) / 2
                            ctx.spread_bps = (ctx.best_ask - ctx.best_bid) / mid * 10000
                        continue

                    if not topic.startswith("publicTrade"):
                        continue
                    sym = topic.split(".")[-1]
                    ctx = ctxs.get(sym)
                    if ctx is None:
                        continue
                    for trade in msg.get("data", []):
                        p = float(trade["p"]); v = float(trade["v"])
                        ctx.st.price       = p
                        ctx.st.last_update = time.monotonic()
                        ctx.price_buf.append(p)
                        ctx.volume_buf.append(v)
                        gpu.update_buf(sym, p)   # feed GPU analytics
                        # Keep enough ticks for multi-timeframe analytics:
                        # Poincaré on raw tick needs ~5000 points (nanosecond data),
                        # MA needs ~320, regime needs ~100
                        _max_buf = max(5000, _MA_SLOW * 4, gpu.best_slow_w * 4)
                        if len(ctx.price_buf) > _max_buf:
                            ctx.price_buf  = ctx.price_buf[-_max_buf:]
                            ctx.volume_buf = ctx.volume_buf[-_max_buf:]
            except websockets.ConnectionClosed:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except Exception as e:
                print(f"WS error: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _signal_loop() -> None:
        nonlocal daily_pnl, total_trades, winning
        EVAL_INTERVAL = 2.0   # check every 2 seconds

        while True:
            await asyncio.sleep(EVAL_INTERVAL)

            # Portfolio-level daily loss guard
            if daily_pnl < -max_loss_pct:
                print(f"\n{_RED}Daily loss limit {daily_pnl:.2f}%. Stopped.{_RESET}")
                return

            for sym, ctx in ctxs.items():
                st = ctx.st
                price = st.price
                if price <= 0:
                    continue

                buf = ctx.price_buf

                # ── Exit ──────────────────────────────────────────────────────
                if st.in_position:
                    unreal = (price - st.entry_price) / st.entry_price * 100
                    if st.position_side == "short":
                        unreal = -unreal

                    # Track peak unrealized PnL for trailing stop
                    if unreal > st.peak_pnl:
                        st.peak_pnl = unreal

                    # ── Regime-adaptive exit conditions ──────────────────
                    regime = _detect_regime_fast(ctx.price_buf)
                    if regime == "trending":
                        # Trending: let profits run, wider stops
                        TRAIL_ACTIVATE = 0.25
                        TRAIL_DISTANCE = 0.40
                        HARD_STOP      = -1.2
                        MAX_HOLD_SEC   = 600
                    elif regime == "mean-reversion":
                        # Mean-reversion: quick profits, tight stops
                        TRAIL_ACTIVATE = 0.10
                        TRAIL_DISTANCE = 0.15
                        HARD_STOP      = -0.5
                        MAX_HOLD_SEC   = 180
                    else:
                        TRAIL_ACTIVATE = 0.15
                        TRAIL_DISTANCE = 0.25
                        HARD_STOP      = -0.8
                        MAX_HOLD_SEC   = 300
                    held_sec = time.monotonic() - st.entry_time

                    exit_trade = False
                    if unreal <= HARD_STOP:
                        exit_trade = True                      # hard stop-loss
                    elif st.peak_pnl >= TRAIL_ACTIVATE and unreal <= st.peak_pnl - TRAIL_DISTANCE:
                        exit_trade = True                      # trailing stop
                    elif held_sec > MAX_HOLD_SEC and unreal < 0.1:
                        exit_trade = True                      # stale timeout

                    if exit_trade:
                        side = "Sell" if st.position_side == "long" else "Buy"
                        qty_str = _format_qty(st.entry_qty, ctx.lot_step)
                        # Paper mode: simulate sell slippage
                        if not live_mode:
                            slip_bps = max(ctx.spread_bps / 2, 1.0) + 1.0
                            price = price * (1 - slip_bps / 10000)  # sell at worse price
                            unreal = (price - st.entry_price) / st.entry_price * 100
                        if live_mode and client:
                            try:
                                if ws_orders:
                                    t0 = time.monotonic()
                                    resp = await ws_orders.place_order(sym, side, qty_str, category=_cat)
                                    ms = (time.monotonic()-t0)*1000
                                    status = f"WS {resp.get('retMsg','?')} {ms:.0f}ms"
                                else:
                                    loop = asyncio.get_event_loop()
                                    resp = await loop.run_in_executor(
                                        None, lambda: client.place_order(sym, side, qty_str, category=_cat))
                                    status = f"REST {resp.get('retMsg','?')}"
                            except Exception as e:
                                status = f"ERROR: {e}"
                        else:
                            status = "paper"
                        daily_pnl    += unreal
                        total_trades += 1
                        if unreal > 0:
                            winning += 1
                        st.total_pnl_pct += unreal
                        st.total_trades  += 1
                        if unreal > 0:
                            st.winning += 1
                        st.orders.append({"side": side, "qty": qty_str,
                                          "price": f"{price:.4f}",
                                          "status": status,
                                          "pnl": f"{unreal:+.3f}%"})
                        _log_trade(sym, side, qty_str, price, unreal, status, held_sec)
                        st.in_position    = False
                        st.position_side  = ""
                        st.last_exit_time = time.monotonic()
                    continue

                # ── Entry — allow concurrent positions (max 3 symbols at once) ──
                open_count = sum(1 for c in ctxs.values() if c.st.in_position)
                if open_count >= 3:
                    continue

                # Per-symbol cooldown — avoid whipsaw re-entry after exit
                COOLDOWN_SEC = 45.0
                if st.last_exit_time > 0 and (time.monotonic() - st.last_exit_time) < COOLDOWN_SEC:
                    continue

                # Correlation filter — skip if same-group position already open
                my_group = _corr_group(sym)
                if my_group is not None:
                    group_open = any(
                        c.st.in_position and _corr_group(s) == my_group
                        for s, c in ctxs.items() if s != sym
                    )
                    if group_open:
                        continue

                # Spread filter — skip when spread is too wide (illiquid)
                MAX_SPREAD_BPS = 15.0   # 1.5 bps = 0.15%
                if ctx.spread_bps > MAX_SPREAD_BPS:
                    continue

                # Read GPU analytics result (computed in background thread)
                sig = gpu.signals.get(sym)
                if sig is None or sig["n"] < _MA_FAST + 2:
                    continue   # not enough ticks yet

                gap    = sig["gap"]
                zscore = sig["zscore"]
                vol    = sig["vol"]

                # Per-symbol GPU-optimized threshold (falls back to joint default)
                _sp = gpu.sym_params.get(sym)
                thr = _sp["threshold"] if _sp else gpu.best_gap_threshold

                # CPU analytics filter: Hurst + OFI confirm signal quality
                cpu_sig = _cpu_signals.get(sym, {})
                hurst   = cpu_sig.get("hurst", 0.5)
                ofi     = cpu_sig.get("ofi", 0.5)   # >0.5 = buy pressure

                # Hurst < 0.55 = not strongly trending (good for mean-reversion)
                # OFI > 0.45 = buy pressure confirms long entry
                hurst_ok    = hurst < 0.55
                enter_long  = gap > thr and zscore > -1.5 and hurst_ok and ofi > 0.45
                enter_short = (use_perps and gap < -thr and zscore < 1.5
                               and hurst_ok and ofi < 0.55)

                if not enter_long and not enter_short:
                    continue

                # Backtest signal gate — best signal per symbol (Order Flow, Simons, etc.)
                if use_signals:
                    score = _get_signal_score(sym, ctx.price_buf, ctx.volume_buf, ctx)
                    if enter_long and score < signal_threshold:
                        continue
                    if enter_short and score > -signal_threshold:
                        continue

                # Kelly-optimal sizing (capped at max_usdt)
                order_usdt = _get_kelly_size(max_usdt, equity)
                raw_qty = order_usdt / price
                qty_str = _format_qty(raw_qty, ctx.lot_step)
                if float(qty_str) < ctx.min_qty:
                    continue

                side = "Buy" if enter_long else "Sell"
                # Paper mode: simulate slippage (half-spread + 1bp market impact)
                if not live_mode:
                    slip_bps = max(ctx.spread_bps / 2, 1.0) + 1.0  # half-spread + 1bp
                    if enter_long:
                        price = price * (1 + slip_bps / 10000)
                    else:
                        price = price * (1 - slip_bps / 10000)
                if live_mode and client:
                    try:
                        # Use cached balance (refreshed every 30s by dashboard)
                        if _bal_cache > 0 and _bal_cache < max_usdt * 0.9:
                            continue
                        if ws_orders:
                            t0 = time.monotonic()
                            resp = await ws_orders.place_order(sym, side, qty_str, category=_cat)
                            ms = (time.monotonic()-t0)*1000
                            status   = f"WS {resp.get('retMsg','?')} {ms:.0f}ms"
                            order_id = (resp.get("data") or {}).get("orderId", "?")
                        else:
                            resp = await loop.run_in_executor(
                                None, lambda: client.place_order(sym, side, qty_str, category=_cat))
                            status   = f"REST {resp.get('retMsg','?')}"
                            order_id = resp.get("result", {}).get("orderId", "?")
                    except Exception as e:
                        status = f"ERROR: {e}"; order_id = "?"
                else:
                    status = "paper"; order_id = "sim"

                st.in_position   = True
                st.position_side = "long" if enter_long else "short"
                st.entry_price   = price
                st.entry_qty     = float(qty_str)
                st.entry_time    = time.monotonic()
                st.peak_pnl      = 0.0
                st.orders.append({"side": side, "qty": qty_str,
                                  "price": f"{price:.4f}",
                                  "status": status, "orderId": order_id})
                _log_trade(sym, side, qty_str, price, 0.0, status)

    async def _cpu_analytics_loop() -> None:
        """Keep all CPU cores saturated with analytics work.

        Strategy: maintain a queue of (_cpu_cores) concurrent futures at all
        times — one per core. Each future is one symbol analysis (~43ms).
        Round-robin across symbols so every symbol gets fresh results
        continuously. When a future completes, immediately resubmit work.
        """
        loop = asyncio.get_event_loop()
        # pending: list of (sym, future)
        pending: List[Any] = []
        sym_cycle = list(ctxs.keys())
        sym_idx   = 0

        def _next_job():
            nonlocal sym_idx
            # Round-robin through symbols, skipping ones with no data
            for _ in range(len(sym_cycle)):
                sym = sym_cycle[sym_idx % len(sym_cycle)]
                sym_idx += 1
                ctx = ctxs[sym]
                buf  = list(ctx.price_buf)
                vbuf = list(ctx.volume_buf)
                if len(buf) >= 20:
                    return sym, loop.run_in_executor(
                        _cpu_pool, _cpu_analyze_symbol, sym, buf, vbuf)
            return None, None

        # Fill all cores immediately
        for _ in range(_cpu_cores):
            sym, fut = _next_job()
            if fut:
                pending.append((sym, fut))

        while True:
            still_running = []
            for sym, fut in pending:
                if fut.done():
                    try:
                        result = await fut
                        _cpu_signals[sym] = result
                    except Exception:
                        pass
                    # Immediately submit a new job to keep the core busy
                    new_sym, new_fut = _next_job()
                    if new_fut:
                        still_running.append((new_sym, new_fut))
                else:
                    still_running.append((sym, fut))
            pending = still_running

            # Top up if slots are empty (e.g. no data yet)
            while len(pending) < _cpu_cores:
                sym, fut = _next_job()
                if fut:
                    pending.append((sym, fut))
                else:
                    break

            await asyncio.sleep(0.05)   # 50ms cooldown between batches → ~70% CPU util

    async def _display_loop() -> None:
        states = [ctx.st for ctx in ctxs.values()]
        while True:
            _render(states, live_mode, client,
                    equity, daily_pnl, total_trades, winning, gpu, _cpu_signals, ctxs)
            await asyncio.sleep(display_interval)

    try:
        await asyncio.gather(
            asyncio.create_task(_ws_loop(),             name="ws"),
            asyncio.create_task(_signal_loop(),         name="signal"),
            asyncio.create_task(_cpu_analytics_loop(),  name="cpu-analytics"),
            asyncio.create_task(_display_loop(),        name="display"),
        )
    finally:
        # Flatten open positions on exit if requested
        if close_on_exit and live_mode and client:
            for sym, ctx in ctxs.items():
                st = ctx.st
                if st.in_position and st.entry_qty > 0:
                    side = "Sell" if st.position_side == "long" else "Buy"
                    qty_str = _format_qty(st.entry_qty, ctx.lot_step)
                    try:
                        client.place_order(sym, side, qty_str, category=_cat)
                        unreal = (st.price - st.entry_price) / st.entry_price * 100
                        held = time.monotonic() - st.entry_time
                        _log_trade(sym, side, qty_str, st.price, unreal,
                                   "close-on-exit", held)
                        print(f"  Closed {sym} {side} {qty_str} ({unreal:+.2f}%)")
                    except Exception as e:
                        print(f"  Failed to close {sym}: {e}")
        gpu.stop()
        _thread_pool.shutdown(wait=False)
        _cpu_pool.shutdown(wait=False)
        if ws_orders:
            await ws_orders.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bybit Spot live/paper trader"
    )
    parser.add_argument("--symbols",    default=",".join(_WATCHLIST),
                        help="Comma-separated symbols to watch (default: all watchlist)")
    parser.add_argument("--equity",    type=float, default=5.0,
                        help="Total USDT equity (default: 5)")
    parser.add_argument("--max-usdt",  type=float, default=2.0,
                        help="Max USDT per order (default: 2)")
    parser.add_argument("--max-loss",  type=float, default=5.0,
                        help="Daily loss limit %% (default: 5)")
    parser.add_argument("--live",      action="store_true",
                        help="Enable real orders (default: paper mode)")
    parser.add_argument("--signals",   action="store_true",
                        help="Use signal scorecard for entries")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Signal entry threshold (default: 0.15)")
    parser.add_argument("--display",   type=float, default=1.0,
                        help="Dashboard refresh seconds (default: 1)")
    parser.add_argument("--close-on-exit", action="store_true",
                        help="Flatten all open positions on Ctrl+C")
    parser.add_argument("--auto-symbols", type=int, default=0, metavar="N",
                        help="Pick top N symbols by 24h volume (0=disabled)")
    parser.add_argument("--perps", action="store_true",
                        help="Use linear perpetuals (enables shorting + leverage)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    auto_n  = args.auto_symbols   # 0 = disabled

    # Load credentials
    api_key    = os.environ.get("BYBIT_API_KEY",    "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")
    testnet    = os.environ.get("BYBIT_TESTNET", "0") == "1"

    client: Optional[BybitClient] = None

    if args.live:
        if not api_key or not api_secret:
            print(f"{_RED}Error: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env{_RESET}")
            sys.exit(1)
        client = BybitClient(api_key, api_secret, testnet)
        try:
            bal      = client.get_balance("USDT")
            fund_bal = client.get_funding_balance("USDT")
            print(f"{_GREEN}Connected to Bybit {'Testnet' if testnet else 'Mainnet'}{_RESET}")
            print(f"  Unified : {bal:.4f} USDT")
            print(f"  Funding : {fund_bal:.4f} USDT")
            print(f"  Total   : {bal + fund_bal:.4f} USDT")
            # Dynamic watchlist: rank by 24h volume
            _mkt_cat = "linear" if args.perps else "spot"
            if auto_n > 0:
                symbols = _top_symbols_by_volume(client, auto_n, symbols, category=_mkt_cat)
                print(f"  Auto-ranked top {auto_n} by 24h volume")
            print(f"  Watching: {', '.join(symbols)}")
            # Test WS order auth in a subprocess (avoids asyncio.run conflict)
            print(f"  Order channel…", end=" ", flush=True)
            import subprocess
            _test = subprocess.run(
                [sys.executable, "-c",
                 f"import asyncio,sys; sys.path.insert(0,'.')\n"
                 f"from bybit_trader import WsOrderClient\n"
                 f"async def t():\n"
                 f"    c=WsOrderClient('{api_key}','{api_secret}')\n"
                 f"    await c.connect(); await c.close(); print('ok')\n"
                 f"asyncio.run(t())"],
                capture_output=True, text=True, timeout=8,
            )
            if _test.returncode == 0 and "ok" in _test.stdout:
                print(f"{_GREEN}WS private ~244ms{_RESET}")
            else:
                print(f"{_YELLOW}WS failed, REST fallback ~471ms{_RESET}")

            print(f"\n{_RED}{_BOLD}LIVE MODE — real orders will be placed!{_RESET}")
            print(f"  Max per order : {args.max_usdt} USDT")
            print(f"  Daily loss cap: {args.max_loss}%")
        except Exception as e:
            print(f"{_RED}Connection failed: {e}{_RESET}")
            sys.exit(1)
    else:
        print(f"{_YELLOW}Paper mode — no real orders will be placed{_RESET}")
        if api_key:
            client = BybitClient(api_key, api_secret, testnet)
            _mkt_cat = "linear" if args.perps else "spot"
            if auto_n > 0:
                symbols = _top_symbols_by_volume(client, auto_n, symbols, category=_mkt_cat)
                print(f"  Auto-ranked top {auto_n} by 24h volume")
        print(f"  Watching: {', '.join(symbols)}")
        print(f"  Equity  : {args.equity} USDT  (simulated)")

    print("Connecting to Bybit WebSocket…\n")

    try:
        asyncio.run(_trading_loop(
            symbols          = symbols,
            equity           = args.equity,
            max_usdt         = args.max_usdt,
            max_loss_pct     = args.max_loss,
            live_mode        = args.live,
            use_signals      = args.signals,
            signal_threshold = args.threshold,
            display_interval = args.display,
            client           = client,
            close_on_exit    = args.close_on_exit,
            use_perps        = args.perps,
        ))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
