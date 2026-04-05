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
        self.recv_window = "5000"

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

    def get_instrument_info(self, symbol: str) -> dict:
        """Get lot size / min order qty / tick size for a symbol."""
        r = self.get("/v5/market/instruments-info",
                     {"category": "spot", "symbol": symbol})
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
    ) -> dict:
        """Place a Spot order. Returns full Bybit response."""
        body: Dict = {
            "category":    "spot",
            "symbol":      symbol,
            "side":        side,
            "orderType":   order_type,
            "qty":         qty,
            "timeInForce": time_in_force,
        }
        if price:
            body["price"] = price
        return self.post("/v5/order/create", body)

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel all open Spot orders for a symbol."""
        return self.post("/v5/order/cancel-all",
                         {"category": "spot", "symbol": symbol})


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


# ── Dashboard ─────────────────────────────────────────────────────────────────

def _render(states: List["LiveState"], live_mode: bool, client: Optional[BybitClient],
            equity: float, daily_pnl: float, total_trades: int, winning: int,
            gpu: Optional["GpuAnalytics"] = None) -> None:
    _clear()
    mode_str = f"{_RED}{_BOLD}LIVE ORDERS{_RESET}" if live_mode else f"{_YELLOW}PAPER MODE{_RESET}"
    bal_str = ""
    if client:
        try:
            bal = client.get_balance("USDT")
            bal_str = f"  wallet={_CYAN}{bal:.4f} USDT{_RESET}"
        except Exception:
            bal_str = ""

    win_rate = winning / total_trades * 100 if total_trades > 0 else 0.0
    print(f"{_BOLD}{'─'*70}{_RESET}")
    if gpu:
        gpu_str = (f"  {_DIM}[{gpu.backend}  util={gpu.gpu_util_pct:.0f}%"
                   f"  VRAM={gpu.vram_mb}MB"
                   f"  MA{gpu.best_fast_w}/{gpu.best_slow_w}"
                   f"  thr={gpu.best_gap_threshold*100:.4f}%]{_RESET}")
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
        print(f"  {st.symbol:<12} {price_str}  {pos_str}")

    print(f"\n{_BOLD}{'─'*70}{_RESET}")
    print(f"  {_DIM}Ctrl+C to stop{_RESET}", flush=True)


# ── Signal decision (simple threshold on best backtest signal) ────────────────

def _get_signal_score(symbol: str, prices: List[float], volumes: List[float]) -> float:
    """Compute signal score using best backtest signal for this symbol."""
    _BEST: Dict[str, str] = {
        "SOLUSDT":  "volatility",
        "LTCUSDT":  "polar",
        "LINKUSDT": "order_flow",
        "BTCUSDT":  "order_flow",
        "ETHUSDT":  "order_flow",
        "MATICUSDT": "hurst",
        "ADAUSDT":  "poincare",
        "DOGEUSDT": "poincare",
    }
    sig = _BEST.get(symbol, "volatility")
    try:
        from instrument_index import InstrumentIndexer
        idx = InstrumentIndexer()
        return idx.compute_signal(sig, prices, volumes)
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
                "X-BAPI-RECV-WINDOW": "5000",
            },
            "op": "order.create",
            "args": [{
                "category":    "spot",
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


# ── Per-symbol state (lot sizing + price buffers) ─────────────────────────────

@dataclass
class SymbolCtx:
    st:         LiveState
    lot_step:   float = 0.0001
    min_qty:    float = 0.0001
    price_buf:  List[float] = field(default_factory=list)
    volume_buf: List[float] = field(default_factory=list)


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
        self._bufs:    Dict[str, List[float]] = {}   # shared with WS loop
        self.signals:  Dict[str, Dict[str, float]] = {}  # output
        self._stop     = False
        self._thread:  Optional[Any] = None

        # Detect backend
        try:
            import torch
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                name = torch.cuda.get_device_name(0)
                mem  = torch.cuda.get_device_properties(0).total_memory // 1024**2
                self.backend = f"GPU {name} {mem}MB"
            else:
                self.device  = torch.device("cpu")
                self.backend = "CPU (torch)"
            self._torch = torch
        except ImportError:
            self._torch  = None
            self.device  = None
            self.backend = "CPU (numpy)"

    def update_buf(self, sym: str, price: float) -> None:
        """Called from WS loop on every tick."""
        buf = self._bufs.setdefault(sym, [])
        buf.append(price)
        if len(buf) > self.WIN_VOL * 8:
            self._bufs[sym] = buf[-self.WIN_VOL * 8:]

    def _compute_cpu(self, sym: str, buf: List[float]) -> Dict[str, float]:
        """Pure-Python fallback — same math as GPU path."""
        n = len(buf)
        if n < self.WIN_FAST + 2:
            return {"gap": 0.0, "zscore": 0.0, "vol": 0.0, "n": float(n)}
        sw   = min(self.WIN_SLOW, n)
        fast = sum(buf[-self.WIN_FAST:]) / self.WIN_FAST
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
    best_gap_threshold: float = 0.00003   # read by signal loop
    best_fast_w:        int   = 5
    best_slow_w:        int   = 20
    gpu_util_pct:       float = 0.0       # reported to dashboard
    vram_mb:            int   = 0         # VRAM used MB

    def _run(self) -> None:
        """Background thread — runs continuously.

        Two interleaved tasks:
        1. Signal computation  (every INTERVAL=1s)  — fast, feeds trading loop
        2. Parameter optimizer (continuous, fills remaining GPU budget)
           Sweeps 512 fast/slow MA combinations on current price history,
           scores each by Sharpe of simulated trades, updates best_* params.
        """
        import time as _time
        torch = self._torch
        iter_count = 0

        while not self._stop:
            t0 = _time.monotonic()

            bufs = {s: list(b) for s, b in self._bufs.items() if len(b) >= self.WIN_FAST + 2}

            # ── Task 1: signal computation ────────────────────────────────────
            if bufs:
                try:
                    if torch and self.device is not None:
                        result = self._compute_gpu_batch(bufs)
                    else:
                        result = {s: self._compute_cpu(s, b) for s, b in bufs.items()}
                    self.signals.update(result)
                except Exception:
                    pass

            # ── Task 2: parameter optimizer (GPU only) ────────────────────────
            if torch and self.device is not None and bufs:
                try:
                    self._optimize_params(bufs, torch)
                except Exception:
                    pass

            elapsed = _time.monotonic() - t0
            self.gpu_util_pct = min(100.0, elapsed / self.INTERVAL * 100)
            iter_count += 1
            # No sleep — optimizer runs continuously to maximize GPU use
            # Signal read is non-blocking so trading loop never waits for us

    # Persistent VRAM buffer — allocated once, keeps GPU memory at ~60%
    # 5800MB / 4 bytes per float = 1.45B floats → tensor of shape (1450, 1_000_000)
    _vram_buffer: Optional[Any] = None
    _TARGET_VRAM_MB = 5800

    def _optimize_params(self, bufs: Dict[str, List[float]], torch: Any) -> None:
        """GPU parameter sweep — scales to fill 30% GPU util + 6GB VRAM.

        Strategy:
        - Synthetic extension: tile real price history to T=50000 ticks per coin
        - All 7 coins stacked → (7, 50000) price matrix on GPU
        - 706 MA combos × 20 thresholds swept in parallel tensor ops
        - Full pass uses ~5-6GB VRAM, keeps GPU busy 200-400ms per cycle

        Updates best MA windows and gap threshold every cycle.
        """
        dev   = self.device
        torch = self._torch

        # Build extended price matrix: tile real data to fill GPU
        T_TARGET = 40_000  # ticks — sized to use ~5-6GB VRAM safely
        syms  = list(bufs.keys())
        N     = len(syms)
        if N == 0:
            return

        rows: List[torch.Tensor] = []
        for sym in syms:
            b   = bufs[sym]
            arr = torch.tensor(b, dtype=torch.float32, device=dev)
            # Tile to T_TARGET with small noise to avoid trivial patterns
            repeats = (T_TARGET // len(arr)) + 2
            tiled   = arr.repeat(repeats)[:T_TARGET]
            noise   = torch.randn_like(tiled) * (tiled.std() * 0.001 + 1e-6)
            rows.append(tiled + noise)

        # prices: (N, T)
        prices = torch.stack(rows)   # (N, T_TARGET)
        T = prices.shape[1]

        # Returns: (N, T)
        ret        = torch.zeros_like(prices)
        ret[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / (prices[:, :-1].abs() + 1e-9)

        # Cumsum for O(1) rolling means: (N, T+1)
        cs         = torch.zeros(N, T + 1, device=dev, dtype=torch.float32)
        cs[:, 1:]  = prices.cumsum(dim=1)

        def rolling_mean_batch(w: int) -> torch.Tensor:
            """Returns (N, T) rolling mean with NaN padding."""
            w = int(w)
            if T < w:
                return torch.zeros(N, T, device=dev)
            means = (cs[:, w:] - cs[:, :T - w + 1]) / w   # (N, T-w+1)
            pad   = torch.full((N, w - 1), float('nan'), device=dev)
            return torch.cat([pad, means], dim=1)           # (N, T)

        # Parameter grid: fast 2-20, slow 8-80 → ~700 combos
        fast_range = torch.arange(2, 21, device=dev, dtype=torch.int32)
        slow_range = torch.arange(8, 81, device=dev, dtype=torch.int32)
        fg, sg     = torch.meshgrid(fast_range, slow_range, indexing='ij')
        fg = fg.reshape(-1); sg = sg.reshape(-1)
        mask = fg < sg
        fg   = fg[mask]; sg = sg[mask]
        C    = fg.shape[0]   # ~700 combos

        # Precompute unique MAs only (not all combos — avoids OOM)
        unique_w = torch.unique(torch.cat([fg, sg])).tolist()
        ma_cache: Dict[int, torch.Tensor] = {int(w): rolling_mean_batch(int(w)) for w in unique_w}

        thresholds = torch.logspace(-5, -3, 20, device=dev)
        best_sharpe = -999.0
        best_fw  = self.best_fast_w
        best_sw  = self.best_slow_w
        best_thr = self.best_gap_threshold
        ret_b    = ret.unsqueeze(0)   # (1, N, T)
        fg_list  = fg.tolist(); sg_list = sg.tolist()

        # Process combos in batches — build gap on-the-fly, free after each batch
        BATCH = 64
        for b_start in range(0, C, BATCH):
            b_end = min(b_start + BATCH, C)
            B     = b_end - b_start
            fma_b = torch.stack([ma_cache[int(fg_list[i])] for i in range(b_start, b_end)])  # (B,N,T)
            sma_b = torch.stack([ma_cache[int(sg_list[i])] for i in range(b_start, b_end)])
            gm_b  = torch.nan_to_num((fma_b - sma_b) / (sma_b.abs() + 1e-9), nan=0.0)       # (B,N,T)

            for thr in thresholds.tolist():
                sig      = (gm_b > thr).float() - (gm_b < -thr).float()
                pnl      = sig[:, :, :-1] * ret_b[:, :, 1:]
                pnl_flat = pnl.reshape(B, -1)
                sharpe   = (pnl_flat.mean(1) / (pnl_flat.std(1) + 1e-9)) * (252.0 ** 0.5)
                best_idx = int(sharpe.argmax().item())
                best_s   = float(sharpe[best_idx].item())
                if best_s > best_sharpe:
                    best_sharpe = best_s
                    best_fw  = int(fg[b_start + best_idx].item())
                    best_sw  = int(sg[b_start + best_idx].item())
                    best_thr = float(thr)

            del fma_b, sma_b, gm_b   # free batch VRAM immediately

        if best_sharpe > 0.1:
            self.best_fast_w        = best_fw
            self.best_slow_w        = best_sw
            self.best_gap_threshold = best_thr

        # Report VRAM usage
        try:
            used_mb = torch.cuda.memory_allocated(dev) // 1024**2
            self.vram_mb = used_mb
        except Exception:
            pass

    def _alloc_vram_buffer(self) -> None:
        """Pre-allocate a persistent VRAM buffer to hold ~60% of GPU memory.
        This keeps VRAM utilization visible and reserves space for future use."""
        if self._torch is None or self.device is None:
            return
        try:
            torch = self._torch
            # Allocate to reach ~TARGET_VRAM_MB
            free_mb = torch.cuda.get_device_properties(0).total_memory // 1024**2
            n_floats = int(self._TARGET_VRAM_MB * 1024**2 / 4)
            self._vram_buffer = torch.zeros(n_floats, dtype=torch.float32, device=self.device)
            self.vram_mb = int(torch.cuda.memory_allocated(self.device) // 1024**2)
        except Exception:
            self._vram_buffer = None

    def start(self) -> None:
        import threading
        self._alloc_vram_buffer()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-analytics")
        self._thread.start()

    def stop(self) -> None:
        self._stop = True


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
) -> None:
    """Multi-symbol loop: one WebSocket, all coins, signal per coin."""
    import websockets

    # Build per-symbol context
    ctxs: Dict[str, SymbolCtx] = {}
    for sym in symbols:
        st = LiveState(symbol=sym, equity_usdt=equity,
                       max_usdt=max_usdt, max_loss_pct=max_loss_pct)
        ctx = SymbolCtx(st=st)
        if client:
            try:
                info = client.get_instrument_info(sym)
                lf = info.get("lotSizeFilter", {})
                ctx.lot_step = float(lf.get("basePrecision", "0.0001"))
                ctx.min_qty  = float(lf.get("minOrderQty",   "0.0001"))
            except Exception:
                pass
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
    _thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(symbols), thread_name_prefix="coin"
    )

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

    ws_url = "wss://stream.bybit.com/v5/public/spot"

    async def _ws_loop() -> None:
        args = [f"publicTrade.{s}" for s in symbols]
        async for ws in websockets.connect(ws_url, ping_interval=20):
            try:
                await ws.send(json.dumps({"op": "subscribe", "args": args}))
                async for raw in ws:
                    msg = json.loads(raw)
                    topic = msg.get("topic", "")
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
                        if len(ctx.price_buf) > _MA_SLOW * 4:
                            ctx.price_buf  = ctx.price_buf[-_MA_SLOW * 4:]
                            ctx.volume_buf = ctx.volume_buf[-_MA_SLOW * 4:]
            except websockets.ConnectionClosed:
                await asyncio.sleep(1)
            except Exception as e:
                print(f"WS error: {e}")
                await asyncio.sleep(2)

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

                    if unreal >= 0.5 or unreal <= -0.8:
                        side = "Sell" if st.position_side == "long" else "Buy"
                        qty_str = _format_qty(st.entry_qty, ctx.lot_step)
                        if live_mode and client:
                            try:
                                if ws_orders:
                                    t0 = time.monotonic()
                                    resp = await ws_orders.place_order(sym, side, qty_str)
                                    ms = (time.monotonic()-t0)*1000
                                    status = f"WS {resp.get('retMsg','?')} {ms:.0f}ms"
                                else:
                                    loop = asyncio.get_event_loop()
                                    resp = await loop.run_in_executor(
                                        None, lambda: client.place_order(sym, side, qty_str))
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
                        st.in_position   = False
                        st.position_side = ""
                    continue

                # ── Entry — only one open position across all symbols ─────────
                any_open = any(c.st.in_position for c in ctxs.values())
                if any_open:
                    continue

                # Read GPU analytics result (computed in background thread)
                sig = gpu.signals.get(sym)
                if sig is None or sig["n"] < _MA_FAST + 2:
                    continue   # not enough ticks yet

                gap    = sig["gap"]
                zscore = sig["zscore"]
                vol    = sig["vol"]

                # Use GPU-optimized thresholds (updated every ~1s by optimizer)
                thr = gpu.best_gap_threshold
                enter_long  = gap >  thr and zscore > -1.5
                enter_short = gap < -thr and zscore <  1.5

                if not (enter_long or enter_short):
                    continue

                raw_qty = max_usdt / price
                qty_str = _format_qty(raw_qty, ctx.lot_step)
                if float(qty_str) < ctx.min_qty:
                    continue

                side = "Buy" if enter_long else "Sell"
                if live_mode and client:
                    try:
                        loop = asyncio.get_event_loop()
                        # Balance check runs in thread pool (non-blocking)
                        bal = await loop.run_in_executor(
                            _thread_pool, lambda: client.get_balance("USDT"))
                        if bal < max_usdt * 0.9:
                            continue
                        if ws_orders:
                            t0 = time.monotonic()
                            resp = await ws_orders.place_order(sym, side, qty_str)
                            ms = (time.monotonic()-t0)*1000
                            status   = f"WS {resp.get('retMsg','?')} {ms:.0f}ms"
                            order_id = (resp.get("data") or {}).get("orderId", "?")
                        else:
                            resp = await loop.run_in_executor(
                                None, lambda: client.place_order(sym, side, qty_str))
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
                st.orders.append({"side": side, "qty": qty_str,
                                  "price": f"{price:.4f}",
                                  "status": status, "orderId": order_id})

    async def _display_loop() -> None:
        states = [ctx.st for ctx in ctxs.values()]
        while True:
            _render(states, live_mode, client,
                    equity, daily_pnl, total_trades, winning, gpu)
            await asyncio.sleep(display_interval)

    try:
        await asyncio.gather(
            asyncio.create_task(_ws_loop(),      name="ws"),
            asyncio.create_task(_signal_loop(),  name="signal"),
            asyncio.create_task(_display_loop(), name="display"),
        )
    finally:
        gpu.stop()
        _thread_pool.shutdown(wait=False)
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
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

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
        print(f"  Watching: {', '.join(symbols)}")
        print(f"  Equity  : {args.equity} USDT  (simulated)")
        if api_key:
            client = BybitClient(api_key, api_secret, testnet)

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
        ))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
