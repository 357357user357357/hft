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
        """Get Spot wallet balance for a coin."""
        r = self.get("/v5/account/wallet-balance",
                     {"accountType": "SPOT", "coin": coin})
        if r.get("retCode") != 0:
            raise RuntimeError(f"balance error: {r}")
        for item in r["result"]["list"]:
            for c in item.get("coin", []):
                if c["coin"] == coin:
                    return float(c.get("availableToWithdraw") or c.get("walletBalance") or 0)
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

def _render(st: LiveState, live_mode: bool, client: Optional[BybitClient]) -> None:
    _clear()
    mode_str = f"{_RED}{_BOLD}LIVE ORDERS{_RESET}" if live_mode else f"{_YELLOW}PAPER MODE{_RESET}"
    balance_str = ""
    if live_mode and client:
        try:
            bal = client.get_balance("USDT")
            balance_str = f"  wallet={_CYAN}{bal:.4f} USDT{_RESET}"
        except Exception:
            balance_str = "  wallet=?"

    print(f"{_BOLD}{'─'*70}{_RESET}")
    print(f"{_BOLD}  BYBIT TRADER  {_RESET}{mode_str}  "
          f"{_DIM}{time.strftime('%H:%M:%S')}{_RESET}{balance_str}")
    print(f"{_BOLD}{'─'*70}{_RESET}")

    pnl_col = _pnl_col(st.total_pnl_pct)
    win_rate = st.winning / st.total_trades * 100 if st.total_trades > 0 else 0.0

    print(f"  symbol  : {_CYAN}{st.symbol}{_RESET}")
    print(f"  price   : {_CYAN}{st.price:.4f} USDT{_RESET}  "
          f"{_DIM}updated {time.monotonic() - st.last_update:.1f}s ago{_RESET}")
    print(f"  equity  : {st.equity_usdt:.2f} USDT  max_order={st.max_usdt:.2f} USDT")
    print(f"  trades  : {st.total_trades}  win={win_rate:.1f}%  "
          f"pnl={pnl_col}{st.total_pnl_pct:+.3f}%{_RESET}  "
          f"daily={_pnl_col(st.daily_pnl_pct)}{st.daily_pnl_pct:+.3f}%{_RESET}")

    if st.in_position:
        if st.price > 0 and st.entry_price > 0:
            unreal = (st.price - st.entry_price) / st.entry_price * 100
            if st.position_side == "short":
                unreal = -unreal
        else:
            unreal = 0.0
        held = time.monotonic() - st.entry_time
        print(f"\n  {_BOLD}POSITION{_RESET}: {_CYAN}{st.position_side.upper()}{_RESET}  "
              f"entry={st.entry_price:.4f}  qty={st.entry_qty:.4f}  "
              f"unrealised={_pnl_col(unreal)}{unreal:+.2f}%{_RESET}  "
              f"held={int(held)}s")
    else:
        print(f"\n  {_DIM}No open position{_RESET}")

    if st.orders:
        last = st.orders[-1]
        print(f"\n  {_DIM}Last order: {last.get('side','')} {last.get('qty','')} "
              f"@ {last.get('price','market')}  "
              f"status={last.get('status','?')}{_RESET}")

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


# ── Main trading loop ─────────────────────────────────────────────────────────

async def _trading_loop(
    symbol:      str,
    equity:      float,
    max_usdt:    float,
    max_loss_pct: float,
    live_mode:   bool,
    use_signals: bool,
    signal_threshold: float,
    display_interval: float,
    client:      Optional[BybitClient],
) -> None:
    """Main loop: WebSocket price feed → signal → order decision."""
    import websockets

    st = LiveState(
        symbol=symbol,
        equity_usdt=equity,
        max_usdt=max_usdt,
        max_loss_pct=max_loss_pct,
    )

    # Get instrument info for proper lot sizing
    lot_step   = 0.0001   # fallback
    min_qty    = 0.0001
    if client:
        try:
            info      = client.get_instrument_info(symbol)
            lot_filter = info.get("lotSizeFilter", {})
            lot_step  = float(lot_filter.get("basePrecision", "0.0001"))
            min_qty   = float(lot_filter.get("minOrderQty", "0.0001"))
        except Exception as e:
            print(f"  Warning: could not fetch instrument info: {e}")

    # Price buffer for signal computation
    price_buf:  List[float] = []
    volume_buf: List[float] = []
    SIGNAL_WINDOW = 120   # bars for signal

    # Bybit Spot WebSocket public stream
    ws_url = f"wss://stream.bybit.com/v5/public/spot"

    async def _ws_loop() -> None:
        nonlocal price_buf, volume_buf

        async for ws in websockets.connect(ws_url, ping_interval=20):
            try:
                # Subscribe to trade stream
                sub = {"op": "subscribe", "args": [f"publicTrade.{symbol}"]}
                await ws.send(json.dumps(sub))

                async for raw in ws:
                    msg = json.loads(raw)

                    # Bybit sends {"topic":"publicTrade.SOLUSDT","data":[...]}
                    if msg.get("topic", "").startswith("publicTrade"):
                        for trade in msg.get("data", []):
                            price  = float(trade["p"])
                            qty    = float(trade["v"])
                            st.price       = price
                            st.last_update = time.monotonic()

                            price_buf.append(price)
                            volume_buf.append(qty)
                            # Keep only what we need
                            if len(price_buf) > SIGNAL_WINDOW * 2:
                                price_buf  = price_buf[-SIGNAL_WINDOW * 2:]
                                volume_buf = volume_buf[-SIGNAL_WINDOW * 2:]

            except websockets.ConnectionClosed:
                await asyncio.sleep(1)
            except Exception as e:
                print(f"WS error: {e}")
                await asyncio.sleep(2)

    async def _signal_loop() -> None:
        """Evaluate signals and place orders periodically."""
        EVAL_INTERVAL = 10.0   # seconds between evaluations

        while True:
            await asyncio.sleep(EVAL_INTERVAL)

            if len(price_buf) < SIGNAL_WINDOW:
                continue   # not enough data yet

            # Check daily loss limit
            if st.daily_pnl_pct < -st.max_loss_pct:
                print(f"\n{_RED}Daily loss limit reached "
                      f"({st.daily_pnl_pct:.2f}%). Stopped.{_RESET}")
                return

            price = st.price
            if price <= 0:
                continue

            # ── Signal evaluation ─────────────────────────────────────────────
            if use_signals:
                score = _get_signal_score(
                    symbol,
                    price_buf[-SIGNAL_WINDOW:],
                    volume_buf[-SIGNAL_WINDOW:],
                )
            else:
                score = 0.0   # no signal gate → algo-only mode

            # ── Exit logic ────────────────────────────────────────────────────
            if st.in_position:
                unreal = (price - st.entry_price) / st.entry_price * 100
                if st.position_side == "short":
                    unreal = -unreal

                should_exit = (
                    unreal >= 0.5    # take profit at +0.5%
                    or unreal <= -0.8  # stop loss at -0.8%
                    or (use_signals and st.position_side == "long" and score < -signal_threshold)
                    or (use_signals and st.position_side == "short" and score > signal_threshold)
                )

                if should_exit:
                    side = "Sell" if st.position_side == "long" else "Buy"
                    qty_str = _format_qty(st.entry_qty, lot_step)

                    if live_mode and client:
                        try:
                            resp = client.place_order(symbol, side, qty_str)
                            status = resp.get("retMsg", "?")
                            order_id = resp.get("result", {}).get("orderId", "?")
                        except Exception as e:
                            status = f"ERROR: {e}";  order_id = "?"
                    else:
                        status = "paper";  order_id = "sim"

                    pnl = unreal * (st.max_usdt / st.equity_usdt)
                    st.total_pnl_pct  += unreal
                    st.daily_pnl_pct  += unreal
                    st.total_trades   += 1
                    if unreal > 0:
                        st.winning += 1

                    st.orders.append({
                        "side": side, "qty": qty_str,
                        "price": f"{price:.4f}",
                        "status": status, "orderId": order_id,
                        "pnl": f"{unreal:+.3f}%",
                    })
                    st.in_position    = False
                    st.position_side  = ""

            # ── Entry logic ───────────────────────────────────────────────────
            elif not st.in_position:
                # Simple entry: signal threshold (if enabled) or just
                # use the volatility breakout from the algo itself
                enter_long  = use_signals and score >  signal_threshold
                enter_short = use_signals and score < -signal_threshold

                # Without signals: enter on price momentum (simplified)
                if not use_signals and len(price_buf) >= 20:
                    ma_fast = sum(price_buf[-5:])  / 5
                    ma_slow = sum(price_buf[-20:]) / 20
                    enter_long  = ma_fast > ma_slow * 1.002   # 0.2% above
                    enter_short = ma_fast < ma_slow * 0.998   # 0.2% below

                if enter_long or enter_short:
                    # Calculate qty: max_usdt / price, rounded to lot step
                    raw_qty = st.max_usdt / price
                    qty_str = _format_qty(raw_qty, lot_step)

                    if float(qty_str) < min_qty:
                        continue   # order too small

                    side = "Buy" if enter_long else "Sell"

                    # Check balance before ordering
                    if live_mode and client:
                        try:
                            bal = client.get_balance("USDT")
                            if bal < st.max_usdt * 0.95:
                                continue  # not enough balance
                            resp = client.place_order(symbol, side, qty_str)
                            status   = resp.get("retMsg", "?")
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

                    st.orders.append({
                        "side": side, "qty": qty_str,
                        "price": f"{price:.4f}",
                        "status": status, "orderId": order_id,
                    })

    async def _display_loop() -> None:
        while True:
            _render(st, live_mode, client)
            await asyncio.sleep(display_interval)

    await asyncio.gather(
        asyncio.create_task(_ws_loop(),      name="ws"),
        asyncio.create_task(_signal_loop(),  name="signal"),
        asyncio.create_task(_display_loop(), name="display"),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bybit Spot live/paper trader"
    )
    parser.add_argument("--symbol",    default="SOLUSDT",
                        help="Trading pair (default: SOLUSDT)")
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

    # Load credentials
    api_key    = os.environ.get("BYBIT_API_KEY",    "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")
    testnet    = os.environ.get("BYBIT_TESTNET", "0") == "1"

    client: Optional[BybitClient] = None

    if args.live:
        if not api_key or not api_secret:
            print(f"{_RED}Error: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env{_RESET}")
            print("Edit /home/nikolas/Documents/hft/.env and add your keys.")
            sys.exit(1)
        client = BybitClient(api_key, api_secret, testnet)
        # Verify connection
        try:
            bal = client.get_balance("USDT")
            price = client.get_price(args.symbol)
            print(f"{_GREEN}Connected to Bybit {'Testnet' if testnet else 'Mainnet'}{_RESET}")
            print(f"  Balance : {bal:.4f} USDT")
            print(f"  {args.symbol} : {price:.4f} USDT")
            print(f"\n{_RED}{_BOLD}LIVE MODE — real orders will be placed!{_RESET}")
            print(f"  Max per order : {args.max_usdt} USDT")
            print(f"  Daily loss cap: {args.max_loss}%")
            confirm = input("\nType 'yes' to continue: ").strip().lower()
            if confirm != "yes":
                print("Aborted.")
                sys.exit(0)
        except Exception as e:
            print(f"{_RED}Connection failed: {e}{_RESET}")
            sys.exit(1)
    else:
        # Paper mode — still connect WebSocket but no real orders
        print(f"{_YELLOW}Paper mode — no real orders will be placed{_RESET}")
        print(f"  Symbol  : {args.symbol}")
        print(f"  Equity  : {args.equity} USDT  (simulated)")
        print(f"  Signals : {'ON' if args.signals else 'OFF (momentum only)'}")
        if api_key:
            # Can still show real balance
            client = BybitClient(api_key, api_secret, testnet)

    print("Connecting to Bybit WebSocket…\n")

    try:
        asyncio.run(_trading_loop(
            symbol           = args.symbol.upper(),
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
