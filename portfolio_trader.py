#!/usr/bin/env python3
"""Portfolio Paper Trader — trades multiple symbols simultaneously.

Runs SOLUSDT, LTCUSDT, LINKUSDT (best performers from backtest) in parallel,
each with its own Binance Futures WebSocket + full 19-dim signal scorecard.
Best signals per instrument (from backtest):
  SOLUSDT  → Composite / Volatility / Whitehead Torsion
  LTCUSDT  → Polar Coords / Volatility / Frenet-Serret
  LINKUSDT → Order Flow / Simons SDEs / Spectral-Hecke

Pair arbitrage mode (--pairs):
  Trades statistical spread between cointegrated pairs in parallel with the
  single-instrument strategies.  The Kalman filter hedge-ratio tracker is
  Rust-accelerated (ricci_rs).  Each pair has its own WebSocket tasks for
  both legs and a combined spread display.

Usage:
    python portfolio_trader.py                          # default 3 symbols
    python portfolio_trader.py --symbols SOLUSDT LTCUSDT
    python portfolio_trader.py --equity 30000 --interval 5
    python portfolio_trader.py --no-signals             # faster, algo-only
    python portfolio_trader.py --pairs                  # add pair arb mode
    python portfolio_trader.py --pairs --pair-zscore 2.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque

sys.path.insert(0, os.path.dirname(__file__))

from pair_arb import PairArb, PairConfig, PairSignal, CANDIDATE_PAIRS
from live_trader import (
    _futures_ws_url, _spot_ws_url, _parse_agg_trade,
    _build_algos, AlgoBundle,
    _RESET, _BOLD, _GREEN, _RED, _YELLOW, _CYAN, _WHITE, _DIM,
    _pnl_color, _clear,
)
from signal_gate import SignalGate, GateConfig
from slippage import SlippageModel, SlippageConfig
from instrument_index import InstrumentIndexer

# Best instruments from backtest (ordered by total PnL)
DEFAULT_SYMBOLS = ["SOLUSDT", "LTCUSDT", "LINKUSDT"]

# Primary signal highlight per instrument (from backtest top-3)
_BEST_SIGNAL: Dict[str, str] = {
    "SOLUSDT":  "volatility",
    "LTCUSDT":  "polar",
    "LINKUSDT": "order_flow",
    "BNBUSDT":  "order_flow",
    "BTCUSDT":  "order_flow",
    "ETHUSDT":  "order_flow",
    "XRPUSDT":  "simons",
    "ADAUSDT":  "hurst",
    "DOGEUSDT": "torsion",
}


# ── Per-symbol state ──────────────────────────────────────────────────────────

@dataclass
class SymbolState:
    symbol:       str
    algos:        AlgoBundle
    gate:         Optional[SignalGate]
    price:        float = 0.0
    trade_count:  int   = 0
    start_time:   float = field(default_factory=time.monotonic)
    connected:    bool  = False
    error:        str   = ""


# ── Combined dashboard ────────────────────────────────────────────────────────

def _render_portfolio(states: Dict[str, SymbolState], equity_per_symbol: float) -> None:
    _clear()
    now = time.monotonic()
    print(f"{_BOLD}{'─'*90}{_RESET}")
    print(f"{_BOLD}  PORTFOLIO PAPER TRADER   "
          f"{_DIM}{time.strftime('%H:%M:%S')}{_RESET}   "
          f"equity/symbol: ${equity_per_symbol:,.0f}   "
          f"symbols: {len(states)}")
    print(f"{_BOLD}{'─'*90}{_RESET}")

    total_pnl = 0.0
    total_trades = 0

    for sym, st in states.items():
        uptime = now - st.start_time
        sym_pnl  = st.algos.total_pnl()
        sym_trades = st.algos.total_trades()
        total_pnl    += sym_pnl
        total_trades += sym_trades
        pnl_col = _pnl_color(sym_pnl)

        status = f"{_GREEN}LIVE{_RESET}" if st.connected else f"{_RED}connecting…{_RESET}"
        print(
            f"  {_BOLD}{sym:<10}{_RESET}  "
            f"price={_CYAN}{st.price:>12.4f}{_RESET}  "
            f"trades={sym_trades:>5}  "
            f"pnl={pnl_col}{sym_pnl:>+8.3f}%{_RESET}  "
            f"uptime={int(uptime//60):02d}:{int(uptime%60):02d}  "
            f"{status}"
        )

        # Signal gate score if available
        if st.gate and st.gate.last_scorecard:
            sc = st.gate.last_scorecard
            best_sig = _BEST_SIGNAL.get(sym, "simons")
            try:
                from instrument_index import InstrumentIndexer
                sig_score = InstrumentIndexer._extract_score(sc, best_sig)
            except Exception:
                sig_score = 0.0
            sig_col = _GREEN if sig_score > 0.1 else _RED if sig_score < -0.1 else _WHITE
            print(
                f"    {_DIM}signals:{_RESET}  "
                f"order_flow={_pnl_color(sc.order_flow.score)}{sc.order_flow.score:+.3f}{_RESET}  "
                f"simons={_pnl_color(sc.simons.score)}{sc.simons.score:+.3f}{_RESET}  "
                f"hurst={_pnl_color(sc.hurst.score)}{sc.hurst.score:+.3f}{_RESET}  "
                f"{best_sig}={sig_col}{sig_score:+.3f}{_RESET}  "
                f"regime={_YELLOW}{sc.order_flow.regime if hasattr(sc.order_flow,'regime') else '?'}{_RESET}"
            )

        # Top algo P&L
        rows = st.algos.rows()
        best_algo = max(rows, key=lambda r: r[1].total_pnl_pct)
        worst_algo = min(rows, key=lambda r: r[1].total_pnl_pct)
        print(
            f"    {_DIM}algos:{_RESET}  "
            f"best={_GREEN}{best_algo[0]}{_RESET} {_GREEN}{best_algo[1].total_pnl_pct:+.3f}%{_RESET}  "
            f"worst={_RED}{worst_algo[0]}{_RESET} {_RED}{worst_algo[1].total_pnl_pct:+.3f}%{_RESET}"
        )
        print()

    print(f"{'─'*90}")
    pnl_col = _pnl_color(total_pnl)
    print(
        f"  {_BOLD}TOTAL{_RESET}  "
        f"trades={total_trades}  "
        f"pnl={pnl_col}{_BOLD}{total_pnl:+.3f}%{_RESET}  "
        f"across all {len(states)} symbols"
    )
    print(f"{'─'*90}")
    print(f"{_DIM}  Press Ctrl+C to stop{_RESET}", flush=True)


# ── Per-symbol async worker ───────────────────────────────────────────────────

async def _run_symbol(
    state: SymbolState,
    futures: bool,
    signal_interval: float,
    equity: float,
) -> None:
    import websockets

    url = _futures_ws_url(state.symbol) if futures else _spot_ws_url(state.symbol)

    async for ws in websockets.connect(url, ping_interval=20, ping_timeout=10):
        try:
            state.connected = True
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("e") != "aggTrade":
                    continue
                trade = _parse_agg_trade(msg)
                state.price = trade.price
                state.trade_count += 1
                state.algos.on_trade(trade)
        except websockets.ConnectionClosed:
            state.connected = False
            await asyncio.sleep(1)
        except Exception as e:
            state.connected = False
            state.error = str(e)
            await asyncio.sleep(2)


# ── Display loop (runs on main task, refreshes all symbols) ──────────────────

async def _display_loop(
    states: Dict[str, SymbolState],
    equity_per_symbol: float,
    display_interval: float,
) -> None:
    while True:
        _render_portfolio(states, equity_per_symbol)
        await asyncio.sleep(display_interval)


# ── Main ──────────────────────────────────────────────────────────────────────


# ── Pair arbitrage live state ─────────────────────────────────────────────────

@dataclass
class PairState:
    """Live state for one actively traded pair."""
    arb:        PairArb
    leg_a:      str
    leg_b:      str
    price_a:    float = 0.0
    price_b:    float = 0.0
    last_signal: str   = "hold"
    trade_count: int   = 0
    pnl_pct:    float  = 0.0
    entry_spread: float = 0.0
    entry_price_a: float = 0.0
    in_position: bool  = False
    position_side: str = ""


async def _run_pair_price_feed(
    pair_state: PairState,
    symbol_states: Dict[str, "SymbolState"],
    equity_per_pair: float,
    futures: bool,
) -> None:
    """Feeds live prices into PairArb engine from existing symbol WebSocket streams.

    Rather than opening duplicate WebSocket connections, we poll the prices
    already collected by the per-symbol tasks.  This avoids extra connections
    and keeps the pair state in sync with the algo state.
    """
    ps  = pair_state
    arb = ps.arb
    cfg = arb.config

    while True:
        # Poll prices from the main symbol states (already streaming)
        sym_a = symbol_states.get(cfg.leg_a)
        sym_b = symbol_states.get(cfg.leg_b)

        if sym_a and sym_b and sym_a.price > 0 and sym_b.price > 0:
            pa = sym_a.price
            pb = sym_b.price
            ps.price_a = pa
            ps.price_b = pb

            prev_in_pos = arb.state.in_position
            sig = arb.on_prices(pa, pb)

            if sig != PairSignal.Hold:
                ps.last_signal = sig.value
                ps.trade_count += 1

                sp = arb.state.spread_raw
                normaliser = abs(pa) if abs(pa) > 1e-6 else 1.0

                if sig in (PairSignal.LongSpread, PairSignal.ShortSpread):
                    ps.in_position   = True
                    ps.position_side = sig.value
                    ps.entry_spread  = sp
                    ps.entry_price_a = pa

                elif sig == PairSignal.Close and prev_in_pos:
                    # Calculate realised P&L on close
                    if ps.position_side == PairSignal.LongSpread.value:
                        raw = (sp - ps.entry_spread) / normaliser * 100
                    else:
                        raw = (ps.entry_spread - sp) / normaliser * 100
                    # Deduct fees (4 legs × maker fee ≈ 4 × 4bps)
                    ps.pnl_pct    += raw - 0.016
                    ps.in_position = False
                    ps.position_side = ""

        await asyncio.sleep(0.5)   # check twice per second


def _render_pairs(pair_states: List[PairState]) -> None:
    """Print pair arb section of dashboard."""
    if not pair_states:
        return
    print(f"  {_DIM}{'─'*86}{_RESET}")
    print(f"  {_BOLD}PAIR ARB{_RESET}")
    for ps in pair_states:
        st  = ps.arb.state
        col = _GREEN if ps.pnl_pct > 0 else _RED if ps.pnl_pct < 0 else _WHITE
        pos_str = (
            f"{_GREEN}LONG {_RESET}" if ps.position_side == "long_spread" else
            f"{_RED}SHORT{_RESET}"   if ps.position_side == "short_spread" else
            f"{_DIM}     {_RESET}"
        )
        coint_col = _GREEN if st.coint_ok else _YELLOW
        print(
            f"  {_CYAN}{ps.leg_a}/{ps.leg_b:<8}{_RESET}  "
            f"β={st.beta:+.3f}  "
            f"z={st.zscore:>+5.2f}  "
            f"pos={pos_str}  "
            f"trades={ps.trade_count:>3d}  "
            f"pnl={col}{ps.pnl_pct:>+7.3f}%{_RESET}  "
            f"coint={coint_col}{'OK' if st.coint_ok else '??'}{_RESET}"
        )


async def _run_portfolio(
    symbols: List[str],
    equity_total: float,
    use_signals: bool,
    signal_interval: float,
    futures: bool,
    display_interval: float,
    run_pairs: bool = False,
    pair_zscore: float = 2.0,
) -> None:
    equity_per = equity_total / len(symbols)
    slippage = SlippageModel(SlippageConfig(enabled=False))

    states: Dict[str, SymbolState] = {}
    for sym in symbols:
        gate = None
        if use_signals:
            gate = SignalGate(sym, GateConfig(
                eval_interval_secs=signal_interval,
                window_size=500,
                min_prices=60,
                hard_block=False,
                regime_enabled=True,
                regime_interval_secs=30.0,
            ))
        algos = _build_algos(sym, equity_per, gate, slippage)
        states[sym] = SymbolState(symbol=sym, algos=algos, gate=gate)

    # ── Pair arbitrage setup ──────────────────────────────────────────────────
    pair_states: List[PairState] = []
    if run_pairs:
        equity_per_pair = equity_total * 0.1   # 10% of total equity per pair leg
        for leg_a, leg_b, _ in CANDIDATE_PAIRS:
            # Only create pair if at least one leg is already in our symbol set
            # (we'll subscribe to both legs via the main symbol streams)
            pcfg = PairConfig(
                leg_a=leg_a, leg_b=leg_b,
                zscore_entry=pair_zscore,
                zscore_exit=0.3,
                min_warmup_bars=100,
                zscore_window=60,
            )
            ps = PairState(arb=PairArb(pcfg), leg_a=leg_a, leg_b=leg_b)
            pair_states.append(ps)

        # Ensure all pair legs are also subscribed to WebSocket streams
        all_pair_symbols = {leg for ps in pair_states for leg in (ps.leg_a, ps.leg_b)}
        for sym in all_pair_symbols:
            if sym not in states:
                gate = None
                if use_signals:
                    gate = SignalGate(sym, GateConfig(
                        eval_interval_secs=signal_interval,
                        window_size=500,
                        min_prices=60,
                        hard_block=False,
                        regime_enabled=True,
                        regime_interval_secs=30.0,
                    ))
                algos = _build_algos(sym, equity_per_pair, gate, slippage)
                states[sym] = SymbolState(symbol=sym, algos=algos, gate=gate)

    # ── Patch display to include pairs ────────────────────────────────────────
    _original_render = _render_portfolio

    async def _display_loop_with_pairs(
        st: Dict[str, SymbolState],
        eqs: float,
        interval: float,
    ) -> None:
        while True:
            _render_portfolio(st, eqs)
            _render_pairs(pair_states)
            await asyncio.sleep(interval)

    # Launch one WebSocket task per symbol + pair feed tasks + display task
    tasks = [
        asyncio.create_task(
            _run_symbol(states[sym], futures, signal_interval, equity_per),
            name=f"ws-{sym}",
        )
        for sym in states
    ]
    if run_pairs and pair_states:
        for ps in pair_states:
            tasks.append(asyncio.create_task(
                _run_pair_price_feed(ps, states, equity_total * 0.1, futures),
                name=f"pair-{ps.leg_a}-{ps.leg_b}",
            ))

    display_fn = _display_loop_with_pairs if run_pairs else _display_loop
    tasks.append(asyncio.create_task(
        display_fn(states, equity_per, display_interval),
        name="display",
    ))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio paper trader — multiple symbols in parallel"
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help=f"Symbols to trade (default: {' '.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--equity",  type=float, default=30_000.0,
                        help="Total paper equity split across symbols (default: 30000)")
    parser.add_argument("--no-signals", action="store_true",
                        help="Disable signal scorecard (faster)")
    parser.add_argument("--interval",  type=float, default=5.0,
                        help="Signal re-eval interval secs (default: 5)")
    parser.add_argument("--display",   type=float, default=1.0,
                        help="Dashboard refresh interval secs (default: 1)")
    parser.add_argument("--spot",       action="store_true",
                        help="Use Spot instead of Futures WebSocket")
    parser.add_argument("--pairs",      action="store_true",
                        help="Enable pair arbitrage mode alongside single-leg trading")
    parser.add_argument("--pair-zscore", type=float, default=2.0,
                        help="Z-score entry threshold for pair arb (default: 2.0)")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols]
    equity_per = args.equity / len(symbols)

    print(f"{_BOLD}Portfolio Paper Trader{_RESET}")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Equity  : ${args.equity:,.0f} total  (${equity_per:,.0f}/symbol)")
    print(f"  Signals : {'ON' if not args.no_signals else 'OFF'}")
    print(f"  Market  : {'Futures' if not args.spot else 'Spot'}")
    if args.pairs:
        active_pairs = [(a, b) for a, b, _ in CANDIDATE_PAIRS
                        if a in symbols or b in symbols]
        print(f"  Pairs   : {len(active_pairs)} pairs active (z={args.pair_zscore})")
    print("Connecting…\n")

    try:
        asyncio.run(_run_portfolio(
            symbols=symbols,
            equity_total=args.equity,
            use_signals=not args.no_signals,
            signal_interval=args.interval,
            futures=not args.spot,
            display_interval=args.display,
            run_pairs=args.pairs,
            pair_zscore=args.pair_zscore,
        ))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
